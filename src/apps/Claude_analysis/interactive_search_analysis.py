"""
Interactive Search Analysis Page
Permite analizar el proceso de búsqueda y reranking de forma interactiva
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
from urllib.parse import urlparse, urlunparse
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client
from src.config.config import CHROMADB_COLLECTION_CONFIG, EMBEDDING_MODELS
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ranx import Qrels, Run, evaluate
    RANX_AVAILABLE = True
except ImportError:
    RANX_AVAILABLE = False


def normalize_url(url: str) -> str:
    """Normalize URL by removing query parameters and fragments"""
    if not url or not url.strip():
        return ""
    try:
        parsed = urlparse(url.strip())
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        return normalized
    except:
        return url.strip()


@st.cache_resource
def load_crossencoder():
    """Load CrossEncoder model once - using ms-marco-MiniLM-L-6-v2"""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def create_metrics_comparison_plot(metrics_manual_before: Dict, metrics_manual_after: Dict,
                                   metrics_ranx_before: Dict, metrics_ranx_after: Dict,
                                   k_values: List[int]):
    """Create line plots comparing metrics: Manual vs Ranx, Before vs After"""

    # Metrics to plot
    metric_names = ['precision', 'recall', 'f1', 'ndcg', 'map']

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Define colors and markers for 4 lines
    styles = [
        {'label': 'Manual Antes', 'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'alpha': 0.8},
        {'label': 'Manual Después', 'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'alpha': 0.8},
        {'label': 'Ranx Antes', 'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'alpha': 0.7},
        {'label': 'Ranx Después', 'color': '#d62728', 'marker': 'v', 'linestyle': '--', 'alpha': 0.7},
    ]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        # Extract values for this metric
        manual_before_values = [float(metrics_manual_before.get(f'{metric}@{k}', 0)) for k in k_values]
        manual_after_values = [float(metrics_manual_after.get(f'{metric}@{k}', 0)) for k in k_values]
        ranx_before_values = [float(metrics_ranx_before.get(f'{metric}@{k}', 0)) for k in k_values]
        ranx_after_values = [float(metrics_ranx_after.get(f'{metric}@{k}', 0)) for k in k_values]

        # Plot 4 lines
        ax.plot(k_values, manual_before_values, marker=styles[0]['marker'], label=styles[0]['label'],
                color=styles[0]['color'], linestyle=styles[0]['linestyle'], linewidth=2, markersize=7, alpha=styles[0]['alpha'])
        ax.plot(k_values, manual_after_values, marker=styles[1]['marker'], label=styles[1]['label'],
                color=styles[1]['color'], linestyle=styles[1]['linestyle'], linewidth=2, markersize=7, alpha=styles[1]['alpha'])
        ax.plot(k_values, ranx_before_values, marker=styles[2]['marker'], label=styles[2]['label'],
                color=styles[2]['color'], linestyle=styles[2]['linestyle'], linewidth=2, markersize=7, alpha=styles[2]['alpha'])
        ax.plot(k_values, ranx_after_values, marker=styles[3]['marker'], label=styles[3]['label'],
                color=styles[3]['color'], linestyle=styles[3]['linestyle'], linewidth=2, markersize=7, alpha=styles[3]['alpha'])

        # Styling
        ax.set_xlabel('Top-K', fontsize=11)
        ax.set_ylabel('Valor', fontsize=11)
        ax.set_title(f'{metric.upper()}@k', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values[::2] if len(k_values) > 8 else k_values)  # Show every other tick if too many

    # MRR (single value, shown as bar with 4 bars)
    ax = axes[5]
    manual_mrr_before = float(metrics_manual_before.get('mrr', 0))
    manual_mrr_after = float(metrics_manual_after.get('mrr', 0))
    ranx_mrr_before = float(metrics_ranx_before.get('mrr', 0))
    ranx_mrr_after = float(metrics_ranx_after.get('mrr', 0))

    x_pos = [0, 1, 2.5, 3.5]
    values = [manual_mrr_before, manual_mrr_after, ranx_mrr_before, ranx_mrr_after]
    colors = [styles[0]['color'], styles[1]['color'], styles[2]['color'], styles[3]['color']]
    labels = ['Manual\nAntes', 'Manual\nDespués', 'Ranx\nAntes', 'Ranx\nDespués']

    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, width=0.8)
    ax.set_ylabel('Valor', fontsize=11)
    ax.set_title('MRR', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(x_pos[i], v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def calculate_retrieval_metrics(ground_truth_links: List[str], retrieved_docs: List[Dict],
                                k_values: List[int]) -> Dict:
    """Calculate precision, recall, F1, MAP, MRR, NDCG for different k values"""

    # Normalize ground truth links
    gt_normalized = {normalize_url(link) for link in ground_truth_links if link}

    if not gt_normalized:
        return {}

    # Extract retrieved links
    retrieved_links = [normalize_url(doc.get('link', '')) for doc in retrieved_docs]

    metrics = {}

    for k in k_values:
        top_k_links = set(retrieved_links[:k])

        # True positives
        tp = len(top_k_links & gt_normalized)

        # Precision@k
        precision = tp / k if k > 0 else 0.0

        # Recall@k
        recall = tp / len(gt_normalized) if len(gt_normalized) > 0 else 0.0

        # F1@k
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'f1@{k}'] = f1

        # NDCG@k
        dcg = 0.0
        idcg = 0.0
        for i, link in enumerate(retrieved_links[:k], 1):
            rel = 1.0 if normalize_url(link) in gt_normalized else 0.0
            dcg += rel / np.log2(i + 1)

        # Ideal DCG (all relevant docs at top)
        for i in range(1, min(k, len(gt_normalized)) + 1):
            idcg += 1.0 / np.log2(i + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f'ndcg@{k}'] = ndcg

        # MAP@k
        sum_precisions = 0.0
        relevant_count = 0
        for i, link in enumerate(retrieved_links[:k], 1):
            if normalize_url(link) in gt_normalized:
                relevant_count += 1
                sum_precisions += relevant_count / i

        map_k = sum_precisions / len(gt_normalized) if len(gt_normalized) > 0 else 0.0
        metrics[f'map@{k}'] = map_k

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, link in enumerate(retrieved_links, 1):
        if normalize_url(link) in gt_normalized:
            mrr = 1.0 / i
            break
    metrics['mrr'] = mrr

    return metrics


def calculate_ranx_metrics(ground_truth_links: List[str], retrieved_docs: List[Dict],
                           k_values: List[int], query_id: str = "q1", use_crossencoder: bool = False) -> Dict:
    """Calculate metrics using ranx library

    Args:
        ground_truth_links: List of relevant document URLs
        retrieved_docs: List of retrieved documents with scores
        k_values: List of k values to calculate metrics for
        query_id: Query identifier for ranx
        use_crossencoder: If True, use crossencoder_score; if False, use cosine_similarity
    """

    if not RANX_AVAILABLE:
        return {}

    # Normalize ground truth links
    gt_normalized = {normalize_url(link) for link in ground_truth_links if link}

    if not gt_normalized:
        return {}

    # Create qrels (ground truth) - binary relevance
    qrels_dict = {query_id: {}}
    for link in gt_normalized:
        qrels_dict[query_id][link] = 1

    qrels = Qrels(qrels_dict)

    # Create run (retrieved results with scores)
    run_dict = {query_id: {}}
    for doc in retrieved_docs:
        link = normalize_url(doc.get('link', ''))
        # Use the appropriate score based on the flag
        if use_crossencoder:
            score = doc.get('crossencoder_score', 0.0)
        else:
            score = doc.get('cosine_similarity', 0.0)
        run_dict[query_id][link] = float(score)

    run = Run(run_dict)

    # Define metrics to evaluate
    metrics_to_eval = ['mrr']
    for k in k_values:
        metrics_to_eval.extend([f'precision@{k}', f'recall@{k}', f'f1@{k}',
                                f'ndcg@{k}', f'map@{k}'])

    # Evaluate
    try:
        results = evaluate(qrels, run, metrics_to_eval)
        # Convert all values to native Python float for compatibility
        converted_results = {}
        for key, value in results.items():
            if isinstance(value, (np.ndarray, np.generic)):
                converted_results[key] = float(value)
            else:
                converted_results[key] = float(value) if value is not None else 0.0
        return converted_results
    except Exception as e:
        st.warning(f"⚠️ Error calculando métricas con ranx: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {}


def search_documents(client, collection_name: str, query_embedding: np.ndarray,
                     top_k: int = 15) -> List[Dict]:
    """Search documents using cosine similarity"""
    try:
        collection = client.get_collection(collection_name)

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        documents = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                doc = {
                    'rank': i + 1,
                    'link': results['metadatas'][0][i].get('link', ''),
                    'title': results['metadatas'][0][i].get('title', ''),
                    'content': results['documents'][0][i] if results['documents'] else '',
                    'cosine_similarity': 1 - results['distances'][0][i],  # ChromaDB returns distance
                    'original_rank': i + 1
                }
                documents.append(doc)

        return documents

    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return []


def get_optimal_batch_size(documents: List[Dict], max_content_length: int = 1200) -> int:
    """
    Calculate optimal batch size based on document length
    Same logic as Colab for consistency
    """
    if not documents:
        return 32

    avg_length = sum(len(doc.get('content', '')[:max_content_length]) for doc in documents) / len(documents)

    if avg_length > 1000:
        return 16
    elif avg_length > 500:
        return 32
    else:
        return 64


def apply_crossencoder_reranking(question: str, documents: List[Dict],
                                 cross_encoder: CrossEncoder) -> List[Dict]:
    """
    Apply CrossEncoder reranking to documents
    IDENTICAL to Colab implementation for consistency and determinism

    Mejoras:
    - Usa título + contenido (no solo contenido)
    - Batch size adaptativo según longitud de documentos
    - Ordenamiento determinístico con tie-breaking
    """
    if not documents:
        return []

    try:
        # MEJORA 1: Prepare query-document pairs WITH TITLE + CONTENT
        pairs = []
        for doc in documents:
            title = doc.get('title', '').strip()
            content = doc.get('content', '').strip()

            # Combinar título y contenido limitado
            if title:
                # Título + primeros 1200 caracteres de contenido
                combined_text = f"{title}: {content[:1200]}"
            else:
                # Solo contenido limitado si no hay título
                combined_text = content[:1500]

            pairs.append([question, combined_text])

        # MEJORA 2: Batch size adaptativo según longitud de documentos
        optimal_batch_size = get_optimal_batch_size(documents, max_content_length=1200)

        # Get CrossEncoder scores con batch size optimizado
        scores = cross_encoder.predict(pairs, batch_size=optimal_batch_size)

        # Apply Min-Max normalization to convert logits to [0, 1] range
        scores = np.array(scores)
        if len(scores) > 1 and scores.max() != scores.min():
            # Standard Min-Max normalization
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            # Fallback for edge cases (all scores identical)
            normalized_scores = np.full_like(scores, 0.5)

        # Add scores to documents
        for i, doc in enumerate(documents):
            doc['crossencoder_score'] = float(normalized_scores[i])

        # DETERMINISTIC: Sort by CrossEncoder score desc, then by link asc (tie-breaking)
        reranked_docs = sorted(
            documents,
            key=lambda x: (-x['crossencoder_score'], x.get('link', ''))
        )

        # Update ranks
        for i, doc in enumerate(reranked_docs, 1):
            doc['rank'] = i

        return reranked_docs

    except Exception as e:
        st.warning(f"⚠️ CrossEncoder reranking error: {e}")
        return documents


def show_interactive_search_analysis_page():
    """Main page function"""

    st.title("🔍 Análisis Interactivo de Búsqueda y Reranking")
    st.markdown("""
    Esta página permite analizar el proceso completo de búsqueda vectorial y reranking de forma interactiva.
    Puedes ver cómo cambian las métricas antes y después del CrossEncoder.

    **CrossEncoder utilizado:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

    **Mejoras implementadas (idénticas al Colab):**
    - ✅ Usa **título + contenido** para reranking (no solo contenido)
    - ✅ Batch size adaptativo según longitud de documentos
    - ✅ Ordenamiento determinístico con tie-breaking por URL
    - ✅ Normalización Min-Max de scores CrossEncoder

    **Métricas calculadas:** Precision@k, Recall@k, F1@k, NDCG@k, MAP@k, MRR

    **Implementaciones comparadas:** Cálculo Manual vs Ranx
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuración")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=list(EMBEDDING_MODELS.keys()),
        index=0
    )

    # Top-k selection
    top_k = st.sidebar.slider("Top-K documentos:", 5, 20, 15)

    # K values for metrics
    k_values = st.sidebar.multiselect(
        "Valores de k para métricas:",
        options=[1, 3, 5, 10, 15],
        default=[3, 5, 10, 15]
    )

    # Initialize ChromaDB
    try:
        config = ChromaDBConfig.from_env()
        client = get_chromadb_client(config)
        st.sidebar.success("✅ Conectado a ChromaDB")
    except Exception as e:
        st.error(f"❌ Error conectando a ChromaDB: {e}")
        return

    # Load questions with validated links
    try:
        questions_collection = client.get_collection("questions_withlinks")

        # Get all questions (without embeddings initially - too heavy)
        all_questions = questions_collection.get(
            include=['documents', 'metadatas']
        )

        num_questions = len(all_questions['ids'])
        st.sidebar.info(f"📊 {num_questions} preguntas con enlaces validados disponibles")

    except Exception as e:
        st.error(f"❌ Error cargando preguntas: {e}")
        return

    # Question selection
    st.header("1️⃣ Selección de Pregunta(s)")

    col1, col2 = st.columns([1, 3])

    with col1:
        question_range_input = st.text_input(
            "Índice o rango de preguntas:",
            value="0",
            help="Ingresa un número único (ej: 25) o un rango (ej: 20-30, 100-150)"
        )

    # Parse question range
    try:
        if '-' in question_range_input:
            # Range format: "20-30"
            parts = question_range_input.split('-')
            start_idx = int(parts[0].strip())
            end_idx = int(parts[1].strip())

            if start_idx < 0 or end_idx >= num_questions or start_idx > end_idx:
                st.error(f"❌ Rango inválido. Debe ser entre 0 y {num_questions - 1}, y start ≤ end")
                return

            question_indices = list(range(start_idx, end_idx + 1))
            is_range = True
        else:
            # Single question
            question_idx = int(question_range_input.strip())
            if question_idx < 0 or question_idx >= num_questions:
                st.error(f"❌ Índice inválido. Debe ser entre 0 y {num_questions - 1}")
                return

            question_indices = [question_idx]
            is_range = False
    except ValueError:
        st.error("❌ Formato inválido. Usa un número (ej: 25) o un rango (ej: 20-30)")
        return

    # Display info about selection
    with col2:
        if is_range:
            st.info(f"📊 Analizando {len(question_indices)} preguntas (índices {question_indices[0]} a {question_indices[-1]})")
            st.caption("Se mostrarán métricas promedio agregadas")
        else:
            question_idx = question_indices[0]
            selected_question = all_questions['documents'][question_idx]
            selected_question_id = all_questions['ids'][question_idx]
            metadata = all_questions['metadatas'][question_idx]
            validated_links = metadata.get('validated_links', [])
            if isinstance(validated_links, str):
                try:
                    validated_links = json.loads(validated_links)
                except:
                    validated_links = [validated_links]

            st.text_area("Pregunta seleccionada:", selected_question, height=100)
            st.info(f"🔗 Ground Truth: {len(validated_links)} enlaces validados")

            # Show ground truth links
            with st.expander("Ver enlaces de ground truth"):
                for link in validated_links:
                    st.markdown(f"- [{normalize_url(link)}]({link})")

    # Search button
    if st.button("🚀 Ejecutar Búsqueda y Análisis", type="primary", use_container_width=True):

        # Get collection names
        collection_config = CHROMADB_COLLECTION_CONFIG.get(model_name, {})
        questions_collection_name = collection_config.get('questions', '')
        docs_collection_name = collection_config.get('documents', '')

        if not questions_collection_name or not docs_collection_name:
            st.error(f"❌ No se encontró colección para el modelo {model_name}")
            return

        # Initialize accumulators for range analysis
        all_metrics_before = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_metrics_before['mrr'] = []
        all_metrics_after = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_metrics_after['mrr'] = []

        all_ranx_metrics_before = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_ranx_metrics_before['mrr'] = []
        all_ranx_metrics_after = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_ranx_metrics_after['mrr'] = []

        # For single question, keep original docs for display
        single_retrieved_docs = None
        single_reranked_docs = None
        single_validated_links = None

        # Process each question in the range
        progress_bar = st.progress(0)
        for idx, q_idx in enumerate(question_indices):
            progress_bar.progress((idx + 1) / len(question_indices))

            # Get question data
            selected_question = all_questions['documents'][q_idx]
            selected_question_id = all_questions['ids'][q_idx]
            metadata = all_questions['metadatas'][q_idx]
            validated_links = metadata.get('validated_links', [])
            if isinstance(validated_links, str):
                try:
                    validated_links = json.loads(validated_links)
                except:
                    validated_links = [validated_links]

                # Get the question from the model-specific collection
                # Match by 'url' metadata (both collections have it)
                model_questions_collection = client.get_collection(questions_collection_name)

                try:

                question_url = metadata.get('url', '')

                if not question_url:
                    st.error("❌ La pregunta no tiene URL en metadata")
                    return

                # Get all questions and find by URL match
                st.info(f"🔍 Buscando pregunta por URL en {questions_collection_name}...")
                all_model_questions = model_questions_collection.get(
                    include=['embeddings', 'documents', 'metadatas']
                )

                # Find the question by URL match
                question_embedding = None
                matched_text = None
                for i, meta in enumerate(all_model_questions['metadatas']):
                    if meta.get('url', '') == question_url:
                        question_embedding = np.array(all_model_questions['embeddings'][i])
                        matched_text = all_model_questions['documents'][i]
                        st.success(f"✅ Pregunta encontrada en posición {i} por URL match")
                        break

                if question_embedding is None:
                    st.error("❌ No se encontró la pregunta en la colección del modelo seleccionado")
                    st.info(f"💡 URL buscada: {question_url}")
                    st.info(f"💡 La pregunta existe en questions_withlinks pero no en {questions_collection_name}")
                    st.info("💡 Verifica que todas las colecciones tengan las mismas preguntas")
                    return

            except Exception as e:
                st.error(f"❌ Error obteniendo embedding: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        with st.spinner("Buscando documentos..."):
            # Search documents
            retrieved_docs = search_documents(
                client,
                docs_collection_name,
                question_embedding,
                top_k
            )

            if not retrieved_docs:
                st.warning("⚠️ No se encontraron documentos")
                return

        # Calculate metrics before
        metrics_before = calculate_retrieval_metrics(validated_links, retrieved_docs, k_values)

        # Calculate ranx metrics before (use cosine similarity)
        ranx_metrics_before = {}
        if RANX_AVAILABLE:
            ranx_metrics_before = calculate_ranx_metrics(validated_links, retrieved_docs, k_values,
                                                         query_id="before", use_crossencoder=False)

        # Apply CrossEncoder reranking
        with st.spinner("Aplicando CrossEncoder reranking..."):
            cross_encoder = load_crossencoder()
            reranked_docs = apply_crossencoder_reranking(
                selected_question,
                retrieved_docs.copy(),
                cross_encoder
            )

        # Show consolidated document ranking table
        st.header("2️⃣ Comparación de Rankings: Antes vs Después del CrossEncoder")

        # Debug: Show score verification
        with st.expander("🔍 Verificación de Scores (Debug)", expanded=False):
            st.write("**Primeros 3 documentos - Antes del Reranking:**")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                st.write(f"{i}. Cosine: {doc.get('cosine_similarity', 'N/A'):.4f}, CE: {doc.get('crossencoder_score', 'N/A')}")

            st.write("\n**Primeros 3 documentos - Después del Reranking:**")
            for i, doc in enumerate(reranked_docs[:3], 1):
                st.write(f"{i}. Cosine: {doc.get('cosine_similarity', 'N/A'):.4f}, CE: {doc.get('crossencoder_score', 'N/A'):.4f}")

        # Create consolidated table
        ranking_comparison = []
        gt_normalized = {normalize_url(link) for link in validated_links if link}

        # Use top-K documents (up to what user selected)
        for doc in reranked_docs[:top_k]:
            link = doc['link']
            normalized_link = normalize_url(link)
            is_relevant = normalized_link in gt_normalized

            # Find original rank (before reranking)
            original_rank = doc['original_rank']

            # Current rank (after reranking)
            new_rank = doc['rank']

            # Rank change
            rank_change = original_rank - new_rank
            if rank_change > 0:
                change_indicator = f"🔼 +{rank_change}"
            elif rank_change < 0:
                change_indicator = f"🔽 {rank_change}"
            else:
                change_indicator = "➡️ 0"

            # Extract URL path for better differentiation
            from urllib.parse import urlparse
            parsed_url = urlparse(link)
            url_path = parsed_url.path.split('/')[-1] if parsed_url.path else link

            ranking_comparison.append({
                'Relevante': '✅' if is_relevant else '❌',
                'Título': doc['title'][:50] + '...' if len(doc['title']) > 50 else doc['title'],
                'URL Path': url_path[:40] + '...' if len(url_path) > 40 else url_path,
                'Pre-CE': original_rank,
                'Post-CE': new_rank,
                'Cambio': change_indicator,
                'Cos': f"{doc['cosine_similarity']:.3f}",
                'CE': f"{doc['crossencoder_score']:.3f}",
                'URL Completo': link
            })

        df_ranking = pd.DataFrame(ranking_comparison)
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)

        # Show results AFTER reranking
        st.header("3️⃣ Métricas: Antes vs Después del Reranking")

        # Calculate metrics after
        metrics_after = calculate_retrieval_metrics(validated_links, reranked_docs, k_values)

        # Calculate ranx metrics after (use crossencoder score)
        ranx_metrics_after = {}
        if RANX_AVAILABLE:
            ranx_metrics_after = calculate_ranx_metrics(validated_links, reranked_docs, k_values,
                                                        query_id="after", use_crossencoder=True)

        # Display comparison: Custom vs Ranx
        if RANX_AVAILABLE and ranx_metrics_after:
            st.subheader("📊 Comparación: Implementación Manual vs Ranx (Después Reranking)")

            comparison_data = []
            for k in k_values:
                # Precision
                manual_prec_after = float(metrics_after.get(f'precision@{k}', 0))
                manual_prec_before = float(metrics_before.get(f'precision@{k}', 0))
                ranx_prec_after = float(ranx_metrics_after.get(f'precision@{k}', 0))
                ranx_prec_before = float(ranx_metrics_before.get(f'precision@{k}', 0))

                comparison_data.append({
                    'k': str(k),
                    'Métrica': 'Precision',
                    'Manual': f"{manual_prec_after:.4f}",
                    'Ranx': f"{ranx_prec_after:.4f}",
                    'Δ Manual': f"{manual_prec_after - manual_prec_before:+.4f}",
                    'Δ Ranx': f"{ranx_prec_after - ranx_prec_before:+.4f}",
                })

                # Recall
                manual_rec_after = float(metrics_after.get(f'recall@{k}', 0))
                manual_rec_before = float(metrics_before.get(f'recall@{k}', 0))
                ranx_rec_after = float(ranx_metrics_after.get(f'recall@{k}', 0))
                ranx_rec_before = float(ranx_metrics_before.get(f'recall@{k}', 0))

                comparison_data.append({
                    'k': str(k),
                    'Métrica': 'Recall',
                    'Manual': f"{manual_rec_after:.4f}",
                    'Ranx': f"{ranx_rec_after:.4f}",
                    'Δ Manual': f"{manual_rec_after - manual_rec_before:+.4f}",
                    'Δ Ranx': f"{ranx_rec_after - ranx_rec_before:+.4f}",
                })

                # F1
                manual_f1_after = float(metrics_after.get(f'f1@{k}', 0))
                manual_f1_before = float(metrics_before.get(f'f1@{k}', 0))
                ranx_f1_after = float(ranx_metrics_after.get(f'f1@{k}', 0))
                ranx_f1_before = float(ranx_metrics_before.get(f'f1@{k}', 0))

                comparison_data.append({
                    'k': str(k),
                    'Métrica': 'F1',
                    'Manual': f"{manual_f1_after:.4f}",
                    'Ranx': f"{ranx_f1_after:.4f}",
                    'Δ Manual': f"{manual_f1_after - manual_f1_before:+.4f}",
                    'Δ Ranx': f"{ranx_f1_after - ranx_f1_before:+.4f}",
                })

                # NDCG
                manual_ndcg_after = float(metrics_after.get(f'ndcg@{k}', 0))
                manual_ndcg_before = float(metrics_before.get(f'ndcg@{k}', 0))
                ranx_ndcg_after = float(ranx_metrics_after.get(f'ndcg@{k}', 0))
                ranx_ndcg_before = float(ranx_metrics_before.get(f'ndcg@{k}', 0))

                comparison_data.append({
                    'k': str(k),
                    'Métrica': 'NDCG',
                    'Manual': f"{manual_ndcg_after:.4f}",
                    'Ranx': f"{ranx_ndcg_after:.4f}",
                    'Δ Manual': f"{manual_ndcg_after - manual_ndcg_before:+.4f}",
                    'Δ Ranx': f"{ranx_ndcg_after - ranx_ndcg_before:+.4f}",
                })

                # MAP
                manual_map_after = float(metrics_after.get(f'map@{k}', 0))
                manual_map_before = float(metrics_before.get(f'map@{k}', 0))
                ranx_map_after = float(ranx_metrics_after.get(f'map@{k}', 0))
                ranx_map_before = float(ranx_metrics_before.get(f'map@{k}', 0))

                comparison_data.append({
                    'k': str(k),
                    'Métrica': 'MAP',
                    'Manual': f"{manual_map_after:.4f}",
                    'Ranx': f"{ranx_map_after:.4f}",
                    'Δ Manual': f"{manual_map_after - manual_map_before:+.4f}",
                    'Δ Ranx': f"{ranx_map_after - ranx_map_before:+.4f}",
                })

            # Add MRR
            manual_mrr_after = float(metrics_after.get('mrr', 0))
            manual_mrr_before = float(metrics_before.get('mrr', 0))
            ranx_mrr_after = float(ranx_metrics_after.get('mrr', 0))
            ranx_mrr_before = float(ranx_metrics_before.get('mrr', 0))

            comparison_data.append({
                'k': '-',
                'Métrica': 'MRR',
                'Manual': f"{manual_mrr_after:.4f}",
                'Ranx': f"{ranx_mrr_after:.4f}",
                'Δ Manual': f"{manual_mrr_after - manual_mrr_before:+.4f}",
                'Δ Ranx': f"{ranx_mrr_after - ranx_mrr_before:+.4f}",
            })

            df_comparison_after = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison_after, use_container_width=True, hide_index=True)

            # Add line plots comparing before and after
            st.subheader("📈 Gráficas Comparativas: Manual vs Ranx | Antes vs Después (k=1 a 15)")

            # Calculate metrics for full range k=1 to 15 for plots
            full_k_range = list(range(1, 16))

            with st.spinner("Calculando métricas para k=1 a 15..."):
                # Manual metrics for full range
                metrics_before_full = calculate_retrieval_metrics(validated_links, retrieved_docs, full_k_range)
                metrics_after_full = calculate_retrieval_metrics(validated_links, reranked_docs, full_k_range)

                # Ranx metrics for full range
                ranx_metrics_before_full = {}
                ranx_metrics_after_full = {}
                if RANX_AVAILABLE:
                    ranx_metrics_before_full = calculate_ranx_metrics(validated_links, retrieved_docs, full_k_range,
                                                                      query_id="before_full", use_crossencoder=False)
                    ranx_metrics_after_full = calculate_ranx_metrics(validated_links, reranked_docs, full_k_range,
                                                                     query_id="after_full", use_crossencoder=True)

            # Combined plot with all 4 lines
            if ranx_metrics_before_full and ranx_metrics_after_full:
                fig_combined = create_metrics_comparison_plot(
                    metrics_before_full,
                    metrics_after_full,
                    ranx_metrics_before_full,
                    ranx_metrics_after_full,
                    full_k_range
                )
                st.pyplot(fig_combined)
                plt.close(fig_combined)
            else:
                st.warning("⚠️ Ranx no disponible - mostrando solo métricas manuales")
                # Fallback: show manual only (duplicate for ranx)
                fig_manual_only = create_metrics_comparison_plot(
                    metrics_before_full,
                    metrics_after_full,
                    metrics_before_full,  # Use manual as fallback
                    metrics_after_full,   # Use manual as fallback
                    full_k_range
                )
                st.pyplot(fig_manual_only)
                plt.close(fig_manual_only)

        else:
            # Fallback to original display if ranx not available
            cols = st.columns(len(k_values))
            for i, k in enumerate(k_values):
                with cols[i]:
                    before_val = metrics_before.get(f'precision@{k}', 0)
                    after_val = metrics_after.get(f'precision@{k}', 0)
                    delta = after_val - before_val

                    st.metric(
                        f"Precision@{k}",
                        f"{after_val:.3f}",
                        f"{delta:+.3f}"
                    )

                    before_val = metrics_before.get(f'recall@{k}', 0)
                    after_val = metrics_after.get(f'recall@{k}', 0)
                    delta = after_val - before_val

                    st.metric(
                        f"Recall@{k}",
                        f"{after_val:.3f}",
                        f"{delta:+.3f}"
                    )

                    before_val = metrics_before.get(f'f1@{k}', 0)
                    after_val = metrics_after.get(f'f1@{k}', 0)
                    delta = after_val - before_val

                    st.metric(
                        f"F1@{k}",
                        f"{after_val:.3f}",
                        f"{delta:+.3f}"
                    )

            # Show MRR with delta
            mrr_before = metrics_before.get('mrr', 0)
            mrr_after = metrics_after.get('mrr', 0)
            st.metric("MRR", f"{mrr_after:.3f}", f"{mrr_after - mrr_before:+.3f}")


if __name__ == "__main__":
    show_interactive_search_analysis_page()
