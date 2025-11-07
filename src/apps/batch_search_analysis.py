"""
Batch Search Analysis Page - Question Ranges
Permite analizar el proceso de búsqueda y reranking para rangos de preguntas
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client
from src.config.config import CHROMADB_COLLECTION_CONFIG, EMBEDDING_MODELS
from src.apps.search_utils import (
    normalize_url, load_crossencoder, apply_crossencoder_reranking,
    calculate_retrieval_metrics, calculate_ranx_metrics, search_documents,
    create_metrics_comparison_plot, get_question_embedding, RANX_AVAILABLE,
    expand_query
)


def show_batch_search_analysis_page():
    """Main page function for batch question analysis"""

    st.title("📊 Análisis por Lotes - Rangos de Preguntas")
    st.markdown("""
    Analiza el proceso de búsqueda vectorial y reranking para **múltiples preguntas**.

    Las métricas se promedian sobre todas las preguntas del rango.

    **🚀 Mejoras Implementadas:**
    - ✅ **Multi-stage retrieval:** Recupera más candidatos antes de reranking
    - ✅ **CrossEncoder mejorado:** `ms-marco-electra-base` (mejor precisión)
    - ✅ **Query expansion:** Expansión de consultas con terminología Azure

    **Formato de entrada:**
    - **Pregunta única:** `25`
    - **Rango:** `20-30` o `100-150`
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuración")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=list(EMBEDDING_MODELS.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Mejoras de Rendimiento")

    # Multi-stage retrieval configuration
    use_multistage = st.sidebar.checkbox(
        "Multi-stage retrieval",
        value=True,
        help="Recupera más documentos antes de reranking para mejorar recall"
    )

    if use_multistage:
        retrieval_k = st.sidebar.slider(
            "Candidatos a recuperar:",
            min_value=15,
            max_value=100,
            value=15,
            step=5,
            help="Número de documentos a recuperar antes de reranking (15=reranking simple, 30-50=multi-stage para mayor recall)"
        )
    else:
        retrieval_k = 15

    # Query expansion configuration
    use_query_expansion = st.sidebar.checkbox(
        "Query expansion",
        value=True,
        help="Expande consultas con terminología Azure para mejorar reranking"
    )

    if use_query_expansion:
        max_expansions = st.sidebar.slider(
            "Sinónimos por término:",
            min_value=1,
            max_value=5,
            value=2,
            help="Número máximo de sinónimos a agregar por término"
        )

    # CrossEncoder model selection
    crossencoder_model = st.sidebar.selectbox(
        "Modelo CrossEncoder:",
        options=[
            'cross-encoder/ms-marco-electra-base',
            'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        ],
        index=0,
        help="Modelo para reranking (electra-base es mejor pero más lento)"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Métricas")

    # Top-k selection for final results
    top_k = st.sidebar.slider(
        "Top-K documentos finales:",
        min_value=5,
        max_value=20,
        value=15,
        help="Documentos a retornar después de reranking"
    )

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
        all_questions = questions_collection.get(
            include=['documents', 'metadatas']
        )
        num_questions = len(all_questions['ids'])
        st.sidebar.info(f"📊 {num_questions} preguntas validadas disponibles")
    except Exception as e:
        st.error(f"❌ Error cargando preguntas: {e}")
        return

    # Question range selection
    st.header("1️⃣ Selección de Rango de Preguntas")

    question_range_input = st.text_input(
        "Índice o rango de preguntas:",
        value="0-10",
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
    if is_range:
        st.info(f"📊 Analizando {len(question_indices)} preguntas (índices {question_indices[0]} a {question_indices[-1]})")
        st.caption("Se mostrarán métricas promedio agregadas")
    else:
        st.info(f"📊 Analizando 1 pregunta (índice {question_indices[0]})")

    # Search button
    if st.button("🚀 Ejecutar Análisis por Lotes", type="primary", use_container_width=True):

        # Get collection names
        collection_config = CHROMADB_COLLECTION_CONFIG.get(model_name, {})
        questions_collection_name = collection_config.get('questions', '')
        docs_collection_name = collection_config.get('documents', '')

        if not questions_collection_name or not docs_collection_name:
            st.error(f"❌ No se encontró colección para el modelo {model_name}")
            return

        # Initialize accumulators
        all_metrics_before = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_metrics_before['mrr'] = []
        all_metrics_after = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_metrics_after['mrr'] = []

        all_ranx_metrics_before = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_ranx_metrics_before['mrr'] = []
        all_ranx_metrics_after = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in k_values}
        all_ranx_metrics_after['mrr'] = []

        # Get model questions collection
        model_questions_collection = client.get_collection(questions_collection_name)

        # Process each question in the range
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, q_idx in enumerate(question_indices):
            progress_bar.progress((idx + 1) / len(question_indices))
            status_text.text(f"Procesando pregunta {idx + 1}/{len(question_indices)} (índice {q_idx})...")

            # Get question data
            selected_question = all_questions['documents'][q_idx]
            metadata = all_questions['metadatas'][q_idx]
            validated_links = metadata.get('validated_links', [])
            if isinstance(validated_links, str):
                try:
                    validated_links = json.loads(validated_links)
                except:
                    validated_links = [validated_links]

            # Get embedding
            question_url = metadata.get('url', '')
            question_embedding = get_question_embedding(
                client, model_questions_collection, question_url, questions_collection_name
            )

            if question_embedding is None:
                st.warning(f"⚠️ Pregunta {q_idx} no encontrada en colección del modelo, omitida")
                continue

            # Apply query expansion if enabled
            search_question = selected_question
            if use_query_expansion:
                search_question = expand_query(selected_question, max_expansions=max_expansions)

            # Search documents (multi-stage retrieval)
            retrieved_docs = search_documents(client, docs_collection_name, question_embedding, retrieval_k)

            if not retrieved_docs:
                st.warning(f"⚠️ No se encontraron documentos para pregunta {q_idx}, omitida")
                continue

            # Calculate metrics before
            metrics_before = calculate_retrieval_metrics(
                validated_links, retrieved_docs, k_values, score_key='cosine_similarity'
            )
            for key, value in metrics_before.items():
                all_metrics_before[key].append(value)

            if RANX_AVAILABLE:
                ranx_metrics_before = calculate_ranx_metrics(
                    validated_links, retrieved_docs, k_values,
                    query_id=f"before_{q_idx}", use_crossencoder=False
                )
                for key, value in ranx_metrics_before.items():
                    all_ranx_metrics_before[key].append(value)

            # Apply CrossEncoder reranking
            cross_encoder = load_crossencoder(crossencoder_model)
            # Use expanded question for better reranking
            reranked_docs = apply_crossencoder_reranking(search_question, retrieved_docs.copy(), cross_encoder)

            # Calculate metrics after
            metrics_after = calculate_retrieval_metrics(
                validated_links, reranked_docs, k_values, score_key='crossencoder_score'
            )
            for key, value in metrics_after.items():
                all_metrics_after[key].append(value)

            if RANX_AVAILABLE:
                ranx_metrics_after = calculate_ranx_metrics(
                    validated_links, reranked_docs, k_values,
                    query_id=f"after_{q_idx}", use_crossencoder=True
                )
                for key, value in ranx_metrics_after.items():
                    all_ranx_metrics_after[key].append(value)

        progress_bar.empty()
        status_text.empty()

        # Calculate averages
        avg_metrics_before = {key: np.mean(values) if values else 0.0 for key, values in all_metrics_before.items()}
        avg_metrics_after = {key: np.mean(values) if values else 0.0 for key, values in all_metrics_after.items()}
        avg_ranx_metrics_before = {key: np.mean(values) if values else 0.0 for key, values in all_ranx_metrics_before.items()}
        avg_ranx_metrics_after = {key: np.mean(values) if values else 0.0 for key, values in all_ranx_metrics_after.items()}

        # Show results
        st.success(f"✅ Análisis completado para {len(all_metrics_before['mrr'])} preguntas")

        st.header("2️⃣ Métricas Promedio: Antes vs Después del Reranking")

        if RANX_AVAILABLE and all_ranx_metrics_after['mrr']:
            st.subheader("📊 Comparación: Implementación Manual vs Ranx (Promedios)")

            comparison_data = []
            for k in k_values:
                for metric_name in ['precision', 'recall', 'f1', 'ndcg', 'map']:
                    manual_before = float(avg_metrics_before.get(f'{metric_name}@{k}', 0))
                    manual_after = float(avg_metrics_after.get(f'{metric_name}@{k}', 0))
                    ranx_before = float(avg_ranx_metrics_before.get(f'{metric_name}@{k}', 0))
                    ranx_after = float(avg_ranx_metrics_after.get(f'{metric_name}@{k}', 0))

                    comparison_data.append({
                        'k': str(k),
                        'Métrica': metric_name.capitalize(),
                        'Manual Antes': f"{manual_before:.4f}",
                        'Manual Después': f"{manual_after:.4f}",
                        'Δ Manual': f"{manual_after - manual_before:+.4f}",
                        'Ranx Antes': f"{ranx_before:.4f}",
                        'Ranx Después': f"{ranx_after:.4f}",
                        'Δ Ranx': f"{ranx_after - ranx_before:+.4f}",
                    })

            # Add MRR
            manual_mrr_before = float(avg_metrics_before.get('mrr', 0))
            manual_mrr_after = float(avg_metrics_after.get('mrr', 0))
            ranx_mrr_before = float(avg_ranx_metrics_before.get('mrr', 0))
            ranx_mrr_after = float(avg_ranx_metrics_after.get('mrr', 0))

            comparison_data.append({
                'k': '-',
                'Métrica': 'MRR',
                'Manual Antes': f"{manual_mrr_before:.4f}",
                'Manual Después': f"{manual_mrr_after:.4f}",
                'Δ Manual': f"{manual_mrr_after - manual_mrr_before:+.4f}",
                'Ranx Antes': f"{ranx_mrr_before:.4f}",
                'Ranx Después': f"{ranx_mrr_after:.4f}",
                'Δ Ranx': f"{ranx_mrr_after - ranx_mrr_before:+.4f}",
            })

            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # Gráficas con rango completo k=1 a 15
        st.header("3️⃣ Gráficas Comparativas (k=1 a 15)")

        full_k_range = list(range(1, 16))

        with st.spinner("Calculando métricas promedio para k=1 a 15..."):
            # Calculate metrics for full range
            all_metrics_before_full = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in full_k_range}
            all_metrics_before_full['mrr'] = []
            all_metrics_after_full = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in full_k_range}
            all_metrics_after_full['mrr'] = []

            all_ranx_metrics_before_full = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in full_k_range}
            all_ranx_metrics_before_full['mrr'] = []
            all_ranx_metrics_after_full = {f'{metric}@{k}': [] for metric in ['precision', 'recall', 'f1', 'ndcg', 'map'] for k in full_k_range}
            all_ranx_metrics_after_full['mrr'] = []

            # Re-calculate for full k range (using cached embeddings and docs from previous loop would be complex)
            # For simplicity, recalculate (could be optimized later)
            for idx, q_idx in enumerate(question_indices):
                selected_question = all_questions['documents'][q_idx]
                metadata = all_questions['metadatas'][q_idx]
                validated_links = metadata.get('validated_links', [])
                if isinstance(validated_links, str):
                    try:
                        validated_links = json.loads(validated_links)
                    except:
                        validated_links = [validated_links]

                question_url = metadata.get('url', '')
                question_embedding = get_question_embedding(
                    client, model_questions_collection, question_url, questions_collection_name
                )

                if question_embedding is None:
                    continue

                # Apply query expansion if enabled
                search_question = selected_question
                if use_query_expansion:
                    search_question = expand_query(selected_question, max_expansions=max_expansions)

                # Multi-stage retrieval
                retrieved_docs = search_documents(client, docs_collection_name, question_embedding, retrieval_k)

                if not retrieved_docs:
                    continue

                metrics_before_full = calculate_retrieval_metrics(
                    validated_links, retrieved_docs, full_k_range, score_key='cosine_similarity'
                )
                for key, value in metrics_before_full.items():
                    all_metrics_before_full[key].append(value)

                if RANX_AVAILABLE:
                    ranx_metrics_before_full = calculate_ranx_metrics(
                        validated_links, retrieved_docs, full_k_range,
                        query_id=f"before_full_{q_idx}", use_crossencoder=False
                    )
                    for key, value in ranx_metrics_before_full.items():
                        all_ranx_metrics_before_full[key].append(value)

                cross_encoder = load_crossencoder(crossencoder_model)
                # Use expanded question
                reranked_docs = apply_crossencoder_reranking(search_question, retrieved_docs.copy(), cross_encoder)

                metrics_after_full = calculate_retrieval_metrics(
                    validated_links, reranked_docs, full_k_range, score_key='crossencoder_score'
                )
                for key, value in metrics_after_full.items():
                    all_metrics_after_full[key].append(value)

                if RANX_AVAILABLE:
                    ranx_metrics_after_full = calculate_ranx_metrics(
                        validated_links, reranked_docs, full_k_range,
                        query_id=f"after_full_{q_idx}", use_crossencoder=True
                    )
                    for key, value in ranx_metrics_after_full.items():
                        all_ranx_metrics_after_full[key].append(value)

            # Calculate averages
            avg_metrics_before_full = {key: np.mean(values) if values else 0.0 for key, values in all_metrics_before_full.items()}
            avg_metrics_after_full = {key: np.mean(values) if values else 0.0 for key, values in all_metrics_after_full.items()}
            avg_ranx_metrics_before_full = {key: np.mean(values) if values else 0.0 for key, values in all_ranx_metrics_before_full.items()}
            avg_ranx_metrics_after_full = {key: np.mean(values) if values else 0.0 for key, values in all_ranx_metrics_after_full.items()}

        if avg_ranx_metrics_before_full and avg_ranx_metrics_after_full:
            fig_combined = create_metrics_comparison_plot(
                avg_metrics_before_full, avg_metrics_after_full,
                avg_ranx_metrics_before_full, avg_ranx_metrics_after_full,
                full_k_range
            )
            st.pyplot(fig_combined)
            plt.close(fig_combined)
        else:
            st.warning("⚠️ Ranx no disponible - mostrando solo métricas manuales")


if __name__ == "__main__":
    show_batch_search_analysis_page()
