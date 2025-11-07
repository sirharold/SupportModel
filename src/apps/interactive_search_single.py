"""
Interactive Search Analysis Page - Single Question
Permite analizar el proceso de búsqueda y reranking para una pregunta individual
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


def show_interactive_search_single_page():
    """Main page function for single question analysis"""

    st.title("🔍 Análisis Interactivo - Pregunta Individual")
    st.markdown("""
    Analiza el proceso completo de búsqueda vectorial y reranking para **una pregunta específica**.

    **🚀 Mejoras Implementadas:**
    - ✅ **Multi-stage retrieval:** Recupera más candidatos antes de reranking
    - ✅ **CrossEncoder mejorado:** `ms-marco-electra-base` (mejor precisión)
    - ✅ **Query expansion:** Expansión de consultas con terminología Azure
    - ✅ **Normalización Min-Max** y ordenamiento determinístico

    **Métricas:** Precision@k, Recall@k, F1@k, NDCG@k, MAP@k, MRR (Manual vs Ranx)
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
        help="Expande la consulta con terminología Azure para mejorar reranking"
    )

    if use_query_expansion:
        max_expansions = st.sidebar.slider(
            "Sinónimos por término:",
            min_value=1,
            max_value=5,
            value=2,
            help="Número máximo de sinónimos a agregar por término encontrado"
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
        help="Modelo para reranking (electra-base es el mejor pero más lento)"
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

    # Question selection
    st.header("1️⃣ Selección de Pregunta")

    col1, col2 = st.columns([1, 3])

    with col1:
        question_idx = st.number_input(
            "Índice de pregunta:",
            min_value=0,
            max_value=num_questions - 1,
            value=0,
            step=1
        )

    # Get question data
    if question_idx >= len(all_questions['ids']):
        st.error("❌ Índice de pregunta inválido")
        return

    selected_question = all_questions['documents'][question_idx]
    selected_question_id = all_questions['ids'][question_idx]
    metadata = all_questions['metadatas'][question_idx]
    validated_links = metadata.get('validated_links', [])
    if isinstance(validated_links, str):
        try:
            validated_links = json.loads(validated_links)
        except:
            validated_links = [validated_links]

    # Display question info
    with col2:
        st.text_area("Pregunta seleccionada:", selected_question, height=100)
        st.info(f"🔗 Ground Truth: {len(validated_links)} enlaces validados")

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

        with st.spinner("Obteniendo embedding de la pregunta..."):
            model_questions_collection = client.get_collection(questions_collection_name)
            question_url = metadata.get('url', '')

            question_embedding = get_question_embedding(
                client, model_questions_collection, question_url, questions_collection_name
            )

            if question_embedding is None:
                st.error("❌ No se encontró la pregunta en la colección del modelo")
                return

            st.success("✅ Embedding obtenido")

        # Apply query expansion if enabled
        search_question = selected_question
        if use_query_expansion:
            with st.spinner("Expandiendo consulta..."):
                search_question = expand_query(
                    selected_question,
                    max_expansions=max_expansions,
                    debug=True
                )
                if search_question != selected_question:
                    st.info(f"📝 Consulta expandida: {len(search_question)} caracteres (+{len(search_question) - len(selected_question)} chars)")

        with st.spinner(f"Buscando documentos (recuperando top-{retrieval_k})..."):
            retrieved_docs = search_documents(client, docs_collection_name, question_embedding, retrieval_k)

            if not retrieved_docs:
                st.warning("⚠️ No se encontraron documentos")
                return

            st.success(f"✅ {len(retrieved_docs)} documentos recuperados")

        # Calculate metrics before
        st.subheader("🔍 Debug: Antes de CrossEncoder")
        metrics_before = calculate_retrieval_metrics(
            validated_links, retrieved_docs, k_values, score_key='cosine_similarity', debug=True
        )
        ranx_metrics_before = {}
        if RANX_AVAILABLE:
            ranx_metrics_before = calculate_ranx_metrics(
                validated_links, retrieved_docs, k_values, query_id="before", use_crossencoder=False, debug=True
            )

        # Apply CrossEncoder reranking
        with st.spinner(f"Aplicando CrossEncoder reranking con {crossencoder_model.split('/')[-1]}..."):
            cross_encoder = load_crossencoder(crossencoder_model)
            # Use expanded question for better reranking
            reranked_docs = apply_crossencoder_reranking(search_question, retrieved_docs.copy(), cross_encoder)

            # If multi-stage, show how many docs we're working with
            if use_multistage and len(reranked_docs) > top_k:
                st.info(f"🎯 Multi-stage: Reranked {len(reranked_docs)} docs, retornando top-{top_k}")

        # Calculate metrics after
        st.subheader("🔍 Debug: Después de CrossEncoder")
        metrics_after = calculate_retrieval_metrics(
            validated_links, reranked_docs, k_values, score_key='crossencoder_score', debug=True
        )
        ranx_metrics_after = {}
        if RANX_AVAILABLE:
            ranx_metrics_after = calculate_ranx_metrics(
                validated_links, reranked_docs, k_values, query_id="after", use_crossencoder=True, debug=True
            )

        # Show consolidated document ranking table
        st.header("2️⃣ Comparación de Rankings: Antes vs Después del CrossEncoder")

        ranking_comparison = []
        gt_normalized = {normalize_url(link) for link in validated_links if link}

        for doc in reranked_docs[:top_k]:
            link = doc['link']
            normalized_link = normalize_url(link)
            is_relevant = normalized_link in gt_normalized
            original_rank = doc['original_rank']
            new_rank = doc['rank']
            rank_change = original_rank - new_rank

            if rank_change > 0:
                change_indicator = f"🔼 +{rank_change}"
            elif rank_change < 0:
                change_indicator = f"🔽 {rank_change}"
            else:
                change_indicator = "➡️ 0"

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

        # Show metrics comparison
        st.header("3️⃣ Métricas: Antes vs Después del Reranking")

        if RANX_AVAILABLE and ranx_metrics_after:
            st.subheader("📊 Comparación: Implementación Manual vs Ranx")

            comparison_data = []
            for k in k_values:
                for metric_name in ['precision', 'recall', 'f1', 'ndcg', 'map']:
                    manual_before = float(metrics_before.get(f'{metric_name}@{k}', 0))
                    manual_after = float(metrics_after.get(f'{metric_name}@{k}', 0))
                    ranx_before = float(ranx_metrics_before.get(f'{metric_name}@{k}', 0))
                    ranx_after = float(ranx_metrics_after.get(f'{metric_name}@{k}', 0))

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
            manual_mrr_before = float(metrics_before.get('mrr', 0))
            manual_mrr_after = float(metrics_after.get('mrr', 0))
            ranx_mrr_before = float(ranx_metrics_before.get('mrr', 0))
            ranx_mrr_after = float(ranx_metrics_after.get('mrr', 0))

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
        st.header("4️⃣ Gráficas Comparativas (k=1 a 15)")

        full_k_range = list(range(1, 16))

        with st.spinner("Calculando métricas para k=1 a 15..."):
            metrics_before_full = calculate_retrieval_metrics(
                validated_links, retrieved_docs, full_k_range, score_key='cosine_similarity'
            )
            metrics_after_full = calculate_retrieval_metrics(
                validated_links, reranked_docs, full_k_range, score_key='crossencoder_score'
            )

            ranx_metrics_before_full = {}
            ranx_metrics_after_full = {}
            if RANX_AVAILABLE:
                ranx_metrics_before_full = calculate_ranx_metrics(
                    validated_links, retrieved_docs, full_k_range, query_id="before_full", use_crossencoder=False
                )
                ranx_metrics_after_full = calculate_ranx_metrics(
                    validated_links, reranked_docs, full_k_range, query_id="after_full", use_crossencoder=True
                )

        if ranx_metrics_before_full and ranx_metrics_after_full:
            fig_combined = create_metrics_comparison_plot(
                metrics_before_full, metrics_after_full,
                ranx_metrics_before_full, ranx_metrics_after_full,
                full_k_range
            )
            st.pyplot(fig_combined)
            plt.close(fig_combined)
        else:
            st.warning("⚠️ Ranx no disponible - mostrando solo métricas manuales")

        # Ejemplos de cálculo de métricas para k=10
        st.header("5️⃣ Ejemplo Detallado: Cálculo de Métricas para k=10")
        st.markdown("""
        Esta sección muestra **cómo se calculan las métricas** usando los datos reales de esta búsqueda para k=10.
        """)

        # Usar datos después del CrossEncoder para el ejemplo
        example_k = 10

        # Preparar datos para el ejemplo
        gt_normalized = {normalize_url(link) for link in validated_links if link}

        # Ordenar y dedupilicar documentos (igual que en calculate_retrieval_metrics)
        sorted_docs = sorted(
            reranked_docs,
            key=lambda x: (-x.get('crossencoder_score', 0.0), x.get('link', ''))
        )

        seen_urls = set()
        dedup_docs = []
        for doc in sorted_docs:
            norm_url = normalize_url(doc.get('link', ''))
            if norm_url and norm_url not in seen_urls:
                seen_urls.add(norm_url)
                dedup_docs.append(doc)

        # Top-10 documentos únicos
        top_10_docs = dedup_docs[:example_k]
        top_10_links = [normalize_url(doc.get('link', '')) for doc in top_10_docs]

        # Identificar relevantes
        relevant_in_top10 = [link in gt_normalized for link in top_10_links]
        tp = sum(relevant_in_top10)

        st.subheader("📊 Datos para k=10")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ground Truth (Documentos Relevantes):**")
            st.code(f"Total relevantes: {len(gt_normalized)}")
            with st.expander("Ver URLs relevantes"):
                for i, link in enumerate(gt_normalized, 1):
                    st.text(f"{i}. {link[:70]}...")

        with col2:
            st.markdown("**Top-10 Documentos Recuperados:**")
            st.code(f"Documentos únicos recuperados: {len(top_10_docs)}")

            # Mostrar top-10 con marcadores de relevancia
            top10_display = []
            for i, (doc, is_rel) in enumerate(zip(top_10_docs, relevant_in_top10), 1):
                score = doc.get('crossencoder_score', 0.0)
                link = normalize_url(doc.get('link', ''))
                top10_display.append({
                    'Pos': i,
                    'Relevante': '✅' if is_rel else '❌',
                    'Score': f"{score:.4f}",
                    'URL': link[:50] + '...' if len(link) > 50 else link
                })

            df_top10 = pd.DataFrame(top10_display)
            st.dataframe(df_top10, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🧮 Cálculo Manual de Métricas (k=10)")

        # Calcular métricas paso a paso
        import numpy as np

        # 1. Precision@10
        st.markdown("### 1️⃣ Precision@10")
        st.latex(r"\text{Precision@k} = \frac{\text{TP}}{k} = \frac{\text{documentos relevantes en top-k}}{k}")
        st.code(f"""
Cálculo:
  - TP (True Positives) = {tp} documentos relevantes encontrados en top-10
  - k = {example_k}

  Precision@10 = {tp} / {example_k} = {tp / example_k:.4f}
        """)
        precision_10 = tp / example_k
        st.success(f"**Resultado: Precision@10 = {precision_10:.4f}**")

        # 2. Recall@10
        st.markdown("### 2️⃣ Recall@10")
        st.latex(r"\text{Recall@k} = \frac{\text{TP}}{\text{Total Relevantes}} = \frac{\text{documentos relevantes en top-k}}{\text{total de documentos relevantes}}")
        st.code(f"""
Cálculo:
  - TP (True Positives) = {tp} documentos relevantes encontrados en top-10
  - Total Relevantes = {len(gt_normalized)} documentos en ground truth

  Recall@10 = {tp} / {len(gt_normalized)} = {tp / len(gt_normalized):.4f}
        """)
        recall_10 = tp / len(gt_normalized) if len(gt_normalized) > 0 else 0.0
        st.success(f"**Resultado: Recall@10 = {recall_10:.4f}**")

        # 3. F1@10
        st.markdown("### 3️⃣ F1@10")
        st.latex(r"\text{F1@k} = 2 \times \frac{\text{Precision@k} \times \text{Recall@k}}{\text{Precision@k} + \text{Recall@k}}")
        f1_10 = (2 * precision_10 * recall_10) / (precision_10 + recall_10) if (precision_10 + recall_10) > 0 else 0.0
        st.code(f"""
Cálculo:
  - Precision@10 = {precision_10:.4f}
  - Recall@10 = {recall_10:.4f}

  F1@10 = 2 × ({precision_10:.4f} × {recall_10:.4f}) / ({precision_10:.4f} + {recall_10:.4f})
        = 2 × {precision_10 * recall_10:.4f} / {precision_10 + recall_10:.4f}
        = {f1_10:.4f}
        """)
        st.success(f"**Resultado: F1@10 = {f1_10:.4f}**")

        # 4. NDCG@10
        st.markdown("### 4️⃣ NDCG@10 (Normalized Discounted Cumulative Gain)")
        st.latex(r"\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}")
        st.latex(r"\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}")

        # Calcular DCG
        dcg_parts = []
        dcg = 0.0
        for i, is_rel in enumerate(relevant_in_top10, 1):
            rel = 1.0 if is_rel else 0.0
            gain = rel / np.log2(i + 1)
            dcg += gain
            if is_rel:
                dcg_parts.append(f"  Posición {i}: 1 / log2({i}+1) = 1 / {np.log2(i + 1):.3f} = {gain:.4f}")

        # Calcular IDCG (ideal: todos los relevantes al principio)
        idcg = 0.0
        for i in range(1, min(example_k, len(gt_normalized)) + 1):
            idcg += 1.0 / np.log2(i + 1)

        ndcg_10 = dcg / idcg if idcg > 0 else 0.0

        st.code(f"""
Cálculo DCG (Discounted Cumulative Gain):
{chr(10).join(dcg_parts) if dcg_parts else "  (No hay documentos relevantes en top-10)"}

  DCG@10 = {dcg:.4f}

Cálculo IDCG (Ideal DCG - todos los relevantes al inicio):
  Número de relevantes = {len(gt_normalized)}
  IDCG@10 = suma de 1/log2(i+1) para i=1 hasta min(10, {len(gt_normalized)})
  IDCG@10 = {idcg:.4f}

  NDCG@10 = DCG / IDCG = {dcg:.4f} / {idcg:.4f} = {ndcg_10:.4f}
        """)
        st.success(f"**Resultado: NDCG@10 = {ndcg_10:.4f}**")

        # 5. MAP@10
        st.markdown("### 5️⃣ MAP@10 (Mean Average Precision)")
        st.latex(r"\text{MAP@k} = \frac{1}{\min(k, |\text{Relevantes}|)} \sum_{i=1}^{k} \text{Precision@i} \times \text{rel}_i")

        map_parts = []
        sum_precisions = 0.0
        relevant_count = 0
        for i, is_rel in enumerate(relevant_in_top10, 1):
            if is_rel:
                relevant_count += 1
                precision_at_i = relevant_count / i
                sum_precisions += precision_at_i
                map_parts.append(f"  Posición {i}: Precision@{i} = {relevant_count}/{i} = {precision_at_i:.4f}")

        map_10 = sum_precisions / len(gt_normalized) if len(gt_normalized) > 0 else 0.0

        st.code(f"""
Cálculo (se suma Precision@i solo cuando el doc en posición i es relevante):
{chr(10).join(map_parts) if map_parts else "  (No hay documentos relevantes en top-10)"}

  Suma de Precisiones = {sum_precisions:.4f}
  Total Relevantes = {len(gt_normalized)}

  MAP@10 = {sum_precisions:.4f} / {len(gt_normalized)} = {map_10:.4f}
        """)
        st.success(f"**Resultado: MAP@10 = {map_10:.4f}**")

        # 6. MRR (Mean Reciprocal Rank)
        st.markdown("### 6️⃣ MRR (Mean Reciprocal Rank)")
        st.latex(r"\text{MRR} = \frac{1}{\text{rank del primer relevante}}")

        first_relevant_pos = None
        for i, is_rel in enumerate(relevant_in_top10, 1):
            if is_rel:
                first_relevant_pos = i
                break

        mrr = 1.0 / first_relevant_pos if first_relevant_pos else 0.0

        if first_relevant_pos:
            st.code(f"""
Cálculo:
  - Primer documento relevante encontrado en posición: {first_relevant_pos}

  MRR = 1 / {first_relevant_pos} = {mrr:.4f}
            """)
        else:
            st.code("""
Cálculo:
  - No se encontró ningún documento relevante en top-10

  MRR = 0.0
            """)
        st.success(f"**Resultado: MRR = {mrr:.4f}**")

        # Comparación con valores calculados
        st.markdown("---")
        st.subheader("✅ Verificación con Valores Calculados")

        verification_data = {
            'Métrica': ['Precision@10', 'Recall@10', 'F1@10', 'NDCG@10', 'MAP@10', 'MRR'],
            'Valor Manual (Mostrado)': [
                f"{precision_10:.4f}",
                f"{recall_10:.4f}",
                f"{f1_10:.4f}",
                f"{ndcg_10:.4f}",
                f"{map_10:.4f}",
                f"{mrr:.4f}"
            ],
            'Valor Calculado (Función)': [
                f"{metrics_after_full.get('precision@10', 0):.4f}",
                f"{metrics_after_full.get('recall@10', 0):.4f}",
                f"{metrics_after_full.get('f1@10', 0):.4f}",
                f"{metrics_after_full.get('ndcg@10', 0):.4f}",
                f"{metrics_after_full.get('map@10', 0):.4f}",
                f"{metrics_after_full.get('mrr', 0):.4f}"
            ]
        }

        df_verification = pd.DataFrame(verification_data)
        st.dataframe(df_verification, use_container_width=True, hide_index=True)

        # Sección Ranx
        if RANX_AVAILABLE and ranx_metrics_after_full:
            st.markdown("---")
            st.subheader("📦 Cálculo con Ranx (Librería de IR)")

            st.markdown("""
            **Ranx** es una librería de Python para métricas de Information Retrieval. Requiere:
            1. Un `Qrels` (Query Relevance) con los documentos relevantes
            2. Un `Run` con los documentos recuperados y sus scores
            """)

            # Mostrar estructura de datos para Ranx
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**1. Qrels (Ground Truth):**")
                qrels_example = {
                    "q1": {}
                }
                for link in list(gt_normalized)[:3]:
                    qrels_example["q1"][link[:40] + "..."] = 1
                if len(gt_normalized) > 3:
                    qrels_example["q1"]["..."] = f"({len(gt_normalized) - 3} más)"

                st.code(f"""
from ranx import Qrels

qrels_dict = {{
  "q1": {{
    {chr(10).join([f'    "{link[:40]}...": 1,' for link in list(gt_normalized)[:3]])}
    ... ({len(gt_normalized)} docs relevantes total)
  }}
}}

qrels = Qrels(qrels_dict)
                """, language="python")

            with col2:
                st.markdown("**2. Run (Documentos Recuperados):**")
                run_example = []
                for i, doc in enumerate(top_10_docs[:3], 1):
                    link = normalize_url(doc.get('link', ''))
                    score = doc.get('crossencoder_score', 0.0)
                    run_example.append(f'    "{link[:40]}...": {score:.4f},')

                st.code(f"""
from ranx import Run

run_dict = {{
  "q1": {{
{chr(10).join(run_example)}
    ... ({len(top_10_docs)} docs total)
  }}
}}

run = Run(run_dict)
                """, language="python")

            st.markdown("**3. Calcular Métricas:**")
            st.code("""
from ranx import evaluate

metrics = evaluate(
    qrels,
    run,
    ["precision@10", "recall@10", "f1@10",
     "ndcg@10", "map@10", "mrr"]
)
            """, language="python")

            st.markdown("**4. Resultados de Ranx (k=10):**")

            ranx_results = {
                'Métrica': ['Precision@10', 'Recall@10', 'F1@10', 'NDCG@10', 'MAP@10', 'MRR'],
                'Valor Ranx': [
                    f"{ranx_metrics_after_full.get('precision@10', 0):.4f}",
                    f"{ranx_metrics_after_full.get('recall@10', 0):.4f}",
                    f"{ranx_metrics_after_full.get('f1@10', 0):.4f}",
                    f"{ranx_metrics_after_full.get('ndcg@10', 0):.4f}",
                    f"{ranx_metrics_after_full.get('map@10', 0):.4f}",
                    f"{ranx_metrics_after_full.get('mrr', 0):.4f}"
                ],
                'Valor Manual': [
                    f"{precision_10:.4f}",
                    f"{recall_10:.4f}",
                    f"{f1_10:.4f}",
                    f"{ndcg_10:.4f}",
                    f"{map_10:.4f}",
                    f"{mrr:.4f}"
                ],
                'Diferencia': [
                    f"{abs(ranx_metrics_after_full.get('precision@10', 0) - precision_10):.6f}",
                    f"{abs(ranx_metrics_after_full.get('recall@10', 0) - recall_10):.6f}",
                    f"{abs(ranx_metrics_after_full.get('f1@10', 0) - f1_10):.6f}",
                    f"{abs(ranx_metrics_after_full.get('ndcg@10', 0) - ndcg_10):.6f}",
                    f"{abs(ranx_metrics_after_full.get('map@10', 0) - map_10):.6f}",
                    f"{abs(ranx_metrics_after_full.get('mrr', 0) - mrr):.6f}"
                ]
            }

            df_ranx = pd.DataFrame(ranx_results)
            st.dataframe(df_ranx, use_container_width=True, hide_index=True)

            st.info("ℹ️ Las pequeñas diferencias (< 0.0001) son normales debido a precisión de punto flotante. Los valores deben ser prácticamente idénticos.")


if __name__ == "__main__":
    show_interactive_search_single_page()
