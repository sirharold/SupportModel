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
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="Número de documentos a recuperar antes de reranking"
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


if __name__ == "__main__":
    show_interactive_search_single_page()
