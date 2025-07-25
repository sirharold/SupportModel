"""
Página de Comparación de Recuperación: Pregunta vs. Respuesta
Compara documentos recuperados usando la pregunta original vs la respuesta aceptada
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.services.auth.clients import initialize_clients
from src.services.storage.chromadb_utils import ChromaDBConfig, get_chromadb_client, ChromaDBClientWrapper
from src.data.embedding_safe import get_embedding_client
from src.config.config import EMBEDDING_MODELS, CHROMADB_COLLECTION_CONFIG, GENERATIVE_MODELS, DEFAULT_GENERATIVE_MODEL, GENERATIVE_MODEL_DESCRIPTIONS

# Map short model names to config keys (aligned with QUERY_MODELS mapping)
MODEL_NAME_MAPPING = {
    "mpnet": "multi-qa-mpnet-base-dot-v1",  # sentence-transformers/multi-qa-mpnet-base-dot-v1 → 768D
    "minilm": "all-MiniLM-L6-v2",           # sentence-transformers/all-MiniLM-L6-v2 → 384D
    "ada": "ada",                           # text-embedding-ada-002 → 1536D  
    "e5-large": "e5-large-v2"               # intfloat/e5-large-v2 → 1024D
}
from src.utils.comparison_utils import (
    calculate_jaccard_similarity,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_composite_score
)
from src.evaluation.metrics.rag import (
    calculate_hallucination_score,
    calculate_context_utilization,
    calculate_answer_completeness,
    calculate_user_satisfaction_proxy
)
from src.core.qa_pipeline import answer_question_with_rag


def initialize_clients_no_cache(model_name: str):
    """Initialize clients without caching to avoid embedding dimension conflicts"""
    config = ChromaDBConfig.from_env()
    client = get_chromadb_client(config)
    
    chromadb_collections = CHROMADB_COLLECTION_CONFIG[model_name]
    chromadb_wrapper = ChromaDBClientWrapper(
        client,
        documents_class=chromadb_collections["documents"],
        questions_class=chromadb_collections["questions"],
        retry_attempts=3
    )
    
    embedding_client = get_embedding_client(
        model_name=EMBEDDING_MODELS[model_name],
        huggingface_api_key=config.huggingface_api_key,
        openai_api_key=config.openai_api_key
    )
    
    return chromadb_wrapper, embedding_client


def calculate_rag_metrics_for_documents(question: str, answer: str, documents: List[Dict], generative_model: str, chromadb_wrapper, embedding_client) -> Dict:
    """
    Calculate comprehensive RAG metrics using retrieved documents
    """
    try:
        # Use the expected answer as the generated answer for evaluation
        # This simulates what would happen if we generated an answer from these documents
        generated_answer = answer
        
        # Calculate advanced RAG metrics directly
        hallucination_metrics = calculate_hallucination_score(
            answer=generated_answer,
            context_docs=documents,
            question=question
        )
        
        context_metrics = calculate_context_utilization(
            answer=generated_answer,
            context_docs=documents,
            question=question
        )
        
        completeness_metrics = calculate_answer_completeness(
            answer=generated_answer,
            question=question,
            context_docs=documents
        )
        
        satisfaction_metrics = calculate_user_satisfaction_proxy(
            answer=generated_answer,
            question=question,
            context_docs=documents
        )
        
        return {
            'faithfulness': 1.0 - hallucination_metrics.get('hallucination_score', 0.5),  # Invert hallucination
            'answer_relevance': context_metrics.get('context_utilization_score', 0.5),
            'answer_correctness': completeness_metrics.get('completeness_score', 0.5),
            'answer_similarity': satisfaction_metrics.get('satisfaction_score', 0.5),
            'generated_answer': generated_answer
        }
        
    except Exception as e:
        # Return neutral scores on error
        return {
            'faithfulness': 0.5,
            'answer_relevance': 0.5,
            'answer_correctness': 0.5,
            'answer_similarity': 0.5,
            'generated_answer': answer
        }


def evaluate_retrieved_content_quality(question: str, answer: str, documents: List[Dict], generative_model: str) -> float:
    """
    Use generative model to evaluate how well the retrieved documents support answering the question
    """
    try:
        # Initialize clients with the selected generative model
        _, _, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(
            "all-MiniLM-L6-v2",  # We just need the generative clients
            generative_model
        )
        
        # Prepare the context from retrieved documents
        context = "\n\n".join([f"Document {i+1}: {doc['content'][:300]}..." 
                              for i, doc in enumerate(documents[:5])])  # Use top 5 docs
        
        # Create evaluation prompt
        prompt = f"""
Please evaluate how well the following retrieved documents support answering the given question.

Question: {question}
Expected Answer: {answer}

Retrieved Documents:
{context}

Rate the quality of the retrieved documents on a scale of 0.0 to 1.0 based on:
- Relevance to the question
- Coverage of the expected answer
- Overall helpfulness for answering the question

Respond with only a number between 0.0 and 1.0 (e.g., 0.85)
"""
        
        # Get evaluation from the selected generative model
        response = None
        if generative_model == "gpt-4" and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            score_text = response.choices[0].message.content.strip()
        elif generative_model == "gemini-1.5-flash" and gemini_client:
            response = gemini_client.generate_content(prompt)
            score_text = response.text.strip()
        elif generative_model in ["llama-3.3-70b", "deepseek-v3-chat"] and openrouter_client:
            response = openrouter_client.chat.completions.create(
                model=GENERATIVE_MODELS[generative_model],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            score_text = response.choices[0].message.content.strip()
        elif generative_model == "tinyllama-1.1b" and local_tinyllama_client:
            response = local_tinyllama_client.generate(prompt, max_length=20)
            score_text = response.strip()
        else:
            # Fallback - return a neutral score
            return 0.5
        
        # Parse the score
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            # If we can't parse the response, return neutral score
            return 0.5
            
    except Exception as e:
        # On any error, return neutral score
        return 0.5


def show_question_answer_comparison_page():
    """Página principal del comparador de recuperación pregunta vs respuesta"""
    
    st.title("🔄 Comparador de Recuperación: Pregunta vs. Respuesta")
    st.markdown("""
    Esta herramienta compara qué tan distintos son los documentos recuperados cuando se usa:
    - **🔍 La pregunta original** como query
    - **✅ La respuesta aceptada** como query
    
    Analiza el rendimiento de 4 modelos de embedding diferentes.
    """)
    
    # Information about metrics
    with st.expander("📚 Información sobre las Métricas Utilizadas"):
        st.markdown("""
        **🔢 Métricas Calculadas:**
        
        - **Jaccard Similarity**: Mide la similitud entre dos conjuntos de documentos (intersección/unión)
        - **nDCG@10**: Normalized Discounted Cumulative Gain, evalúa el ranking usando respuesta como ground truth
        - **Precision@5**: Fracción de documentos relevantes en los primeros 5 resultados
        - **Score Compuesto**: Combinación ponderada: `0.5×Jaccard + 0.3×nDCG@10 + 0.2×Precision@5`
        
        **🎯 Interpretación:**
        - Valores más altos indican mejor alineación entre pregunta y respuesta
        - Score compuesto cercano a 1.0 = excelente consistencia
        - Score compuesto < 0.3 = baja consistencia entre query types
        """)
    
    # Initialize session state
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    if 'questions_loaded' not in st.session_state:
        st.session_state.questions_loaded = False
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuración")
    
    # Number of questions to analyze
    num_questions = st.sidebar.slider(
        "Número de preguntas a analizar",
        min_value=5,
        max_value=50,
        value=30,
        step=5,
        help="Cantidad de preguntas aleatorias a seleccionar de la base de datos"
    )
    
    # Top-k documents to retrieve
    top_k = st.sidebar.slider(
        "Documentos a recuperar (top-k)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Cantidad de documentos a recuperar para cada query"
    )
    
    # Reranking option
    use_reranking = st.sidebar.checkbox(
        "🔄 Usar Reranking",
        value=True,  # Default to True
        help="Aplicar reranking a los documentos recuperados para mejorar la relevancia"
    )
    
    # Generative model selection
    generative_model = st.sidebar.selectbox(
        "🤖 Modelo Generativo",
        options=list(GENERATIVE_MODELS.keys()),
        index=list(GENERATIVE_MODELS.keys()).index(DEFAULT_GENERATIVE_MODEL),
        help="Modelo generativo para evaluar la calidad de las respuestas"
    )
    
    
    # Load questions button
    if st.sidebar.button("🎲 Cargar Preguntas Aleatorias", type="primary"):
        with st.spinner("Cargando preguntas aleatorias..."):
            questions = load_random_questions(num_questions)
            st.session_state.selected_questions = questions
            st.session_state.comparison_results = {}  # Reset results
            st.session_state.questions_loaded = True  # Mark as loaded
            
            if questions:
                st.sidebar.success(f"✅ {len(questions)} preguntas cargadas")
            else:
                st.sidebar.error("❌ No se pudieron cargar preguntas")
    
    # Auto-load questions on first visit
    if not st.session_state.questions_loaded:
        with st.spinner("Cargando preguntas aleatorias..."):
            questions = load_random_questions(num_questions)
            st.session_state.selected_questions = questions
            st.session_state.questions_loaded = True
            if questions:
                st.sidebar.success(f"✅ {len(questions)} preguntas cargadas")
            else:
                st.sidebar.error("❌ No se pudieron cargar preguntas")
    
    # Main content area
    if st.session_state.selected_questions:
        display_questions_and_comparisons(st.session_state.selected_questions, top_k, use_reranking, generative_model)
    else:
        st.info("👆 Haz clic en 'Cargar Preguntas Aleatorias' para comenzar")


def load_random_questions(num_questions: int) -> List[Dict[str, Any]]:
    """Carga preguntas aleatorias desde ChromaDB"""
    
    try:
        # Initialize ChromaDB for MiniLM (we'll use this just to get questions, not for embeddings)
        full_model_name = MODEL_NAME_MAPPING["minilm"]
        chromadb_wrapper, embedding_client, _, _, _, _, _, _ = initialize_clients(full_model_name, "tinyllama-1.1b")
        
        # List available collections
        try:
            collections = chromadb_wrapper.client.list_collections()
            collection_names = [col.name for col in collections]
        except Exception as e:
            st.error(f"❌ Error listando colecciones: {e}")
            return []
        
        # Get questions collection
        questions_collection = CHROMADB_COLLECTION_CONFIG[full_model_name]["questions"]
        
        # Get questions collection directly
        try:
            questions_collection_obj = chromadb_wrapper.client.get_collection(name=questions_collection)
            
            # Get collection info
            collection_count = questions_collection_obj.count()
            
        except Exception as e:
            st.error(f"❌ Error accediendo colección {questions_collection}: {e}")
            st.error(f"Colecciones disponibles: {collection_names}")
            return []
        
        # Get all questions by querying with a generic term
        results = questions_collection_obj.query(
            query_texts=["Azure question"],  # Generic query to get questions
            n_results=min(1000, collection_count),  # Get many questions but not more than available
            include=["metadatas", "documents"]
        )
        
        if not results or not results.get('metadatas'):
            # Try alternative query
            results = questions_collection_obj.query(
                query_texts=["Microsoft"],  # Different query
                n_results=min(500, collection_count),
                include=["metadatas", "documents"]
            )
            
            if not results or not results.get('metadatas'):
                st.error("No se pudieron cargar preguntas con ningún query")
                return []
        
        # Process results
        all_questions = []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        documents = results['documents'][0] if results['documents'] else []
        
        questions_with_answers = 0
        for i, metadata in enumerate(metadatas):
            if metadata and 'title' in metadata:
                question_data = {
                    'id': metadata.get('id', f'q_{i}'),
                    'title': metadata.get('title', 'Sin título'),
                    'content': documents[i] if i < len(documents) else metadata.get('question_content', ''),
                    'accepted_answer': metadata.get('accepted_answer', ''),
                    'ms_links': metadata.get('ms_links', [])
                }
                
                # Only include questions with accepted answers
                if question_data['accepted_answer']:
                    all_questions.append(question_data)
                    questions_with_answers += 1
        
        if not all_questions:
            st.error("❌ No se encontraron preguntas con respuestas aceptadas")
            return []
        
        # Random sample
        if len(all_questions) > num_questions:
            selected = random.sample(all_questions, num_questions)
        else:
            selected = all_questions
        return selected
        
    except Exception as e:
        st.error(f"Error cargando preguntas: {str(e)}")
        return []


def display_questions_and_comparisons(questions: List[Dict], top_k: int, use_reranking: bool, generative_model: str):
    """Muestra las preguntas en un dropdown y permite hacer comparaciones"""
    
    st.subheader(f"📋 {len(questions)} Preguntas Cargadas")
    
    # Summary metrics if comparisons have been done
    if st.session_state.comparison_results:
        display_summary_metrics()
        
        # Export functionality
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📥 Exportar Resultados"):
                export_results_to_csv()
        with col1:
            st.markdown("")  # Spacer
    
    # Create dropdown for question selection
    st.markdown("### 📝 Seleccionar Pregunta para Analizar")
    
    # Create options for selectbox
    question_options = [f"{idx + 1}. {q['title'][:80]}..." for idx, q in enumerate(questions)]
    
    selected_idx = st.selectbox(
        "Selecciona una pregunta:",
        range(len(questions)),
        format_func=lambda x: question_options[x],
        help="Selecciona una pregunta de la lista para ver su contenido y realizar la comparación"
    )
    
    # Display selected question
    if selected_idx is not None:
        selected_question = questions[selected_idx]
        
        st.markdown("---")
        
        # Question details
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {selected_idx + 1}. {selected_question['title']}")
            
            # Question content in expandable section
            with st.expander("📝 Ver contenido completo de la pregunta", expanded=True):
                st.text(selected_question['content'])
            
            # Answer in expandable section
            if selected_question.get('accepted_answer'):
                with st.expander("✅ Ver respuesta aceptada", expanded=True):
                    st.text(selected_question['accepted_answer'])
            else:
                st.warning("⚠️ Esta pregunta no tiene respuesta aceptada")
        
        with col2:
            # Comparison button
            if st.button(
                "🔍 Comparar recuperación",
                key=f"compare_{selected_idx}",
                disabled=not selected_question.get('accepted_answer'),
                type="primary"
            ):
                perform_comparison(selected_question, selected_idx, top_k, use_reranking, generative_model)
        
        # Display results if comparison was done
        question_id = selected_question['id']
        if question_id in st.session_state.comparison_results:
            st.markdown("---")
            display_comparison_results(
                st.session_state.comparison_results[question_id],
                selected_question
            )


def perform_comparison(question: Dict, idx: int, top_k: int, use_reranking: bool, generative_model: str):
    """Realiza la comparación de recuperación para una pregunta"""
    
    with st.spinner(f"Comparando recuperación para pregunta {idx + 1}..."):
        results = {}
        
        # For each embedding model
        for short_model_name in MODEL_NAME_MAPPING.keys():
            try:
                # Get full model name
                full_model_name = MODEL_NAME_MAPPING[short_model_name]
                
                # Validate that the model key exists in both configs
                if full_model_name not in EMBEDDING_MODELS:
                    st.error(f"❌ Modelo {full_model_name} no encontrado en EMBEDDING_MODELS")
                    continue
                if full_model_name not in CHROMADB_COLLECTION_CONFIG:
                    st.error(f"❌ Modelo {full_model_name} no encontrado en CHROMADB_COLLECTION_CONFIG")
                    continue
                
                # Initialize clients for this specific model WITHOUT caching
                chromadb_wrapper, embedding_client = initialize_clients_no_cache(full_model_name)
                
                # Verify embedding dimensions match collection expectations
                expected_dimensions = {
                    "multi-qa-mpnet-base-dot-v1": 768,
                    "all-MiniLM-L6-v2": 384, 
                    "ada": 1536,
                    "e5-large-v2": 1024
                }
                expected_dim = expected_dimensions.get(full_model_name, "unknown")
                
                # Get collection name
                docs_collection = CHROMADB_COLLECTION_CONFIG[full_model_name]["documents"]
                
                # Validate documents collection exists
                try:
                    collections = chromadb_wrapper.client.list_collections()
                    collection_names = [col.name for col in collections]
                    
                    if docs_collection not in collection_names:
                        st.error(f"❌ Colección {docs_collection} no encontrada para modelo {short_model_name}")
                        st.error(f"Colecciones disponibles: {collection_names}")
                        continue
                    
                    # Get documents collection directly
                    docs_collection_obj = chromadb_wrapper.client.get_collection(name=docs_collection)
                    doc_count = docs_collection_obj.count()
                    
                except Exception as e:
                    st.error(f"❌ Error accediendo colección {docs_collection}: {e}")
                    continue
                
                # Test embedding dimension before queries
                try:
                    test_embedding = embedding_client.generate_query_embedding("test")
                    actual_dim = len(test_embedding)
                    
                    if actual_dim != expected_dim:
                        st.error(f"❌ **DIMENSIÓN INCORRECTA**: Esperado {expected_dim}D, obtenido {actual_dim}D")
                        continue
                        
                except Exception as embed_e:
                    st.error(f"❌ **Error generando embedding de prueba**: {embed_e}")
                    continue
                
                # Start timing
                start_time = time.time()
                
                # Generate embeddings manually with the correct model instead of using query_texts
                question_query = f"{question['title']} {question['content']}"
                
                # Generate question embedding with the correct model
                question_embedding = embedding_client.generate_query_embedding(question_query)
                
                # Generate answer embedding with the correct model  
                answer_embedding = embedding_client.generate_query_embedding(question['accepted_answer'])
                
                # Query using embeddings directly (not query_texts)
                # For reranking, retrieve more documents initially
                n_retrieve = top_k * 2 if use_reranking else top_k
                
                question_results = docs_collection_obj.query(
                    query_embeddings=[question_embedding],
                    n_results=n_retrieve,
                    include=["metadatas", "distances", "documents"]
                )
                
                answer_results = docs_collection_obj.query(
                    query_embeddings=[answer_embedding],
                    n_results=n_retrieve,
                    include=["metadatas", "distances", "documents"]
                )
                
                # Apply reranking if enabled
                if use_reranking:
                    try:
                        from src.core.reranker import rerank_with_llm
                    
                        # Process initial results
                        question_docs_pre = process_query_results(question_results, "question")
                        answer_docs_pre = process_query_results(answer_results, "answer")
                        
                        # Prepare documents for reranking - rerank_with_llm expects specific format
                        question_docs_for_rerank = [
                            {
                                'content': doc['content'], 
                                'metadata': {
                                    'title': doc['title'], 
                                    'chunk_index': doc['chunk_index'],
                                    'link': doc.get('link', '')
                                }
                            }
                            for doc in question_docs_pre
                        ]
                        answer_docs_for_rerank = [
                            {
                                'content': doc['content'], 
                                'metadata': {
                                    'title': doc['title'], 
                                    'chunk_index': doc['chunk_index'],
                                    'link': doc.get('link', '')
                                }
                            }
                            for doc in answer_docs_pre
                        ]
                        
                        # Rerank documents - rerank_with_llm returns docs with 'relevance_score'
                        # Note: We pass None for openai_client as the function uses CrossEncoder
                        reranked_question_docs = rerank_with_llm(
                            question_query, 
                            question_docs_for_rerank, 
                            openai_client=None,
                            top_k=top_k,
                            embedding_model=full_model_name
                        )
                        reranked_answer_docs = rerank_with_llm(
                            question['accepted_answer'], 
                            answer_docs_for_rerank,
                            openai_client=None, 
                            top_k=top_k,
                            embedding_model=full_model_name
                        )
                    
                        # Convert reranked results back to our format
                        question_docs = []
                        for i, doc in enumerate(reranked_question_docs):
                            score = doc.get('score', 0.5)  # rerank_with_llm returns 'score', not 'relevance_score'
                            question_docs.append({
                                'title': doc['metadata']['title'],
                                'chunk_index': doc['metadata']['chunk_index'],
                                'link': doc['metadata'].get('link', ''),
                                'content': doc['content'],
                                'distance': 1 - score,
                                'similarity': max(0, min(1, score)),
                                'source': 'question',
                                'rank': i + 1
                            })
                        
                        answer_docs = []
                        for i, doc in enumerate(reranked_answer_docs):
                            score = doc.get('score', 0.5)  # rerank_with_llm returns 'score', not 'relevance_score'
                            answer_docs.append({
                                'title': doc['metadata']['title'],
                                'chunk_index': doc['metadata']['chunk_index'],
                                'link': doc['metadata'].get('link', ''),
                                'content': doc['content'],
                                'distance': 1 - score,
                                'similarity': max(0, min(1, score)),
                                'source': 'answer',
                                'rank': i + 1
                            })
                            
                    except Exception as rerank_error:
                        st.warning(f"⚠️ Error al reranking para {short_model_name}: {str(rerank_error)}")
                        st.info("Usando resultados sin reranking...")
                        # Fallback to non-reranked results
                        question_docs = process_query_results(question_results, "question")[:top_k]
                        answer_docs = process_query_results(answer_results, "answer")[:top_k]
                else:
                    # Process results without reranking
                    question_docs = process_query_results(question_results, "question")[:top_k]
                    answer_docs = process_query_results(answer_results, "answer")[:top_k]
                
                # Calculate RAG metrics for both question and answer retrievals
                question_full = f"{question['title']} {question['content']}"
                
                # RAG metrics for question-based retrieval
                question_rag_metrics = calculate_rag_metrics_for_documents(
                    question_full,
                    question['accepted_answer'],
                    question_docs,
                    generative_model,
                    chromadb_wrapper,
                    embedding_client
                )
                
                # RAG metrics for answer-based retrieval  
                answer_rag_metrics = calculate_rag_metrics_for_documents(
                    question_full,
                    question['accepted_answer'], 
                    answer_docs,
                    generative_model,
                    chromadb_wrapper,
                    embedding_client
                )
                
                # Evaluate content quality using generative model (keep existing metric)
                question_quality = evaluate_retrieved_content_quality(
                    question_full, 
                    question['accepted_answer'], 
                    question_docs, 
                    generative_model
                )
                
                answer_quality = evaluate_retrieved_content_quality(
                    question_full, 
                    question['accepted_answer'], 
                    answer_docs, 
                    generative_model
                )
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Calculate metrics
                metrics = calculate_comparison_metrics(question_docs, answer_docs, top_k)
                
                # Add generative evaluation to metrics
                metrics['question_quality'] = question_quality
                metrics['answer_quality'] = answer_quality
                metrics['avg_quality'] = (question_quality + answer_quality) / 2
                
                # Add RAG metrics (averaged from question and answer retrievals)
                metrics['faithfulness'] = (question_rag_metrics['faithfulness'] + answer_rag_metrics['faithfulness']) / 2
                metrics['answer_relevance'] = (question_rag_metrics['answer_relevance'] + answer_rag_metrics['answer_relevance']) / 2
                metrics['answer_correctness'] = (question_rag_metrics['answer_correctness'] + answer_rag_metrics['answer_correctness']) / 2
                metrics['answer_similarity'] = (question_rag_metrics['answer_similarity'] + answer_rag_metrics['answer_similarity']) / 2
                
                # Store individual RAG metrics for detailed analysis
                metrics['question_rag_metrics'] = question_rag_metrics
                metrics['answer_rag_metrics'] = answer_rag_metrics
                
                # Update composite score to include generative evaluation
                # New formula: 0.4×Jaccard + 0.25×nDCG@10 + 0.15×Precision@5 + 0.2×AvgQuality
                original_composite = metrics['composite_score']
                metrics['composite_score'] = (
                    0.4 * metrics['jaccard_similarity'] + 
                    0.25 * metrics['ndcg_at_10'] + 
                    0.15 * metrics['precision_at_5'] +
                    0.2 * metrics['avg_quality']
                )
                metrics['original_composite'] = original_composite  # Keep original for reference
                
                # If reranking was used, also calculate metrics without reranking for comparison
                metrics_without_rerank = None
                if use_reranking:
                    # Get non-reranked docs
                    question_docs_no_rerank = process_query_results(question_results, "question")[:top_k]
                    answer_docs_no_rerank = process_query_results(answer_results, "answer")[:top_k]
                    metrics_without_rerank = calculate_comparison_metrics(question_docs_no_rerank, answer_docs_no_rerank, top_k)
                    
                    # Add generative evaluation for non-reranked
                    question_quality_no_rerank = evaluate_retrieved_content_quality(
                        question_full, 
                        question['accepted_answer'], 
                        question_docs_no_rerank, 
                        generative_model
                    )
                    answer_quality_no_rerank = evaluate_retrieved_content_quality(
                        question_full, 
                        question['accepted_answer'], 
                        answer_docs_no_rerank, 
                        generative_model
                    )
                    
                    # RAG metrics for non-reranked documents
                    question_rag_no_rerank = calculate_rag_metrics_for_documents(
                        question_full,
                        question['accepted_answer'],
                        question_docs_no_rerank,
                        generative_model,
                        chromadb_wrapper,
                        embedding_client
                    )
                    
                    answer_rag_no_rerank = calculate_rag_metrics_for_documents(
                        question_full,
                        question['accepted_answer'],
                        answer_docs_no_rerank,
                        generative_model,
                        chromadb_wrapper,
                        embedding_client
                    )
                    
                    metrics_without_rerank['question_quality'] = question_quality_no_rerank
                    metrics_without_rerank['answer_quality'] = answer_quality_no_rerank
                    metrics_without_rerank['avg_quality'] = (question_quality_no_rerank + answer_quality_no_rerank) / 2
                    
                    # Add RAG metrics for non-reranked
                    metrics_without_rerank['faithfulness'] = (question_rag_no_rerank['faithfulness'] + answer_rag_no_rerank['faithfulness']) / 2
                    metrics_without_rerank['answer_relevance'] = (question_rag_no_rerank['answer_relevance'] + answer_rag_no_rerank['answer_relevance']) / 2
                    metrics_without_rerank['answer_correctness'] = (question_rag_no_rerank['answer_correctness'] + answer_rag_no_rerank['answer_correctness']) / 2
                    metrics_without_rerank['answer_similarity'] = (question_rag_no_rerank['answer_similarity'] + answer_rag_no_rerank['answer_similarity']) / 2
                    
                    # Update composite score for non-reranked
                    metrics_without_rerank['composite_score'] = (
                        0.4 * metrics_without_rerank['jaccard_similarity'] + 
                        0.25 * metrics_without_rerank['ndcg_at_10'] + 
                        0.15 * metrics_without_rerank['precision_at_5'] +
                        0.2 * metrics_without_rerank['avg_quality']
                    )
                
                results[short_model_name] = {
                    'question_docs': question_docs,
                    'answer_docs': answer_docs,
                    'metrics': metrics,
                    'metrics_without_rerank': metrics_without_rerank,
                    'elapsed_time': elapsed_time,
                    'used_reranking': use_reranking
                }
                
            except Exception as e:
                st.error(f"Error con modelo {short_model_name}: {str(e)}")
                results[short_model_name] = {
                    'question_docs': [],
                    'answer_docs': [],
                    'metrics': {}
                }
        
        # Store results
        st.session_state.comparison_results[question['id']] = results


def process_query_results(results: Dict, source: str) -> List[Dict]:
    """Procesa los resultados de una query"""
    
    docs = []
    
    if results and results.get('metadatas'):
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if results.get('distances') else []
        documents = results['documents'][0] if results.get('documents') else []
        
        for i, metadata in enumerate(metadatas):
            if metadata:
                doc = {
                    'title': metadata.get('title', 'Sin título'),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'link': metadata.get('link', ''),
                    'content': documents[i] if i < len(documents) else '',
                    'distance': distances[i] if i < len(distances) else 1.0,
                    'similarity': max(0, min(1, 1 - (distances[i] if i < len(distances) else 1.0))),
                    'source': source,
                    'rank': i + 1
                }
                docs.append(doc)
    
    return docs


def calculate_comparison_metrics(question_docs: List[Dict], answer_docs: List[Dict], top_k: int) -> Dict:
    """Calcula métricas de comparación entre dos sets de documentos"""
    
    # Extract document IDs (title + chunk_index)
    question_ids = set(f"{doc['title']}_{doc['chunk_index']}" for doc in question_docs)
    answer_ids = set(f"{doc['title']}_{doc['chunk_index']}" for doc in answer_docs)
    
    # Common documents
    common_ids = question_ids.intersection(answer_ids)
    
    # Jaccard similarity
    jaccard = calculate_jaccard_similarity(question_ids, answer_ids)
    
    # nDCG@10 (using answer docs as ground truth)
    ndcg = calculate_ndcg_at_k(question_docs, answer_docs, k=10)
    
    # Precision@5
    precision = calculate_precision_at_k(question_docs, answer_docs, k=5)
    
    # Composite score
    composite = calculate_composite_score(jaccard, ndcg, precision)
    
    return {
        'jaccard_similarity': jaccard,
        'ndcg_at_10': ndcg,
        'precision_at_5': precision,
        'common_docs': len(common_ids),
        'composite_score': composite
    }


def display_comparison_results(results: Dict, question: Dict):
    """Muestra los resultados de la comparación"""
    
    st.subheader("📊 Resultados de Comparación")
    
    # Create 4 columns for models
    cols = st.columns(4)
    
    model_names = list(MODEL_NAME_MAPPING.keys())
    
    for idx, model_name in enumerate(model_names):
        with cols[idx]:
            st.markdown(f"**{model_name.upper()}**")
            
            if model_name in results:
                model_results = results[model_name]
                
                # Display metrics summary
                metrics = model_results['metrics']
                if metrics:
                    st.metric(
                        "Score Compuesto",
                        f"{metrics.get('composite_score', 0):.3f}",
                        help="0.4×Jaccard + 0.25×nDCG@10 + 0.15×Precision@5 + 0.2×Calidad_LLM"
                    )
                    
                    # Display elapsed time and non-reranked score if reranking was used
                    elapsed_time = model_results.get('elapsed_time', 0)
                    metrics_without_rerank = model_results.get('metrics_without_rerank')
                    
                    if metrics_without_rerank:
                        # Show both time and non-reranked score
                        st.markdown(
                            f"<small style='color: gray;'>⏱️ {elapsed_time:.2f}s | "
                            f"Sin rerank: {metrics_without_rerank.get('composite_score', 0):.3f}</small>", 
                            unsafe_allow_html=True
                        )
                    else:
                        # Just show time
                        st.markdown(f"<small style='color: gray;'>⏱️ {elapsed_time:.2f}s</small>", unsafe_allow_html=True)
                
                # Display documents (showing all top-k)
                st.markdown(f"**🔍 Desde Pregunta (Top-{len(model_results['question_docs'])}):**")
                for doc in model_results['question_docs']:
                    display_doc_card(doc, is_common=is_doc_common(doc, model_results['answer_docs']))
                
                st.markdown(f"**✅ Desde Respuesta (Top-{len(model_results['answer_docs'])}):**")
                for doc in model_results['answer_docs']:
                    display_doc_card(doc, is_common=is_doc_common(doc, model_results['question_docs']))
            else:
                st.error("Sin resultados")
    
    # Metrics table
    st.subheader("📈 Tabla Comparativa de Métricas")
    
    metrics_data = []
    for model_name in model_names:
        if model_name in results and results[model_name]['metrics']:
            metrics = results[model_name]['metrics']
            metrics_data.append({
                'Modelo': model_name.upper(),
                'Jaccard': f"{metrics.get('jaccard_similarity', 0):.3f}",
                'nDCG@10': f"{metrics.get('ndcg_at_10', 0):.3f}",
                'Precision@5': f"{metrics.get('precision_at_5', 0):.3f}",
                'Calidad LLM': f"{metrics.get('avg_quality', 0):.3f}",
                'Faithfulness': f"{metrics.get('faithfulness', 0):.3f}",
                'Relevancia': f"{metrics.get('answer_relevance', 0):.3f}",
                'Correctness': f"{metrics.get('answer_correctness', 0):.3f}",
                'Similarity': f"{metrics.get('answer_similarity', 0):.3f}",
                'Score Final': f"{metrics.get('composite_score', 0):.3f}"
            })
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # RAG Metrics Details
        with st.expander("📊 Detalle de Métricas RAG"):
            st.markdown("### Métricas RAG por Modelo")
            
            for model_name in model_names:
                if model_name in results and results[model_name]['metrics']:
                    metrics = results[model_name]['metrics']
                    
                    st.markdown(f"#### {model_name.upper()}")
                    
                    # Create 4 columns for RAG metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}", 
                                help="Qué tan libre de alucinaciones es la respuesta")
                    with col2:
                        st.metric("Relevancia", f"{metrics.get('answer_relevance', 0):.3f}",
                                help="Qué tan relevante es la respuesta a la pregunta")
                    with col3:
                        st.metric("Correctness", f"{metrics.get('answer_correctness', 0):.3f}",
                                help="Qué tan correcta y completa es la respuesta")
                    with col4:
                        st.metric("Similarity", f"{metrics.get('answer_similarity', 0):.3f}",
                                help="Similitud semántica con la respuesta esperada")
        
        # Visualization
        display_metrics_visualization(results)


def display_doc_card(doc: Dict, is_common: bool):
    """Muestra una tarjeta de documento"""
    
    # Color based on whether it's common
    border_color = "#28a745" if is_common else "#6c757d"
    
    st.markdown(f"""
    <div style="
        border-left: 3px solid {border_color};
        padding: 5px;
        margin: 5px 0;
        background-color: rgba(0,0,0,0.02);
    ">
        <small><b>{doc['title'][:50]}...</b></small><br>
        <small>Chunk: {doc['chunk_index']} | Score: {doc['similarity']:.3f}</small>
    </div>
    """, unsafe_allow_html=True)


def is_doc_common(doc: Dict, other_docs: List[Dict]) -> bool:
    """Verifica si un documento está en otro conjunto"""
    
    doc_id = f"{doc['title']}_{doc['chunk_index']}"
    other_ids = set(f"{d['title']}_{d['chunk_index']}" for d in other_docs)
    
    return doc_id in other_ids


def display_summary_metrics():
    """Muestra métricas resumen de todas las comparaciones realizadas"""
    
    all_metrics = []
    
    for question_id, results in st.session_state.comparison_results.items():
        for model_name, model_results in results.items():
            if model_results.get('metrics'):
                metrics = model_results['metrics']
                all_metrics.append({
                    'model': model_name,
                    'composite_score': metrics.get('composite_score', 0),
                    'jaccard': metrics.get('jaccard_similarity', 0),
                    'ndcg': metrics.get('ndcg_at_10', 0),
                    'precision': metrics.get('precision_at_5', 0)
                })
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        avg_scores = df.groupby('model').agg({
            'composite_score': 'mean',
            'jaccard': 'mean',
            'ndcg': 'mean',
            'precision': 'mean'
        }).round(3)
        
        # Sort by composite score
        avg_scores = avg_scores.sort_values('composite_score', ascending=False)
        
        st.success(f"📊 **Resumen Global** ({len(st.session_state.comparison_results)} comparaciones realizadas)")
        
        # Display top metrics
        cols = st.columns(len(avg_scores))
        for idx, (model, metrics) in enumerate(avg_scores.iterrows()):
            with cols[idx]:
                rank_emoji = ["🥇", "🥈", "🥉", "🏅"][idx] if idx < 4 else "📊"
                st.metric(
                    f"{rank_emoji} {model.upper()}", 
                    f"{metrics['composite_score']:.3f}",
                    help=f"Jaccard: {metrics['jaccard']:.3f} | nDCG: {metrics['ndcg']:.3f} | Precision: {metrics['precision']:.3f}"
                )


def display_metrics_visualization(results: Dict):
    """Muestra visualización de métricas"""
    
    # Prepare data for radar chart
    models = []
    jaccard_scores = []
    ndcg_scores = []
    precision_scores = []
    
    for model_name, model_results in results.items():
        if model_results['metrics']:
            metrics = model_results['metrics']
            models.append(model_name.upper())
            jaccard_scores.append(metrics.get('jaccard_similarity', 0))
            ndcg_scores.append(metrics.get('ndcg_at_10', 0))
            precision_scores.append(metrics.get('precision_at_5', 0))
    
    if models:
        # Create radar chart
        fig = go.Figure()
        
        # Add traces for each model
        for i, model in enumerate(models):
            fig.add_trace(go.Scatterpolar(
                r=[jaccard_scores[i], ndcg_scores[i], precision_scores[i]],
                theta=['Jaccard', 'nDCG@10', 'Precision@5'],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparación de Métricas por Modelo"
        )
        
        st.plotly_chart(fig, use_container_width=True)



def export_results_to_csv():
    """Exporta los resultados a CSV"""
    
    if not st.session_state.comparison_results:
        st.warning("No hay resultados para exportar")
        return
    
    # Prepare data for export
    export_data = []
    
    for question_id, results in st.session_state.comparison_results.items():
        # Find the question data
        question_data = next(
            (q for q in st.session_state.selected_questions if q['id'] == question_id), 
            {}
        )
        
        for model_name, model_results in results.items():
            if model_results.get('metrics'):
                metrics = model_results['metrics']
                
                export_data.append({
                    'question_id': question_id,
                    'question_title': question_data.get('title', ''),
                    'model': model_name,
                    'jaccard_similarity': metrics.get('jaccard_similarity', 0),
                    'ndcg_at_10': metrics.get('ndcg_at_10', 0),
                    'precision_at_5': metrics.get('precision_at_5', 0),
                    'common_docs': metrics.get('common_docs', 0),
                    'composite_score': metrics.get('composite_score', 0)
                })
    
    if export_data:
        df = pd.DataFrame(export_data)
        
        # Create CSV download
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"comparison_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Show preview
        with st.expander("👀 Vista previa de datos a exportar"):
            st.dataframe(df, use_container_width=True)
    else:
        st.error("No se pudieron preparar los datos para exportar")


if __name__ == "__main__":
    show_question_answer_comparison_page()