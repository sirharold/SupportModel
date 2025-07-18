"""
Página de Métricas Acumulativas - Evalúa múltiples preguntas y calcula promedios (REFACTORIZADA)
"""

import streamlit as st
import time
from typing import List, Dict, Any
from config import EMBEDDING_MODELS, GENERATIVE_MODELS, WEAVIATE_CLASS_CONFIG

# Importar utilidades refactorizadas
from utils.memory_utils import get_memory_usage, cleanup_memory
# from utils.data_processing import filter_questions_with_links  # No longer needed
from utils.metrics_display import display_cumulative_metrics, display_models_comparison
from utils.cumulative_evaluation import run_cumulative_metrics_evaluation, run_cumulative_metrics_for_models
from utils.file_utils import display_download_section
from utils.metrics import validate_data_integrity


def show_cumulative_metrics_page():
    """Función principal que muestra la página de métricas acumulativas."""
    st.title("📊 Métricas Acumulativas de Recuperación RAG")
    st.markdown("""
    Esta página permite evaluar múltiples preguntas y calcular métricas promedio para diferentes modelos de embedding.
    """)
    
    # Configuración inicial
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Configuración de Evaluación")
        
        # Información sobre la fuente de datos
        st.info("📊 Las preguntas se extraen aleatoriamente desde Weaviate, filtrando solo aquellas que tienen enlaces de Microsoft Learn en la respuesta aceptada.")
        
        # Número de preguntas
        num_questions = st.number_input(
            "🔢 Número de preguntas a evaluar:",
            min_value=1,
            max_value=2000,
            value=50,
            step=1,
            help="Número total de preguntas para la evaluación"
        )
        
        # Configuración de modelo generativo
        generative_model_name = st.selectbox(
            "🤖 Modelo Generativo:",
            list(GENERATIVE_MODELS.keys()),
            index=0,
            help="Modelo usado para reranking LLM"
        )
        
        # Configuración de recuperación
        top_k = st.number_input(
            "🔝 Top-K documentos:",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Número máximo de documentos a recuperar"
        )
        
        # Usar reranking LLM
        use_llm_reranker = st.checkbox(
            "🤖 Usar Reranking LLM",
            value=True,
            help="Activar reordenamiento de documentos con LLM"
        )
        
        # Tamaño de lote
        batch_size = st.number_input(
            "📦 Tamaño de lote:",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Número de preguntas a procesar por lote (para gestión de memoria)"
        )
    
    with col2:
        st.subheader("🎯 Selección de Modelos")
        
        # Opción para evaluar todos los modelos
        evaluate_all_models = st.checkbox(
            "🔄 Evaluar todos los modelos",
            value=False,
            help="Evaluar todos los modelos de embedding disponibles"
        )
        
        if evaluate_all_models:
            selected_models = list(EMBEDDING_MODELS.keys())
            st.info(f"📋 Se evaluarán {len(selected_models)} modelos")
            for model in selected_models:
                st.markdown(f"• {model}")
        else:
            # Selección manual de modelos
            selected_models = st.multiselect(
                "🎛️ Modelos de Embedding:",
                list(EMBEDDING_MODELS.keys()),
                default=[list(EMBEDDING_MODELS.keys())[0]],
                help="Selecciona uno o más modelos para evaluar"
            )
        
        if not selected_models:
            st.error("❌ Debes seleccionar al menos un modelo")
            return
        
        # Información sobre memoria estimada
        estimated_memory = len(selected_models) * num_questions * 0.1  # MB aproximados
        st.info(f"💾 Memoria estimada: ~{estimated_memory:.1f} MB")
    
    # Botones de control
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Ejecutar Evaluación", type="primary"):
            run_evaluation(
                num_questions, selected_models, 
                generative_model_name, top_k, use_llm_reranker, 
                batch_size, evaluate_all_models
            )
    
    with col2:
        if st.button("🧹 Limpiar Caché"):
            for key in list(st.session_state.keys()):
                if 'cumulative_results' in key:
                    del st.session_state[key]
            st.success("✅ Caché limpiado")
            st.rerun()
    
    with col3:
        if st.button("📊 Mostrar Estadísticas de Memoria"):
            show_memory_stats()
    
    # Mostrar resultados si existen
    show_cached_results()


def run_evaluation(num_questions: int, selected_models: List[str],
                  generative_model_name: str, top_k: int, use_llm_reranker: bool,
                  batch_size: int, evaluate_all_models: bool):
    """Ejecuta la evaluación de métricas."""
    
    # Mostrar información de memoria inicial
    initial_memory = get_memory_usage()
    st.info(f"💾 Memoria inicial: {initial_memory:.1f} MB")
    
    # Ejecutar evaluación
    start_time = time.time()
    evaluation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if evaluate_all_models:
            with st.spinner(f"🔄 Evaluando {len(selected_models)} modelos..."):
                results = run_cumulative_metrics_for_models(
                    num_questions=num_questions,
                    model_names=selected_models,
                    generative_model_name=generative_model_name,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    batch_size=batch_size
                )
        else:
            # Evaluación de modelo único
            model_name = selected_models[0]
            with st.spinner(f"⚙️ Evaluando {model_name}..."):
                single_result = run_cumulative_metrics_evaluation(
                    num_questions=num_questions,
                    model_name=model_name,
                    generative_model_name=generative_model_name,
                    top_k=top_k,
                    use_llm_reranker=use_llm_reranker,
                    batch_size=batch_size
                )
                results = {model_name: single_result}
        
        # Validar integridad de datos
        for model_name, result in results.items():
            results[model_name] = validate_data_integrity(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Guardar en caché
        cache_key = f"cumulative_results_{len(selected_models)}_{num_questions}_{int(start_time)}"
        st.session_state[cache_key] = {
            'results': results,
            'evaluation_time': evaluation_time,
            'execution_time': execution_time,
            'evaluate_all_models': evaluate_all_models,
            'params': {
                'num_questions': num_questions,
                'selected_models': selected_models,
                'embedding_model_name': selected_models[0] if len(selected_models) == 1 else 'Multi-Model',
                'generative_model_name': generative_model_name,
                'top_k': top_k,
                'use_llm_reranker': use_llm_reranker,
                'batch_size': batch_size
            }
        }
        
        st.success(f"✅ Evaluación completada en {execution_time:.2f} segundos")
        
        # Mostrar memoria final
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        st.info(f"💾 Memoria final: {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Forzar rerun para mostrar resultados
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error durante la evaluación: {str(e)}")
        st.exception(e)
    finally:
        # Limpieza de memoria
        cleanup_memory()


def show_cached_results():
    """Muestra los resultados cacheados si existen."""
    cached_keys = [k for k in st.session_state.keys() if 'cumulative_results' in k]
    
    if not cached_keys:
        st.info("ℹ️ No hay resultados de evaluación. Ejecuta una evaluación para ver los resultados.")
        return
    
    # Seleccionar qué resultados mostrar
    if len(cached_keys) > 1:
        st.subheader("📋 Resultados Disponibles")
        selected_key = st.selectbox(
            "Selecciona resultados:",
            cached_keys,
            format_func=lambda x: f"Evaluación {x.split('_')[-1]} ({st.session_state[x]['params']['embedding_model_name']})"
        )
    else:
        selected_key = cached_keys[0]
    
    if selected_key not in st.session_state:
        return
    
    cached_results = st.session_state[selected_key]
    results = cached_results['results']
    evaluate_all_models = cached_results['evaluate_all_models']
    params = cached_results['params']
    
    st.markdown("---")
    
    # Mostrar resultados
    if evaluate_all_models:
        st.subheader("📈 Comparación Multi-Modelo")
        display_models_comparison(results, params['use_llm_reranker'])
    else:
        st.subheader(f"📊 Resultados para {params['embedding_model_name']}")
        model_name = list(results.keys())[0]
        display_cumulative_metrics(results[model_name], model_name, params['use_llm_reranker'])
    
    # Sección de descarga
    st.markdown("---")
    display_download_section(cached_results)


def show_memory_stats():
    """Muestra estadísticas detalladas de memoria."""
    current_memory = get_memory_usage()
    
    st.subheader("📊 Estadísticas de Memoria")
    st.metric("Memoria Actual", f"{current_memory:.1f} MB")
    
    # Contar elementos en caché
    cache_count = len([k for k in st.session_state.keys() if 'cumulative_results' in k])
    st.metric("Resultados en Caché", cache_count)
    
    if cache_count > 0:
        st.info("💡 Usa 'Limpiar Caché' para liberar memoria")


if __name__ == "__main__":
    show_cumulative_metrics_page()