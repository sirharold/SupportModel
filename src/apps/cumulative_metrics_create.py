"""
Página de Creación de Configuración para Métricas Acumulativas
Permite configurar evaluaciones y enviarlas a Google Colab
"""

import streamlit as st
import time
import json
import os
from typing import List, Dict, Any
from src.config.config import EMBEDDING_MODELS, GENERATIVE_MODELS, CHROMADB_COLLECTION_CONFIG

# Importar utilidades
from src.data.memory_utils import get_memory_usage, cleanup_memory
from src.services.storage.real_gdrive_integration import (
    show_gdrive_status, create_evaluation_config_in_drive,
    check_evaluation_status_in_drive, show_gdrive_authentication_instructions, 
    show_gdrive_debug_info
)
from src.apps.cumulative_metrics_page import display_current_colab_status


def show_cumulative_metrics_create_page():
    """Página principal para crear configuraciones de métricas acumulativas."""
    
    st.title("⚙️ Configuración Métricas Acumulativas - ACTUALIZADO")
    st.markdown("""
    Esta página permite configurar evaluaciones de múltiples preguntas y enviarlas a Google Colab para procesamiento con GPU.
    """)
    
    # Mostrar información del flujo
    st.info("""
    📋 **Flujo de trabajo:**
    1. **Configurar evaluación** en esta página
    2. **Enviar a Google Drive** para Colab
    3. **Ejecutar en Google Colab** con GPU
    4. **Ver resultados** en la página de resultados
    """)
    
    # Configuración inicial
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Configuración de Datos")
        
        # Información sobre la fuente de datos
        st.info("🚀 Las preguntas se extraen desde la colección optimizada 'questions_withlinks' que contiene solo preguntas PRE-VALIDADAS con enlaces de Microsoft Learn que existen en la collection de documentos. Esto garantiza máxima velocidad y precisión.")
        
        # Número de preguntas
        num_questions = st.number_input(
            "🔢 Número de preguntas a evaluar:",
            min_value=1,
            max_value=3100,
            value=600,
            step=1,
            help="Número total de preguntas para la evaluación"
        )
        
        # Tamaño de lote
        batch_size = st.number_input(
            "📦 Tamaño de lote:",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Número de preguntas a procesar por lote (afecta uso de memoria)"
        )
        
    with col2:
        st.subheader("🤖 Configuración de Modelos")
        
        # Configuración de modelo generativo
        generative_model_name = st.selectbox(
            "🤖 Modelo Generativo:",
            list(GENERATIVE_MODELS.keys()),
            index=list(GENERATIVE_MODELS.keys()).index("gpt-4") if "gpt-4" in GENERATIVE_MODELS else 0,
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
        
        # Método de reranking
        reranking_method = st.selectbox(
            "🔄 Método de Reranking:",
            options=["crossencoder", "standard", "none"],
            index=0,  # CrossEncoder por defecto
            format_func=lambda x: {
                "crossencoder": "🧠 CrossEncoder (Recomendado)",
                "standard": "📊 Reranking Estándar",
                "none": "❌ Sin Reranking"
            }[x],
            help="Método de reranking: CrossEncoder usa ms-marco-MiniLM-L-6-v2 para mejor calidad"
        )
        
        # Mantener compatibilidad hacia atrás
        use_llm_reranker = reranking_method != "none"
        
        # NUEVO: Generar métricas RAG
        generate_rag_metrics = st.checkbox(
            "📝 Generar Métricas RAG",
            value=True,
            help="Generar respuestas y calcular métricas RAG (faithfulness, answer_relevance, etc.). Aumenta significativamente el tiempo de procesamiento."
        )
        
        if generate_rag_metrics:
            st.warning("⚠️ **Nota:** Activar métricas RAG aumentará considerablemente el tiempo de evaluación ya que se deben generar respuestas para cada pregunta.")
    
    # Selección de modelos de embedding
    st.subheader("📈 Selección de Modelos de Embedding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Modelos Disponibles:**")
        
        # Opción para evaluar todos los modelos
        evaluate_all_models = st.checkbox(
            "🔄 Evaluar todos los modelos",
            value=True,
            help="Evalúa automáticamente todos los modelos disponibles"
        )
        
        if not evaluate_all_models:
            # Selección manual de modelos
            selected_models = st.multiselect(
                "📋 Seleccionar modelos específicos:",
                options=list(EMBEDDING_MODELS.keys()),
                default=[list(EMBEDDING_MODELS.keys())[0]],
                help="Selecciona uno o más modelos para evaluar"
            )
        else:
            selected_models = list(EMBEDDING_MODELS.keys())
            st.write(f"✅ Se evaluarán {len(selected_models)} modelos:")
            for model in selected_models:
                st.write(f"   • {model}")
    
    with col2:
        st.markdown("**📊 Información de la Evaluación:**")
        
        if selected_models:
            total_evaluations = len(selected_models) * num_questions
            estimated_time_minutes = total_evaluations * 0.1 / 60  # Estimación aproximada
            
            st.metric("🔢 Modelos seleccionados", len(selected_models))
            st.metric("🔢 Total evaluaciones", f"{total_evaluations:,}")
            st.metric("⏱️ Tiempo estimado (GPU)", f"{estimated_time_minutes:.1f} min")
            
            # Mostrar detalles de configuración
            st.markdown("**⚙️ Configuración:**")
            st.write(f"• Preguntas: {num_questions:,}")
            st.write(f"• Top-K: {top_k}")
            st.write(f"• Reranking: {reranking_method}")
            st.write(f"• Métricas RAG: {'✅' if generate_rag_metrics else '❌'}")
            st.write(f"• Lote: {batch_size}")
        else:
            st.warning("⚠️ Selecciona al menos un modelo para continuar")
    
    # Verificar Google Drive
    st.subheader("☁️ Configuración de Google Drive")
    
    gdrive_ok = show_gdrive_status()
    
    if not gdrive_ok:
        st.error("❌ Configuración de Google Drive requerida")
        show_gdrive_authentication_instructions()
        
        col1, col2 = st.columns(2)
        with col1:
            # Verificar Estado button removed per user request
            st.empty()
        with col2:
            if st.button("🔍 Debug Google Drive"):
                st.markdown("---")
                show_gdrive_debug_info()
        
        st.stop()
    
    # Crear configuración
    if selected_models:
        
        # Preparar configuración de evaluación
        evaluation_config = {
            'num_questions': num_questions,
            'selected_models': selected_models,
            'generative_model_name': generative_model_name,
            'top_k': top_k,
            'use_llm_reranker': use_llm_reranker,  # Compatibilidad hacia atrás
            'reranking_method': reranking_method,  # Nuevo campo
            'generate_rag_metrics': generate_rag_metrics,
            'batch_size': batch_size,
            'evaluate_all_models': evaluate_all_models,
            'evaluation_type': 'cumulative_metrics_colab',
            'created_timestamp': time.time()
        }
        
        st.markdown("---")
        st.subheader("🚀 Crear y Enviar Configuración")
        
        # Mostrar resumen de configuración
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Resumen de Configuración:**")
            st.json({
                'num_questions': num_questions,
                'selected_models': selected_models,
                'generative_model': generative_model_name,
                'top_k': top_k,
                'reranking_method': reranking_method,
                'llm_reranker': use_llm_reranker,
                'rag_metrics': generate_rag_metrics,
                'batch_size': batch_size
            })
        
        with col2:
            st.markdown("**📁 Google Drive:**")
            st.info("📂 Carpeta: `/TesisMagister/acumulative/`")
            st.info("📄 Se creará: `evaluation_config_TIMESTAMP.json`")
        
        # Botones de acción
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Crear Configuración y Enviar a Google Drive", type="primary"):
                create_config_and_send_to_drive(evaluation_config)
        
        with col2:
            # Verificar Estado button removed per user request
            st.empty()
        
        with col3:
            # Ver Resultados button removed per user request
            st.empty()
        
        # Mostrar estado actual
        st.markdown("---")
        display_current_colab_status()
    
    else:
        st.warning("⚠️ Por favor selecciona al menos un modelo para continuar")


def create_config_and_send_to_drive(evaluation_config: Dict):
    """Crea y envía configuración a Google Drive real - Compatible con cumulative metrics y N questions"""
    
    st.info("📤 Creando configuración y enviando a Google Drive...")
    
    try:
        # Detectar tipo de evaluación
        evaluation_type = evaluation_config.get('evaluation_type', 'cumulative_metrics')
        
        # Obtener preguntas reales de la base de datos solo si no están ya incluidas
        if 'questions_data' not in evaluation_config or evaluation_config['questions_data'] is None:
            with st.spinner("📥 Obteniendo preguntas de ChromaDB..."):
                from src.data.processing import fetch_random_questions_from_chromadb
                from src.services.auth.clients import initialize_clients
                
                try:
                    # Determinar modelo y configuración según el tipo
                    if evaluation_type == 'n_questions_cumulative_analysis':
                        # Para N questions, usar la configuración anidada
                        first_model = list(evaluation_config['model_config']['embedding_models'].keys())[0]
                        num_questions = evaluation_config['data_config']['num_questions']
                        generative_model = evaluation_config['model_config']['generative_model']
                    else:
                        # Para cumulative metrics, usar la configuración plana
                        first_model = evaluation_config['selected_models'][0]
                        num_questions = evaluation_config['num_questions']
                        generative_model = evaluation_config['generative_model_name']
                    
                    # Inicializar clientes
                    chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client = initialize_clients(
                        model_name=first_model,
                        generative_model_name=generative_model
                    )
                    
                    # Obtener preguntas optimizadas de la colección pre-validada
                    with st.spinner(f"🚀 Obteniendo {num_questions} preguntas optimizadas..."):
                        from src.data.optimized_questions import get_optimized_questions_batch
                        
                        questions = get_optimized_questions_batch(
                            chromadb_wrapper=chromadb_wrapper,
                            num_questions=num_questions,
                            embedding_model_name=first_model
                        )
                        
                        if questions:
                            # Mostrar estadísticas de las preguntas obtenidas
                            total_links = sum(q.get('total_links', 0) for q in questions)
                            total_valid_links = sum(q.get('valid_links', 0) for q in questions)
                            avg_success_rate = sum(q.get('validation_success_rate', 0) for q in questions) / len(questions) * 100
                            
                            st.write(f"✅ Obtenidas {len(questions)} preguntas optimizadas")
                            st.write(f"📊 Total de links: {total_links}, Links válidos: {total_valid_links}")
                            st.write(f"🎯 Tasa promedio de validación: {avg_success_rate:.1f}%")
                        else:
                            st.warning("⚠️ No se pudieron obtener preguntas de la colección optimizada")
                    
                    if questions:
                        evaluation_config['questions_data'] = questions
                        st.success(f"✅ Obtenidas {len(questions)} preguntas con enlaces MS Learn")
                    else:
                        st.warning("⚠️ No se encontraron preguntas, usando configuración sin datos")
                        evaluation_config['questions_data'] = None
                        
                except Exception as e:
                    st.warning(f"⚠️ Error obteniendo preguntas: {e}")
                    evaluation_config['questions_data'] = None
        else:
            st.info("📝 Usando preguntas ya incluidas en la configuración")
        
        # Enviar a Google Drive real
        with st.spinner("☁️ Enviando configuración a Google Drive..."):
            # Generar nombre de archivo basado en el tipo de evaluación
            import time
            timestamp = int(time.time())
            
            if evaluation_type == 'n_questions_cumulative_analysis':
                filename = f"n_questions_config_{timestamp}.json"
            else:
                filename = f"evaluation_config_{timestamp}.json"
            
            result = create_evaluation_config_in_drive(evaluation_config, filename)
            
            if result['success']:
                st.success("✅ ¡Configuración enviada exitosamente a Google Drive!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📄 Archivo creado:**")
                    st.code(result['config_filename'])
                    if 'web_link' in result:
                        st.markdown(f"[🔗 Ver en Drive]({result['web_link']})")
                
                with col2:
                    st.markdown("**📋 Próximos pasos:**")
                    st.markdown("""
                    1. Abrir Google Colab
                    2. Subir `Universal_Colab_Evaluator.ipynb` 
                    3. Activar GPU (T4)
                    4. Ejecutar todas las celdas
                    5. Volver a la página de resultados
                    """)
                
                # Results navigation button removed per user request
                st.info("Results button removed per user request")
                
                # Botón para descargar notebook si está disponible
                try:
                    notebook_path = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Universal_Colab_Evaluator.ipynb"
                    if os.path.exists(notebook_path):
                        with open(notebook_path, 'rb') as f:
                            notebook_content = f.read()
                        
                        st.download_button(
                            label="📥 Descargar Universal_Colab_Evaluator.ipynb",
                            data=notebook_content,
                            file_name="Universal_Colab_Evaluator.ipynb",
                            mime="application/json",
                            help="Descarga el notebook para ejecutar en Google Colab"
                        )
                except Exception as e:
                    st.warning(f"⚠️ Error preparando descarga de notebook: {e}")
                
            else:
                st.error(f"❌ Error enviando configuración: {result.get('error', 'Error desconocido')}")
                
    except Exception as e:
        st.error(f"❌ Error crítico: {e}")
        st.exception(e)


def check_colab_evaluation_status():
    """Verifica el estado de la evaluación en Google Drive"""
    
    with st.spinner("🔍 Verificando estado de evaluación..."):
        status_result = check_evaluation_status_in_drive()
        
        if status_result['success']:
            status_data = status_result['status_data']
            
            st.markdown("### 📊 Estado de Evaluación")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔄 Estado", status_data.get('status', 'Desconocido'))
                st.metric("📅 Última actualización", status_data.get('timestamp', 'N/A')[:19] if status_data.get('timestamp') else 'N/A')
            
            with col2:
                st.metric("🤖 Modelos", status_data.get('models_to_evaluate', 'N/A'))
                st.metric("❓ Preguntas", status_data.get('questions_total', 'N/A'))
            
            with col3:
                st.metric("🚀 GPU", "✅" if status_data.get('gpu_used') else "❌")
                st.metric("⏱️ Tiempo", f"{status_data.get('total_time_seconds', 0):.1f}s" if status_data.get('total_time_seconds') else 'N/A')
            
            # Mostrar detalles adicionales
            if status_data.get('status') == 'completed':
                st.success("🎉 ¡Evaluación completada! Puedes ver los resultados en la página de resultados.")
                
                # Results button removed per user request
                st.info("Results button removed per user request")
                    
            elif status_data.get('status') == 'running':
                st.info("⏳ Evaluación en progreso... Verifica nuevamente en unos minutos.")
                
            elif status_data.get('status') == 'error':
                st.error("❌ Error en la evaluación. Revisa los logs de Colab.")
                
        else:
            st.warning("⚠️ No se pudo verificar el estado. Asegúrate de que Google Drive esté configurado correctamente.")


if __name__ == "__main__":
    show_cumulative_metrics_create_page()