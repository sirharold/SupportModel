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
        st.info("📊 Las preguntas se extraen aleatoriamente desde ChromaDB, filtrando solo aquellas que tienen enlaces de Microsoft Learn en la respuesta aceptada.")
        
        # Número de preguntas
        num_questions = st.number_input(
            "🔢 Número de preguntas a evaluar:",
            min_value=1,
            max_value=2000,
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
        
        # Usar reranking LLM
        use_llm_reranker = st.checkbox(
            "🤖 Usar Reranking LLM",
            value=True,
            help="Activar reordenamiento de documentos con LLM"
        )
        
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
            st.write(f"• LLM Reranking: {'✅' if use_llm_reranker else '❌'}")
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
            'use_llm_reranker': use_llm_reranker,
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
    """Crea y envía configuración a Google Drive real"""
    
    st.info("📤 Creando configuración y enviando a Google Drive...")
    
    try:
        # Obtener preguntas reales de la base de datos
        with st.spinner("📥 Obteniendo preguntas de ChromaDB..."):
            from src.data.processing import fetch_random_questions_from_chromadb
            from src.services.auth.clients import initialize_clients
            
            try:
                # Usar el primer modelo seleccionado para obtener las preguntas
                first_model = evaluation_config['selected_models'][0]
                chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, client = initialize_clients(
                    model_name=first_model,
                    generative_model_name=evaluation_config['generative_model_name']
                )
                questions = fetch_random_questions_from_chromadb(
                    chromadb_wrapper=chromadb_wrapper,
                    embedding_model_name=first_model,
                    num_questions=evaluation_config['num_questions']
                )
                
                if questions:
                    evaluation_config['questions_data'] = questions
                    st.success(f"✅ Obtenidas {len(questions)} preguntas con enlaces MS Learn")
                else:
                    st.warning("⚠️ No se encontraron preguntas, usando configuración sin datos")
                    evaluation_config['questions_data'] = None
                    
            except Exception as e:
                st.warning(f"⚠️ Error obteniendo preguntas: {e}")
                evaluation_config['questions_data'] = None
        
        # Enviar a Google Drive real
        with st.spinner("☁️ Enviando configuración a Google Drive..."):
            result = create_evaluation_config_in_drive(evaluation_config)
            
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