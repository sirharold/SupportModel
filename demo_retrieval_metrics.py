#!/usr/bin/env python3
"""
Demo script para mostrar cómo usar las métricas de recuperación en el sistema RAG.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.qa_pipeline_with_metrics import answer_question_with_retrieval_metrics
from src.evaluation.metrics.retrieval import format_metrics_for_display
from src.services.auth.clients import initialize_clients
from src.config.config import EMBEDDING_MODELS


def demo_single_question_metrics():
    """
    Demostración de métricas para una sola pregunta.
    """
    print("🎯 DEMO: MÉTRICAS DE RECUPERACIÓN PARA UNA PREGUNTA")
    print("=" * 80)
    
    # Pregunta de ejemplo
    question = "¿Cómo configurar Azure Blob Storage?"
    
    # Respuesta aceptada simulada (ground truth)
    ground_truth_answer = """
    Para configurar Azure Blob Storage, sigue estos pasos:
    
    1. Crear una cuenta de almacenamiento: https://learn.microsoft.com/azure/storage/common/storage-account-create
    2. Configurar contenedores: https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction
    3. Establecer permisos: https://learn.microsoft.com/azure/storage/blobs/storage-blob-container-properties-metadata
    
    También puedes consultar la documentación general de Azure Storage.
    """
    
    # Enlaces de Microsoft Learn extraídos
    ms_links = [
        "https://learn.microsoft.com/azure/storage/common/storage-account-create",
        "https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction",
        "https://learn.microsoft.com/azure/storage/blobs/storage-blob-container-properties-metadata"
    ]
    
    print(f"📋 Pregunta: {question}")
    print(f"📋 Ground truth links: {len(ms_links)}")
    print(f"📋 Modelos a evaluar: {list(EMBEDDING_MODELS.keys())}")
    print()
    
    # Evaluar cada modelo
    for model_key in EMBEDDING_MODELS.keys():
        print(f"🔍 Evaluando modelo: {model_key}")
        print("-" * 60)
        
        try:
            # Inicializar clientes
            chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(model_key)
            
            # Ejecutar pipeline con métricas
            result = answer_question_with_retrieval_metrics(
                question=question,
                chromadb_wrapper=chromadb_wrapper,
                embedding_client=embedding_client,
                openai_client=openai_client,
                gemini_client=gemini_client,
                local_tinyllama_client=local_tinyllama_client,
                local_mistral_client=local_mistral_client,
                openrouter_client=openrouter_client,
                top_k=10,
                use_llm_reranker=True,
                generate_answer=False,  # Solo documentos para métricas
                calculate_metrics=True,
                ground_truth_answer=ground_truth_answer,
                ms_links=ms_links,
                generative_model_name='llama-4-scout'
            )
            
            # Mostrar resultados
            if len(result) >= 3:
                docs, debug_info, retrieval_metrics = result
                
                print(f"📊 Documentos recuperados: {len(docs)}")
                print(f"📊 Ground truth detectado: {retrieval_metrics.get('ground_truth_links_count', 0)} enlaces")
                
                # Mostrar métricas formateadas
                formatted_metrics = format_metrics_for_display(retrieval_metrics)
                print("\n" + formatted_metrics)
                
            else:
                print("❌ Error: Resultado incompleto del pipeline")
                
        except Exception as e:
            print(f"❌ Error evaluando {model_key}: {e}")
        
        print("\n" + "=" * 80)
    
    return True


def demo_batch_metrics():
    """
    Demostración de métricas para múltiples preguntas.
    """
    print("🎯 DEMO: MÉTRICAS DE RECUPERACIÓN PARA MÚLTIPLES PREGUNTAS")
    print("=" * 80)
    
    # Preguntas de ejemplo
    questions_and_answers = [
        {
            'question': "¿Cómo crear una máquina virtual en Azure?",
            'accepted_answer': """
                Para crear una VM en Azure:
                1. Accede al portal: https://learn.microsoft.com/azure/virtual-machines/windows/quick-create-portal
                2. Configura la red: https://learn.microsoft.com/azure/virtual-network/virtual-networks-overview
                3. Establecer seguridad: https://learn.microsoft.com/azure/virtual-machines/security-policy
                """,
            'ms_links': [
                "https://learn.microsoft.com/azure/virtual-machines/windows/quick-create-portal",
                "https://learn.microsoft.com/azure/virtual-network/virtual-networks-overview",
                "https://learn.microsoft.com/azure/virtual-machines/security-policy"
            ]
        },
        {
            'question': "¿Cómo configurar Azure Functions?",
            'accepted_answer': """
                Para configurar Azure Functions:
                1. Crear función: https://learn.microsoft.com/azure/azure-functions/functions-create-first-function-vs-code
                2. Configurar triggers: https://learn.microsoft.com/azure/azure-functions/functions-triggers-bindings
                3. Deploy: https://learn.microsoft.com/azure/azure-functions/functions-deployment-technologies
                """,
            'ms_links': [
                "https://learn.microsoft.com/azure/azure-functions/functions-create-first-function-vs-code",
                "https://learn.microsoft.com/azure/azure-functions/functions-triggers-bindings",
                "https://learn.microsoft.com/azure/azure-functions/functions-deployment-technologies"
            ]
        }
    ]
    
    print(f"📋 Preguntas a evaluar: {len(questions_and_answers)}")
    print(f"📋 Modelo a usar: multi-qa-mpnet-base-dot-v1 (ejemplo)")
    print()
    
    # Usar solo un modelo para el demo
    model_key = "multi-qa-mpnet-base-dot-v1"
    
    try:
        # Inicializar clientes
        chromadb_wrapper, embedding_client, openai_client, gemini_client, local_tinyllama_client, local_mistral_client, openrouter_client, _ = initialize_clients(model_key)
        
        # Evaluar cada pregunta
        all_results = []
        
        for i, qa_pair in enumerate(questions_and_answers):
            print(f"🔍 Evaluando pregunta {i+1}/{len(questions_and_answers)}")
            print(f"📋 Pregunta: {qa_pair['question']}")
            
            try:
                result = answer_question_with_retrieval_metrics(
                    question=qa_pair['question'],
                    chromadb_wrapper=chromadb_wrapper,
                    embedding_client=embedding_client,
                    openai_client=openai_client,
                    gemini_client=gemini_client,
                    local_tinyllama_client=local_tinyllama_client,
                    local_mistral_client=local_mistral_client,
                    openrouter_client=openrouter_client,
                    top_k=10,
                    use_llm_reranker=True,
                    generate_answer=False,
                    calculate_metrics=True,
                    ground_truth_answer=qa_pair['accepted_answer'],
                    ms_links=qa_pair['ms_links'],
                    generative_model_name='llama-4-scout'
                )
                
                if len(result) >= 3:
                    docs, debug_info, retrieval_metrics = result
                    all_results.append(retrieval_metrics)
                    
                    # Mostrar métricas principales
                    before = retrieval_metrics.get('before_reranking', {})
                    after = retrieval_metrics.get('after_reranking', {})
                    
                    print(f"  📊 MRR: {before.get('MRR', 0):.4f} → {after.get('MRR', 0):.4f}")
                    print(f"  📊 Precision@5: {before.get('Precision@5', 0):.4f} → {after.get('Precision@5', 0):.4f}")
                    print(f"  📊 Documentos: {retrieval_metrics.get('docs_before_count', 0)} → {retrieval_metrics.get('docs_after_count', 0)}")
                    
                else:
                    print("  ❌ Error: Resultado incompleto")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print()
        
        # Mostrar resumen agregado
        if all_results:
            print("📊 RESUMEN AGREGADO:")
            print("-" * 50)
            
            # Calcular promedios
            metrics_keys = ['MRR', 'Recall@1', 'Recall@5', 'Precision@1', 'Precision@5', 'F1@1', 'F1@5']
            
            for metric_key in metrics_keys:
                before_values = [r['before_reranking'].get(metric_key, 0) for r in all_results if 'before_reranking' in r]
                after_values = [r['after_reranking'].get(metric_key, 0) for r in all_results if 'after_reranking' in r]
                
                if before_values and after_values:
                    before_avg = sum(before_values) / len(before_values)
                    after_avg = sum(after_values) / len(after_values)
                    improvement = after_avg - before_avg
                    
                    print(f"{metric_key}: {before_avg:.4f} → {after_avg:.4f} (Δ{improvement:+.4f})")
            
        else:
            print("❌ No se pudieron calcular métricas para ninguna pregunta")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
    
    print("\n" + "=" * 80)
    return True


def main():
    """
    Ejecuta las demostraciones.
    """
    print("🚀 DEMO DE MÉTRICAS DE RECUPERACIÓN")
    print("=" * 80)
    print("⚠️  NOTA: Esta demo requiere conexión a Weaviate y modelos configurados")
    print("⚠️  Si no tienes la configuración completa, algunos tests pueden fallar")
    print("=" * 80)
    
    demos = [
        ("📊 Single Question Demo", demo_single_question_metrics),
        ("📊 Batch Questions Demo", demo_batch_metrics)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n🎬 EJECUTANDO: {demo_name}")
        print("-" * 80)
        
        try:
            demo_func()
            print(f"✅ {demo_name} completado")
        except Exception as e:
            print(f"❌ {demo_name} falló: {e}")
    
    print("\n" + "=" * 80)
    print("🏁 DEMO COMPLETADO")
    print("📋 Para usar en producción:")
    print("   1. Configura las variables de entorno (Weaviate, OpenAI, etc.)")
    print("   2. Importa answer_question_with_retrieval_metrics")
    print("   3. Usa calculate_metrics=True para obtener métricas")
    print("   4. Usa format_metrics_for_display para mostrar resultados")


if __name__ == "__main__":
    main()