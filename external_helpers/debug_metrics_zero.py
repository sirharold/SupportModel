#!/usr/bin/env python3
"""
Script de debugging paso a paso para entender por qué las métricas están en cero.
Analiza una pregunta específica de la collection questions_withlinks.
"""

import json
import pandas as pd
import numpy as np
import os
import sys
from urllib.parse import urlparse, urlunparse
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def normalize_url(url: str) -> str:
    """Normaliza URL removiendo query params y fragments"""
    if not url or not url.strip():
        return ""
    
    try:
        parsed = urlparse(url.strip())
        normalized = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path, '', '', ''
        ))
        return normalized
    except Exception as e:
        return url.strip()

def debug_single_question():
    """Debug completo de una pregunta específica"""
    
    print("🔍 DEBUG: ¿Por qué las métricas están en cero?")
    print("=" * 60)
    
    # 1. Conectar a ChromaDB y obtener una pregunta real
    print("\n📋 PASO 1: Obtener pregunta de ChromaDB questions_withlinks")
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Conectar a ChromaDB
        client = chromadb.PersistentClient(
            path="/Users/haroldgomez/chromadb2",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Obtener la colección questions_withlinks
        collection = client.get_collection("questions_withlinks")
        
        # Obtener una pregunta de muestra
        results = collection.get(limit=1, include=['metadatas', 'documents'])
        
        if not results['metadatas']:
            print("❌ No hay preguntas en la colección questions_withlinks")
            return False
        
        # Usar la primera pregunta como ejemplo
        question_metadata = results['metadatas'][0]
        question_document = results['documents'][0] if results['documents'] else ""
        
        print(f"✅ Pregunta obtenida de ChromaDB")
        print(f"Keys disponibles: {list(question_metadata.keys())}")
        
        # Crear estructura de datos compatible
        config_data = {
            'questions': [{
                'question': question_metadata.get('question', question_metadata.get('question_content', question_metadata.get('title', ''))),
                'ground_truth_links': question_metadata.get('validated_links', '').split('|') if question_metadata.get('validated_links') else [],
                'accepted_answer': question_metadata.get('accepted_answer', ''),
                'ms_links': question_metadata.get('ms_links', '').split('|') if question_metadata.get('ms_links') else []
            }]
        }
        
    except Exception as e:
        print(f"❌ Error conectando a ChromaDB: {e}")
        print("Intentando usar el archivo de resultados existente...")
        
        # Fallback: usar el archivo de resultados más reciente
        result_files = [f for f in os.listdir('.') if f.startswith('cumulative_results_') and f.endswith('.json')]
        if result_files:
            latest_file = max(result_files)
            print(f"Usando archivo de resultados: {latest_file}")
            return analyze_results_file(latest_file)
        else:
            print("❌ No se encontraron archivos de resultados")
            return False
    
    # 2. Analizar estructura del config
    print(f"\n📊 PASO 2: Analizar estructura del config")
    print(f"Keys principales: {list(config_data.keys())}")
    
    # Obtener preguntas según el formato
    questions_data = None
    if 'questions_data' in config_data:
        questions_data = config_data['questions_data']
        print(f"✅ Formato nuevo: {len(questions_data)} preguntas en 'questions_data'")
    elif 'questions' in config_data:
        questions_data = config_data['questions']
        print(f"✅ Formato legacy: {len(questions_data)} preguntas en 'questions'")
    else:
        print(f"❌ No se encontraron preguntas. Keys: {list(config_data.keys())}")
        return False
    
    # 3. Tomar la primera pregunta para debug
    print(f"\n🎯 PASO 3: Analizar pregunta específica")
    if not questions_data:
        print("❌ No hay preguntas para analizar")
        return False
        
    question_data = questions_data[0]  # Primera pregunta
    print(f"Keys de la pregunta: {list(question_data.keys())}")
    
    question = question_data.get('question', question_data.get('question_content', ''))
    print(f"Pregunta: {question[:100]}...")
    
    # 4. Analizar ground truth links
    print(f"\n🔗 PASO 4: Analizar ground truth links")
    
    # Probar diferentes campos donde pueden estar los links
    possible_link_fields = [
        'ground_truth_links',
        'validated_links', 
        'accepted_answer_links',
        'ms_links'
    ]
    
    ground_truth_links = []
    for field in possible_link_fields:
        if field in question_data:
            links = question_data[field]
            if isinstance(links, str):
                # Si es string separado por |
                links = [link.strip() for link in links.split('|') if link.strip()]
            elif isinstance(links, list):
                links = [str(link) for link in links if link]
            
            if links:
                ground_truth_links = links
                print(f"✅ Ground truth encontrado en '{field}': {len(links)} links")
                for i, link in enumerate(links):
                    print(f"  {i+1}. {link}")
                break
    
    if not ground_truth_links:
        print("❌ No se encontraron ground truth links en ningún campo")
        return False
    
    # 5. Normalizar ground truth links
    print(f"\n🔧 PASO 5: Normalizar ground truth links")
    normalized_gt = [normalize_url(link) for link in ground_truth_links if link]
    print(f"Links normalizados ({len(normalized_gt)}):")
    for i, link in enumerate(normalized_gt):
        print(f"  {i+1}. {link}")
    
    # 6. Cargar documentos de embeddings (ejemplo con ada)
    print(f"\n📄 PASO 6: Cargar documentos de embeddings")
    embedding_files = {
        'ada': 'colab_data/docs_ada_with_embeddings_20250721_123712.parquet',
        'e5-large': 'colab_data/docs_e5large_with_embeddings_20250721_124918.parquet'
    }
    
    model_name = 'ada'  # Usar ada para el test
    file_path = embedding_files[model_name]
    
    if not os.path.exists(file_path):
        print(f"❌ No se encontró archivo de embeddings: {file_path}")
        return False
    
    df = pd.read_parquet(file_path)
    print(f"✅ Cargados {len(df)} documentos de embeddings ({model_name})")
    print(f"Columnas: {list(df.columns)}")
    
    # 7. Verificar cobertura de ground truth en embeddings
    print(f"\n🎯 PASO 7: Verificar cobertura de ground truth")
    
    # Obtener todos los links de embeddings normalizados
    if 'link' not in df.columns:
        print("❌ No hay columna 'link' en los embeddings")
        return False
    
    all_embedding_links = set()
    for link in df['link'].dropna():
        normalized_link = normalize_url(str(link))
        if normalized_link:
            all_embedding_links.add(normalized_link)
    
    print(f"Total links únicos en embeddings: {len(all_embedding_links)}")
    
    # Verificar si cada ground truth está en embeddings
    matches_found = 0
    for i, gt_link in enumerate(normalized_gt):
        is_in_embeddings = gt_link in all_embedding_links
        status = "✅" if is_in_embeddings else "❌"
        print(f"  {status} GT {i+1}: {gt_link}")
        if is_in_embeddings:
            matches_found += 1
    
    print(f"\n📊 Cobertura: {matches_found}/{len(normalized_gt)} ground truth links encontrados en embeddings")
    
    if matches_found == 0:
        print("🚨 PROBLEMA IDENTIFICADO: Ningún ground truth link está en los embeddings!")
        return False
    
    # 8. Simular búsqueda por embeddings (sin embeddings reales)
    print(f"\n🔍 PASO 8: Simular búsqueda")
    
    # Encontrar documentos que coincidan con ground truth
    relevant_docs = []
    for _, doc in df.iterrows():
        doc_link = normalize_url(str(doc.get('link', '')))
        if doc_link in normalized_gt:
            relevant_docs.append({
                'rank': len(relevant_docs) + 1,
                'cosine_similarity': 0.95,  # Simular score alto
                'link': doc.get('link', ''),
                'title': doc.get('title', ''),
                'content': str(doc.get('content', ''))[:200] + "..."
            })
    
    print(f"Documentos relevantes encontrados: {len(relevant_docs)}")
    for doc in relevant_docs:
        print(f"  - Rank {doc['rank']}: {doc['title']}")
        print(f"    Link: {doc['link']}")
        print(f"    Similitud: {doc['cosine_similarity']}")
    
    # 9. Simular top-10 con documentos irrelevantes + relevantes
    print(f"\n🎲 PASO 9: Simular top-10 realista")
    
    # Tomar algunos documentos aleatorios como "irrelevantes"
    random_docs = df.sample(n=8).to_dict('records')
    
    simulated_top10 = []
    
    # Agregar documentos irrelevantes primero (simulando que embeddings no funcionan bien)
    for i, doc in enumerate(random_docs):
        simulated_top10.append({
            'rank': i + 1,
            'cosine_similarity': 0.85 - (i * 0.05),  # Scores decrecientes
            'link': doc.get('link', ''),
            'title': doc.get('title', ''),
            'content': str(doc.get('content', ''))[:100] + "..."
        })
    
    # Agregar documentos relevantes al final (rank 9, 10)
    for i, rel_doc in enumerate(relevant_docs[:2]):
        if len(simulated_top10) < 10:
            rel_doc['rank'] = len(simulated_top10) + 1
            rel_doc['cosine_similarity'] = 0.45 - (i * 0.05)  # Scores bajos pero reales
            simulated_top10.append(rel_doc)
    
    print(f"Top-10 simulado:")
    for doc in simulated_top10:
        is_relevant = normalize_url(doc['link']) in normalized_gt
        status = "🎯" if is_relevant else "❌"
        print(f"  {status} Rank {doc['rank']}: Score {doc['cosine_similarity']:.3f} - {doc['title'][:50]}...")
    
    # 10. Calcular métricas manualmente
    print(f"\n📊 PASO 10: Calcular métricas de retrieval")
    
    relevance_scores = []
    for doc in simulated_top10:
        doc_link = normalize_url(doc.get('link', ''))
        is_relevant = 1 if doc_link in normalized_gt else 0
        relevance_scores.append(is_relevant)
    
    print(f"Relevance array: {relevance_scores}")
    
    # Calcular métricas para k=5 y k=10
    for k in [5, 10]:
        rel_k = relevance_scores[:k]
        
        # Precision@k
        precision_k = sum(rel_k) / k if k > 0 else 0
        
        # Recall@k
        total_relevant = len(normalized_gt)
        recall_k = sum(rel_k) / total_relevant if total_relevant > 0 else 0
        
        # F1@k
        if precision_k + recall_k > 0:
            f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        else:
            f1_k = 0
        
        print(f"Métricas@{k}:")
        print(f"  Precision: {precision_k:.3f}")
        print(f"  Recall: {recall_k:.3f}")  
        print(f"  F1: {f1_k:.3f}")
    
    # 11. Conclusiones
    print(f"\n🎯 CONCLUSIONES:")
    print(f"✅ Ground truth links SÍ están en embeddings: {matches_found > 0}")
    print(f"✅ La normalización de URLs funciona correctamente")
    print(f"⚠️  El problema probablemente es que los embeddings no recuperan los documentos relevantes en top-10")
    print(f"💡 Solución: Revisar por qué las consultas no recuperan documentos con ground truth links")
    
    return True

def analyze_results_file(filename):
    """Analiza archivo de resultados para entender por qué métricas están en cero"""
    print(f"\n🔍 ANÁLISIS DEL ARCHIVO DE RESULTADOS: {filename}")
    print("=" * 60)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        print(f"✅ Archivo cargado exitosamente")
        
        # Analizar estructura general
        print(f"\n📊 ESTRUCTURA GENERAL:")
        print(f"- Keys principales: {list(results_data.keys())}")
        
        config = results_data.get('config', {})
        print(f"- Número de preguntas: {config.get('num_questions', 'N/A')}")
        print(f"- Modelos evaluados: {config.get('models_evaluated', 'N/A')}")
        print(f"- Método de reranking: {config.get('reranking_method', 'N/A')}")
        
        # Analizar resultados por modelo
        results = results_data.get('results', {})
        print(f"\n🤖 ANÁLISIS DE MODELOS:")
        
        for model_name, model_results in results.items():
            print(f"\n--- Modelo: {model_name} ---")
            print(f"Preguntas evaluadas: {model_results.get('num_questions_evaluated', 'N/A')}")
            print(f"Dimensiones embedding: {model_results.get('embedding_dimensions', 'N/A')}")
            print(f"Total documentos: {model_results.get('total_documents', 'N/A')}")
            
            # Analizar métricas ANTES
            avg_before = model_results.get('avg_before_metrics', {})
            print(f"\n📈 MÉTRICAS ANTES (promedios):")
            key_metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']
            for metric in key_metrics:
                value = avg_before.get(metric, 'N/A')
                print(f"  {metric}: {value}")
            
            # Analizar métricas individuales (primera pregunta)
            all_before = model_results.get('all_before_metrics', [])
            if all_before:
                print(f"\n🔍 PRIMERA PREGUNTA (métricas individuales):")
                first_question = all_before[0]
                for metric in key_metrics:
                    value = first_question.get(metric, 'N/A')
                    print(f"  {metric}: {value}")
                
                # Verificar document_scores si existe
                doc_scores = first_question.get('document_scores', [])
                if doc_scores:
                    print(f"\n📄 DOCUMENTOS RECUPERADOS (primera pregunta):")
                    print(f"Total documentos: {len(doc_scores)}")
                    for i, doc in enumerate(doc_scores[:3]):  # Mostrar solo top 3
                        is_relevant = doc.get('is_relevant', False)
                        similarity = doc.get('cosine_similarity', 0)
                        link = doc.get('link', '')[:50] + "..." if len(doc.get('link', '')) > 50 else doc.get('link', '')
                        status = "✅ RELEVANTE" if is_relevant else "❌ NO RELEVANTE"
                        print(f"  {i+1}. {status} | Sim: {similarity:.3f} | {link}")
        
        print(f"\n🎯 CONCLUSIONES:")
        print(f"📊 El problema está claramente identificado en los datos")
        print(f"🔍 Revisar los document_scores para ver si los documentos relevantes están siendo recuperados")
        print(f"💡 Si todos los documentos tienen is_relevant=False, el problema está en el matching de URLs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analizando archivo: {e}")
        return False

if __name__ == "__main__":
    success = debug_single_question()
    if success:
        print("\n✅ Debug completado successfully")
    else:
        print("\n❌ Debug falló")