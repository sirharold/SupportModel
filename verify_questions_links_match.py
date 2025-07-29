#!/usr/bin/env python3
"""
Script para verificar que los links de las preguntas en questions_withlinks
existan en la colección docs_ada.

Este script analiza:
1. Cuántas preguntas hay en questions_withlinks
2. Para cada pregunta, extrae los links de accepted_answer
3. Normaliza los links usando la misma función
4. Verifica si existen en docs_ada
5. Reporta estadísticas de cobertura
"""

import os
import sys
import chromadb
from chromadb.config import Settings
from urllib.parse import urlparse, urlunparse
import re
from typing import Set, List, Dict

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def normalize_url(url: str) -> str:
    """
    Normaliza URL removiendo query params y fragments - igual que en el código del Colab
    """
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

def extract_ms_links(text: str) -> List[str]:
    """
    Extrae links de Microsoft Learn del texto - igual que en create_questions_withlinks_collection.py
    """
    if not text:
        return []
    
    # Patrón para detectar URLs de Microsoft Learn
    ms_patterns = [
        r'https?://learn\.microsoft\.com/[^\s\)\]\"\'\,\;]*',
        r'https?://docs\.microsoft\.com/[^\s\)\]\"\'\,\;]*',
        r'https?://technet\.microsoft\.com/[^\s\)\]\"\'\,\;]*',
        r'https?://support\.microsoft\.com/[^\s\)\]\"\'\,\;]*'
    ]
    
    all_links = []
    for pattern in ms_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        all_links.extend(matches)
    
    # Limpiar links
    cleaned_links = []
    for link in all_links:
        # Remover caracteres de puntuación al final
        link = re.sub(r'[.,;!?]+$', '', link)
        if link:
            cleaned_links.append(link)
    
    return list(set(cleaned_links))  # Remover duplicados

def get_all_doc_links_from_parquet() -> Set[str]:
    """
    Obtiene todos los links normalizados del archivo parquet docs_ada
    """
    import pandas as pd
    
    print("📥 Obteniendo todos los links del archivo parquet docs_ada...")
    
    try:
        # Cargar datos desde parquet
        parquet_file = "colab_data/docs_ada_with_embeddings_20250721_123712.parquet"
        
        if not os.path.exists(parquet_file):
            print(f"❌ Archivo parquet no encontrado: {parquet_file}")
            return set()
        
        df = pd.read_parquet(parquet_file)
        print(f"✅ Archivo parquet cargado: {len(df):,} documentos")
        print(f"📊 Columnas disponibles: {list(df.columns)}")
        
        # Verificar que existe la columna link
        if 'link' not in df.columns:
            print("❌ No se encontró columna 'link' en el parquet")
            return set()
        
        # Normalizar todos los links
        all_links = set()
        for link in df['link'].dropna():
            normalized_link = normalize_url(str(link))
            if normalized_link:
                all_links.add(normalized_link)
        
        print(f"✅ Obtenidos {len(all_links)} links únicos normalizados del parquet")
        return all_links
        
    except Exception as e:
        print(f"❌ Error obteniendo links del parquet: {e}")
        return set()

def get_questions_from_files() -> List[Dict]:
    """
    Busca preguntas en archivos disponibles, incluyendo archivos de resultados
    """
    import pandas as pd
    import json
    
    print("📥 Buscando preguntas en archivos disponibles...")
    
    all_questions = []
    
    # 1. Primero intentar extraer preguntas del archivo de resultados
    result_files = [f for f in os.listdir('.') if f.startswith('cumulative_results_') and f.endswith('.json')]
    
    if result_files:
        latest_result = max(result_files)
        print(f"📊 Analizando archivo de resultados: {latest_result}")
        
        try:
            with open(latest_result, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            # Extraer preguntas del primer modelo (ada)
            ada_results = results_data.get('results', {}).get('ada', {})
            individual_rag = ada_results.get('individual_rag_metrics', [])
            
            if individual_rag:
                print(f"✅ Encontradas {len(individual_rag)} preguntas con métricas RAG")
                
                # Para cada métrica RAG individual, necesitamos reconstruir la pregunta
                # Pero no tenemos las preguntas originales en el archivo de resultados
                print("⚠️ El archivo de resultados no contiene las preguntas originales")
        
        except Exception as e:
            print(f"⚠️ Error analizando archivo de resultados: {e}")
    
    # 2. Buscar en archivos de datos específicos
    data_files = [
        "./data/train_set.json",
        "./data/val_set.json", 
        "./data/train_triplets.json"
    ]
    
    for json_file in data_files:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"📁 Analizando {json_file}...")
                
                # Verificar si es lista de preguntas
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if isinstance(first_item, dict):
                        # Verificar si tiene estructura de pregunta
                        if any(key in first_item for key in ['question', 'query', 'title', 'question_content']):
                            print(f"✅ {json_file}: {len(data)} elementos con estructura de pregunta")
                            all_questions.extend(data)
                        else:
                            print(f"⚠️ {json_file}: {len(data)} elementos, pero no parece ser de preguntas")
                            print(f"   Keys encontradas: {list(first_item.keys())}")
                
                # Verificar si tiene estructura anidada
                elif isinstance(data, dict):
                    for key in ['questions', 'questions_data', 'data', 'items']:
                        if key in data and isinstance(data[key], list):
                            questions = data[key]
                            print(f"✅ {json_file}: {len(questions)} preguntas en '{key}'")
                            all_questions.extend(questions)
                            break
            
            except Exception as e:
                print(f"⚠️ Error leyendo {json_file}: {e}")
    
    # 3. Buscar en cualquier JSON que contenga "question"
    print("\n📂 Buscando en otros archivos JSON...")
    json_files = []
    for root, dirs, files in os.walk("."):
        # Evitar directorios de dependencias
        if 'stable_env' in root or 'node_modules' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    # Leer una muestra del archivo para ver si contiene preguntas
                    with open(full_path, 'r', encoding='utf-8') as f:
                        sample = f.read(1000)  # Leer primeros 1000 caracteres
                        if '"question"' in sample.lower() or '"accepted_answer"' in sample.lower():
                            json_files.append(full_path)
                except:
                    continue
    
    print(f"📁 Archivos JSON adicionales con contenido de preguntas: {json_files}")
    
    for json_file in json_files[:3]:  # Limitar a 3 archivos adicionales
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Analizar estructura
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict) and any(key in first_item for key in ['question', 'accepted_answer']):
                    print(f"✅ {json_file}: {len(data)} preguntas adicionales")
                    all_questions.extend(data)
        
        except Exception as e:
            print(f"⚠️ Error leyendo {json_file}: {e}")
    
    print(f"✅ Total preguntas encontradas: {len(all_questions)}")
    return all_questions

def analyze_questions_links():
    """
    Función principal para analizar preguntas y sus links usando archivos parquet
    """
    print("🔍 VERIFICACIÓN DE LINKS: archivos de preguntas vs docs_ada parquet")
    print("=" * 70)
    
    try:
        # Cargar documentos desde parquet
        print("\n📋 PASO 1: Cargando documentos desde parquet...")
        doc_links = get_all_doc_links_from_parquet()
        if not doc_links:
            print("❌ No se pudieron obtener links de documentos")
            return False
        
        # Buscar preguntas en archivos JSON
        print("\n📋 PASO 2: Buscando preguntas...")
        all_questions = get_questions_from_files()
        if not all_questions:
            print("❌ No se encontraron preguntas")
            return False
        
        # Analizar preguntas
        print(f"\n🔍 PASO 3: Analizando {len(all_questions)} preguntas...")
        
        total_questions = len(all_questions)
        questions_with_links = 0
        questions_with_matches = 0
        total_links_found = 0
        total_links_matched = 0
        
        # Estadísticas detalladas
        questions_by_match_count = {0: 0, 1: 0, 2: 0, 3: 0, 'more': 0}
        sample_matches = []
        sample_no_matches = []
        
        # Procesar cada pregunta
        for i, question_data in enumerate(all_questions):
            if (i + 1) % 500 == 0:
                print(f"   Procesadas {i+1:,}/{total_questions:,} preguntas...")
            
            # Obtener accepted_answer desde diferentes campos posibles
            accepted_answer = (
                question_data.get('accepted_answer', '') or
                question_data.get('accepted_answer_text', '') or
                question_data.get('answer', '')
            )
            
            if not accepted_answer:
                continue
            
            # Extraer links del accepted_answer
            ms_links = extract_ms_links(accepted_answer)
            if not ms_links:
                continue
            
            questions_with_links += 1
            total_links_found += len(ms_links)
            
            # Normalizar links y verificar matches
            matched_links = []
            for link in ms_links:
                normalized_link = normalize_url(link)
                if normalized_link and normalized_link in doc_links:
                    matched_links.append(normalized_link)
                    total_links_matched += 1
            
            # Contar preguntas con matches
            if matched_links:
                questions_with_matches += 1
                
                # Categorizar por número de matches
                match_count = len(matched_links)
                if match_count <= 3:
                    questions_by_match_count[match_count] += 1
                else:
                    questions_by_match_count['more'] += 1
                
                # Guardar muestra de matches
                if len(sample_matches) < 3:
                    question_text = (
                        question_data.get('question', '') or
                        question_data.get('question_content', '') or
                        question_data.get('title', '') or
                        question_data.get('question_title', '')
                    )
                    sample_matches.append({
                        'question': question_text[:100] + "..." if len(question_text) > 100 else question_text,
                        'matched_links': matched_links[:2],  # Solo primeros 2
                        'total_matches': len(matched_links)
                    })
            else:
                # Guardar muestra sin matches
                if len(sample_no_matches) < 3:
                    question_text = (
                        question_data.get('question', '') or
                        question_data.get('question_content', '') or
                        question_data.get('title', '') or
                        question_data.get('question_title', '')
                    )
                    sample_no_matches.append({
                        'question': question_text[:100] + "..." if len(question_text) > 100 else question_text,
                        'links': [normalize_url(link) for link in ms_links[:2]]  # Solo primeros 2
                    })
        
        # Reportar resultados
        print(f"\n📊 RESULTADOS FINALES:")
        print("=" * 50)
        print(f"📝 Total preguntas analizadas: {total_questions:,}")
        print(f"🔗 Preguntas con links MS: {questions_with_links:,} ({questions_with_links/total_questions*100:.1f}%)")
        print(f"✅ Preguntas con matches en docs_ada: {questions_with_matches:,} ({questions_with_matches/total_questions*100:.1f}%)")
        print(f"❌ Preguntas SIN matches: {questions_with_links - questions_with_matches:,}")
        
        print(f"\n🔗 ESTADÍSTICAS DE LINKS:")
        print(f"📊 Total links encontrados: {total_links_found:,}")
        print(f"✅ Links que matchean en docs_ada: {total_links_matched:,} ({total_links_matched/total_links_found*100:.1f}%)")
        
        print(f"\n📈 DISTRIBUCIÓN DE MATCHES POR PREGUNTA:")
        print(f"🚫 0 matches: {questions_by_match_count[0]:,} preguntas")
        print(f"1️⃣ 1 match: {questions_by_match_count[1]:,} preguntas")
        print(f"2️⃣ 2 matches: {questions_by_match_count[2]:,} preguntas") 
        print(f"3️⃣ 3 matches: {questions_by_match_count[3]:,} preguntas")
        print(f"🔢 4+ matches: {questions_by_match_count['more']:,} preguntas")
        
        # Mostrar muestras
        print(f"\n✅ MUESTRA DE PREGUNTAS CON MATCHES:")
        for i, sample in enumerate(sample_matches, 1):
            print(f"{i}. {sample['question']}")
            print(f"   Matches ({sample['total_matches']}): {sample['matched_links']}")
        
        print(f"\n❌ MUESTRA DE PREGUNTAS SIN MATCHES:")
        for i, sample in enumerate(sample_no_matches, 1):
            print(f"{i}. {sample['question']}")
            print(f"   Links no encontrados: {sample['links']}")
        
        # Conclusiones
        print(f"\n🎯 CONCLUSIONES:")
        coverage_rate = questions_with_matches / questions_with_links * 100 if questions_with_links > 0 else 0
        
        if coverage_rate > 80:
            print(f"✅ EXCELENTE: {coverage_rate:.1f}% de cobertura - La mayoría de preguntas tienen documentos")
        elif coverage_rate > 60:
            print(f"⚠️ BUENO: {coverage_rate:.1f}% de cobertura - Aceptable pero mejorable")
        elif coverage_rate > 40:
            print(f"⚠️ REGULAR: {coverage_rate:.1f}% de cobertura - Problema moderado de cobertura")
        else:
            print(f"🚨 PROBLEMA: {coverage_rate:.1f}% de cobertura - Cobertura muy baja")
        
        if questions_with_matches > 0:
            print(f"💡 Las métricas cero probablemente se deben a problemas de calidad de embeddings")
            print(f"💡 {questions_with_matches:,} preguntas SÍ deberían tener documentos relevantes")
        else:
            print(f"🚨 NINGUNA pregunta tiene matches - Problema grave en la pipeline de datos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        return False

if __name__ == "__main__":
    success = analyze_questions_links()
    if success:
        print("\n✅ Análisis completado exitosamente")
    else:
        print("\n❌ Análisis falló")