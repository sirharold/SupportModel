#!/usr/bin/env python3
"""
Script para probar que la lectura de configuración funciona correctamente
con el archivo de configuración actualizado.
"""

import json
import os

def test_config_reading(config_path):
    """
    Prueba la lectura de configuración simulando el comportamiento del Colab
    """
    print(f"🔍 TESTING CONFIG READING: {os.path.basename(config_path)}")
    print("=" * 60)
    
    try:
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        print(f"✅ Config loaded successfully")
        
        # Simular lectura como en el código actualizado
        params = config_data.get('params', {})
        
        # Buscar top_k primero en el nivel raíz, luego en params
        top_k = config_data.get('top_k', params.get('top_k', 10))
        
        # Buscar generate_rag_metrics en nivel raíz, luego en params  
        generate_rag = config_data.get('generate_rag_metrics', params.get('generate_rag_metrics', True))
        
        # Buscar questions_data primero, luego questions (compatibilidad)
        questions_data = config_data.get('questions_data', config_data.get('questions', []))
        
        # Otros parámetros importantes
        selected_models = config_data.get('selected_models', [])
        reranking_method = config_data.get('reranking_method', 'standard')
        use_llm_reranker = config_data.get('use_llm_reranker', False)
        
        print(f"\n📊 PARÁMETROS LEÍDOS:")
        print(f"├── top_k: {top_k}")
        print(f"├── generate_rag_metrics: {generate_rag}")
        print(f"├── num_questions: {len(questions_data)}")
        print(f"├── selected_models: {selected_models}")
        print(f"├── reranking_method: {reranking_method}")
        print(f"├── use_llm_reranker: {use_llm_reranker}")
        
        # Verificar que top_k es el esperado
        expected_top_k = 50
        if top_k == expected_top_k:
            print(f"\n✅ TOP-K CORRECTO: {top_k} (esperado: {expected_top_k})")
        else:
            print(f"\n❌ TOP-K INCORRECTO: {top_k} (esperado: {expected_top_k})")
        
        # Verificar que tenemos preguntas
        if len(questions_data) > 0:
            print(f"✅ PREGUNTAS ENCONTRADAS: {len(questions_data)}")
            
            # Mostrar muestra de primera pregunta
            first_question = questions_data[0]
            print(f"\n📝 MUESTRA DE PRIMERA PREGUNTA:")
            print(f"├── Keys: {list(first_question.keys())}")
            question_text = first_question.get('question', first_question.get('question_content', 'N/A'))
            print(f"├── Texto: {question_text[:100]}...")
            
            # Verificar ground truth links
            validated_links = first_question.get('validated_links', [])
            if validated_links:
                print(f"├── Ground truth links: {len(validated_links)}")
                print(f"└── Primer link: {validated_links[0]}")
            else:
                print(f"└── ⚠️  No se encontraron validated_links")
        else:
            print(f"❌ NO SE ENCONTRARON PREGUNTAS")
        
        print(f"\n🎯 SIMULACIÓN DE EVALUACIÓN:")
        print(f"Se evaluarían {len(selected_models)} modelos")
        print(f"Con {len(questions_data)} preguntas")
        print(f"Recuperando top-{top_k} documentos por pregunta")
        if use_llm_reranker:
            print(f"Con reranking usando método: {reranking_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    config_path = "/Users/haroldgomez/Downloads/evaluation_config_1753771775.json"
    
    if os.path.exists(config_path):
        success = test_config_reading(config_path)
        if success:
            print(f"\n✅ Test completado exitosamente")
            print(f"💡 El código actualizado debería leer top_k=50 correctamente")
        else:
            print(f"\n❌ Test falló")
    else:
        print(f"❌ Archivo de configuración no encontrado: {config_path}")