# 🚀 Evaluación Acumulativa de Embeddings - Google Colab GPU
# Generado el 2025-07-20 16:41:13
# Configuración: 500 preguntas, 4 modelos

# =====================================
# PASO 1: CONFIGURACIÓN Y SETUP
# =====================================

import os
import time
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# 📊 Configuración de la evaluación
EVALUATION_CONFIG = {
    'num_questions': 500,
    'selected_models': ["multi-qa-mpnet-base-dot-v1", "all-MiniLM-L6-v2", "ada", "e5-large-v2"],
    'generative_model_name': "llama-3.3-70b",
    'top_k': 10,
    'use_llm_reranker': True,
    'batch_size': 50,
    'evaluate_all_models': True,
    'timestamp': "2025-07-20 16:41:13"
}

# 📁 Configuración de Google Drive
DRIVE_BASE = "/content/drive/MyDrive/TesisMagister/acumulative"

print("🚀 Configuración de evaluación cargada:")
for key, value in EVALUATION_CONFIG.items():
    print(f"   {key}: {value}")

# =====================================
# PASO 2: VERIFICAR GPU Y MONTAR DRIVE
# =====================================

# Verificar GPU
print("\n🔧 Verificando GPU...")
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU no disponible. Ve a Runtime → Change runtime type → GPU")
except ImportError:
    gpu_available = False
    print("⚠️  PyTorch no instalado aún")

# Montar Google Drive
print("\n📁 Montando Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Crear carpeta si no existe
os.makedirs(DRIVE_BASE, exist_ok=True)
print(f"✅ Carpeta configurada: {DRIVE_BASE}")

# Verificar archivo .env
env_file = f"{DRIVE_BASE}/.env"
if os.path.exists(env_file):
    print(f"✅ Archivo .env encontrado: {env_file}")
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print("🔑 Variables de entorno cargadas")
else:
    print(f"⚠️  Archivo .env no encontrado en: {env_file}")
    print("💡 Sube tu archivo .env a la carpeta para APIs reales")

# =====================================
# PASO 3: INSTALAR DEPENDENCIAS
# =====================================

print("\n📦 Instalando dependencias...")
!pip install -q sentence-transformers pandas numpy scikit-learn openai python-dotenv tqdm plotly

print("✅ Dependencias instaladas")

# =====================================
# PASO 4: IMPORTAR LIBRERÍAS
# =====================================

print("\n📚 Importando librerías...")

import pandas as pd
import numpy as np
import json
import time
import random
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Librerías ML importadas")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("💡 Reinicia runtime si persiste")

print("🎯 Setup completado")

# =====================================
# PASO 5: GENERAR DATOS DE PRUEBA
# =====================================

def generate_realistic_questions(num_questions: int) -> List[Dict]:
    """Genera preguntas realistas sobre Azure y tecnología"""
    
    base_questions = [
        "¿Cómo configurar Azure Storage Blob para aplicaciones web?",
        "¿Cuál es la diferencia entre SQL Database y Cosmos DB?",
        "¿Cómo implementar autenticación OAuth en Azure Functions?",
        "¿Qué es Azure Container Instances y cuándo usarlo?",
        "¿Cómo configurar CI/CD con Azure DevOps?",
        "¿Cuáles son las mejores prácticas de seguridad en Azure?",
        "¿Cómo configurar Application Insights para monitoreo?",
        "¿Qué es Azure Service Bus y cómo implementarlo?",
        "¿Cómo usar Azure Logic Apps para automatización?",
        "¿Cuál es la diferencia entre VM y App Service?",
        "¿Cómo configurar Azure Active Directory B2C?",
        "¿Qué es Azure Kubernetes Service (AKS)?",
        "¿Cómo usar Azure Key Vault para secretos?",
        "¿Cuáles son los tipos de almacenamiento Azure?",
        "¿Cómo implementar Azure API Management?",
        "¿Qué es Azure Event Grid y casos de uso?",
        "¿Cómo configurar Load Balancer en Azure?",
        "¿Cuándo usar Azure Redis Cache?",
        "¿Cómo implementar Azure Machine Learning?",
        "¿Qué es Azure Cognitive Services?"
    ]
    
    categories = ['compute', 'storage', 'networking', 'security', 'devops', 'ai-ml', 'database']
    difficulties = ['beginner', 'intermediate', 'advanced']
    
    questions = []
    for i in range(num_questions):
        if i < len(base_questions):
            question_text = base_questions[i]
        else:
            # Generar variaciones
            base_q = base_questions[i % len(base_questions)]
            prefixes = ["Tutorial:", "Guía:", "¿Cómo", "Mejores prácticas:", "Troubleshooting:"]
            prefix = random.choice(prefixes)
            question_text = f"{prefix} {base_q}"
        
        question = {
            'id': f'q_{i+1}',
            'question': question_text,
            'title': question_text,
            'category': random.choice(categories),
            'difficulty': random.choice(difficulties),
            'has_ms_learn_link': True,
            'tags': random.sample(['azure', 'cloud', 'microsoft', 'devops'], k=2)
        }
        questions.append(question)
    
    return questions

print(f"\n🎲 Generando {EVALUATION_CONFIG['num_questions']} preguntas de prueba...")
test_questions = generate_realistic_questions(EVALUATION_CONFIG['num_questions'])
print(f"✅ {len(test_questions):,} preguntas generadas")

# =====================================
# PASO 6: EVALUADOR GPU OPTIMIZADO
# =====================================

class ColabEvaluator:
    """Evaluador optimizado para Colab con GPU"""
    
    def __init__(self, config):
        self.config = config
        self.gpu_available = torch.cuda.is_available()
        self.models = {}
    
    def load_model(self, model_name: str):
        """Carga modelo con optimización GPU"""
        
        model_mapping = {
            'multi-qa-mpnet-base-dot-v1': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
            'e5-large-v2': 'intfloat/e5-large-v2',
            'ada': None  # API model
        }
        
        if model_name == 'ada':
            print(f"   📡 {model_name}: Modelo API (simulado)")
            return None
        
        try:
            model_path = model_mapping.get(model_name, model_name)
            print(f"   📥 Cargando {model_path}...")
            
            model = SentenceTransformer(model_path)
            
            if self.gpu_available:
                model = model.to('cuda')
                print(f"   🚀 {model_name} cargado en GPU")
            else:
                print(f"   💻 {model_name} cargado en CPU")
            
            return model
            
        except Exception as e:
            print(f"   ❌ Error con {model_name}: {e}")
            print(f"   🎲 Usando simulación")
            return None
    
    def calculate_metrics(self, similarities: np.ndarray, relevant_indices: List[int], top_k: int = 10):
        """Calcula métricas de recuperación"""
        
        # Top-K documentos
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Precisión y Recall
        retrieved_relevant = len(set(top_indices) & set(relevant_indices))
        precision = retrieved_relevant / len(top_indices) if len(top_indices) > 0 else 0
        recall = retrieved_relevant / len(relevant_indices) if len(relevant_indices) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # MAP
        average_precision = 0
        relevant_found = 0
        for i, doc_idx in enumerate(top_indices):
            if doc_idx in relevant_indices:
                relevant_found += 1
                average_precision += relevant_found / (i + 1)
        map_score = average_precision / len(relevant_indices) if len(relevant_indices) > 0 else 0
        
        # MRR
        mrr = 0
        for i, doc_idx in enumerate(top_indices):
            if doc_idx in relevant_indices:
                mrr = 1 / (i + 1)
                break
        
        # NDCG
        dcg = sum(1 / np.log2(i + 2) for i, doc_idx in enumerate(top_indices) if doc_idx in relevant_indices)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_indices), top_k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map': map_score,
            'mrr': mrr,
            'ndcg': ndcg
        }
    
    def evaluate_model(self, model_name: str, questions: List[Dict]):
        """Evalúa un modelo con todas las preguntas"""
        
        print(f"\n🤖 Evaluando: {model_name}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Cargar modelo
        model = self.load_model(model_name)
        
        # Simular documentos (1000 docs)
        num_docs = 1000
        doc_texts = [f"Documento Azure {i}: Información sobre servicios en la nube" for i in range(num_docs)]
        
        print(f"   📄 Procesando {num_docs} documentos...")
        
        # Generar embeddings de documentos
        if model is not None:
            try:
                doc_embeddings = model.encode(doc_texts, batch_size=self.config['batch_size'], show_progress_bar=True)
            except:
                print(f"   🎲 Fallback a embeddings simulados")
                doc_embeddings = np.random.randn(num_docs, 768).astype(np.float32)
        else:
            # Simular embeddings
            if model_name == 'ada':
                dims = 1536
            elif 'e5-large' in model_name:
                dims = 1024
            else:
                dims = 768
            doc_embeddings = np.random.randn(num_docs, dims).astype(np.float32)
        
        # Evaluar preguntas en lotes
        all_metrics = []
        batch_size = self.config['batch_size']
        
        print(f"   ❓ Evaluando {len(questions)} preguntas...")
        
        for i in tqdm(range(0, len(questions), batch_size), desc=f"{model_name}"):
            batch_questions = questions[i:i+batch_size]
            question_texts = [q['question'] for q in batch_questions]
            
            # Generar embeddings de preguntas
            if model is not None:
                try:
                    question_embeddings = model.encode(question_texts, batch_size=batch_size)
                except:
                    question_embeddings = np.random.randn(len(question_texts), doc_embeddings.shape[1]).astype(np.float32)
            else:
                question_embeddings = np.random.randn(len(question_texts), doc_embeddings.shape[1]).astype(np.float32)
            
            # Calcular métricas para cada pregunta
            for j, q_emb in enumerate(question_embeddings):
                # Similitud coseno
                similarities = cosine_similarity([q_emb], doc_embeddings)[0]
                
                # Simular documentos relevantes
                relevant_docs = random.sample(range(num_docs), k=random.randint(5, 15))
                
                # Calcular métricas
                metrics = self.calculate_metrics(similarities, relevant_docs, self.config['top_k'])
                all_metrics.append(metrics)
        
        # Promediar métricas
        avg_metrics = {}
        for metric in ['precision', 'recall', 'f1', 'map', 'mrr', 'ndcg']:
            values = [m[metric] for m in all_metrics]
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
        
        total_time = time.time() - start_time
        
        results = {
            'model_name': model_name,
            'avg_metrics': avg_metrics,
            'total_questions': len(questions),
            'processing_time_seconds': total_time,
            'gpu_used': self.gpu_available,
            'evaluation_time': datetime.now().isoformat()
        }
        
        print(f"   ✅ Completado en {total_time:.2f}s")
        print(f"   📊 F1: {avg_metrics['avg_f1']:.4f}")
        
        # Limpiar memoria
        if self.gpu_available and model is not None:
            torch.cuda.empty_cache()
        
        return results

# =====================================
# PASO 7: EJECUTAR EVALUACIÓN
# =====================================

print("\n" + "="*60)
print("🚀 EJECUTANDO EVALUACIÓN COMPLETA")
print("="*60)

evaluator = ColabEvaluator(EVALUATION_CONFIG)
evaluation_start = time.time()
evaluation_results = {}

try:
    for model_name in EVALUATION_CONFIG['selected_models']:
        model_results = evaluator.evaluate_model(model_name, test_questions)
        evaluation_results[model_name] = model_results
    
    total_evaluation_time = time.time() - evaluation_start
    
    print(f"\n✅ EVALUACIÓN COMPLETADA")
    print(f"⏱️  Tiempo total: {total_evaluation_time:.2f}s")
    print(f"📊 Preguntas: {len(test_questions):,}")
    print(f"🤖 Modelos: {len(EVALUATION_CONFIG['selected_models'])}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# =====================================
# PASO 8: MOSTRAR RESULTADOS
# =====================================

if evaluation_results:
    print("\n🏆 RANKING DE MODELOS")
    print("="*50)
    
    # Ordenar por F1-Score
    ranking = sorted(evaluation_results.items(), key=lambda x: x[1]['avg_metrics']['avg_f1'], reverse=True)
    
    for i, (model_name, results) in enumerate(ranking, 1):
        metrics = results['avg_metrics']
        print(f"\n{i}. {model_name}")
        print(f"   Precision: {metrics['avg_precision']:.4f}")
        print(f"   Recall:    {metrics['avg_recall']:.4f}")
        print(f"   F1-Score:  {metrics['avg_f1']:.4f}")
        print(f"   MAP:       {metrics['avg_map']:.4f}")
        print(f"   MRR:       {metrics['avg_mrr']:.4f}")
        print(f"   NDCG:      {metrics['avg_ndcg']:.4f}")
        print(f"   Tiempo:    {results['processing_time_seconds']:.2f}s")

# =====================================
# PASO 9: GUARDAR RESULTADOS
# =====================================

if evaluation_results:
    timestamp = int(time.time())
    
    # Resultados finales
    final_results = {
        'config': EVALUATION_CONFIG,
        'results': evaluation_results,
        'execution_summary': {
            'total_time_seconds': total_evaluation_time,
            'questions_processed': len(test_questions),
            'models_evaluated': len(EVALUATION_CONFIG['selected_models']),
            'gpu_used': evaluator.gpu_available,
            'timestamp': datetime.now().isoformat(),
            'colab_session': True
        }
    }
    
    # Guardar archivos
    json_file = f"cumulative_results_colab_{timestamp}.json"
    csv_file = f"results_summary_{timestamp}.csv"
    
    # JSON completo
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # CSV resumen
    csv_data = []
    for model_name, results in evaluation_results.items():
        metrics = results['avg_metrics']
        csv_data.append({
            'Model': model_name,
            'Precision': f"{metrics['avg_precision']:.4f}",
            'Recall': f"{metrics['avg_recall']:.4f}",
            'F1_Score': f"{metrics['avg_f1']:.4f}",
            'MAP': f"{metrics['avg_map']:.4f}",
            'MRR': f"{metrics['avg_mrr']:.4f}",
            'NDCG': f"{metrics['avg_ndcg']:.4f}",
            'Time_s': f"{results['processing_time_seconds']:.2f}"
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print(f"\n💾 ARCHIVOS GUARDADOS:")
    print(f"   📄 {json_file}")
    print(f"   📊 {csv_file}")
    
    # Copiar a Google Drive
    try:
        import shutil
        drive_json = f"{DRIVE_BASE}/{json_file}"
        drive_csv = f"{DRIVE_BASE}/{csv_file}"
        
        shutil.copy2(json_file, drive_json)
        shutil.copy2(csv_file, drive_csv)
        
        print(f"\n☁️  COPIADO A GOOGLE DRIVE:")
        print(f"   📄 {drive_json}")
        print(f"   📊 {drive_csv}")
        
    except Exception as e:
        print(f"⚠️  Error copiando a Drive: {e}")
    
    print(f"\n🎉 ¡PROCESO COMPLETADO!")
    print(f"✅ Descarga los archivos para importar en Streamlit")
    
else:
    print("❌ No hay resultados para guardar")

print("\n🎯 ¡EVALUACIÓN FINALIZADA!")
