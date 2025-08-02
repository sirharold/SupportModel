# ANEXO C: CONFIGURACIÓN DE AMBIENTE

## Introducción

Este anexo proporciona las instrucciones detalladas para configurar el ambiente de desarrollo y ejecución del sistema RAG para recuperación semántica de documentación técnica de Microsoft Azure. La configuración se basa en las dependencias especificadas en `requirements.txt` y ha sido validada en los entornos utilizados durante la investigación experimental.

## Requisitos del Sistema

### Requisitos Mínimos de Hardware

**Para Desarrollo y Testing:**
- **CPU**: 4 cores mínimo (Intel i5/AMD Ryzen 5 o superior)
- **RAM**: 8GB mínimo, 16GB recomendado
- **Almacenamiento**: 10GB espacio libre (SSD recomendado)
- **Red**: Conexión a internet estable para APIs y descargas de modelos

**Para Producción (Corpus completo):**
- **CPU**: 8+ cores (Intel i7/Xeon o AMD Ryzen 7+)
- **RAM**: 32GB mínimo para ChromaDB con 800K+ vectores
- **Almacenamiento**: 50GB+ SSD (ChromaDB + modelos + datos)
- **GPU**: Opcional, mejora rendimiento de embeddings (CUDA compatible)

### Requisitos de Software

- **Sistema Operativo**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8.0 o superior (3.9.x recomendado)
- **pip**: Última versión
- **Git**: Para clonado del repositorio

## Configuración del Ambiente Python

### 1. Creación de Ambiente Virtual

```bash
# Crear ambiente virtual
python -m venv venv_support_model

# Activar ambiente (Linux/macOS)
source venv_support_model/bin/activate

# Activar ambiente (Windows)
venv_support_model\Scripts\activate
```

### 2. Instalación de Dependencias

#### Instalación Estándar

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias desde requirements.txt
pip install -r requirements.txt
```

#### Dependencias Principales (requirements.txt)

```txt
# Core APIs y Base de Datos
openai==1.93.0                    # API de OpenAI para Ada embeddings y evaluación
chromadb==0.5.23                  # Base de datos vectorial principal
python-dotenv==1.1.1              # Gestión de variables de ambiente

# Interfaz de Usuario y Visualización
streamlit==1.46.1                 # Aplicación web interactiva
plotly==6.2.0                     # Gráficos interactivos
weasyprint==63.1                  # Generación de reportes PDF
kaleido==0.2.1                    # Backend para exportación de gráficos

# Machine Learning y NLP
scikit-learn==1.7.0               # Métricas de evaluación y utilidades ML
torch==2.2.2                      # Backend para modelos de transformers
transformers==4.44.0              # Modelos de lenguaje y CrossEncoders
sentence-transformers==5.0.0      # Modelos de embedding especializados
accelerate==0.32.1                # Optimización de inferencia
bitsandbytes==0.43.0              # Cuantización de modelos

# Computación Científica
numpy==1.26.4                     # Operaciones numéricas fundamentales

# Evaluación y Métricas
bert-score==0.3.13                # Evaluación semántica con BERT
rouge-score==0.1.2                # Métricas de evaluación de texto

# APIs de Google (para modelos alternativos)
google-generativeai==0.8.5        # API de Google Gemini
google-auth==2.40.3               # Autenticación Google
google-auth-oauthlib==1.2.2       # OAuth para Google APIs
google-api-python-client==2.175.0 # Cliente Python para APIs Google
```

### 3. Verificación de Instalación

```bash
# Verificar instalación Python
python --version

# Verificar instalación de paquetes críticos
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
python -c "import sentence_transformers; print('Sentence-Transformers:', sentence_transformers.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

## Configuración de Variables de Ambiente

### 1. Archivo .env

Crear archivo `.env` en el directorio raíz del proyecto:

```bash
# Archivo .env - NO INCLUIR EN CONTROL DE VERSIONES

# API Keys (requeridas)
OPENAI_API_KEY=your_openai_api_key_here

# APIs opcionales (para funcionalidades extendidas)
GOOGLE_API_KEY=your_google_api_key_here

# Configuración de ChromaDB
CHROMADB_PATH=/Users/haroldgomez/chromadb2
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Configuración de modelos
DEFAULT_EMBEDDING_MODEL=multi-qa-mpnet-base-dot-v1
RERANKER_MODEL=ms-marco-MiniLM-L-6-v2

# Configuración de evaluación
EVAL_TOP_K=10
EVAL_BATCH_SIZE=4
```

### 2. Variables de Sistema (Opcional)

```bash
# Configurar variables permanentes (Linux/macOS)
echo 'export OPENAI_API_KEY="your_key_here"' >> ~/.bashrc
echo 'export CHROMADB_PATH="/path/to/chromadb"' >> ~/.bashrc
source ~/.bashrc

# Windows (PowerShell)
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_key_here", "User")
```

## Configuración de ChromaDB

### 1. Inicialización de Base de Datos

```python
# Script de inicialización (initialize_chromadb.py)
import chromadb
from chromadb.config import Settings

# Crear cliente ChromaDB
client = chromadb.PersistentClient(
    path="/Users/haroldgomez/chromadb2",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

print("ChromaDB inicializado correctamente")
print(f"Ubicación: /Users/haroldgomez/chromadb2")
```

### 2. Verificación de Colecciones

```python
# Verificar estado de colecciones
def verify_collections():
    collections = client.list_collections()
    print(f"Total colecciones: {len(collections)}")
    
    expected_collections = [
        'docs_ada', 'docs_mpnet', 'docs_minilm', 'docs_e5large',
        'questions_ada', 'questions_mpnet', 'questions_minilm', 
        'questions_e5large', 'questions_withlinks'
    ]
    
    for name in expected_collections:
        try:
            collection = client.get_collection(name)
            count = collection.count()
            print(f"✅ {name}: {count:,} elementos")
        except Exception as e:
            print(f"❌ {name}: No encontrada - {e}")

verify_collections()
```

## Configuración de Modelos de Embedding

### 1. Descarga Automática de Modelos

Los modelos se descargan automáticamente en el primer uso:

```python
# Test de modelos
from sentence_transformers import SentenceTransformer

# Modelos utilizados en el proyecto
models = [
    'all-MiniLM-L6-v2',           # MiniLM - 384D
    'multi-qa-mpnet-base-dot-v1', # MPNet - 768D  
    'intfloat/e5-large-v2'        # E5-Large - 1024D
]

for model_name in models:
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ {model_name}: Cargado correctamente")
        
        # Test de embedding
        test_text = "Azure Virtual Machine configuration"
        embedding = model.encode([test_text])
        print(f"   Dimensiones: {embedding.shape}")
        
    except Exception as e:
        print(f"❌ {model_name}: Error - {e}")
```

### 2. Configuración de CrossEncoder

```python
# Test de CrossEncoder para reranking
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('ms-marco-MiniLM-L-6-v2')
print("✅ CrossEncoder cargado correctamente")

# Test de reranking
query = "How to configure Azure storage?"
documents = [
    "Azure Storage configuration guide",
    "Virtual machine setup instructions"
]

scores = reranker.predict([(query, doc) for doc in documents])
print(f"   Scores de ejemplo: {scores}")
```

## Configuración de la Aplicación Streamlit

### 1. Configuración Básica

Crear archivo `streamlit_app/.streamlit/config.toml`:

```toml
[global]
dataFrameSerialization = "legacy"

[server]
port = 8501
address = "localhost"
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#0078d4"        # Azure blue
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor = "#000000"
```

### 2. Test de Aplicación

```bash
# Ejecutar aplicación Streamlit
streamlit run streamlit_app/app.py

# Debería abrir automáticamente en: http://localhost:8501
```

## Resolución de Problemas Comunes

### 1. Errores de Instalación

**Error: "Microsoft Visual C++ 14.0 is required" (Windows)**
```bash
# Solución: Instalar Visual Studio Build Tools
# Descargar desde: https://visualstudio.microsoft.com/downloads/
```

**Error: "Failed building wheel for [package]"**
```bash
# Solución: Actualizar pip y setuptools
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt
```

### 2. Errores de Memoria

**Error: "RuntimeError: [enforce fail at CPUAllocator.cpp]"**
```bash
# Solución: Reducir batch size en evaluaciones
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 3. Errores de ChromaDB

**Error: "ConnectionError: Could not connect to ChromaDB"**
```python
# Solución: Verificar permisos y path
import os
chromadb_path = "/Users/haroldgomez/chromadb2"
os.makedirs(chromadb_path, exist_ok=True)
os.chmod(chromadb_path, 0o755)
```

### 4. Errores de API

**Error: "OpenAI API rate limit exceeded"**
```python
# Solución: Implementar rate limiting
import time
from openai import RateLimitError

def safe_api_call(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Configuración para Desarrollo

### 1. Herramientas de Desarrollo

```bash
# Instalación de herramientas de desarrollo
pip install pytest black flake8 jupyter

# Formateo de código
black src/ streamlit_app/

# Linting
flake8 src/ streamlit_app/ --max-line-length=88
```

### 2. Pre-commit Hooks (Opcional)

```bash
# Instalación de pre-commit
pip install pre-commit

# Crear .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: [--max-line-length=88]
EOF

# Instalar hooks
pre-commit install
```

## Configuración para Producción

### 1. Optimizaciones de Performance

```bash
# Variables de ambiente para producción
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0  # Si GPU disponible
```

### 2. Configuración de Logging

```python
# logging_config.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('support_model.log'),
            logging.StreamHandler()
        ]
    )

setup_logging()
```

## Verificación Final del Ambiente

### Script de Verificación Completa

```python
# verify_environment.py
import sys
import subprocess
import importlib

def verify_environment():
    print("🔍 VERIFICACIÓN DEL AMBIENTE")
    print("=" * 50)
    
    # Verificar Python
    print(f"Python: {sys.version}")
    
    # Verificar dependencias críticas
    critical_packages = [
        'openai', 'chromadb', 'streamlit', 'sentence_transformers',
        'torch', 'transformers', 'numpy', 'scikit-learn'
    ]
    
    for package in critical_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: No instalado")
    
    # Verificar APIs
    import os
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API Key: Configurada")
    else:
        print("⚠️  OpenAI API Key: No configurada")
    
    # Verificar ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path="/Users/haroldgomez/chromadb2")
        collections = client.list_collections()
        print(f"✅ ChromaDB: {len(collections)} colecciones")
    except Exception as e:
        print(f"❌ ChromaDB: Error - {e}")
    
    print("\n🎯 ESTADO FINAL:")
    print("El ambiente está listo para ejecutar el sistema RAG")

if __name__ == "__main__":
    verify_environment()
```

Ejecutar verificación:
```bash
python verify_environment.py
```

## Soporte y Contacto

Para problemas de configuración no cubiertos en este anexo:

1. **Revisar logs** del sistema (`support_model.log`)
2. **Verificar versiones** de Python y dependencias
3. **Consultar documentación** de paquetes específicos
4. **Contactar** al equipo de desarrollo con detalles del error

---

**Nota**: Esta configuración ha sido validada en los entornos utilizados durante la investigación experimental (julio 2025). Versiones más recientes de las dependencias pueden requerir ajustes menores.