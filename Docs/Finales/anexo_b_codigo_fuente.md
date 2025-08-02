# ANEXO B: CÓDIGO FUENTE PRINCIPAL

## Repositorio del Proyecto

El código fuente completo del sistema RAG para recuperación semántica de documentación técnica de Microsoft Azure está disponible en el repositorio de GitHub del proyecto.

### Ubicación del Repositorio

**Repositorio GitHub:** [Pendiente de publicación]

### Estructura del Repositorio

El repositorio contiene la implementación completa del sistema, organizada en los siguientes directorios principales:

```
SupportModel/
├── src/                          # Código fuente principal
│   ├── core/                     # Componentes centrales del sistema
│   │   ├── embeddings.py         # Gestión de modelos de embedding
│   │   ├── retriever.py          # Motor de recuperación vectorial
│   │   └── reranker.py           # Reranking con CrossEncoder
│   ├── data/                     # Procesamiento de datos
│   │   ├── extractor.py          # Extracción de Microsoft Learn
│   │   └── preprocessor.py       # Procesamiento de documentos
│   └── evaluation/               # Framework de evaluación
│       ├── metrics.py            # Métricas de evaluación
│       └── ragas_evaluator.py    # Evaluación RAG especializada
├── streamlit_app/                # Interfaz de usuario Streamlit
│   ├── app.py                    # Aplicación principal
│   └── utils/                    # Utilidades de la interfaz
├── colab_data/                   # Notebooks de Google Colab
│   └── Cumulative_Ticket_Evaluation.ipynb
├── data/                         # Datos del proyecto
│   ├── cumulative_results_*.json # Resultados experimentales
│   ├── train_set.json           # Conjunto de entrenamiento
│   └── val_set.json             # Conjunto de validación
├── external_helpers/             # Scripts auxiliares
│   ├── calculate_topic_distribution_v2.py
│   ├── verify_document_statistics.py
│   └── verify_questions_statistics_v2.py
└── requirements.txt              # Dependencies del proyecto
```

### Componentes Principales

#### 1. Motor de Recuperación (`src/core/`)
- **`embeddings.py`**: Gestión de múltiples modelos de embedding (Ada, MPNet, MiniLM, E5-Large)
- **`retriever.py`**: Implementación de búsqueda vectorial sobre ChromaDB
- **`reranker.py`**: CrossEncoder con normalización sigmoid para reranking

#### 2. Procesamiento de Datos (`src/data/`)
- **`extractor.py`**: Extracción automatizada de documentación de Microsoft Learn
- **`preprocessor.py`**: Segmentación y limpieza de documentos técnicos

#### 3. Framework de Evaluación (`src/evaluation/`)
- **`metrics.py`**: Implementación de Precision@k, Recall@k, NDCG, MAP, MRR
- **`ragas_evaluator.py`**: Métricas especializadas RAG (Faithfulness, Context Precision/Recall)

#### 4. Interfaz de Usuario (`streamlit_app/`)
- **`app.py`**: Aplicación Streamlit para exploración interactiva de resultados
- Dashboard con visualizaciones de métricas comparativas

#### 5. Notebooks Experimentales (`colab_data/`)
- **`Cumulative_Ticket_Evaluation.ipynb`**: Notebook principal de evaluación experimental
- Implementación completa del pipeline de evaluación multi-modelo

### Tecnologías y Dependencias

El proyecto utiliza las siguientes tecnologías principales:

- **Python 3.8+**: Lenguaje de programación principal
- **ChromaDB 0.5.23**: Base de datos vectorial
- **Sentence-Transformers 5.0.0**: Modelos de embedding
- **OpenAI API 1.93.0**: Modelo Ada y evaluación RAG
- **Streamlit 1.46.1**: Interfaz de usuario web
- **Transformers 4.44.0**: Arquitecturas de modelos de lenguaje

### Reproducibilidad

El repositorio incluye:

1. **Configuración de ambiente** completa (`requirements.txt`)
2. **Scripts de verificación** para validar configuración
3. **Datos de ejemplo** para testing rápido
4. **Documentación detallada** de instalación y uso
5. **Notebooks ejecutables** en Google Colab

### Instrucciones de Acceso

Para acceder al código fuente completo:

1. **Clonar el repositorio:**
   ```bash
   git clone [URL_del_repositorio]
   cd SupportModel
   ```

2. **Configurar el ambiente:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación:**
   ```bash
   streamlit run streamlit_app/app.py
   ```

### Licencia y Términos de Uso

El código fuente se distribuye bajo los términos establecidos para investigación académica, con las siguientes consideraciones:

- **Uso académico**: Permitido para investigación y educación
- **Datos de Microsoft**: Sujeto a términos de uso de Microsoft Learn
- **Modelos propietarios**: OpenAI Ada requiere API key válida
- **Atribución**: Citar apropiadamente en trabajos derivados

### Contacto y Soporte

Para consultas sobre el código fuente, implementación o extensiones:

- **Autor**: Harold Gómez
- **Institución**: [Institución académica]
- **Email**: [Email de contacto]

### Nota sobre Versiones

El código corresponde a la versión utilizada para la evaluación experimental reportada en este trabajo (julio 2025). Versiones posteriores pueden incluir mejoras y optimizaciones adicionales.