# Documentación del Proyecto RAG - Recuperación Semántica Azure

Este directorio contiene toda la documentación organizada del proyecto de sistema RAG para recuperación semántica de documentación técnica de Microsoft Azure.

## Estructura de Directorios

### 📚 `/Finales/`
Contiene los documentos finales de la tesis, listos para presentación:

- **`Contenidos.md`** - Tabla de contenidos completa del proyecto
- **`capitulo_1.md`** - Introducción y Fundamentos del Proyecto  
- **`capitulo_2_estado_del_arte.md`** - Estado del Arte (pendiente)
- **`capitulo_3_marco_teorico.md`** - Marco Teórico
- **`capitulo_4_metodologia.md`** - Metodología (pendiente)
- **`capitulo_5_implementacion.md`** - Implementación
- **`capitulo_6_resultados_y_analisis.md`** - Resultados y Análisis
- **`capitulo_7_conclusiones_y_trabajo_futuro.md`** - Conclusiones y Trabajo Futuro

#### Anexos
- **`anexo_b_codigo_fuente.md`** - Referencias al código fuente en GitHub
- **`anexo_c_configuracion_ambiente.md`** - Configuración de ambiente basada en requirements.txt
- **`anexo_d_ejemplos_consultas_respuestas.md`** - Ejemplos de consultas y respuestas del sistema
- **`anexo_e_resultados_detallados_metricas.md`** - Resultados detallados por métrica con datos reales
- **`anexo_f_streamlit_app.md`** - Documentación de la aplicación Streamlit

### 🔬 `/Analisis/`
Scripts y resultados de análisis utilizados durante la investigación:

#### Scripts de Análisis de Métricas
- **`analyze_metrics.py`** - Análisis básico de métricas de rendimiento
- **`analyze_metrics_v2.py`** - Versión mejorada del análisis de métricas
- **`analyze_cosine_similarities.py`** - Análisis de similitudes coseno
- **`analyze_sample_docs.py`** - Análisis de documentos de muestra

#### Scripts de Verificación de Estadísticas
- **`verify_document_statistics.py`** - Verificación de estadísticas de documentos
- **`verify_questions_statistics.py`** - Verificación de estadísticas de preguntas
- **`verify_questions_statistics_v2.py`** - Versión mejorada de verificación de preguntas

#### Análisis Estadístico
- **`apply_wilcoxon_test.py`** - Aplicación de tests de Wilcoxon
- **`wilcoxon_detailed_analysis.py`** - Análisis detallado de Wilcoxon
- **`wilcoxon_test_results.csv`** - Resultados de tests estadísticos

#### Análisis de Distribución de Temas
- **`calculate_topic_distribution.py`** - Cálculo inicial de distribución temática
- **`calculate_topic_distribution_v2.py`** - Versión mejorada del cálculo de temas

#### Otros Análisis
- **`comprehensive_embedding_comparison.py`** - Comparación comprehensiva de embeddings
- **`explore_json_structure.py`** - Exploración de estructura de datos JSON

#### Resultados de Análisis (JSON)
- **`document_length_analysis.json`** - Análisis de longitud de documentos
- **`questions_analysis.json`** - Análisis de preguntas
- **`questions_comprehensive_analysis.json`** - Análisis comprehensivo de preguntas
- **`topic_distribution_results.json`** - Resultados de distribución temática v1
- **`topic_distribution_results_v2.json`** - Resultados de distribución temática v2

## Uso de los Documentos

### Para Lectura de la Tesis
Los documentos finales en `/Finales/` están organizados según la estructura académica estándar y pueden leerse en orden secuencial.

### Para Reproducción de Análisis
Los scripts en `/Analisis/` contienen todo el código utilizado para generar las estadísticas y métricas reportadas en la tesis. Ejecutar estos scripts requiere:

1. Configuración del ambiente según `anexo_c_configuracion_ambiente.md`
2. Acceso a los datos experimentales en `/data/`
3. Base de datos ChromaDB configurada

### Para Validación de Resultados
Todos los datos cuantitativos en la tesis pueden verificarse ejecutando los scripts correspondientes:

- **Métricas de rendimiento**: `analyze_metrics_v2.py`
- **Estadísticas de corpus**: `verify_document_statistics.py`, `verify_questions_statistics_v2.py`
- **Distribución temática**: `calculate_topic_distribution_v2.py`
- **Significancia estadística**: `wilcoxon_detailed_analysis.py`

## Versionado de Documentos

- **v1.0**: Documentos finales completados (agosto 2025)
- **Scripts de análisis**: Desarrollados durante julio-agosto 2025
- **Datos base**: Evaluación experimental del 26 de julio de 2025

## Referencias de Datos

Todos los análisis se basan en datos experimentales verificables:
- **Resultados principales**: `/data/cumulative_results_1753578255.json`
- **Ground truth**: `/data/preguntas_con_links_validos.csv`
- **Configuración de evaluación**: Metadata en archivos de resultados

## Contacto

Para consultas sobre la documentación o reprodución de análisis:
- **Autor**: Harold Gómez
- **Proyecto**: Tesis de Magíster en Ciencias de la Computación
- **Fecha**: Agosto 2025