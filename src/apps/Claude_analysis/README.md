# Claude Analysis Files

Esta carpeta contiene archivos de análisis, versiones antiguas y backups que fueron utilizados durante el desarrollo y análisis del proyecto.

## 📂 Contenido

### Análisis de Métricas Cumulativas
- `cumulative_metrics_create.py` - Creación de métricas cumulativas
- `cumulative_metrics_page.py` - Página de visualización de métricas cumulativas
- `cumulative_metrics_results_matplotlib.py` - Visualización con matplotlib
- `cumulative_metrics_results.py` - Resultados de métricas cumulativas

### Análisis de Datos
- `data_analysis_page_real.py` - Página de análisis de datos (versión real)
- `data_analysis_page.py` - Página de análisis de datos (versión antigua)
- `sankey_relevance_flow.py` - Visualización de flujo de relevancia con Sankey

### Versiones Antiguas de Interactive Search
- `interactive_search_analysis_single.py` - Versión antigua (antes de mejoras)
- `interactive_search_analysis.py` - Versión antigua (antes de mejoras)
- `interactive_search_analysis.py.backup` - Backup explícito
- `INTERACTIVE_SEARCH_README.md` - Documentación de versiones anteriores

### Backups y Versiones Antiguas
- `main_qa_app_backup.py` - Backup de la app principal
- `main_qa_app_clean.py` - Versión limpia antigua de la app principal
- `keyword_search_app.py` - App de búsqueda por palabras clave

## 📝 Notas

**Fecha de archivo:** 2025-11-06

**Razón:** Limpieza y organización del proyecto. Los archivos activos de Streamlit están en el directorio padre:

### Archivos activos en `/src/apps/`:
- `main_qa_app.py` - Aplicación principal de Streamlit
- `interactive_search_single.py` - Página de análisis individual (con mejoras Nov 2025)
- `batch_search_analysis.py` - Página de análisis por lotes (con mejoras Nov 2025)
- `search_utils.py` - Librería compartida (con mejoras Nov 2025)

### Mejoras implementadas (Nov 2025):
1. ✅ Multi-stage retrieval (top-50 → rerank → top-15)
2. ✅ CrossEncoder mejorado (ms-marco-electra-base)
3. ✅ Query expansion con diccionario Azure (50+ términos)
4. ✅ URL deduplication fix
5. ✅ Debug mode mejorado

## ⚠️ Importante

Estos archivos se mantienen para referencia histórica y pueden contener código útil para análisis futuros, pero **no se usan en la aplicación activa de Streamlit**.

Si necesitas recuperar algún archivo o funcionalidad, puedes consultarlos aquí.
