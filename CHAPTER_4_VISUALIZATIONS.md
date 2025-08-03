# 📊 Visualizaciones del Capítulo 4: Análisis Exploratorio de Datos

## 🎯 Descripción

Esta página de Streamlit genera todas las visualizaciones mencionadas en el **Capítulo 4: Análisis Exploratorio de Datos** del proyecto de tesis. Incluye 9 figuras hermosas y profesionales que ilustran las características del corpus de documentación Azure y el dataset de preguntas Microsoft Q&A.

## 📈 Figuras Implementadas

### 🔢 Figura 4.1: Histograma de Distribución de Chunks
- **Descripción**: Histograma con estadísticas descriptivas de longitud de chunks
- **Datos**: 187,031 chunks analizados
- **Características**: Líneas estadísticas (media, mediana, cuartiles), caja de estadísticas

### 📊 Figura 4.2: Comparación Chunks vs Documentos
- **Descripción**: Box plots comparativos entre chunks y documentos completos
- **Características**: Doble vista con escalas apropiadas, estadísticas laterales

### 🎯 Figura 4.3: Distribución Temática - Barras
- **Descripción**: Gráfico de barras de distribución temática con porcentajes
- **Categorías**: Development (53.6%), Security (28.6%), Operations (11.9%), Azure Services (5.8%)
- **Características**: Etiquetas de porcentaje, colores temáticos

### 🥧 Figura 4.4: Distribución Temática - Pie
- **Descripción**: Gráfico de torta con etiquetas detalladas
- **Características**: Explosión del sector principal, leyenda informativa, sombras

### ❓ Figura 4.5: Histograma de Preguntas
- **Descripción**: Distribución de longitud de preguntas Microsoft Q&A
- **Datos**: 13,436 preguntas analizadas
- **Características**: Estadísticas comparativas, diseño consistente

### 📝 Figura 4.6: Tipos de Preguntas
- **Descripción**: Distribución por categorías de preguntas
- **Tipos**: Configuración, Troubleshooting, Implementación, Conceptual, API/SDK

### 🔗 Figura 4.7: Flujo de Ground Truth
- **Descripción**: Diagrama de flujo de cobertura de ground truth
- **Características**: Cajas conectadas, flujo visual, métricas de correspondencia

### 🌊 Figura 4.8: Diagrama de Sankey
- **Descripción**: Flujo de correspondencia pregunta-documento usando Plotly
- **Características**: Interactivo, colores distintivos, valores proporcionales

### 📋 Figura 4.9: Dashboard Resumen
- **Descripción**: Panel con métricas clave del corpus
- **Componentes**: Métricas principales, mini-gráficos, comparaciones visuales

## 🚀 Cómo Ejecutar

### Opción 1: Desde la Aplicación Principal
```bash
streamlit run src/apps/main_qa_app.py
```
Luego seleccionar "📊 Visualizaciones Capítulo 4" en el menú de navegación.

### Opción 2: Aplicación Independiente
```bash
streamlit run run_chapter_4_viz.py
```

## 🎨 Características de Diseño

### Paleta de Colores Profesional
- **Azure**: `#0078d4` - Color principal del proyecto
- **Development**: `#28a745` - Verde para desarrollo
- **Security**: `#dc3545` - Rojo para seguridad
- **Operations**: `#ffc107` - Amarillo para operaciones
- **Services**: `#6f42c1` - Púrpura para servicios

### Elementos Visuales
- ✅ Grid semi-transparente para mejor legibilidad
- ✅ Etiquetas de datos en todas las barras
- ✅ Cajas de estadísticas informativas
- ✅ Colores consistentes por categoría
- ✅ Tipografía clara y jerárquica
- ✅ Espaciado profesional

### Interactividad
- 🔄 Selector de visualización individual
- 📊 Métricas en tiempo real del corpus
- 🎯 Opción "Todas las Visualizaciones" para reporte completo
- 📱 Diseño responsivo

## 📁 Fuentes de Datos

### Datos Reales
- **Corpus completo**: `Docs/Analisis/full_corpus_analysis_final.json`
- **Distribución temática**: `Docs/Analisis/topic_distribution_results_v2.json`
- **Análisis de preguntas**: `Docs/Analisis/questions_comprehensive_analysis.json`

### Datos de Respaldo
Si los archivos reales no están disponibles, la aplicación utiliza datos sintéticos basados en las estadísticas del capítulo 4.

## 🔧 Dependencias

```txt
streamlit==1.46.1
matplotlib==3.8.4
seaborn==0.13.2
plotly==6.2.0
numpy==1.26.4
pandas>=1.2
```

## 📋 Estructura de Código

```
src/apps/chapter_4_visualizations.py    # Página principal
run_chapter_4_viz.py                    # Script independiente
CHAPTER_4_VISUALIZATIONS.md             # Esta documentación
```

### Funciones Principales
- `load_data()`: Carga datos reales o genera sintéticos
- `create_*()`: Funciones específicas para cada figura
- `main()`: Función principal de Streamlit

## 🎯 Casos de Uso

### 1. Investigación Académica
- Generar figuras para inclusión en el documento de tesis
- Análisis exploratorio interactivo del corpus
- Validación visual de estadísticas

### 2. Presentaciones
- Dashboard en vivo para presentaciones
- Visualizaciones individuales para slides
- Métricas actualizadas del proyecto

### 3. Desarrollo
- Verificación de análisis de datos
- Debugging de estadísticas del corpus
- Comparación con resultados previos

## 💡 Tips de Uso

### Para Mejores Resultados
1. **Usa "Dashboard Resumen"** para una vista general rápida
2. **Selecciona visualizaciones individuales** para análisis detallado
3. **Usa "Todas las Visualizaciones"** para reportes completos
4. **Revisa las métricas superiores** para contexto del corpus

### Personalización
- Los colores están definidos en `COLORS` para fácil modificación
- Las estadísticas se pueden actualizar modificando los archivos JSON
- El diseño es responsivo y se adapta a diferentes tamaños de pantalla

## 🔍 Troubleshooting

### Problemas Comunes
1. **Error de estilo seaborn**: Manejado automáticamente con fallbacks
2. **Datos no encontrados**: Usa datos sintéticos automáticamente
3. **Memoria insuficiente**: Reduce el número de datos sintéticos generados

### Verificación
```python
# Verificar que los datos se cargan correctamente
python -c "from src.apps.chapter_4_visualizations import load_data; print(load_data())"
```

## 📊 Métricas del Corpus

### Estadísticas Principales
- **Total Chunks**: 187,031
- **Documentos Únicos**: 62,417
- **Preguntas Q&A**: 13,436
- **Ground Truth Válido**: 2,067 (15.4%)

### Distribución Temática
- **Development**: 53.6% (98,584 chunks)
- **Security**: 28.6% (52,667 chunks)
- **Operations**: 11.9% (21,882 chunks)
- **Azure Services**: 5.8% (10,754 chunks)

---

**Desarrollado por**: Harold Gómez  
**Fecha**: Agosto 2025  
**Proyecto**: Sistema RAG para Soporte Técnico Azure