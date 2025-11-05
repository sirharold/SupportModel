# INSTRUCCIONES - Análisis del Capítulo 7

## 📋 Contexto
El Capítulo 7 es el **capítulo más importante del proyecto de grado**, ya que presenta los resultados experimentales del sistema RAG. Los datos actuales en el capítulo son difíciles de apreciar, por lo que se requiere crear visualizaciones adecuadas.

## 🎯 Objetivo
Crear scripts y visualizaciones para mostrar los resultados de manera clara y profesional usando:
- Tablas comparativas
- Gráficos interactivos
- Análisis estadístico

## 📊 Fuente de Datos
- **Archivo de resultados**: `/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json`
- **Datos reales**: NO simulados, NO aleatorios (verificado en `data_verification`)
- **2,067 preguntas evaluadas** por modelo
- **4 modelos**: Ada, MPNet, MiniLM, E5-Large
- **Métricas**: Precision, Recall, F1, NDCG, MAP, MRR, métricas RAG

## 📝 Requisitos de Visualización

### Tablas
- Usar **solo k = 3, 5, 10, 15** para tablas
- Mostrar métricas ANTES y DESPUÉS del reranking
- Incluir diferencias (Δ) y porcentajes de cambio
- Formato profesional para tesis

### Gráficos
- Usar **todos los valores de k disponibles** (1-15)
- Gráficos de líneas para evolución por k
- Gráficos de barras para comparación entre modelos
- Usar colores consistentes por modelo

## 🗂️ Estructura de Archivos Creados

```
capitulo_7_analisis/
├── INSTRUCCIONES.md          (este archivo)
├── generate_tables.py         (script para generar tablas)
├── generate_charts.py         (script para generar gráficos)
├── run_all_analysis.py        (script maestro para ejecutar todo)
├── tables/                    (tablas generadas en formato markdown/csv)
│   ├── tabla_comparativa_modelos.md
│   ├── tabla_precision_por_k.md
│   ├── tabla_recall_por_k.md
│   └── ...
├── charts/                    (gráficos generados en formato PNG/SVG)
│   ├── precision_por_k.png
│   ├── recall_por_k.png
│   ├── comparacion_modelos.png
│   └── ...
└── analysis/                  (análisis estadístico)
    ├── resumen_estadistico.md
    └── insights.md
```

## 🔧 Pasos para Ejecutar

### 1. Generar todas las visualizaciones
```bash
cd /Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/capitulo_7_analisis
python run_all_analysis.py
```

### 2. Generar solo tablas
```bash
python generate_tables.py
```

### 3. Generar solo gráficos
```bash
python generate_charts.py
```

## 📐 Especificaciones Técnicas

### Librerías Utilizadas
- `pandas`: Manipulación de datos y tablas
- `matplotlib`: Gráficos estáticos
- `plotly`: Gráficos interactivos (opcional)
- `seaborn`: Visualizaciones estadísticas
- `json`: Lectura del archivo de resultados

### Formato de Salida
- **Tablas**: Markdown (.md) y CSV (.csv)
- **Gráficos**: PNG (alta resolución) para inclusión en documento
- **Resolución**: 300 DPI para impresión

## 🎨 Estilo Visual

### Colores por Modelo
- **Ada**: #1f77b4 (azul)
- **MPNet**: #ff7f0e (naranja)
- **MiniLM**: #2ca02c (verde)
- **E5-Large**: #d62728 (rojo)

### Formato de Tablas
- Encabezados en negrita
- Valores numéricos con 3 decimales
- Δ con signo (+/-)
- % de cambio con 1 decimal

## 📊 Métricas Principales a Visualizar

### Métricas de Recuperación
1. **Precision@k**: k = 3, 5, 10, 15
2. **Recall@k**: k = 3, 5, 10, 15
3. **F1@k**: k = 3, 5, 10, 15
4. **NDCG@k**: k = 3, 5, 10, 15
5. **MAP@k**: k = 3, 5, 10, 15
6. **MRR**: Valor único

### Métricas RAG (si disponibles)
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
- BERTScore (Precision, Recall, F1)

## ⚠️ IMPORTANTE
- **SIEMPRE usar datos reales** del JSON
- **NO simular ni inventar datos**
- Verificar que `is_real_data: true` en el JSON
- Documentar cualquier anomalía encontrada

## 🔄 Para Retomar la Sesión
Si necesitas continuar el trabajo en una nueva sesión:
1. Lee este archivo de instrucciones
2. Verifica que existan los archivos de resultados
3. Ejecuta `python run_all_analysis.py`
4. Revisa las visualizaciones generadas en `tables/` y `charts/`

## 📅 Historial de Cambios
- **2025-11-04**: Creación inicial del proyecto de análisis del Capítulo 7

---
**Nota**: Este es el capítulo más importante de la tesis. Todas las visualizaciones deben ser de calidad profesional.
