# 🔬 Análisis Interactivo de Búsqueda - Guía de Uso

## 📋 Descripción

Nueva página en Streamlit que permite analizar interactivamente el proceso completo de búsqueda vectorial y reranking, replicando la lógica del Colab pero de forma visual e interactiva.

## 🎯 Funcionalidades

### 1. **Selección de Pregunta**
- Selecciona cualquiera de las **2,067 preguntas validadas** (questions_withlinks)
- Visualiza la pregunta y sus enlaces de ground truth
- Usa un input numérico para navegar rápidamente

### 2. **Búsqueda Vectorial (Antes del Reranking)**
- Busca en ChromaDB usando similitud coseno
- Muestra los top-K documentos recuperados
- Indica cuáles son relevantes (✅) según ground truth
- Muestra scores de similitud coseno

### 3. **Reranking con CrossEncoder**
- Aplica el modelo `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Normalización Min-Max de scores (igual que en Colab)
- Muestra cómo cambian los rankings
- Indica si documentos suben 🔼, bajan 🔽 o se mantienen ➡️

### 4. **Métricas de Recuperación**
Se calculan **antes y después** del reranking:

- **Precision@k**: Proporción de documentos relevantes en top-k
- **Recall@k**: Proporción de documentos relevantes recuperados
- **F1@k**: Media armónica de Precision y Recall
- **MAP@k**: Mean Average Precision
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

### 5. **Comparación Visual**
- Tabla comparativa de métricas antes vs después
- Deltas visuales (▲▼) para ver mejoras/degradaciones
- Colores verde (mejora) / rojo (degradación)

## 🚀 Cómo Usar

### Paso 1: Iniciar la Aplicación
```bash
cd /Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel
streamlit run src/apps/main_qa_app.py
```

### Paso 2: Navegación
1. En el menú lateral, selecciona **"🔬 Análisis Interactivo de Búsqueda"**

### Paso 3: Configuración
En el sidebar:
- **Modelo de Embedding**: Elige entre Ada, MPNet, MiniLM, E5-Large
- **Top-K**: Número de documentos a recuperar (5-20)
- **Valores de k**: Selecciona para qué valores calcular métricas (1, 3, 5, 10, 15)

### Paso 4: Selección de Pregunta
- **Índice de pregunta**: Ingresa un número entre 0 y 2,066
  - Ejemplo: `0` para la primera pregunta
  - Ejemplo: `25` para la pregunta #26
  - Ejemplo: `100` para la pregunta #101
- Visualiza los enlaces de ground truth asociados

### Paso 5: Ejecutar Análisis
1. Click en **"🚀 Ejecutar Búsqueda y Análisis"**
2. El sistema:
   - Busca documentos por similitud coseno
   - Muestra resultados iniciales y métricas
   - Aplica CrossEncoder para reranking
   - Muestra resultados rerankeados y métricas
   - Calcula deltas automáticamente

### Paso 6: Interpretar Resultados

#### Documentos Antes del Reranking
- ✅ = Documento relevante (está en ground truth)
- ❌ = Documento no relevante
- Score de similitud coseno

#### Documentos Después del Reranking
- ✅/❌ = Relevancia
- 🔼 = Subió posiciones (mejoró ranking)
- 🔽 = Bajó posiciones (empeoró ranking)
- ➡️ = Mantuvo posición
- CrossEncoder Score normalizado

#### Métricas
- Verde ▲ = Mejora después del reranking
- Rojo ▼ = Degradación después del reranking
- Valores absolutos y cambios relativos

## 📊 Casos de Uso

### Caso 1: Analizar Pregunta Específica
Quieres ver cómo se comporta el sistema con una pregunta particular:
```
1. Ingresa el índice: 50
2. Ejecuta el análisis
3. Observa si el reranking ayuda o perjudica
```

### Caso 2: Comparar Modelos
Quieres ver cuál modelo funciona mejor para una pregunta:
```
1. Selecciona pregunta: 100
2. Prueba con Ada → ejecuta análisis → anota métricas
3. Prueba con MPNet → ejecuta análisis → anota métricas
4. Compara resultados
```

### Caso 3: Evaluar Impacto del Reranking
Quieres ver si el reranking ayuda en general:
```
1. Prueba varias preguntas (ej: 0, 25, 50, 100, 200)
2. Observa las deltas (▲▼)
3. Identifica patrones: ¿Cuándo ayuda? ¿Cuándo perjudica?
```

### Caso 4: Debugging de Resultados
El sistema no encuentra un documento esperado:
```
1. Ingresa la pregunta problemática
2. Revisa los documentos recuperados
3. Verifica si está en top-K o no fue recuperado
4. Analiza los scores para entender por qué
```

## 🔧 Arquitectura Técnica

### Flujo de Datos
```
1. Usuario selecciona pregunta → questions_withlinks (ChromaDB)
2. Sistema obtiene embedding de la pregunta
3. Búsqueda vectorial → docs_{model} (ChromaDB)
4. Cálculo de métricas → Metrics Before
5. Aplicar CrossEncoder → Reranked docs
6. Cálculo de métricas → Metrics After
7. Comparación y visualización
```

### Colecciones ChromaDB Usadas
- `questions_withlinks`: 2,067 preguntas validadas (embeddings ya generados)
- `docs_ada`: Documentos con embeddings de Ada
- `docs_mpnet`: Documentos con embeddings de MPNet
- `docs_minilm`: Documentos con embeddings de MiniLM
- `docs_e5large`: Documentos con embeddings de E5-Large

### Modelos Cargados
- **Embeddings**: Ya generados, se obtienen de ChromaDB
- **CrossEncoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cacheado)

## ⚡ Rendimiento

- **Primera ejecución**: ~5-10 segundos (carga CrossEncoder)
- **Ejecuciones posteriores**: ~2-3 segundos (modelo cacheado)
- **Memoria**: ~1GB adicional (CrossEncoder en RAM)

## 🔍 Diferencias con Colab

| Aspecto | Colab | Streamlit Interactive |
|---------|-------|----------------------|
| **Interfaz** | Código + outputs | Visual interactiva |
| **Ejecución** | Todas las 2,067 preguntas | Una o varias seleccionadas |
| **Tiempo** | 10+ horas | 2-3 segundos por pregunta |
| **Métricas** | Promedios agregados | Valores por pregunta individual |
| **Visualización** | Prints en consola | Tablas y deltas visuales |
| **Propósito** | Evaluación completa | Análisis y debugging |

## 📝 Notas Importantes

1. **Ground Truth**: Solo preguntas con enlaces validados (2,067 de 13,436)
2. **Normalización URL**: Se aplica igual que en Colab (sin query params ni fragments)
3. **Scores CrossEncoder**: Normalización Min-Max igual que en Colab
4. **Métricas**: Fórmulas idénticas a las del Colab

## 🎓 Para Tesis

Esta herramienta es útil para:
- Ilustrar el funcionamiento del sistema en la presentación
- Analizar casos específicos para discusión en Capítulo 7
- Debugging y validación de resultados del Colab
- Generar screenshots para el documento

## 🐛 Troubleshooting

### Error: "No se puede conectar a ChromaDB"
```bash
# Verifica que ChromaDB esté corriendo
# Verifica la ruta en get_chromadb_client()
```

### Error: "No se encontró colección"
```bash
# Verifica que las colecciones existan:
# - questions_withlinks
# - docs_{model}
```

### La página no aparece en el menú
```bash
# Verifica que agregaste el import:
# from src.apps.interactive_search_analysis import show_interactive_search_analysis_page

# Y que agregaste la opción en el radio:
# "🔬 Análisis Interactivo de Búsqueda"
```

## 📞 Soporte

Para problemas o mejoras, revisar:
- `/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/src/apps/interactive_search_analysis.py`
- Logs de Streamlit en consola

---

**Autor**: Sistema RAG - Proyecto de Magíster
**Fecha**: Noviembre 2025
**Versión**: 1.0
