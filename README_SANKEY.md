# 🔄 Diagrama Sankey - Flujo de Relevancia

## 📊 Descripción

El diagrama Sankey visualiza cómo el **CrossEncoder** modifica el ranking de documentos relevantes e irrelevantes durante el proceso de re-ranking.

## ✨ Características

### 📈 Visualización
- **Diagrama Sankey interactivo** usando Plotly
- Muestra flujos entre recuperación inicial y resultado post-reranking
- Colores codificados para identificar flujos positivos y negativos

### 📊 Métricas Mostradas

#### Flujos de Documentos Relevantes:
- 🟢 **Relevantes Mantenidos**: Documentos relevantes que permanecen en Top-K
- 🔴 **Relevantes Perdidos**: Documentos relevantes que salieron del Top-K (malo)
- 🟢 **Relevantes Ganados**: Documentos relevantes que entraron al Top-K (bueno)

#### Flujos de Documentos Irrelevantes:
- ⚪ **Irrelevantes Mantenidos**: Documentos irrelevantes que permanecen
- 🟢 **Irrelevantes Removidos**: Documentos irrelevantes eliminados (bueno)
- 🔴 **Irrelevantes Añadidos**: Documentos irrelevantes que entraron (malo)

#### Métricas de Impacto:
- **Cambio Neto Relevantes**: Ganados - Perdidos
- **Cambio Neto Irrelevantes**: Removidos - Añadidos
- **Impacto Total**: Mejora total del CrossEncoder

## 🚀 Cómo Usar

### 1. Navegar a la Página
```
Aplicación Principal → Menú Lateral → 🔄 Diagrama Sankey - Flujo de Relevancia
```

### 2. Configuración
- **Modelo de Embedding**: Selecciona ada, e5-large, mpnet o minilm
- **Top-K**: Ajusta el número de documentos top a considerar (1-15)

### 3. Interpretación

#### ✅ CrossEncoder Mejorando (Impacto > 0):
```
El re-ranking está:
- Trayendo documentos RELEVANTES al Top-K
- Removiendo documentos IRRELEVANTES del Top-K
```

#### ⚠️ CrossEncoder Empeorando (Impacto < 0):
```
El re-ranking está:
- Perdiendo documentos RELEVANTES del Top-K
- Añadiendo documentos IRRELEVANTES al Top-K
```

#### ℹ️ CrossEncoder Neutro (Impacto = 0):
```
Los cambios positivos y negativos se compensan
```

## 📁 Estructura de Datos Requerida

El diagrama requiere archivos de resultados con la siguiente estructura:

```json
{
  "results": {
    "ada": {
      "all_before_metrics": [
        {
          "precision@1": 0.5,
          "document_scores": [
            {
              "rank": 1,
              "cosine_similarity": 0.85,
              "link": "https://...",
              "title": "...",
              "is_relevant": true
            }
          ]
        }
      ],
      "all_after_metrics": [
        {
          "precision@1": 0.6,
          "document_scores": [
            {
              "rank": 1,
              "cosine_similarity": 0.92,
              "link": "https://...",
              "title": "...",
              "is_relevant": true
            }
          ]
        }
      ]
    }
  }
}
```

## 🔧 Archivos Involucrados

- **Script Principal**: `src/apps/sankey_relevance_flow.py`
- **Menú de Navegación**: `src/apps/main_qa_app.py`
- **Datos**: Archivos `cumulative_results_*.json` en carpeta `data/`

## 💡 Casos de Uso

### Caso 1: Evaluación de CrossEncoder
**Pregunta**: ¿El CrossEncoder realmente mejora el ranking?

**Respuesta**: Observa el **Impacto Total**:
- `+5.2` → Mejora de ~5 documentos promedio por pregunta
- `-2.1` → Empeora ~2 documentos promedio por pregunta

### Caso 2: Debugging de Modelos
**Pregunta**: ¿Por qué ada tiene mejor precision@5 que e5-large?

**Respuesta**: Compara los flujos:
- Ada: +3.5 relevantes ganados, -1.2 relevantes perdidos
- E5-large: +1.8 relevantes ganados, -3.5 relevantes perdidos

### Caso 3: Optimización de Top-K
**Pregunta**: ¿Qué Top-K es óptimo para mi caso de uso?

**Respuesta**: Prueba diferentes valores de K y observa:
- K pequeño → Mayor impacto del CrossEncoder
- K grande → Menor impacto relativo

## 📊 Ejemplo de Interpretación

```
Modelo: ada
Top-K: 10

Flujos:
- Relevantes Mantenidos: 3.2
- Relevantes Ganados: 1.8
- Relevantes Perdidos: 0.5
- Irrelevantes Removidos: 2.1
- Irrelevantes Añadidos: 0.3

Impacto Total: +4.3

✅ Interpretación:
El CrossEncoder está mejorando significativamente el ranking:
- Trae ~2 documentos relevantes nuevos al Top-10
- Remueve ~2 documentos irrelevantes del Top-10
- Solo pierde 0.5 documentos relevantes promedio
- Impacto neto: +4.3 documentos mejor rankeados por pregunta
```

## 🎯 Ventajas del Diagrama Sankey

1. **Visual e Intuitivo**: Fácil de entender el flujo de documentos
2. **Detallado**: Muestra exactamente QUÉ está haciendo el CrossEncoder
3. **Cuantificado**: Métricas precisas de mejora/deterioro
4. **Comparativo**: Permite comparar modelos fácilmente
5. **Interactivo**: Hover sobre flujos para ver detalles

## 🔄 Flujo de Procesamiento

```
Archivo JSON
    ↓
Cargar datos del modelo seleccionado
    ↓
Extraer document_scores de before y after
    ↓
Para cada pregunta:
    ├─ Identificar documentos relevantes/irrelevantes en Top-K (before)
    ├─ Identificar documentos relevantes/irrelevantes en Top-K (after)
    └─ Calcular flujos (mantenidos, ganados, perdidos, etc.)
    ↓
Agregar flujos sobre todas las preguntas
    ↓
Crear diagrama Sankey con Plotly
    ↓
Mostrar métricas y estadísticas
```

## 📝 Notas Técnicas

- **Agregación**: Flujos se promedian sobre todas las preguntas evaluadas
- **Precisión**: Valores mostrados con 1 decimal
- **Performance**: Diagrama se genera en ~1-2 segundos
- **Tamaño de datos**: Soporta hasta 2067 preguntas sin problemas

## 🚧 Limitaciones

1. Requiere archivos de resultados con `document_scores`
2. Solo analiza Top-K documentos (no todos los recuperados)
3. No muestra detalles por pregunta individual
4. Asume que `is_relevant` está correctamente etiquetado

## 🔮 Futuras Mejoras

- [ ] Filtrar por preguntas específicas
- [ ] Exportar diagrama como imagen
- [ ] Comparación lado a lado de múltiples modelos
- [ ] Análisis de subgrupos (por tipo de pregunta)
- [ ] Animación del flujo de documentos
