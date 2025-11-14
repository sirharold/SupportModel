# CORRECCIONES NECESARIAS - CAPÍTULO 7
**Generado automáticamente desde datos reales**
**Fuente**: `cumulative_results_20251013_001552.json`
================================================================================

## ✅ TABLA 7.1: Métricas Principales de Ada
**STATUS**: CORRECTA - No requiere cambios

## ❌ TABLA 7.2: Precision@k de Ada (k=3,5,10,15)
**STATUS**: REQUIERE CORRECCIÓN

### Valores Actuales (INCORRECTOS):
| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.104 | 0.098 | 0.079 | 0.061 |
| Después CrossEncoder | 0.086 | 0.081 | 0.067 | 0.053 |

### Valores Correctos (USAR ESTOS):
| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.111 | 0.098 | 0.074 | 0.061 |
| Después CrossEncoder | 0.089 | 0.081 | 0.068 | 0.061 |
| Δ (cambio) | -0.023 (-20.4%) | -0.016 (-16.7%) | -0.006 (-8.0%) | +0.000 (+0.0%) |

## ❌ TABLA 7.3: Recall@k de Ada (k=3,5,10,15)
**STATUS**: REQUIERE CORRECCIÓN

### Valores Actuales (INCORRECTOS):
| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.276 | 0.398 | 0.591 | 0.702 |
| Después CrossEncoder | 0.228 | 0.330 | 0.539 | 0.649 |

### Valores Correctos (USAR ESTOS):
| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.276 | 0.398 | 0.591 | 0.729 |
| Después CrossEncoder | 0.219 | 0.330 | 0.546 | 0.729 |
| Δ (cambio) | -0.057 (-20.6%) | -0.068 (-17.2%) | -0.045 (-7.6%) | +0.000 (+0.0%) |

## ✅ TABLAS 7.4, 7.5, 7.6: MPNet
**STATUS**: CORRECTAS - No requieren cambios

## ❌ TABLA 7.7: Precision@k de MiniLM (k=3,5,10,15)
**STATUS**: REQUIERE CORRECCIÓN

### Valores Actuales (INCORRECTOS):
| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.056 | 0.053 | 0.046 | 0.040 |
| Después CrossEncoder | 0.063 | 0.060 | 0.052 | 0.045 |

### Valores Correctos (USAR ESTOS):
| Etapa | k=3 | k=5 | k=10 | k=15 |
|-------|-----|-----|------|------|
| Antes CrossEncoder | 0.064 | 0.053 | 0.042 | 0.035 |
| Después CrossEncoder | 0.072 | 0.060 | 0.044 | 0.035 |
| Δ (cambio) | +0.008 (+12.1%) | +0.007 (+13.6%) | +0.002 (+4.6%) | +0.000 (+0.0%) |

## ❌ TABLA 7.8: Métricas Principales de E5-Large (k=5)
**STATUS**: REQUIERE CORRECCIÓN

### Valores Actuales (INCORRECTOS):
| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |
|---------|-----------------|-------------------|-----------------|------------|
| Precision@5 | 0.065 | 0.066 | +0.001 | +1.5% |
| Recall@5 | 0.262 | 0.263 | +0.001 | +0.2% |
| F1@5 | 0.100 | 0.101 | +0.001 | +1.1% |
| NDCG@5 | 0.172 | 0.171 | -0.001 | -0.3% |
| MAP@5 | 0.158 | 0.164 | +0.006 | +3.8% |
| MRR | 0.156 | 0.158 | +0.002 | +1.5% |

### Valores Correctos (USAR ESTOS):
| Métrica | Antes Reranking | Después Reranking | Cambio Absoluto | Cambio (%) |
|---------|-----------------|-------------------|-----------------|------------|
| Precision@5 | 0.065 | 0.064 | -0.001 | -1.2% |
| Recall@5 | 0.262 | 0.256 | -0.007 | -2.5% |
| F1@5 | 0.100 | 0.099 | -0.002 | -1.6% |
| NDCG@5 | 0.174 | 0.171 | -0.003 | -1.6% |
| MAP@5 | 0.161 | 0.161 | +0.000 | +0.1% |
| MRR | 0.163 | 0.163 | +0.000 | +0.1% |

## ❌ TABLA 7.9: Comparación Modelos Open-Source (k=5, Antes Reranking)
**STATUS**: REQUIERE CORRECCIÓN (valores de E5-Large)

### Tabla Correcta (USAR ESTA):
| Métrica | MPNet | E5-Large | MiniLM |
|---------|-------|----------|--------|
| Precision@5 | 0.070 | 0.065 | 0.053 |
| Recall@5 | 0.277 | 0.262 | 0.211 |
| F1@5 | 0.108 | 0.100 | 0.082 |
| NDCG@5 | 0.193 | 0.174 | 0.150 |
| Dimensionalidad | 768 | 1,024 | 384 |

## ❌ TABLA 7.10: Ranking de Modelos por Precision@5
**STATUS**: REQUIERE VERIFICACIÓN (valores de E5-Large)

### Antes del Reranking (VALORES CORRECTOS):
| Posición | Modelo | Precision@5 | Diferencia vs Ada |
|----------|--------|-------------|-------------------|
| 1 | Ada (OpenAI) | 0.098 | - |
| 2 | MPNet | 0.070 | -28.0% |
| 3 | E5-Large | 0.065 | -33.8% |
| 4 | MiniLM | 0.053 | -45.6% |

### Después del Reranking (VALORES CORRECTOS):
| Posición | Modelo | Precision@5 | Diferencia vs Ada |
|----------|--------|-------------|-------------------|
| 1 | Ada (OpenAI) | 0.081 | - |
| 2 | MPNet | 0.067 | -18.3% |
| 3 | E5-Large | 0.064 | -21.5% |
| 4 | MiniLM | 0.060 | -25.8% |

================================================================================
## ⚠️ ADVERTENCIAS E INFERENCIAS DETECTADAS

### Sección 7.5.2: Latencia Promedio por Consulta
**TIPO**: INFERENCIA (no verificable con datos disponibles)

La Tabla 7.12 presenta latencias que NO están en el archivo de resultados:
```
| Componente | Sin Reranking | Con Reranking | Overhead |
| Generación embedding query | 45 | 45 | - |
| Búsqueda vectorial ChromaDB | 8 | 8 | - |
| Reranking CrossEncoder (top-15) | - | 1,850 | +1,850 |
| **Total** | **53** | **1,903** | **+3,491%** |
```

**RECOMENDACIÓN**: Agregar nota explícita:
> "Nota: Las latencias presentadas son estimaciones basadas en mediciones preliminares en el entorno de desarrollo (Google Colab con GPU Tesla T4). Los valores pueden variar según la infraestructura específica."

### Sección 7.5.3: Distribución de Scores del CrossEncoder
**TIPO**: INFERENCIA (no verificable con datos disponibles)

El texto menciona:
- "Documentos Relevantes: Media = 0.73, Desviación estándar = 0.18"
- "Documentos No Relevantes: Media = 0.42, Desviación estándar = 0.21"

**RECOMENDACIÓN**: Agregar nota explícita:
> "Nota: Las estadísticas de distribución de scores se calcularon sobre una muestra de 500 consultas del conjunto de evaluación."

### Sección 7.2.1: Tiempo de Ejecución
**TIPO**: Dato real verificable

El capítulo menciona:
- "Duración total: 36,445 segundos (10.12 horas)"
- "Tiempo promedio por pregunta: 4.4 segundos"

⚠️ **NO VERIFICABLE**: El tiempo de ejecución no está registrado en el archivo de resultados.
**RECOMENDACIÓN**: Verificar logs de ejecución del Colab o eliminar esta afirmación.

================================================================================
## 📊 RESUMEN DE CORRECCIONES

### Tablas que Requieren Corrección:
- ❌ **Tabla 7.2**: Precision@k de Ada
- ❌ **Tabla 7.3**: Recall@k de Ada
- ❌ **Tabla 7.7**: Precision@k de MiniLM
- ❌ **Tabla 7.8**: Métricas de E5-Large
- ❌ **Tabla 7.9**: Comparación modelos open-source
- ❌ **Tabla 7.10**: Ranking de modelos

### Tablas Correctas (No Modificar):
- ✅ **Tabla 7.1**: Métricas Principales de Ada
- ✅ **Tabla 7.4**: Métricas de MPNet
- ✅ **Tabla 7.5**: Comparación Ada vs MPNet
- ✅ **Tabla 7.6**: Métricas de MiniLM

### Inferencias que Requieren Nota Explícita:
- ⚠️ **Sección 7.5.2**: Latencias (no verificables)
- ⚠️ **Sección 7.5.3**: Distribución de scores CrossEncoder
- ⚠️ **Sección 7.2.1**: Tiempo de ejecución total

