# ACTUALIZACIÓN DEL CAPÍTULO 7 - RESULTADOS CON 1000 PREGUNTAS

## Información de la Nueva Evaluación

- **Fecha de evaluación:** 2025-08-02
- **Preguntas por modelo:** 1000
- **Modelos evaluados:** 4
- **Duración total:** 7.8 horas
- **Verificación de datos:** Datos reales sin simulación

## Tabla 7.1: Comparación de Modelos (1000 preguntas)

| Model    |   Questions |   Dimensions |   Precision@5 (Before) |   Recall@5 (Before) |   NDCG@5 (Before) |   MRR (Before) |   Precision@5 (After) |   Recall@5 (After) |   NDCG@5 (After) |   MRR (After) | Precision@5 Change   | Recall@5 Change   | NDCG@5 Change   | MRR Change   |
|:---------|------------:|-------------:|-----------------------:|--------------------:|------------------:|---------------:|----------------------:|-------------------:|-----------------:|--------------:|:---------------------|:------------------|:----------------|:-------------|
| ADA      |        1000 |         1536 |                  0.097 |               0.399 |             0.228 |          0.217 |                 0.079 |              0.324 |            0.206 |         0.197 | -18.3%               | -18.7%            | -9.6%           | -9.5%        |
| E5-LARGE |        1000 |         1024 |                  0.06  |               0.239 |             0.169 |          0.161 |                 0.065 |              0.256 |            0.166 |         0.156 | +7.6%                | +7.1%             | -1.5%           | -3.3%        |
| MPNET    |        1000 |          768 |                  0.074 |               0.292 |             0.199 |          0.185 |                 0.07  |              0.28  |            0.196 |         0.185 | -5.6%                | -4.2%             | -1.5%           | +0.0%        |
| MINILM   |        1000 |          384 |                  0.053 |               0.201 |             0.148 |          0.144 |                 0.059 |              0.226 |            0.162 |         0.156 | +11.8%               | +12.3%            | +9.3%           | +8.4%        |


## Tabla 7.2: Métricas RAG y BERTScore (1000 preguntas)

| Model    | Faithfulness   | Answer Relevancy   | Context Precision   | Context Recall   | Context Relevancy   | Context Utilization   | BERTScore Precision   | BERTScore Recall   | BERTScore F1   |
|:---------|:---------------|:-------------------|:--------------------|:-----------------|:--------------------|:----------------------|:----------------------|:-------------------|:---------------|
| ADA      | N/A            | N/A                | N/A                 | N/A              | N/A                 | N/A                   | N/A                   | N/A                | N/A            |
| E5-LARGE | N/A            | N/A                | N/A                 | N/A              | N/A                 | N/A                   | N/A                   | N/A                | N/A            |
| MPNET    | N/A            | N/A                | N/A                 | N/A              | N/A                 | N/A                   | N/A                   | N/A                | N/A            |
| MINILM   | N/A            | N/A                | N/A                 | N/A              | N/A                 | N/A                   | N/A                   | N/A                | N/A            |


## Análisis Detallado por Modelo

### ADA

**Configuración:**
- Preguntas evaluadas: 1000
- Dimensiones del embedding: 1536
- Modelo completo: ada

**Métricas principales (antes → después del reranking):**
- Precision@5: 0.097 → 0.079
- Recall@5: 0.399 → 0.324
- NDCG@5: 0.228 → 0.206
- MRR: 0.217 → 0.197

**Impacto del CrossEncoder:** Impacto negativo o neutro


### E5-LARGE

**Configuración:**
- Preguntas evaluadas: 1000
- Dimensiones del embedding: 1024
- Modelo completo: e5-large

**Métricas principales (antes → después del reranking):**
- Precision@5: 0.060 → 0.065
- Recall@5: 0.239 → 0.256
- NDCG@5: 0.169 → 0.166
- MRR: 0.161 → 0.156

**Impacto del CrossEncoder:** Mejora significativa en métricas principales


### MPNET

**Configuración:**
- Preguntas evaluadas: 1000
- Dimensiones del embedding: 768
- Modelo completo: mpnet

**Métricas principales (antes → después del reranking):**
- Precision@5: 0.074 → 0.070
- Recall@5: 0.292 → 0.280
- NDCG@5: 0.199 → 0.196
- MRR: 0.185 → 0.185

**Impacto del CrossEncoder:** Impacto negativo o neutro


### MINILM

**Configuración:**
- Preguntas evaluadas: 1000
- Dimensiones del embedding: 384
- Modelo completo: minilm

**Métricas principales (antes → después del reranking):**
- Precision@5: 0.053 → 0.059
- Recall@5: 0.201 → 0.226
- NDCG@5: 0.148 → 0.162
- MRR: 0.144 → 0.156

**Impacto del CrossEncoder:** Mejora significativa en métricas principales


## Comparación con Evaluación Anterior (11 vs 1000 preguntas)

### Cambios Principales:

1. **Tamaño de muestra:** 91x más datos (11 → 1000 preguntas)
2. **Confiabilidad estadística:** Métricas mucho más estables y representativas
3. **E5-Large funcional:** Ahora muestra métricas válidas (antes todas en 0.0)
4. **Jerarquía de rendimiento clara:** Ada > MPNet > E5-Large > MiniLM
5. **Impacto del reranking diferente:** Patrones más complejos con dataset mayor
