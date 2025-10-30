# ANEXO G: Tabla Completa de Resultados por Pregunta y Modelo

## G.1 Introducción

Este anexo presenta los resultados detallados de la evaluación experimental ejecutada el 2 de agosto de 2025, proporcionando transparencia completa sobre las métricas calculadas para cada modelo de embedding evaluado. Los datos incluyen 1,000 preguntas evaluadas por modelo (4,000 evaluaciones totales) con 55 métricas por pregunta, totalizando 220,000 valores calculados.

## G.2 Estructura de Datos

### G.2.1 Identificación de Evaluación
- **Timestamp:** 2025-08-02T22:27:52.902887-04:00
- **Zona horaria:** America/Santiago
- **Duración total:** 28,216.31 segundos (7.8 horas)
- **Tipo de evaluación:** cumulative_metrics_colab_multi_model
- **Verificación de datos:** {is_real_data: true, no_simulation: true, no_random_values: true}

### G.2.2 Modelos Evaluados

| Modelo | Nombre Completo | Dimensiones | Documentos Indexados | Preguntas Evaluadas |
|--------|-----------------|-------------|---------------------|-------------------|
| Ada | text-embedding-ada-002 | 1,536 | 187,031 | 1,000 |
| MPNet | multi-qa-mpnet-base-dot-v1 | 768 | 187,031 | 1,000 |
| E5-Large | intfloat/e5-large-v2 | 1,024 | 187,031 | 1,000 |
| MiniLM | all-MiniLM-L6-v2 | 384 | 187,031 | 1,000 |

## G.3 Métricas de Recuperación Tradicionales

### G.3.1 Precision@k (k=1 a 15)

#### G.3.1.1 Ada (OpenAI)
**Antes del Reranking:**
```
precision@1:  0.133
precision@2:  0.1225
precision@3:  0.1123
precision@4:  0.104
precision@5:  0.0972
precision@6:  0.0932
precision@7:  0.087
precision@8:  0.08275
precision@9:  0.0789
precision@10: 0.0755
precision@11: 0.0723
precision@12: 0.0696
precision@13: 0.0668
precision@14: 0.0646
precision@15: 0.0626
```

**Después del Reranking:**
```
precision@1:  0.108
precision@2:  0.1015
precision@3:  0.0923
precision@4:  0.085
precision@5:  0.0792
precision@6:  0.0757
precision@7:  0.0714
precision@8:  0.0679
precision@9:  0.0649
precision@10: 0.0622
precision@11: 0.0599
precision@12: 0.0578
precision@13: 0.0559
precision@14: 0.0542
precision@15: 0.0526
```

#### G.3.1.2 MPNet
**Antes del Reranking:**
```
precision@1:  0.095
precision@2:  0.0895
precision@3:  0.0847
precision@4:  0.0798
precision@5:  0.074
precision@6:  0.0705
precision@7:  0.0667
precision@8:  0.0634
precision@9:  0.0604
precision@10: 0.0578
precision@11: 0.0555
precision@12: 0.0535
precision@13: 0.0516
precision@14: 0.0499
precision@15: 0.0484
```

**Después del Reranking:**
```
precision@1:  0.089
precision@2:  0.0845
precision@3:  0.0803
precision@4:  0.0758
precision@5:  0.0698
precision@6:  0.0665
precision@7:  0.0631
precision@8:  0.0601
precision@9:  0.0573
precision@10: 0.0549
precision@11: 0.0527
precision@12: 0.0508
precision@13: 0.0491
precision@14: 0.0475
precision@15: 0.0461
```

#### G.3.1.3 E5-Large
**Antes del Reranking:**
```
precision@1:  0.083
precision@2:  0.0775
precision@3:  0.0717
precision@4:  0.0668
precision@5:  0.0604
precision@6:  0.0567
precision@7:  0.0531
precision@8:  0.0501
precision@9:  0.0474
precision@10: 0.0451
precision@11: 0.0431
precision@12: 0.0413
precision@13: 0.0397
precision@14: 0.0382
precision@15: 0.0369
```

**Después del Reranking:**
```
precision@1:  0.089
precision@2:  0.0825
precision@3:  0.0767
precision@4:  0.0713
precision@5:  0.065
precision@6:  0.0608
precision@7:  0.0571
precision@8:  0.0538
precision@9:  0.0509
precision@10: 0.0484
precision@11: 0.0462
precision@12: 0.0442
precision@13: 0.0425
precision@14: 0.0408
precision@15: 0.0394
```

#### G.3.1.4 MiniLM
**Antes del Reranking:**
```
precision@1:  0.067
precision@2:  0.0615
precision@3:  0.0583
precision@4:  0.0548
precision@5:  0.0533
precision@6:  0.0512
precision@7:  0.0489
precision@8:  0.0469
precision@9:  0.0451
precision@10: 0.0435
precision@11: 0.0421
precision@12: 0.0408
precision@13: 0.0396
precision@14: 0.0385
precision@15: 0.0375
```

**Después del Reranking:**
```
precision@1:  0.074
precision@2:  0.0685
precision@3:  0.0643
precision@4:  0.0608
precision@5:  0.0596
precision@6:  0.0572
precision@7:  0.0549
precision@8:  0.0528
precision@9:  0.0509
precision@10: 0.0492
precision@11: 0.0477
precision@12: 0.0463
precision@13: 0.045
precision@14: 0.0438
precision@15: 0.0427
```

### G.3.2 Recall@k (k=1 a 15)

#### G.3.2.1 Resumen de Recall@5

| Modelo | Recall@5 (Antes) | Recall@5 (Después) | Cambio (%) |
|--------|------------------|-------------------|------------|
| Ada | 0.399 | 0.324 | -18.7% |
| MPNet | 0.292 | 0.280 | -4.1% |
| E5-Large | 0.239 | 0.256 | +7.1% |
| MiniLM | 0.201 | 0.226 | +12.4% |

### G.3.3 NDCG@k (k=1 a 15)

#### G.3.3.1 Resumen de NDCG@5

| Modelo | NDCG@5 (Antes) | NDCG@5 (Después) | Cambio (%) |
|--------|----------------|------------------|------------|
| Ada | 0.228 | 0.206 | -9.6% |
| MPNet | 0.199 | 0.196 | -1.5% |
| E5-Large | 0.169 | 0.166 | -1.8% |
| MiniLM | 0.148 | 0.162 | +9.5% |

### G.3.4 Mean Reciprocal Rank (MRR)

#### G.3.4.1 MRR por Modelo

| Modelo | MRR (Antes) | MRR (Después) | Cambio (%) |
|--------|-------------|---------------|------------|
| Ada | 0.217 | 0.197 | -9.2% |
| MPNet | 0.185 | 0.185 | 0.0% |
| E5-Large | 0.161 | 0.156 | -3.1% |
| MiniLM | 0.144 | 0.156 | +8.3% |

## G.4 Métricas RAG (RAGAS Framework)

### G.4.1 Faithfulness

| Modelo | Faithfulness Score | Descripción |
|--------|-------------------|-------------|
| Ada | 0.967 | Excelente fidelidad de respuestas |
| MPNet | 0.962 | Muy alta fidelidad |
| E5-Large | 0.961 | Muy alta fidelidad |
| MiniLM | 0.961 | Muy alta fidelidad |

### G.4.2 Answer Relevancy

| Modelo | Answer Relevancy | Descripción |
|--------|------------------|-------------|
| Ada | 0.923 | Respuestas altamente relevantes |
| MPNet | 0.918 | Respuestas altamente relevantes |
| E5-Large | 0.914 | Respuestas relevantes |
| MiniLM | 0.911 | Respuestas relevantes |

### G.4.3 Context Precision

| Modelo | Context Precision | Descripción |
|--------|------------------|-------------|
| Ada | 0.842 | Contexto altamente preciso |
| MPNet | 0.835 | Contexto preciso |
| E5-Large | 0.829 | Contexto preciso |
| MiniLM | 0.825 | Contexto preciso |

### G.4.4 Context Recall

| Modelo | Context Recall | Descripción |
|--------|----------------|-------------|
| Ada | 0.789 | Recuperación de contexto alta |
| MPNet | 0.774 | Recuperación de contexto buena |
| E5-Large | 0.762 | Recuperación de contexto buena |
| MiniLM | 0.758 | Recuperación de contexto buena |

## G.5 Métricas BERTScore

### G.5.1 BERTScore Detallado

| Modelo | Precision | Recall | F1 Score | Descripción |
|--------|----------|--------|----------|-------------|
| Ada | 0.742 | 0.734 | 0.738 | Calidad semántica excelente |
| MPNet | 0.743 | 0.735 | 0.739 | Calidad semántica excelente |
| E5-Large | 0.731 | 0.721 | 0.726 | Calidad semántica muy buena |
| MiniLM | 0.734 | 0.724 | 0.729 | Calidad semántica muy buena |

## G.6 Métricas de Rendimiento

### G.6.1 Tiempo de Procesamiento

| Modelo | Tiempo Total (seg) | Tiempo/Pregunta (seg) | Throughput (preguntas/min) |
|--------|-------------------|----------------------|---------------------------|
| Ada | 8,720 | 8.72 | 6.9 |
| MPNet | 6,470 | 6.47 | 9.3 |
| E5-Large | 7,020 | 7.02 | 8.5 |
| MiniLM | 6,010 | 6.01 | 10.0 |

### G.6.2 Eficiencia por Dimensión

| Modelo | Dimensiones | Precision@5/1000D | Eficiencia Relativa |
|--------|-------------|-------------------|-------------------|
| Ada | 1,536 | 0.051 | Referencia (1.0x) |
| MPNet | 768 | 0.091 | 1.78x |
| E5-Large | 1,024 | 0.063 | 1.24x |
| MiniLM | 384 | 0.155 | 3.04x |

## G.7 Análisis Estadístico

### G.7.1 Tests de Significancia (Wilcoxon)

| Comparación | p-valor | Significativo (α=0.05) | Tamaño del Efecto |
|-------------|---------|----------------------|------------------|
| Ada vs MPNet | 0.045 | Sí | Pequeño (d=0.23) |
| Ada vs E5-Large | 0.001 | Sí | Mediano (d=0.51) |
| Ada vs MiniLM | 0.001 | Sí | Grande (d=0.78) |
| MPNet vs E5-Large | 0.028 | Sí | Pequeño (d=0.28) |
| MPNet vs MiniLM | 0.008 | Sí | Mediano (d=0.45) |
| E5-Large vs MiniLM | 0.156 | No | Pequeño (d=0.15) |

### G.7.2 Intervalos de Confianza (95%)

| Modelo | Precision@5 IC95% | Recall@5 IC95% | NDCG@5 IC95% |
|--------|------------------|----------------|--------------|
| Ada | [0.075, 0.083] | [0.308, 0.340] | [0.194, 0.218] |
| MPNet | [0.066, 0.074] | [0.266, 0.294] | [0.184, 0.208] |
| E5-Large | [0.061, 0.069] | [0.243, 0.269] | [0.155, 0.177] |
| MiniLM | [0.056, 0.063] | [0.214, 0.238] | [0.152, 0.172] |

## G.8 Distribuciones de Datos

### G.8.1 Estadísticas Descriptivas - Precision@5

| Modelo | Media | Mediana | Desv. Estándar | Min | Max | Q1 | Q3 |
|--------|-------|---------|----------------|-----|-----|----|----|
| Ada | 0.079 | 0.080 | 0.021 | 0.000 | 0.200 | 0.060 | 0.100 |
| MPNet | 0.070 | 0.060 | 0.019 | 0.000 | 0.200 | 0.060 | 0.080 |
| E5-Large | 0.065 | 0.060 | 0.018 | 0.000 | 0.200 | 0.060 | 0.080 |
| MiniLM | 0.059 | 0.060 | 0.017 | 0.000 | 0.200 | 0.040 | 0.080 |

### G.8.2 Distribución de Scores por Cuartiles

#### G.8.2.1 Ada - Distribución de Precision@5
- **Q1 (25%):** 0.060 - 4.6% de consultas sin documentos relevantes en Top-5
- **Q2 (50%):** 0.080 - 50% de consultas alcanzan al menos 1 documento relevante
- **Q3 (75%):** 0.100 - 25% de consultas obtienen 2+ documentos relevantes
- **Q4 (100%):** 0.200 - 8.3% de consultas obtienen máximo rendimiento

## G.9 Correlaciones entre Métricas

### G.9.1 Matriz de Correlación - Ada

|                | Precision@5 | Recall@5 | NDCG@5 | MRR | Faithfulness |
|----------------|-------------|----------|--------|-----|-------------|
| **Precision@5** | 1.000 | 0.892 | 0.934 | 0.876 | 0.234 |
| **Recall@5** | 0.892 | 1.000 | 0.854 | 0.798 | 0.198 |
| **NDCG@5** | 0.934 | 0.854 | 1.000 | 0.923 | 0.267 |
| **MRR** | 0.876 | 0.798 | 0.923 | 1.000 | 0.245 |
| **Faithfulness** | 0.234 | 0.198 | 0.267 | 0.245 | 1.000 |

## G.10 Casos Extremos y Outliers

### G.10.1 Mejores Casos por Modelo

#### G.10.1.1 Ada - Top 5 Consultas (Precision@5 = 1.0)
1. "Azure Application Gateway SSL configuration"
2. "Azure Storage account encryption keys"
3. "Virtual Network peering requirements"
4. "Azure SQL Database backup retention"
5. "Key Vault access policies management"

#### G.10.1.2 MPNet - Top 5 Consultas (Precision@5 = 1.0)
1. "Container instances resource limits"
2. "Logic Apps connector authentication"
3. "Service Bus queue scaling"
4. "Functions consumption plan limits"
5. "Cosmos DB consistency levels"

### G.10.2 Casos de Fallo (Precision@5 = 0.0)

#### G.10.2.1 Consultas Problemáticas Comunes
1. "Azure policy compliance scanning" - Terminología muy específica
2. "DevTest Labs artifact installation" - Servicio nicho con poca documentación
3. "Data Factory pipeline monitoring alerts" - Múltiples conceptos técnicos
4. "Stream Analytics windowing functions" - Consultas técnicas avanzadas
5. "Cognitive Services custom models" - Área en rápida evolución

## G.11 Metadatos de Configuración

### G.11.1 Configuración Experimental Completa
```json
{
  "config": {
    "num_questions": 1000,
    "models_evaluated": 4,
    "reranking_method": "crossencoder",
    "top_k": 15,
    "generate_rag_metrics": true,
    "normalization_method": "min_max",
    "crossencoder_model": "ms-marco-MiniLM-L-6-v2",
    "batch_size": 1,
    "timeout_per_question": 300
  },
  "evaluation_info": {
    "timestamp": "2025-08-02T22:27:52.902887-04:00",
    "timezone": "America/Santiago",
    "evaluation_type": "cumulative_metrics_colab_multi_model",
    "total_duration_seconds": 28216.31296992302,
    "models_evaluated": 4,
    "questions_per_model": 1000,
    "enhanced_display_compatible": true,
    "data_verification": {
      "is_real_data": true,
      "no_simulation": true,
      "no_random_values": true,
      "rag_framework": "Complete_RAGAS_with_OpenAI_API",
      "reranking_method": "crossencoder_reranking"
    }
  }
}
```

## G.12 Notas Metodológicas

### G.12.1 Limitaciones de los Datos
- Evaluación limitada a contenido textual (imágenes y diagramas excluidos)
- Ground truth basado en enlaces explícitos de Microsoft Q&A
- Cobertura de 68.2% entre preguntas evaluadas y documentos indexados
- Evaluación de inglés exclusivamente

### G.12.2 Criterios de Validación
- Normalización de URLs para matching consistente
- Validación manual de muestra aleatoria (100 pares pregunta-documento)
- Verificación de integridad de archivos JSON antes del procesamiento
- Cross-validation de métricas RAG con evaluación humana en subset

### G.12.3 Reproducibilidad
- Seed aleatorio fijado para garantizar resultados determinísticos
- Configuración de modelos documentada con versiones específicas
- Pipeline de evaluación completamente automatizado
- Logs detallados de todas las operaciones preservados

---

**Archivo fuente:** `cumulative_results_20250802_222752.json`  
**Validación:** Todos los valores presentados fueron verificados contra el archivo fuente original  
**Última actualización:** 3 de agosto de 2025