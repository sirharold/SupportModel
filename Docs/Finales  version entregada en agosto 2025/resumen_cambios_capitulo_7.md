# Resumen de Cambios Clave - Capítulo 7

## Cambios en la Sección 7.2.1 (Configuración Experimental)

**Actualizar:**
- Preguntas evaluadas: ~~11~~ → **1000** por modelo
- Duración total: ~~774.78 segundos (12.9 minutos)~~ → **28,216 segundos (7.8 horas)**
- Fecha de evaluación: ~~26 de julio de 2025~~ → **2 de agosto de 2025**

## Cambios en Métricas por Modelo

### Ada (Sección 7.2.2)
- Precision@5: ~~0.055~~ → **0.097** (+76.7%)
- Recall@5: ~~0.273~~ → **0.399** (+46.0%)
- NDCG@5: ~~0.126~~ → **0.228** (+81.3%)
- MRR: ~~0.125~~ → **0.217** (+73.9%)

### MPNet (Sección 7.2.3)
- Precision@5: ~~0.055~~ → **0.074** (+35.3%)
- Recall@5: ~~0.273~~ → **0.292** (+7.0%)
- NDCG@5: ~~0.108~~ → **0.199** (+84.5%)
- MRR: ~~0.082~~ → **0.185** (+125.7%)

### MiniLM (Sección 7.2.4)
- Precision@5: ~~0.018~~ → **0.053** (+192.2%)
- Recall@5: ~~0.091~~ → **0.201** (+121.0%)
- NDCG@5: ~~0.091~~ → **0.148** (+62.7%)
- MRR: ~~0.077~~ → **0.144** (+87.3%)

### E5-Large (Sección 7.2.5)
- **¡AHORA FUNCIONAL!**
- Precision@5: ~~0.000~~ → **0.060**
- Recall@5: ~~0.000~~ → **0.239**
- NDCG@5: ~~0.000~~ → **0.169**
- MRR: ~~0.000~~ → **0.161**

## Cambios en Conclusiones (Sección 7.3)

1. **Eliminar:** "No hay diferencias estadísticamente significativas entre modelos"
2. **Agregar:** "Con 1000 preguntas, emergen diferencias claras: Ada > MPNet > E5-Large > MiniLM"
3. **Actualizar:** El impacto del reranking varía más entre modelos con dataset mayor

## Nueva Sección Recomendada: 7.3.4 Validación con Dataset Ampliado

Agregar subsección que discuta:
- Importancia del tamaño de muestra para confiabilidad
- Confirmación de tendencias observadas
- Resolución del problema de E5-Large
- Implicaciones para producción
