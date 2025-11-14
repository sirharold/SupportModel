# Análisis y Verificación - Capítulo 7

Esta carpeta contiene todos los scripts, tablas y gráficos generados para el análisis del Capítulo 7 de la tesis.

## 📁 Estructura de Archivos

```
capitulo_7_analisis/
├── README.md                              # Este archivo
├── RESUMEN_EJECUTIVO_REVISION.md          # ⭐ LEER PRIMERO - Resumen completo de la revisión
├── CORRECIONES_NECESARIAS.md              # Tablas corregidas listas para copiar/pegar
├── FIGURAS_VERIFICACION.md                # Estado de todas las figuras
├── validation_report.txt                  # Reporte técnico detallado
│
├── generate_tables.py                     # Genera tablas desde datos reales
├── generate_charts.py                     # Genera gráficos (33 imágenes PNG)
├── validate_chapter_data.py               # Valida valores numéricos del capítulo
├── generate_correction_report.py          # Genera reporte de correcciones
├── verify_figures.py                      # Verifica figuras mencionadas
├── quick_verify.py                        # ⭐ Herramienta interactiva de verificación
├── run_all_analysis.py                    # Ejecuta todos los scripts
│
├── tables/                                # Tablas en formato MD y CSV
│   ├── tabla_precision_por_k.md
│   ├── tabla_recall_por_k.md
│   ├── tabla_f1_por_k.md
│   ├── tabla_ndcg_por_k.md
│   ├── tabla_map_por_k.md
│   ├── tabla_ranking_modelos.md
│   └── tabla_comparativa_modelos.md
│
└── charts/                                # Gráficos PNG (300 DPI)
    ├── precision_por_k_before.png
    ├── precision_por_k_after.png
    ├── delta_heatmap.png
    ├── model_ranking_bars.png
    └── ... (29 archivos más)
```

---

## 🚀 Inicio Rápido

### 1. Leer el Resumen de Revisión
```bash
# Abre el resumen ejecutivo (contiene TODA la información de la revisión)
open RESUMEN_EJECUTIVO_REVISION.md
```

### 2. Ver las Correcciones Necesarias
```bash
# Abre el archivo con las tablas corregidas
open CORRECIONES_NECESARIAS.md
```

### 3. Verificar Valores Específicos (Herramienta Interactiva)
```bash
# Ejecuta el verificador interactivo
python quick_verify.py
```

---

## 🛠️ Scripts Disponibles

### Scripts Principales

#### `validate_chapter_data.py`
Valida TODOS los valores numéricos mencionados en el capítulo contra el archivo de resultados real.

**Uso:**
```bash
python validate_chapter_data.py
```

**Output:**
- Verifica 125+ valores numéricos
- Muestra ✅/❌ para cada valor
- Genera `validation_report.txt`

---

#### `quick_verify.py` ⭐ RECOMENDADO
Herramienta interactiva para verificar cualquier valor del capítulo.

**Uso:**
```bash
python quick_verify.py
```

**Funciones:**
- Ver Precision/Recall/F1/NDCG/MAP/MRR de cualquier modelo
- Comparar dos modelos
- Ver todas las métricas de un modelo
- Ver metadatos de evaluación

**Ejemplo de uso:**
```
Selecciona opción: 1
Selecciona modelo (1-4): 1  # Ada
Ingresa valor de k (1-15): 5

Resultado:
✅ Antes del reranking:   0.0978
✅ Después del reranking: 0.0815
📊 Cambio absoluto:       -0.0163
📊 Cambio porcentual:     -16.67%
📉 El reranking DEGRADA esta métrica
```

---

#### `generate_correction_report.py`
Genera las tablas corregidas listas para copiar/pegar en el capítulo.

**Uso:**
```bash
python generate_correction_report.py
```

**Output:**
- `CORRECIONES_NECESARIAS.md` con tablas corregidas

---

#### `verify_figures.py`
Verifica que todas las figuras mencionadas en el capítulo existan.

**Uso:**
```bash
python verify_figures.py
```

**Output:**
- Lista de figuras existentes
- Lista de figuras faltantes (si las hay)
- Gráficos disponibles no referenciados

---

### Scripts de Generación

#### `generate_tables.py`
Genera TODAS las tablas del capítulo desde los datos reales.

**Uso:**
```bash
python generate_tables.py
```

**Output:**
- 7 archivos `.md` en `tables/`
- 7 archivos `.csv` en `tables/`

---

#### `generate_charts.py`
Genera TODOS los gráficos del capítulo (33 imágenes PNG a 300 DPI).

**Uso:**
```bash
python generate_charts.py
```

**Output:**
- 33 archivos `.png` en `charts/`
- Gráficos de alta calidad para impresión

---

#### `run_all_analysis.py`
Ejecuta todos los scripts de generación en el orden correcto.

**Uso:**
```bash
python run_all_analysis.py
```

---

## 📊 Datos Fuente

**Archivo**: `/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json`

**Tamaño**: 135 MB

**Contenido**:
- Resultados de evaluación de 4 modelos (Ada, MPNet, MiniLM, E5-Large)
- 2,067 preguntas evaluadas
- Métricas calculadas para k=1 hasta k=15
- Datos REALES (no simulados, no aleatorios)

---

## ✅ Estado de la Revisión

### Resumen
- **Total de validaciones**: 125+
- **Errores detectados**: 24 valores incorrectos en 6 tablas
- **Figuras verificadas**: 7/7 existen ✅
- **Datos verificados**: 100% reales ✅

### Tablas Correctas (No Modificar)
- ✅ Tabla 7.1: Métricas Principales de Ada
- ✅ Tabla 7.4: Métricas de MPNet
- ✅ Tabla 7.5: Comparación Ada vs MPNet
- ✅ Tabla 7.6: Métricas de MiniLM

### Tablas que Requieren Corrección
- ❌ Tabla 7.2: Precision@k de Ada
- ❌ Tabla 7.3: Recall@k de Ada
- ❌ Tabla 7.7: Precision@k de MiniLM
- ❌ Tabla 7.8: Métricas de E5-Large
- ❌ Tabla 7.9: Comparación modelos open-source
- ❌ Tabla 7.10: Ranking de modelos

### Inferencias que Requieren Nota
- ⚠️ Sección 7.5.2: Latencias (no verificables)
- ⚠️ Sección 7.5.3: Distribución de scores CrossEncoder
- ⚠️ Sección 7.2.1: Tiempo de ejecución total

---

## 🎯 Flujo de Trabajo Recomendado

### Para Corregir el Capítulo 7:

1. **Leer el resumen ejecutivo**
   ```bash
   open RESUMEN_EJECUTIVO_REVISION.md
   ```

2. **Abrir las correcciones**
   ```bash
   open CORRECIONES_NECESARIAS.md
   ```

3. **Copiar/pegar las 6 tablas corregidas** en el capítulo

4. **Agregar las 3 notas metodológicas** para las inferencias

5. **Verificar que todo esté correcto**
   ```bash
   python validate_chapter_data.py
   ```

### Para Verificar un Valor Específico:

```bash
python quick_verify.py
# Seleccionar opción deseada del menú interactivo
```

### Para Regenerar Todo:

```bash
python run_all_analysis.py
```

---

## 📝 Notas Importantes

### Convención de Nombres de Modelos
En el archivo JSON, el modelo se llama `e5-large` (con guión), no `e5large` (sin guión).

### Precisión de Valores
Todos los valores se reportan con 3-4 decimales. No redondear a menos decimales en las tablas.

### Cambios en k=15
Algunos modelos muestran **cambio cero** en k=15 (Precision@15 y Recall@15 son iguales antes y después del reranking). Esto NO es un error, es un resultado real.

### E5-Large: Cambio Importante
Los datos reales muestran que E5-Large **se degrada** ligeramente con reranking, no mejora. Las tablas actuales del capítulo sugieren mejoras que NO son correctas según los datos.

---

## 🔍 Preguntas Frecuentes

**Q: ¿Por qué algunos valores difieren del capítulo?**
A: Errores de transcripción o redondeo. Los scripts leen directamente del archivo JSON, garantizando valores correctos.

**Q: ¿Puedo confiar en estos scripts?**
A: Sí. Los scripts:
- Leen directamente del archivo JSON oficial
- Verifican que los datos sean reales (flag `is_real_data`)
- No hacen cálculos propios, solo extraen valores

**Q: ¿Qué hago si encuentro un valor dudoso?**
A: Usa `python quick_verify.py` para verificar interactivamente cualquier valor.

**Q: ¿Debo regenerar todas las tablas y gráficos?**
A: No es necesario. Las tablas en `tables/` y gráficos en `charts/` ya están actualizados. Solo copiar las corregidas al capítulo.

---

## 📚 Referencias

- **Capítulo 7**: `/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/capitulo7_resultados.md`
- **Datos**: `/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json`
- **CLAUDE.md**: Directrices del proyecto (nunca usar datos aleatorios/simulados)

---

**Última actualización**: 2025-11-12
**Scripts creados por**: Claude Code
**Datos verificados**: ✅ REALES (2,067 preguntas evaluadas)
