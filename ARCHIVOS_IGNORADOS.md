# Archivos Ignorados en Git

Este documento explica qué archivos están siendo ignorados por Git y por qué.

## 🗂️ Carpetas Completamente Ignoradas

### `data/`
**Razón**: Contiene datasets, modelos entrenados y resultados experimentales grandes
- `cumulative_results_*.json` - Resultados experimentales (pueden regenerarse)
- `*.pt` - Modelos entrenados PyTorch (>100MB)
- `preguntas_con_links_validos.csv` - Dataset procesado
- `embedding_*.json` - Embeddings pre-calculados

### `chroma_db/` y `chroma_db2/`
**Razón**: Bases de datos vectoriales locales (>1GB cada una)
- Contienen los 800,000+ vectores indexados
- Pueden recrearse ejecutando scripts de población

### `Docs_Previous/`
**Razón**: Documentación obsoleta de versiones anteriores
- Arquitectura antigua del sistema
- Presentaciones previas
- Documentos de proyecto preliminares

### `stable_env/` y ambientes virtuales
**Razón**: Ambientes virtuales locales específicos de cada desarrollador
- No deben versionarse
- Cada desarrollador debe crear su propio ambiente

## 🔒 Archivos de Seguridad

### Credenciales y Tokens
```
src/config/credentials.json      # API keys y credenciales
src/config/gdrive_config.json    # Configuración Google Drive
token.pickle                     # Token de autenticación Google
.env                            # Variables de ambiente
```

### Archivos de Desarrollo
```
CLAUDE.md                       # Instrucciones internas de Claude AI
streamlit.log                   # Logs de desarrollo
```

## 📊 Archivos de Datos Grandes

### Embeddings Pre-calculados
```
colab_data/*.parquet            # Embeddings de documentos (>100MB cada uno)
colab_data/*.json               # Metadatos de embeddings
```

### Resultados de Análisis
```
Docs/Analisis/*.json            # Resultados de análisis automatizados
Docs/Analisis/*.csv             # Datos de tests estadísticos
```

## 🔧 Archivos de Sistema

### Python
```
__pycache__/                    # Cache de Python bytecode
*.pyc, *.pyo, *.pyd            # Archivos compilados Python
*.egg-info/                     # Metadatos de paquetes
```

### Sistema Operativo
```
.DS_Store                       # Metadatos de macOS
Thumbs.db                       # Cache de Windows
```

### Editores
```
.vscode/                        # Configuración VS Code
.idea/                          # Configuración PyCharm
```

## 📋 Archivos Esenciales NO Ignorados

### Código Fuente
- `src/` - Todo el código Python del sistema
- `external_helpers/` - Scripts auxiliares
- `colab_data/*.py` - Scripts de Colab (pero no los datos)

### Documentación Final
- `Docs/Finales/` - Documentación final de la tesis
- `Docs/Analisis/*.py` - Scripts de análisis (pero no resultados)
- `README.md` - Documentación principal

### Configuración
- `requirements.txt` - Dependencias Python
- `.gitignore` - Configuración de Git
- `*.ipynb` - Notebooks Jupyter (pero no checkpoints)

## 🔄 Cómo Recuperar Archivos Ignorados

Si necesitas acceder a archivos ignorados:

### Para Datos Experimentales:
```bash
# Regenerar resultados ejecutando evaluación
python colab_data/Cumulative_Ticket_Evaluation.ipynb

# Recrear análisis
python Docs/Analisis/analyze_metrics_v2.py
```

### Para Base de Datos:
```bash
# Repoblar ChromaDB
python external_helpers/populate_questions_e5large.py
```

### Para Configuración:
```bash
# Crear archivo de credenciales
cp src/config/credentials.json.template src/config/credentials.json
# Editar con tus API keys
```

## ⚠️ Archivos Críticos para Reproducibilidad

Aunque estos archivos están ignorados, son **esenciales** para reproducir la investigación:

1. **`data/cumulative_results_1753578255.json`** - Resultados experimentales principales
2. **`data/preguntas_con_links_validos.csv`** - Ground truth validado
3. **`chroma_db2/`** - Base de datos con 187,031 documentos indexados

**Solución**: Estos archivos están disponibles mediante:
- Google Drive compartido del proyecto
- Scripts de regeneración en `external_helpers/`
- Instrucciones en `Docs/Finales/anexo_c_configuracion_ambiente.md`

## 📞 Contacto

Si necesitas acceso a archivos ignorados para reproducir la investigación:
1. Revisa las instrucciones en `Docs/Finales/anexo_c_configuracion_ambiente.md`
2. Ejecuta los scripts de configuración en `external_helpers/`
3. Contacta al autor para acceso a Google Drive compartido

---

**Última actualización**: Agosto 2025  
**Propósito**: Mantener el repositorio limpio y seguro sin comprometer la reproducibilidad