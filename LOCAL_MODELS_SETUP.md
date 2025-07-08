# Configuración de Modelos Locales para Optimización de Costos

Este documento explica cómo configurar y usar modelos locales para eliminar completamente los costos de APIs externas.

## 🎯 Beneficios

- **Costo Cero**: Sin gastos de APIs de OpenAI, Gemini, etc.
- **Privacidad**: Todos los datos permanecen en tu sistema
- **Control Total**: Sin dependencias de servicios externos
- **Rendimiento**: Latencia constante sin límites de rate

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **RAM**: 8GB+ (16GB recomendado)
- **GPU**: NVIDIA con 6GB+ VRAM (opcional pero recomendado)
- **Almacenamiento**: 20GB libres para modelos

### Hardware Recomendado
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 3080/4080 o superior
- **CPU**: Intel i7/AMD Ryzen 7 o superior

## 🚀 Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Instalar PyTorch con CUDA (si tienes GPU NVIDIA)

```bash
# Para CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Configurar Hugging Face Token (opcional)

```bash
# Para acceso a modelos de Meta (Llama)
huggingface-cli login
```

## 🔧 Configuración

### 1. Modelos Disponibles

La aplicación ahora incluye estos modelos locales:

- **Llama 3.1 8B**: Generación de respuestas de alta calidad
- **Mistral 7B**: Refinamiento de queries y generación rápida
- **SentenceTransformers**: Embeddings locales (ya configurados)

### 2. Configuración Automática

Los modelos se descargan automáticamente en el primer uso:

```python
# En config.py - ya configurado
DEFAULT_GENERATIVE_MODEL = "llama-3.1-8b"  # Ahora usa modelo local por defecto
```

### 3. Variables de Entorno

Asegúrate de que estas variables estén configuradas en tu `.env`:

```env
# Opcional: Solo para modelos HuggingFace privados
HUGGINGFACE_API_KEY=tu_token_aqui

# Las siguientes son ahora OPCIONALES
# OPENAI_API_KEY=  # Solo si quieres usar GPT-4 como fallback
# GEMINI_API_KEY=  # Solo si quieres usar Gemini como fallback
```

## 🖥️ Uso

### 1. Selección de Modelos

En la interfaz de Streamlit:

1. Ve a **Configuración Avanzada**
2. Selecciona **Modelo Generativo**: `llama-3.1-8b` o `mistral-7b`
3. Mantén **Modelos de Embedding** en opciones locales (HuggingFace)

### 2. Primera Ejecución

El primer uso descargará los modelos (puede tomar 10-30 minutos):

```
[INFO] Loading model: llama-3.1-8b from meta-llama/Llama-3.1-8B-Instruct
[INFO] Model downloaded successfully
```

### 3. Monitoreo de Recursos

Para monitorear uso de GPU:

```bash
# En terminal separada
nvidia-smi -l 1
```

## ⚡ Optimizaciones

### 1. Cuantización Automática

Los modelos usan cuantización 4-bit automáticamente para optimizar memoria:

```python
# Configuración automática en local_models.py
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

### 2. Gestión de Memoria

Para liberar memoria manualmente:

```python
from utils.local_models import cleanup_models
cleanup_models()
```

### 3. Modelos Más Pequeños

Si tienes recursos limitados, puedes modificar `local_models.py`:

```python
# Cambiar a modelos más pequeños
"mistral-7b": "microsoft/DialoGPT-medium"  # Modelo más pequeño
"llama-3.1-8b": "microsoft/DialoGPT-large"  # Alternativa más ligera
```

## 🔍 Troubleshooting

### Error: "CUDA out of memory"

```python
# Reducir batch size en local_models.py
max_new_tokens=256  # Reducir de 512
```

### Error: "Model not found"

```bash
# Limpiar caché de HuggingFace
rm -rf ~/.cache/huggingface/
```

### Rendimiento Lento

1. **Verificar GPU**: Asegúrate de que PyTorch detecte tu GPU
2. **Aumentar RAM**: Cerrar aplicaciones innecesarias
3. **Usar modelos más pequeños**: Cambiar a Mistral 7B

### Error de Permisos con Llama

```bash
# Obtener acceso a modelos de Meta
huggingface-cli login
# Ir a https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Solicitar acceso si es necesario
```

## 📊 Comparación de Rendimiento

| Métrica | APIs Externas | Modelos Locales |
|---------|---------------|-----------------|
| **Costo por consulta** | $0.10-0.20 | $0.00 |
| **Latencia inicial** | 2-5s | 10-30s (primera carga) |
| **Latencia normal** | 2-5s | 1-3s |
| **Privacidad** | Datos enviados | 100% local |
| **Disponibilidad** | Depende de API | 24/7 |

## ✅ Verificación de Instalación

Ejecuta este script para verificar que todo funciona:

```python
from utils.local_models import get_llama_client, get_mistral_client

# Probar Llama
llama = get_llama_client()
response = llama.generate_answer("¿Qué es Python?", "Python es un lenguaje de programación")
print("Llama respuesta:", response)

# Probar Mistral  
mistral = get_mistral_client()
refined = mistral.refine_query("Hola, ¿podrías ayudarme con Python por favor?")
print("Mistral refinado:", refined)
```

## 🎉 ¡Listo!

Una vez configurado, tu aplicación funcionará completamente sin costos de API. Los modelos locales ofrecen calidad comparable a GPT-4 y Gemini Pro para la mayoría de casos de uso.

Para soporte adicional, revisa los logs en la consola o abre un issue en el repositorio.