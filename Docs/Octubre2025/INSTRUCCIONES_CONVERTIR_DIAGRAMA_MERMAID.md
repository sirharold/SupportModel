# Instrucciones para Convertir el Diagrama Mermaid del Capítulo 5

## Problema Identificado

El diagrama Gantt de Mermaid en el capítulo 5 (líneas 15-52) no se convierte automáticamente a imagen cuando se usa pandoc para generar Word.

## Solución

### Opción 1: Usar Mermaid.live (Recomendado - Más Rápido)

1. Abrir: https://mermaid.live/
2. El código mermaid ya está copiado en el archivo `mermaid_temp.mmd`
3. Copiar todo el contenido de `mermaid_temp.mmd` en el editor de mermaid.live
4. Ajustar zoom/tamaño si es necesario
5. Click en "Actions" → "PNG" → Descargar
6. Guardar como: `img/Capitulo5FlujoMetodologico.png`
7. Ejecutar el script de conversión nuevamente

### Opción 2: Instalar mermaid-cli (Si tienes Node.js)

```bash
# Instalar mermaid-cli globalmente
npm install -g @mermaid-js/mermaid-cli

# Navegar al directorio
cd /Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025

# Generar la imagen con alta calidad
mmdc -i mermaid_temp.mmd -o img/Capitulo5FlujoMetodologico.png -w 1400 -H 800 -b transparent

# Verificar que se creó
ls -lh img/Capitulo5FlujoMetodologico.png
```

### Opción 3: Usar la API de Mermaid (Automático)

```bash
# El siguiente comando usa la API de mermaid.ink
curl -X POST 'https://mermaid.ink/img/' \
  -H 'Content-Type: application/json' \
  -d @mermaid_temp.mmd \
  -o img/Capitulo5FlujoMetodologico.png
```

## Después de Crear la Imagen

Una vez que la imagen esté en `img/Capitulo5FlujoMetodologico.png`, el markdown del capítulo 5 ya está configurado para usarla automáticamente (ya fue actualizado).

## Archivos Generados

- `mermaid_temp.mmd`: Código Mermaid extraído del capítulo 5
- `img/Capitulo5FlujoMetodologico.png`: Imagen que se debe generar
