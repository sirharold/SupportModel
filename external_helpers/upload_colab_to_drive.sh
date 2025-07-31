#!/bin/bash

# Script para subir el notebook Colab fixed a Google Drive
# Requiere rclone configurado con Google Drive

echo "📤 Subiendo Cumulative_N_Questions_Colab_Fixed.ipynb a Google Drive..."

# Configurar variables
NOTEBOOK_PATH="/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Cumulative_N_Questions_Colab_Fixed.ipynb"
DRIVE_PATH="gdrive:/"
BACKUP_FOLDER="colab_notebooks"

# Verificar que el archivo existe
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "❌ Error: No se encontró el archivo $NOTEBOOK_PATH"
    exit 1
fi

# Verificar que rclone está instalado
if ! command -v rclone &> /dev/null; then
    echo "❌ Error: rclone no está instalado"
    echo "💡 Instala rclone: https://rclone.org/install/"
    echo "💡 O usa: brew install rclone"
    exit 1
fi

# Verificar que rclone está configurado con Google Drive
if ! rclone listremotes | grep -q "gdrive:"; then
    echo "❌ Error: rclone no está configurado con Google Drive"
    echo "💡 Configura rclone con: rclone config"
    echo "💡 Usa 'gdrive' como nombre del remote"
    exit 1
fi

# Crear carpeta de backup si no existe
echo "📁 Creando carpeta $BACKUP_FOLDER..."
rclone mkdir "$DRIVE_PATH$BACKUP_FOLDER"

# Subir archivo principal
echo "📤 Subiendo archivo principal..."
rclone copy "$NOTEBOOK_PATH" "$DRIVE_PATH" -v

# Subir backup con timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="Cumulative_N_Questions_Colab_Fixed_${TIMESTAMP}.ipynb"

echo "📤 Subiendo backup: $BACKUP_NAME..."
rclone copy "$NOTEBOOK_PATH" "$DRIVE_PATH$BACKUP_FOLDER" --verbose

# Renombrar backup con timestamp
rclone moveto "$DRIVE_PATH$BACKUP_FOLDER/Cumulative_N_Questions_Colab_Fixed.ipynb" "$DRIVE_PATH$BACKUP_FOLDER/$BACKUP_NAME"

# Verificar que se subió correctamente
if rclone lsf "$DRIVE_PATH" | grep -q "Cumulative_N_Questions_Colab_Fixed.ipynb"; then
    echo "✅ Archivo principal subido exitosamente"
else
    echo "❌ Error subiendo archivo principal"
    exit 1
fi

if rclone lsf "$DRIVE_PATH$BACKUP_FOLDER" | grep -q "$BACKUP_NAME"; then
    echo "✅ Backup subido exitosamente: $BACKUP_NAME"
else
    echo "❌ Error subiendo backup"
fi

echo ""
echo "🎉 ¡Completado!"
echo "📄 Archivo principal: $DRIVE_PATH/Cumulative_N_Questions_Colab_Fixed.ipynb"
echo "💾 Backup: $DRIVE_PATH$BACKUP_FOLDER/$BACKUP_NAME"
echo ""
echo "📋 Próximos pasos:"
echo "1. Ve a Google Drive"
echo "2. Busca: Cumulative_N_Questions_Colab_Fixed.ipynb"
echo "3. Ábrelo en Google Colab"
echo "4. ¡Listo para ejecutar!"