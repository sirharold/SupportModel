#!/bin/bash

# Script para ejecutar la aplicación Azure Q&A Expert System
# Autor: Harold Gómez
# Fecha: 2025-10-01

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Iniciando Azure Q&A Expert System...${NC}"

# Directorio del proyecto
PROJECT_DIR="/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel"

# Verificar que estamos en el directorio correcto
if [ ! -f "$PROJECT_DIR/src/apps/main_qa_app.py" ]; then
    echo -e "${YELLOW}⚠️  Advertencia: No se encuentra main_qa_app.py en la ruta esperada${NC}"
    echo "Verificando directorio actual..."
fi

# Cambiar al directorio del proyecto
cd "$PROJECT_DIR" || {
    echo "❌ Error: No se pudo acceder al directorio del proyecto"
    exit 1
}

# Verificar que el entorno virtual existe
if [ ! -f "stable_env/bin/streamlit" ]; then
    echo "❌ Error: No se encuentra el entorno virtual en stable_env/"
    echo "Verifica que el entorno virtual esté instalado correctamente"
    exit 1
fi

echo -e "${GREEN}✅ Directorio del proyecto: $PROJECT_DIR${NC}"
echo -e "${GREEN}✅ Entorno virtual encontrado${NC}"
echo -e "${BLUE}🌐 La aplicación se abrirá en tu navegador...${NC}"
echo ""

# Ejecutar la aplicación con las variables de entorno correctas
PYTHONPATH="$PROJECT_DIR" "$PROJECT_DIR/stable_env/bin/streamlit" run src/apps/main_qa_app.py

echo -e "${BLUE}👋 Aplicación cerrada${NC}"