#!/usr/bin/env python3
"""
Script para subir el notebook Colab fixed a Google Drive
Usa las mismas credenciales que el sistema Streamlit
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def upload_notebook_to_drive():
    """Subir notebook a Google Drive usando las funciones existentes."""
    
    print("📤 Subiendo Cumulative_N_Questions_Colab_Fixed.ipynb a Google Drive...")
    
    # Verificar que el archivo existe
    notebook_path = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Cumulative_N_Questions_Colab_Fixed.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ Error: No se encontró el archivo {notebook_path}")
        return False
    
    try:
        # Importar las funciones del sistema
        sys.path.append('/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel')
        from src.services.storage.real_gdrive_integration import upload_file_to_drive
        
        # Leer el contenido del notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()
        
        # Generar nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_filename = "Cumulative_N_Questions_Colab_Fixed.ipynb"
        backup_filename = f"Cumulative_N_Questions_Colab_Fixed_backup_{timestamp}.ipynb"
        
        print(f"📄 Archivo principal: {main_filename}")
        print(f"💾 Backup: {backup_filename}")
        
        # Subir archivo principal
        print("📤 Subiendo archivo principal...")
        result_main = upload_file_to_drive(
            content=notebook_content,
            filename=main_filename,
            content_type='application/json'
        )
        
        if result_main.get('success'):
            print("✅ Archivo principal subido exitosamente")
            if 'web_link' in result_main:
                print(f"🔗 Link: {result_main['web_link']}")
        else:
            print(f"❌ Error subiendo archivo principal: {result_main.get('error', 'Error desconocido')}")
            return False
        
        # Subir backup
        print("📤 Subiendo backup...")
        result_backup = upload_file_to_drive(
            content=notebook_content,
            filename=backup_filename,
            content_type='application/json'
        )
        
        if result_backup.get('success'):
            print("✅ Backup subido exitosamente")
            if 'web_link' in result_backup:
                print(f"🔗 Link backup: {result_backup['web_link']}")
        else:
            print(f"⚠️ Error subiendo backup: {result_backup.get('error', 'Error desconocido')}")
        
        print("\n🎉 ¡Completado!")
        print("📋 Próximos pasos:")
        print("1. Ve a Google Drive")
        print("2. Busca: Cumulative_N_Questions_Colab_Fixed.ipynb")
        print("3. Ábrelo en Google Colab")
        print("4. ¡Listo para ejecutar con gemini-1.5-flash!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("💡 Asegúrate de estar en el directorio correcto del proyecto")
        return False
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return False

def check_file_info():
    """Mostrar información del archivo a subir."""
    notebook_path = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Cumulative_N_Questions_Colab_Fixed.ipynb"
    
    if os.path.exists(notebook_path):
        file_stats = os.stat(notebook_path)
        file_size = file_stats.st_size
        mod_time = datetime.fromtimestamp(file_stats.st_mtime)
        
        print(f"📊 Información del archivo:")
        print(f"   📄 Archivo: Cumulative_N_Questions_Colab_Fixed.ipynb")
        print(f"   📏 Tamaño: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"   📅 Modificado: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📁 Ruta: {notebook_path}")
        return True
    else:
        print(f"❌ Archivo no encontrado: {notebook_path}")
        return False

if __name__ == "__main__":
    print("🚀 Script de subida de notebook a Google Drive")
    print("=" * 50)
    
    # Mostrar información del archivo
    if not check_file_info():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Confirmar antes de subir
    response = input("¿Continuar con la subida? (y/N): ").strip().lower()
    
    if response in ['y', 'yes', 'sí', 'si']:
        success = upload_notebook_to_drive()
        sys.exit(0 if success else 1)
    else:
        print("📋 Subida cancelada")
        sys.exit(0)