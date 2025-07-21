#!/usr/bin/env python3
"""
Check the structure of the test results file
"""

from utils.real_gdrive_integration import authenticate_gdrive, load_gdrive_config, find_file_in_drive, download_json_from_drive
import json

def main():
    print("🔍 Verificando estructura del archivo de resultados test...")
    
    # Autenticar
    service = authenticate_gdrive()
    folder_id = load_gdrive_config()
    
    if not service or not folder_id:
        print("❌ Error de autenticación")
        return
    
    # Encontrar carpeta results
    results_folder_result = find_file_in_drive(service, folder_id, 'results')
    if not results_folder_result['success'] or not results_folder_result['found']:
        print("❌ No se encontró carpeta results")
        return
    
    results_folder_id = results_folder_result['file_id']
    
    # Encontrar archivo test
    file_result = find_file_in_drive(service, results_folder_id, 'cumulative_results_test.json')
    if not file_result['success'] or not file_result['found']:
        print("❌ No se encontró archivo test")
        return
    
    # Descargar y analizar
    download_result = download_json_from_drive(service, file_result['file_id'])
    if not download_result['success']:
        print(f"❌ Error descargando: {download_result['error']}")
        return
    
    data = download_result['data']
    print("📊 Estructura del archivo:")
    print("-" * 50)
    
    def print_structure(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}: ({type(value).__name__})")
                    print_structure(value, indent + 1)
                else:
                    print(f"{prefix}{key}: {type(value).__name__} = {str(value)[:100]}")
        elif isinstance(obj, list):
            print(f"{prefix}[Lista con {len(obj)} elementos]")
            if obj:
                print(f"{prefix}Primer elemento:")
                print_structure(obj[0], indent + 1)
    
    print_structure(data)
    
    print(f"\n📋 Claves principales: {list(data.keys()) if isinstance(data, dict) else 'No es dict'}")

if __name__ == "__main__":
    main()