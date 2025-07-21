#!/usr/bin/env python3
"""
Check current evaluation status in Google Drive
"""

from utils.real_gdrive_integration import check_evaluation_status_in_drive, download_json_from_drive, authenticate_gdrive, load_gdrive_config

def main():
    print("🔍 Verificando estado de evaluación...")
    
    # Verificar estado
    status_result = check_evaluation_status_in_drive()
    
    if not status_result['success']:
        print(f"❌ Error: {status_result['error']}")
        return
    
    print(f"📊 Estado: {status_result['status']}")
    
    if 'data' in status_result:
        data = status_result['data']
        print(f"📅 Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"📁 Config file: {data.get('config_file', 'N/A')}")
        print(f"📄 Results file: {data.get('results_file', 'N/A')}")
        print(f"✅ Evaluation pending: {data.get('evaluation_pending', 'N/A')}")
        
        # Si hay archivo de resultados especificado, verificar si existe
        if 'results_file' in data:
            print(f"\n🎯 Buscando archivo de resultados: {data['results_file']}")
            
            # Autenticar y buscar
            service = authenticate_gdrive()
            folder_id = load_gdrive_config()
            
            if service and folder_id:
                # Buscar en carpeta results
                from utils.real_gdrive_integration import find_file_in_drive
                
                results_folder_result = find_file_in_drive(service, folder_id, 'results')
                if results_folder_result['success'] and results_folder_result['found']:
                    results_folder_id = results_folder_result['file_id']
                    
                    file_result = find_file_in_drive(service, results_folder_id, data['results_file'])
                    if file_result['success'] and file_result['found']:
                        print(f"✅ Archivo de resultados encontrado!")
                        print(f"   ID: {file_result['file_id']}")
                        print(f"   Modificado: {file_result['modified_time']}")
                    else:
                        print(f"❌ Archivo de resultados NO encontrado")
                else:
                    print(f"❌ Carpeta 'results' no encontrada")

if __name__ == "__main__":
    main()