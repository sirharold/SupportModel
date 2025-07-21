#!/usr/bin/env python3
"""
Fix evaluation status to point to existing test results file
"""

from utils.real_gdrive_integration import authenticate_gdrive, load_gdrive_config, upload_json_to_drive
from datetime import datetime

def main():
    print("🔧 Corrigiendo estado de evaluación...")
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        print("❌ No se pudo autenticar")
        return
    
    folder_id = load_gdrive_config()
    if not folder_id:
        print("❌ No se pudo cargar configuración")
        return
    
    # Crear nuevo status que apunte al archivo test existente
    status_data = {
        'status': 'completed',
        'timestamp': datetime.now().isoformat(),
        'config_file': 'evaluation_config_test.json',
        'results_file': 'cumulative_results_test.json',
        'evaluation_pending': False,
        'completed_at': datetime.now().isoformat(),
        'note': 'Using test results file for demonstration'
    }
    
    # Subir nuevo archivo de status
    result = upload_json_to_drive(service, folder_id, 'evaluation_status.json', status_data)
    
    if result['success']:
        print("✅ Estado de evaluación actualizado")
        print(f"📄 Ahora apunta a: cumulative_results_test.json")
        print("🎯 Puedes probar el botón 'Mostrar Resultados' en Streamlit")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()