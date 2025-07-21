#!/usr/bin/env python3
"""
Script para verificar qué archivos hay en la carpeta results/ de Google Drive
"""

import os
import json
from utils.real_gdrive_integration import authenticate_gdrive, load_gdrive_config, find_file_in_drive

def list_files_in_folder(service, folder_id, folder_name=""):
    """Lista todos los archivos en una carpeta"""
    try:
        query = f"'{folder_id}' in parents"
        results = service.files().list(
            q=query, 
            fields="files(id,name,modifiedTime,mimeType,size)",
            pageSize=100
        ).execute()
        files = results.get('files', [])
        
        print(f"\n📁 Archivos en {folder_name}:")
        print("-" * 50)
        
        if not files:
            print("   (vacía)")
        else:
            for file in files:
                print(f"   📄 {file['name']}")
                print(f"      ID: {file['id']}")
                print(f"      Tipo: {file.get('mimeType', 'N/A')}")
                print(f"      Modificado: {file.get('modifiedTime', 'N/A')}")
                if 'size' in file:
                    print(f"      Tamaño: {file['size']} bytes")
                print()
        
        return files
        
    except Exception as e:
        print(f"❌ Error listando archivos: {e}")
        return []

def main():
    print("🔍 Verificando archivos en Google Drive...")
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        print("❌ No se pudo autenticar con Google Drive")
        return
    
    # Obtener carpeta principal
    folder_id = load_gdrive_config()
    if not folder_id:
        print("❌ No se pudo cargar configuración de carpetas")
        return
    
    print(f"✅ Carpeta principal ID: {folder_id}")
    
    # Listar archivos en carpeta principal
    main_files = list_files_in_folder(service, folder_id, "acumulative (principal)")
    
    # Buscar carpeta results
    results_folder_result = find_file_in_drive(service, folder_id, 'results')
    
    if results_folder_result['success'] and results_folder_result['found']:
        results_folder_id = results_folder_result['file_id']
        print(f"✅ Carpeta results encontrada ID: {results_folder_id}")
        
        # Listar archivos en carpeta results
        results_files = list_files_in_folder(service, results_folder_id, "results")
        
        # Buscar específicamente archivos de resultados
        cumulative_files = [f for f in results_files if f['name'].startswith('cumulative_results_')]
        
        if cumulative_files:
            print(f"\n🎯 Archivos de resultados encontrados: {len(cumulative_files)}")
            for file in cumulative_files:
                print(f"   📊 {file['name']} (Modificado: {file.get('modifiedTime', 'N/A')})")
        else:
            print(f"\n⚠️ No se encontraron archivos cumulative_results_* en la carpeta results")
            
    else:
        print("❌ No se encontró la carpeta 'results'")
        print("📋 Archivos disponibles en carpeta principal:")
        for file in main_files:
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                print(f"   📁 Carpeta: {file['name']}")

if __name__ == "__main__":
    main()