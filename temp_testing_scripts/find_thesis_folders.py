#!/usr/bin/env python3
"""
Busca todas las carpetas que podrían ser la carpeta de tesis original
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    """Autenticar con Google Drive"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def search_folders_by_keywords(service):
    """Busca carpetas por palabras clave relacionadas con tesis"""
    
    print("🔍 Buscando carpetas relacionadas con tesis/magister...")
    
    keywords = [
        'tesis', 'magister', 'thesis', 'master', 'acumulative', 'cumulative',
        'support', 'modelo', 'model', 'investigacion', 'research'
    ]
    
    all_folders = []
    
    for keyword in keywords:
        print(f"🔎 Buscando: '{keyword}'")
        
        # Buscar en nombres de carpetas
        query = f"name contains '{keyword}' and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query, fields="files(id,name,createdTime,modifiedTime)").execute()
        folders = results.get('files', [])
        
        for folder in folders:
            if folder not in all_folders:
                all_folders.append(folder)
                print(f"   📁 {folder['name']} (ID: {folder['id']})")
    
    return all_folders

def list_all_folders(service):
    """Lista TODAS las carpetas para ver qué hay"""
    
    print("\n📂 Listando TODAS las carpetas en tu Google Drive:")
    print("=" * 60)
    
    query = "mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(
        q=query, 
        fields="files(id,name,createdTime,modifiedTime,parents)",
        pageSize=50
    ).execute()
    
    folders = results.get('files', [])
    
    if not folders:
        print("❌ No se encontraron carpetas")
        return []
    
    print(f"📁 Total de carpetas encontradas: {len(folders)}")
    print()
    
    # Separar carpetas raíz de subcarpetas
    root_folders = []
    subfolders = []
    
    for folder in folders:
        if not folder.get('parents'):  # Sin padre = carpeta raíz
            root_folders.append(folder)
        else:
            subfolders.append(folder)
    
    # Mostrar carpetas raíz
    print("📁 CARPETAS RAÍZ:")
    for folder in root_folders:
        print(f"   📂 {folder['name']}")
        print(f"      ID: {folder['id']}")
        print(f"      Creada: {folder['createdTime']}")
        print()
    
    # Mostrar subcarpetas agrupadas por padre
    print("📁 SUBCARPETAS:")
    parent_map = {}
    for folder in subfolders:
        parent_id = folder['parents'][0]
        if parent_id not in parent_map:
            parent_map[parent_id] = []
        parent_map[parent_id].append(folder)
    
    for parent_id, children in parent_map.items():
        # Buscar nombre del padre
        parent_name = "Unknown"
        for root in root_folders:
            if root['id'] == parent_id:
                parent_name = root['name']
                break
        
        print(f"   📂 {parent_name}/")
        for child in children:
            print(f"      └── {child['name']} (ID: {child['id']})")
        print()
    
    return folders

def check_folder_content_detailed(service, folder_id, folder_name):
    """Analiza el contenido de una carpeta en detalle"""
    
    print(f"\n🔍 ANÁLISIS DETALLADO: {folder_name}")
    print("-" * 40)
    
    # Subcarpetas
    subfolder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    subfolders = service.files().list(q=subfolder_query, fields="files(id,name)").execute()
    
    # Archivos
    file_query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder'"
    files = service.files().list(q=file_query, fields="files(id,name,size,mimeType)").execute()
    
    subfolders_list = subfolders.get('files', [])
    files_list = files.get('files', [])
    
    print(f"📁 Subcarpetas ({len(subfolders_list)}):")
    for sf in subfolders_list:
        print(f"   - {sf['name']}")
    
    print(f"📄 Archivos ({len(files_list)}):")
    for f in files_list:
        size = f.get('size', 'N/A')
        mime_type = f.get('mimeType', 'unknown')
        print(f"   - {f['name']} ({size} bytes, {mime_type})")
    
    total_items = len(subfolders_list) + len(files_list)
    print(f"📊 Total de elementos: {total_items}")
    
    return total_items

def main():
    """Función principal"""
    
    print("🔍 BÚSQUEDA COMPLETA: Carpetas de Tesis")
    print("=" * 50)
    
    service = authenticate()
    
    # 1. Buscar por palabras clave
    keyword_folders = search_folders_by_keywords(service)
    
    # 2. Listar todas las carpetas
    all_folders = list_all_folders(service)
    
    # 3. Analizar carpetas que podrían ser la original
    print("\n🎯 ANÁLISIS DE CANDIDATOS:")
    print("=" * 40)
    
    candidates = []
    
    # Añadir carpetas encontradas por palabras clave
    for folder in keyword_folders:
        content_count = check_folder_content_detailed(service, folder['id'], folder['name'])
        candidates.append((folder, content_count))
    
    # Si no hay candidatos, analizar carpetas raíz con más contenido
    if not candidates:
        print("\n🔍 No se encontraron carpetas por palabras clave.")
        print("Analizando carpetas raíz con contenido...")
        
        root_folders = [f for f in all_folders if not f.get('parents')]
        for folder in root_folders[:10]:  # Solo las primeras 10
            content_count = check_folder_content_detailed(service, folder['id'], folder['name'])
            if content_count > 0:
                candidates.append((folder, content_count))
    
    # 4. Mostrar recomendaciones
    if candidates:
        print(f"\n🎯 CANDIDATOS PARA CARPETA ORIGINAL:")
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (folder, count) in enumerate(candidates[:5]):
            print(f"{i+1}. {folder['name']} (ID: {folder['id']})")
            print(f"   📊 Contenido: {count} elementos")
            print(f"   📅 Creada: {folder['createdTime']}")
            print()
        
        print("💡 RECOMENDACIÓN:")
        if candidates[0][1] > 0:
            print(f"✅ Usar: {candidates[0][0]['name']} (más contenido)")
        else:
            print("❓ Todas las carpetas están vacías")
            print("💡 Crear nueva estructura o verificar carpeta manualmente")
    
    print(f"\n📁 Total de carpetas en tu Drive: {len(all_folders)}")

if __name__ == "__main__":
    main()