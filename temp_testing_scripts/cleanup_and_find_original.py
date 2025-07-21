#!/usr/bin/env python3
"""
Encuentra la carpeta TesisMagister original y limpia duplicados
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

def find_all_tesis_folders(service):
    """Encuentra todas las carpetas TesisMagister"""
    
    print("🔍 Buscando todas las carpetas 'TesisMagister'...")
    
    query = "name='TesisMagister' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id,name,createdTime,modifiedTime)").execute()
    folders = results.get('files', [])
    
    print(f"📁 Encontradas {len(folders)} carpetas 'TesisMagister':")
    
    for i, folder in enumerate(folders):
        print(f"   {i+1}. ID: {folder['id']}")
        print(f"      Creada: {folder['createdTime']}")
        print(f"      Modificada: {folder['modifiedTime']}")
        
        # Verificar contenido
        subfolder_query = f"'{folder['id']}' in parents and mimeType='application/vnd.google-apps.folder'"
        subfolders = service.files().list(q=subfolder_query).execute()
        subfolder_names = [sf['name'] for sf in subfolders.get('files', [])]
        
        file_query = f"'{folder['id']}' in parents and mimeType!='application/vnd.google-apps.folder'"
        files = service.files().list(q=file_query).execute()
        file_count = len(files.get('files', []))
        
        print(f"      Subcarpetas: {subfolder_names}")
        print(f"      Archivos: {file_count}")
        print()
    
    return folders

def check_folder_contents(service, folder_id, folder_name):
    """Verifica el contenido de una carpeta"""
    
    print(f"📂 Contenido de {folder_name} (ID: {folder_id}):")
    
    # Listar subcarpetas
    subfolder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    subfolders = service.files().list(q=subfolder_query, fields="files(id,name)").execute()
    
    # Listar archivos
    file_query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder'"
    files = service.files().list(q=file_query, fields="files(id,name,size)").execute()
    
    subfolders_list = subfolders.get('files', [])
    files_list = files.get('files', [])
    
    print(f"   📁 Subcarpetas ({len(subfolders_list)}):")
    for sf in subfolders_list:
        print(f"      - {sf['name']} (ID: {sf['id']})")
    
    print(f"   📄 Archivos ({len(files_list)}):")
    for f in files_list:
        size = f.get('size', 'N/A')
        print(f"      - {f['name']} ({size} bytes)")
    
    return len(subfolders_list) + len(files_list)

def delete_folder(service, folder_id, folder_name):
    """Elimina una carpeta"""
    
    try:
        service.files().delete(fileId=folder_id).execute()
        print(f"🗑️  Carpeta '{folder_name}' eliminada (ID: {folder_id})")
        return True
    except Exception as e:
        print(f"❌ Error eliminando carpeta: {e}")
        return False

def ensure_acumulative_folder(service, tesis_folder_id):
    """Asegura que existe la subcarpeta 'acumulative'"""
    
    print(f"🔍 Verificando subcarpeta 'acumulative' en TesisMagister...")
    
    query = f"name='acumulative' and '{tesis_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query).execute()
    acum_folders = results.get('files', [])
    
    if acum_folders:
        acum_id = acum_folders[0]['id']
        print(f"✅ Carpeta 'acumulative' ya existe (ID: {acum_id})")
        return acum_id
    else:
        print("📁 Creando carpeta 'acumulative'...")
        acum_metadata = {
            'name': 'acumulative',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [tesis_folder_id]
        }
        
        acum_folder = service.files().create(body=acum_metadata, fields='id').execute()
        acum_id = acum_folder.get('id')
        print(f"✅ Carpeta 'acumulative' creada (ID: {acum_id})")
        return acum_id

def main():
    """Función principal"""
    
    print("🧹 LIMPIEZA: Carpetas TesisMagister Duplicadas")
    print("=" * 50)
    
    service = authenticate()
    
    # 1. Encontrar todas las carpetas TesisMagister
    folders = find_all_tesis_folders(service)
    
    if len(folders) <= 1:
        print("✅ Solo hay una carpeta TesisMagister, no hay duplicados")
        if len(folders) == 1:
            folder_id = folders[0]['id']
            acum_id = ensure_acumulative_folder(service, folder_id)
            print(f"\n🎯 Usar carpeta: {folder_id}")
            print(f"📁 Subcarpeta acumulative: {acum_id}")
        return
    
    # 2. Analizar contenido de cada carpeta
    print("\n📊 Analizando contenido de cada carpeta:")
    folder_scores = []
    
    for i, folder in enumerate(folders):
        folder_id = folder['id']
        content_count = check_folder_contents(service, folder_id, f"TesisMagister #{i+1}")
        folder_scores.append((folder, content_count))
        print()
    
    # 3. Identificar carpeta original (la que tiene más contenido)
    folder_scores.sort(key=lambda x: x[1], reverse=True)
    original_folder = folder_scores[0][0]
    original_count = folder_scores[0][1]
    
    print("🎯 RECOMENDACIÓN:")
    print(f"✅ Mantener: TesisMagister (ID: {original_folder['id']})")
    print(f"   📊 Contenido: {original_count} elementos")
    print(f"   📅 Creada: {original_folder['createdTime']}")
    
    # 4. Eliminar carpetas duplicadas
    for folder, count in folder_scores[1:]:
        print(f"🗑️  Eliminar: TesisMagister (ID: {folder['id']}) - {count} elementos")
    
    print(f"\n🤔 ¿Proceder con la limpieza? Se mantendrá la carpeta con más contenido.")
    print("   La carpeta vacía recién creada será eliminada.")
    
    # Para automatizar, eliminar las carpetas vacías
    deleted_count = 0
    for folder, count in folder_scores[1:]:
        if count == 0:  # Solo eliminar carpetas vacías
            if delete_folder(service, folder['id'], "TesisMagister (duplicada)"):
                deleted_count += 1
    
    print(f"\n✅ Limpieza completada:")
    print(f"🗑️  Eliminadas: {deleted_count} carpetas vacías")
    print(f"✅ Mantenida: TesisMagister (ID: {original_folder['id']})")
    
    # 5. Asegurar subcarpeta acumulative
    acum_id = ensure_acumulative_folder(service, original_folder['id'])
    
    print(f"\n🎯 CONFIGURACIÓN FINAL:")
    print(f"📁 TesisMagister: {original_folder['id']}")
    print(f"📁 acumulative: {acum_id}")
    
    # 6. Crear archivo de configuración para futuros scripts
    config = {
        "tesis_folder_id": original_folder['id'],
        "acumulative_folder_id": acum_id
    }
    
    import json
    with open('gdrive_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Configuración guardada en: gdrive_config.json")
    
    return original_folder['id'], acum_id

if __name__ == "__main__":
    main()