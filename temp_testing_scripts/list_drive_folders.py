#!/usr/bin/env python3
"""
Lista todas las carpetas en Google Drive para encontrar la estructura correcta
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def list_all_folders():
    """Lista todas las carpetas en Google Drive"""
    
    print("📂 Listando carpetas en Google Drive...")
    
    # Autenticar
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
    
    try:
        service = build('drive', 'v3', credentials=creds)
        
        # Buscar todas las carpetas
        query = "mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query, fields="files(id,name,parents)").execute()
        folders = results.get('files', [])
        
        print(f"📁 Encontradas {len(folders)} carpetas:")
        print("=" * 50)
        
        # Mostrar carpetas organizadas
        root_folders = []
        child_folders = []
        
        for folder in folders:
            folder_name = folder['name']
            folder_id = folder['id']
            parents = folder.get('parents', [])
            
            if not parents:  # Carpeta raíz
                root_folders.append((folder_name, folder_id))
                print(f"📁 {folder_name} (ID: {folder_id})")
            else:
                child_folders.append((folder_name, folder_id, parents[0]))
        
        # Mostrar subcarpetas
        print("\n📂 Subcarpetas:")
        for folder_name, folder_id, parent_id in child_folders:
            # Buscar nombre del padre
            parent_name = "Unknown"
            for parent_folder_name, parent_folder_id in root_folders:
                if parent_folder_id == parent_id:
                    parent_name = parent_folder_name
                    break
            
            print(f"   📁 {parent_name}/{folder_name} (ID: {folder_id})")
        
        # Buscar específicamente carpetas relacionadas con tesis
        print("\n🔍 Buscando carpetas relacionadas con 'tesis', 'magister', 'acumulative':")
        keywords = ['tesis', 'magister', 'acumulative', 'support', 'thesis']
        
        found_relevant = False
        for folder in folders:
            folder_name = folder['name'].lower()
            for keyword in keywords:
                if keyword in folder_name:
                    print(f"✅ Encontrada: {folder['name']} (ID: {folder['id']})")
                    found_relevant = True
                    break
        
        if not found_relevant:
            print("❌ No se encontraron carpetas relacionadas")
            print("💡 Tip: Crea la carpeta manualmente en Google Drive")
        
        # Crear carpeta de prueba si no existe
        print("\n🔧 ¿Crear carpeta de prueba 'TesisMagister/acumulative'? (y/n)")
        # return folders para uso programático
        return folders, service
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], None

def create_test_folder_structure(service):
    """Crea la estructura de carpetas necesaria"""
    
    print("\n🏗️  Creando estructura de carpetas...")
    
    try:
        # Crear TesisMagister
        tesis_metadata = {
            'name': 'TesisMagister',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        tesis_folder = service.files().create(body=tesis_metadata, fields='id').execute()
        tesis_id = tesis_folder.get('id')
        print(f"✅ Carpeta 'TesisMagister' creada (ID: {tesis_id})")
        
        # Crear acumulative dentro de TesisMagister
        acum_metadata = {
            'name': 'acumulative',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [tesis_id]
        }
        
        acum_folder = service.files().create(body=acum_metadata, fields='id').execute()
        acum_id = acum_folder.get('id')
        print(f"✅ Carpeta 'acumulative' creada (ID: {acum_id})")
        
        return tesis_id, acum_id
        
    except Exception as e:
        print(f"❌ Error creando carpetas: {e}")
        return None, None

if __name__ == "__main__":
    folders, service = list_all_folders()
    
    if service:
        print("\n" + "="*50)
        print("🤔 ¿Quieres crear la estructura TesisMagister/acumulative?")
        print("Esto creará las carpetas necesarias para el proyecto.")
        
        # Para automatizar, vamos a crear las carpetas
        tesis_id, acum_id = create_test_folder_structure(service)
        
        if tesis_id and acum_id:
            print(f"\n🎉 ¡Estructura creada exitosamente!")
            print(f"📁 TesisMagister: {tesis_id}")
            print(f"📁 acumulative: {acum_id}")
            print(f"\n✅ Ahora podemos probar la subida de archivos")