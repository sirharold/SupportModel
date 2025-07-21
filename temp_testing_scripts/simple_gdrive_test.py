#!/usr/bin/env python3
"""
Prueba simple de Google Drive con configuración mínima
"""

import os
import json
import pickle
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scope mínimo para archivos que creamos
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def simple_auth():
    """Autenticación simple"""
    creds = None
    
    # Cargar token si existe
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # Si no hay credenciales válidas, autenticar
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Configuración manual del flow para evitar problemas de verificación
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', 
                SCOPES,
                redirect_uri='http://localhost:8080'  # Puerto específico
            )
            
            print("🔐 Proceso de autenticación iniciado...")
            print("⚠️  Si aparece 'Access blocked', click en 'Advanced' → 'Go to SupportModel (unsafe)'")
            
            # Usar run_local_server con configuración específica
            creds = flow.run_local_server(
                port=8080,
                access_type='offline',
                include_granted_scopes='true'
            )
        
        # Guardar credenciales
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def create_simple_test_file():
    """Crear archivo de prueba simple"""
    content = f"""Prueba Google Drive - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
Este archivo fue subido desde Python usando la API de Google Drive.
Carpeta destino: TesisMagister/acumulative/

¡Funciona! ✅
"""
    
    filename = "drive_test.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filename

def upload_to_existing_folder():
    """Subir archivo a carpeta existente TesisMagister/acumulative"""
    
    print("🚀 Prueba Simple: Google Drive Upload")
    print("=" * 40)
    
    try:
        # 1. Autenticar
        print("🔐 Autenticando...")
        creds = simple_auth()
        service = build('drive', 'v3', credentials=creds)
        print("✅ Autenticación exitosa")
        
        # 2. Buscar carpeta acumulative existente
        print("📁 Buscando carpeta 'acumulative'...")
        
        query = "name='acumulative' and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query).execute()
        folders = results.get('files', [])
        
        if not folders:
            print("❌ No se encontró la carpeta 'acumulative'")
            print("💡 Tip: Verifica que existe /TesisMagister/acumulative/ en tu Drive")
            return False
        
        folder_id = folders[0]['id']
        print(f"✅ Carpeta encontrada: {folders[0]['name']} (ID: {folder_id})")
        
        # 3. Crear archivo de prueba
        print("📄 Creando archivo de prueba...")
        test_file = create_simple_test_file()
        
        # 4. Subir archivo
        print("📤 Subiendo archivo...")
        
        file_metadata = {
            'name': 'drive_test.txt',
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(test_file)
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,webViewLink'
        ).execute()
        
        print("🎉 ¡ÉXITO!")
        print(f"   📄 Archivo: {file.get('name')}")
        print(f"   🆔 ID: {file.get('id')}")
        print(f"   🔗 Link: {file.get('webViewLink')}")
        
        # Limpiar archivo local
        os.remove(test_file)
        print(f"🧹 Archivo local eliminado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = upload_to_existing_folder()
    
    if success:
        print("\n✅ La integración con Google Drive funciona!")
        print("🚀 Lista para integrar con Streamlit")
    else:
        print("\n❌ La prueba falló")
        print("🔧 Revisar configuración OAuth2")