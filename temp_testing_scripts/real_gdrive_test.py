#!/usr/bin/env python3
"""
Prueba real de subida de archivos a Google Drive
Requiere autenticación OAuth2
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

# Scopes requeridos para Google Drive (modo desarrollo)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """Autentica con Google Drive usando OAuth2"""
    
    creds = None
    token_file = 'token.pickle'
    credentials_file = 'credentials.json'
    
    # Verificar si existe el archivo de credenciales
    if not os.path.exists(credentials_file):
        print("❌ Error: No se encontró 'credentials.json'")
        print("📋 Instrucciones para obtener credentials.json:")
        print("1. Ve a https://console.cloud.google.com/")
        print("2. Crea un proyecto o selecciona uno existente")
        print("3. Habilita la API de Google Drive")
        print("4. Ve a 'Credenciales' → 'Crear credenciales' → 'ID de cliente OAuth 2.0'")
        print("5. Tipo de aplicación: 'Aplicación de escritorio'")
        print("6. Descarga el archivo JSON y renómbralo a 'credentials.json'")
        print("7. Coloca 'credentials.json' en esta carpeta")
        return None
    
    # Cargar token existente si está disponible
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # Si no hay credenciales válidas disponibles, permite al usuario autenticarse
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refrescando token de acceso...")
            creds.refresh(Request())
        else:
            print("🔐 Iniciando proceso de autenticación OAuth2...")
            print("Se abrirá una ventana del navegador para autenticarte")
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Guardar las credenciales para la próxima ejecución
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def find_or_create_folder(service, folder_name, parent_id=None):
    """Busca una carpeta o la crea si no existe"""
    
    # Buscar la carpeta
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query).execute()
    items = results.get('files', [])
    
    if items:
        print(f"📁 Carpeta encontrada: {folder_name}")
        return items[0]['id']
    else:
        # Crear la carpeta
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f"📁 Carpeta creada: {folder_name}")
        return folder.get('id')

def upload_file_to_drive(service, file_path, folder_id=None):
    """Sube un archivo a Google Drive"""
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Archivo no encontrado: {file_path}")
        return None
    
    file_name = os.path.basename(file_path)
    
    # Metadatos del archivo
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    # Subir archivo
    media = MediaFileUpload(file_path, resumable=True)
    
    try:
        print(f"📤 Subiendo archivo: {file_name}")
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id,name,webViewLink'
        ).execute()
        
        file_id = file.get('id')
        file_link = file.get('webViewLink')
        
        print(f"✅ Archivo subido exitosamente!")
        print(f"   📄 Nombre: {file_name}")
        print(f"   🆔 ID: {file_id}")
        print(f"   🔗 Link: {file_link}")
        
        return file_id
        
    except Exception as e:
        print(f"❌ Error al subir archivo: {e}")
        return None

def create_test_file():
    """Crea un archivo de prueba para subir"""
    
    test_content = f"""🚀 Prueba Real de Google Drive
===============================

Timestamp: {datetime.now().isoformat()}
Sistema: Streamlit ↔ Google Drive Integration
Estado: ¡Funcionando! ✅

Este archivo fue subido exitosamente desde Python usando la API real de Google Drive.

Carpeta destino: /TesisMagister/acumulative/
Propósito: Verificar conectividad antes del flujo completo Streamlit-Colab-Drive

Próximos pasos:
1. ✅ Subida básica funcionando
2. ⏳ Integrar con Streamlit
3. ⏳ Configurar notebook de Colab
4. ⏳ Flujo completo de evaluación

¡La integración real con Google Drive está funcionando! 🎉
"""
    
    test_file = "test_real_upload.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"📄 Archivo de prueba creado: {test_file}")
    return test_file

def test_real_google_drive():
    """Función principal de prueba"""
    
    print("🚀 PRUEBA REAL: Subida a Google Drive")
    print("=" * 50)
    
    # 1. Autenticar
    print("\n🔐 Paso 1: Autenticación")
    creds = authenticate_google_drive()
    if not creds:
        return False
    
    try:
        # Construir el servicio de Google Drive
        service = build('drive', 'v3', credentials=creds)
        print("✅ Servicio de Google Drive inicializado")
        
        # 2. Crear estructura de carpetas
        print("\n📁 Paso 2: Crear estructura de carpetas")
        
        # Buscar/crear TesisMagister
        tesis_folder_id = find_or_create_folder(service, "TesisMagister")
        
        # Buscar/crear acumulative dentro de TesisMagister
        acumulative_folder_id = find_or_create_folder(service, "acumulative", tesis_folder_id)
        
        print(f"✅ Estructura de carpetas lista")
        print(f"   📁 TesisMagister ID: {tesis_folder_id}")
        print(f"   📁 acumulative ID: {acumulative_folder_id}")
        
        # 3. Crear archivo de prueba
        print("\n📄 Paso 3: Crear archivo de prueba")
        test_file = create_test_file()
        
        # 4. Subir archivo
        print("\n📤 Paso 4: Subir archivo a Google Drive")
        file_id = upload_file_to_drive(service, test_file, acumulative_folder_id)
        
        if file_id:
            print(f"\n🎉 ¡ÉXITO! Archivo subido a Google Drive")
            print(f"Carpeta: /TesisMagister/acumulative/")
            print(f"Archivo ID: {file_id}")
            
            # Limpiar archivo local
            os.remove(test_file)
            print(f"🧹 Archivo local eliminado: {test_file}")
            
            return True
        else:
            print(f"\n❌ Error: No se pudo subir el archivo")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        return False

def verify_folder_access(service, folder_name="acumulative"):
    """Verifica que podemos acceder a la carpeta de trabajo"""
    
    print(f"\n🔍 Verificando acceso a carpeta '{folder_name}'...")
    
    try:
        # Buscar carpeta acumulative
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query).execute()
        items = results.get('files', [])
        
        if items:
            folder_id = items[0]['id']
            print(f"✅ Carpeta encontrada: {folder_name} (ID: {folder_id})")
            
            # Listar contenido de la carpeta
            query = f"'{folder_id}' in parents"
            results = service.files().list(q=query, fields="files(id,name,mimeType)").execute()
            files = results.get('files', [])
            
            print(f"📂 Contenido de la carpeta ({len(files)} archivos):")
            for file in files:
                print(f"   📄 {file['name']} (ID: {file['id']})")
            
            return folder_id
        else:
            print(f"❌ Carpeta '{folder_name}' no encontrada")
            return None
            
    except Exception as e:
        print(f"❌ Error al verificar carpeta: {e}")
        return None

if __name__ == "__main__":
    success = test_real_google_drive()
    
    if success:
        print("\n" + "="*50)
        print("🎊 PRUEBA COMPLETADA EXITOSAMENTE")
        print("✅ La integración con Google Drive está funcionando")
        print("🚀 Lista para integrar con Streamlit y Colab")
    else:
        print("\n" + "="*50)
        print("❌ PRUEBA FALLÓ")
        print("🔧 Revisar configuración y credenciales")