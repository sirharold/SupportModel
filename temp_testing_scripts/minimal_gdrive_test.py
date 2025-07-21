#!/usr/bin/env python3
"""
Prueba mínima para verificar Google Drive API
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scope más básico
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

def test_api_access():
    """Prueba básica de acceso a la API"""
    
    print("🔍 Verificando acceso a Google Drive API...")
    
    creds = None
    
    # Cargar token existente
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # Autenticar si es necesario
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    try:
        # Construir servicio
        service = build('drive', 'v3', credentials=creds)
        
        # Prueba básica: listar archivos (solo metadatos)
        print("📋 Probando acceso básico...")
        results = service.files().list(pageSize=5).execute()
        items = results.get('files', [])
        
        print(f"✅ API funcionando! Encontrados {len(items)} archivos")
        
        # Buscar carpeta específica
        print("📁 Buscando carpeta 'TesisMagister'...")
        query = "name='TesisMagister' and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query).execute()
        folders = results.get('files', [])
        
        if folders:
            print(f"✅ Carpeta TesisMagister encontrada: {folders[0]['id']}")
            
            # Buscar subcarpeta acumulative
            tesis_id = folders[0]['id']
            query = f"name='acumulative' and '{tesis_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            results = service.files().list(q=query).execute()
            subfolders = results.get('files', [])
            
            if subfolders:
                print(f"✅ Carpeta acumulative encontrada: {subfolders[0]['id']}")
                return True
            else:
                print("❌ Carpeta 'acumulative' no encontrada dentro de TesisMagister")
        else:
            print("❌ Carpeta 'TesisMagister' no encontrada")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_api_access()
    
    if success:
        print("\n🎉 ¡API de Google Drive está funcionando!")
        print("✅ Podemos proceder con la subida de archivos")
    else:
        print("\n❌ Problema con la API")
        print("⏳ Espera unos minutos más si acabas de habilitar la API")