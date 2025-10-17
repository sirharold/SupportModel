#!/usr/bin/env python3
"""
Script de autenticación manual para Google Drive
Ejecutar cuando se requiera nueva autenticación con scopes expandidos
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes expandidos para acceder a archivos creados por otros procesos (Colab)
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',  # Leer archivos creados por otros (Colab)
    'https://www.googleapis.com/auth/drive.file'       # Crear/editar archivos de la app
]

# Rutas de archivos
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(PROJECT_ROOT, 'token.pickle')
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, 'src', 'config', 'credentials.json')

def main():
    """Función principal de autenticación"""
    
    print("🔐 SCRIPT DE AUTENTICACIÓN GOOGLE DRIVE")
    print("=" * 50)
    print("📊 Scopes solicitados:")
    for scope in SCOPES:
        print(f"   - {scope}")
    print("=" * 50)
    
    # Verificar que existe el archivo de credenciales
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"❌ ERROR: Archivo de credenciales no encontrado")
        print(f"📁 Esperado en: {CREDENTIALS_FILE}")
        print("\n💡 PASOS PARA OBTENER CREDENCIALES:")
        print("1. Ve a: https://console.cloud.google.com/")
        print("2. Selecciona tu proyecto o crea uno nuevo")
        print("3. Ve a APIs & Services > Credentials")
        print("4. Click 'Create Credentials' > 'OAuth 2.0 Client IDs'")
        print("5. Selecciona 'Desktop application'")
        print("6. Descarga el archivo JSON como 'credentials.json'")
        print("7. Colócalo en: src/config/credentials.json")
        return False
    
    print(f"✅ Archivo de credenciales encontrado: {CREDENTIALS_FILE}")
    
    creds = None
    
    # Verificar si existe token previo (pero puede ser con scopes limitados)
    if os.path.exists(TOKEN_FILE):
        print(f"🔍 Token existente encontrado: {TOKEN_FILE}")
        try:
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
                print("✅ Token cargado exitosamente")
        except Exception as e:
            print(f"⚠️ Error cargando token existente: {e}")
            creds = None
    
    # Verificar si las credenciales son válidas y tienen los scopes correctos
    need_reauth = False
    if creds:
        # Verificar si el token es válido
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                print("🔄 Intentando refrescar token...")
                try:
                    creds.refresh(Request())
                    print("✅ Token refrescado exitosamente")
                except Exception as e:
                    print(f"❌ Error refrescando token: {e}")
                    need_reauth = True
            else:
                print("⚠️ Token inválido, requiere nueva autenticación")
                need_reauth = True
        
        # Verificar scopes (importante para el fix de permisos)
        if creds and hasattr(creds, 'scopes'):
            current_scopes = set(creds.scopes) if creds.scopes else set()
            required_scopes = set(SCOPES)
            
            if not required_scopes.issubset(current_scopes):
                print("⚠️ Token existente no tiene los scopes requeridos")
                print(f"   Scopes actuales: {list(current_scopes)}")
                print(f"   Scopes requeridos: {list(required_scopes)}")
                need_reauth = True
            else:
                print("✅ Scopes correctos verificados")
    else:
        need_reauth = True
    
    # Realizar nueva autenticación si es necesario
    if need_reauth or not creds:
        print("\n🔐 INICIANDO PROCESO DE AUTENTICACIÓN")
        print("-" * 40)
        
        if os.path.exists(TOKEN_FILE):
            print(f"🗑️ Eliminando token anterior: {TOKEN_FILE}")
            os.remove(TOKEN_FILE)
        
        try:
            print("📱 Creando flujo de autenticación OAuth2...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            
            print("🌐 Iniciando servidor local para autenticación...")
            print("\n" + "="*60)
            print("🔗 Se abrirá tu navegador para autorizar la aplicación")
            print("📋 IMPORTANTE: Debes autorizar AMBOS scopes:")
            print("   - Leer archivos de Google Drive")
            print("   - Crear/modificar archivos en Google Drive")
            print("🔒 Esto permitirá leer archivos creados por Colab")
            print("="*60 + "\n")
            
            # Ejecutar el flujo de autenticación
            creds = flow.run_local_server(port=0)
            
            print("✅ ¡Autenticación completada exitosamente!")
            
        except Exception as e:
            print(f"❌ Error durante autenticación: {e}")
            print("\n💡 POSIBLES SOLUCIONES:")
            print("1. Verifica que tienes conexión a internet")
            print("2. Asegúrate de que el archivo credentials.json es válido")
            print("3. Verifica que has habilitado Google Drive API en tu proyecto")
            print("4. Intenta desde un navegador diferente")
            return False
    
    # Guardar credenciales
    if creds:
        try:
            print(f"💾 Guardando credenciales en: {TOKEN_FILE}")
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
            print("✅ Credenciales guardadas exitosamente")
            
            # Verificar el token guardado
            file_size = os.path.getsize(TOKEN_FILE)
            print(f"📏 Tamaño del token: {file_size} bytes")
            
        except Exception as e:
            print(f"❌ Error guardando credenciales: {e}")
            return False
    
    # Verificación final
    print("\n🧪 VERIFICACIÓN FINAL")
    print("-" * 30)
    
    try:
        from googleapiclient.discovery import build
        
        print("🔧 Creando servicio de Google Drive...")
        service = build('drive', 'v3', credentials=creds)
        
        print("📊 Probando acceso básico...")
        results = service.files().list(pageSize=1).execute()
        
        print("✅ ¡Acceso a Google Drive verificado!")
        print(f"🔑 Usuario autenticado: {creds.token}")
        
        # Verificar acceso a la carpeta específica
        try:
            import json
            config_file = os.path.join(PROJECT_ROOT, 'src', 'config', 'gdrive_config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                folder_id = config.get('acumulative_folder_id')
                
                if folder_id:
                    print(f"🔍 Probando acceso a carpeta acumulative: {folder_id}")
                    folder_contents = service.files().list(
                        q=f"'{folder_id}' in parents",
                        pageSize=5
                    ).execute()
                    
                    files = folder_contents.get('files', [])
                    print(f"📁 Archivos encontrados en acumulative: {len(files)}")
                    
                    # Buscar específicamente archivos cumulative_results
                    cumulative_files = [f for f in files if 'cumulative_results_' in f['name']]
                    print(f"📊 Archivos cumulative_results: {len(cumulative_files)}")
                    
                    for file in cumulative_files[:3]:  # Mostrar primeros 3
                        print(f"   - {file['name']}")
                else:
                    print("⚠️ No se encontró folder_id en configuración")
            
        except Exception as e:
            print(f"⚠️ Error verificando carpeta específica: {e}")
    
    except Exception as e:
        print(f"❌ Error en verificación final: {e}")
        return False
    
    print("\n🎉 AUTENTICACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 50)
    print("✅ Token guardado con scopes expandidos")
    print("✅ Acceso a Google Drive verificado")
    print("✅ Permisos para leer archivos de Colab: SÍ")
    print("\n👉 Ahora puedes ejecutar Streamlit:")
    print("   streamlit run src/apps/main_qa_app.py")
    print("\n📊 El dropdown debería mostrar tus archivos cumulative_results")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Autenticación falló")
        exit(1)
    else:
        print("\n✅ Autenticación exitosa")
        exit(0)