#!/usr/bin/env python3
"""
Integración real con Google Drive para el sistema Streamlit-Colab
"""

import os
import json
import pickle
import streamlit as st
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from io import BytesIO

# Scopes necesarios para Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Configuración de carpetas
GDRIVE_CONFIG_FILE = 'gdrive_config.json'
TOKEN_FILE = 'token.pickle'
CREDENTIALS_FILE = 'credentials.json'

def load_gdrive_config():
    """Carga la configuración de carpetas de Google Drive"""
    try:
        with open(GDRIVE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config.get('acumulative_folder_id')
    except FileNotFoundError:
        st.error(f"❌ Archivo de configuración no encontrado: {GDRIVE_CONFIG_FILE}")
        return None
    except Exception as e:
        st.error(f"❌ Error cargando configuración: {e}")
        return None

def authenticate_gdrive():
    """Autentica con Google Drive"""
    
    if not os.path.exists(CREDENTIALS_FILE):
        st.error(f"❌ Archivo de credenciales no encontrado: {CREDENTIALS_FILE}")
        st.info("""
        📋 Para configurar Google Drive:
        1. Ve a Google Cloud Console
        2. Habilita Google Drive API
        3. Crea credenciales OAuth2
        4. Descarga como 'credentials.json'
        5. Coloca el archivo en la carpeta del proyecto
        """)
        return None
    
    creds = None
    
    # Cargar token existente
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            st.warning(f"⚠️ Error cargando token: {e}")
    
    # Autenticar si es necesario
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                st.success("🔄 Token de acceso refrescado")
            except Exception as e:
                st.error(f"❌ Error refrescando token: {e}")
                creds = None
        
        if not creds:
            st.error("🔐 Requiere nueva autenticación con Google Drive")
            st.info("""
            ⚠️ Se necesita autenticación manual:
            1. Ejecuta el script de autenticación en terminal
            2. Completa el proceso OAuth2
            3. Regresa a Streamlit
            """)
            return None
        
        # Guardar credenciales
        try:
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            st.warning(f"⚠️ No se pudo guardar token: {e}")
    
    try:
        service = build('drive', 'v3', credentials=creds)
        # Prueba básica de conectividad
        service.files().list(pageSize=1).execute()
        return service
    except Exception as e:
        st.error(f"❌ Error conectando con Google Drive: {e}")
        return None

def upload_json_to_drive(service, folder_id, filename, data):
    """Sube un archivo JSON a Google Drive"""
    
    try:
        # Convertir datos a JSON string
        json_content = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Crear stream de bytes
        json_bytes = BytesIO(json_content.encode('utf-8'))
        
        # Metadatos del archivo
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        # Crear media upload
        media = MediaIoBaseUpload(json_bytes, mimetype='application/json', resumable=True)
        
        # Subir archivo
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,webViewLink'
        ).execute()
        
        return {
            'success': True,
            'file_id': file.get('id'),
            'file_name': file.get('name'),
            'web_link': file.get('webViewLink')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def download_json_from_drive(service, file_id):
    """Descarga un archivo JSON desde Google Drive"""
    
    try:
        # Descargar contenido del archivo
        request = service.files().get_media(fileId=file_id)
        content = request.execute()
        
        # Decodificar JSON
        json_content = content.decode('utf-8')
        data = json.loads(json_content)
        
        return {
            'success': True,
            'data': data
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def find_file_in_drive(service, folder_id, filename):
    """Busca un archivo o carpeta específica en una carpeta de Drive"""
    
    try:
        # Buscar tanto archivos como carpetas
        query = f"name='{filename}' and '{folder_id}' in parents"
        results = service.files().list(q=query, fields="files(id,name,modifiedTime,mimeType)").execute()
        files = results.get('files', [])
        
        if files:
            return {
                'success': True,
                'found': True,
                'file_id': files[0]['id'],
                'file_name': files[0]['name'],
                'modified_time': files[0]['modifiedTime'],
                'mime_type': files[0]['mimeType']
            }
        else:
            return {
                'success': True,
                'found': False
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_evaluation_config_in_drive(config_data):
    """Crea archivo de configuración de evaluación en Google Drive"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    # Nombre del archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_config_{timestamp}.json"
    
    # Agregar timestamp a los datos
    config_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_data['created_by'] = 'streamlit'
    config_data['config_file'] = filename
    
    # Subir archivo
    result = upload_json_to_drive(service, folder_id, filename, config_data)
    
    if result['success']:
        # También crear/actualizar archivo de status
        status_data = {
            'status': 'config_created',
            'timestamp': datetime.now().isoformat(),
            'config_file': filename,
            'evaluation_pending': True
        }
        
        status_result = upload_json_to_drive(service, folder_id, 'evaluation_status.json', status_data)
        
        return {
            'success': True,
            'config_file_id': result['file_id'],
            'config_filename': filename,
            'web_link': result['web_link'],
            'status_updated': status_result['success']
        }
    else:
        return result

def check_evaluation_status_in_drive():
    """Verifica el estado de la evaluación en Google Drive"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    # Buscar archivo de status
    find_result = find_file_in_drive(service, folder_id, 'evaluation_status.json')
    
    if not find_result['success']:
        return find_result
    
    if not find_result['found']:
        return {
            'success': True,
            'status': 'no_status_file',
            'message': 'No se encontró archivo de estado'
        }
    
    # Descargar y leer archivo de status
    download_result = download_json_from_drive(service, find_result['file_id'])
    
    if not download_result['success']:
        return download_result
    
    status_data = download_result['data']
    
    return {
        'success': True,
        'status': status_data.get('status', 'unknown'),
        'data': status_data
    }

def get_evaluation_results_from_drive():
    """Obtiene los resultados de evaluación desde Google Drive"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    # Verificar estado primero
    status_result = check_evaluation_status_in_drive()
    if not status_result['success']:
        return status_result
    
    if status_result['status'] != 'completed':
        return {
            'success': False,
            'error': f"Evaluación no completada. Estado: {status_result['status']}"
        }
    
    status_data = status_result['data']
    results_file = status_data.get('results_file')
    
    if not results_file:
        return {'success': False, 'error': 'No se especificó archivo de resultados'}
    
    # Buscar carpeta 'results' dentro de acumulative
    results_folder_result = find_file_in_drive(service, folder_id, 'results')
    
    if not results_folder_result['success'] or not results_folder_result['found']:
        return {'success': False, 'error': 'No se encontró la carpeta results/ en Google Drive'}
    
    results_folder_id = results_folder_result['file_id']
    
    # Buscar archivo de resultados dentro de la carpeta 'results'
    find_result = find_file_in_drive(service, results_folder_id, results_file)
    
    if not find_result['success'] or not find_result['found']:
        return {'success': False, 'error': f'No se encontró archivo de resultados: {results_file} en carpeta results/'}
    
    # Descargar resultados
    download_result = download_json_from_drive(service, find_result['file_id'])
    
    if not download_result['success']:
        return download_result
    
    return {
        'success': True,
        'results': download_result['data'],
        'status_data': status_data
    }

def test_gdrive_connection():
    """Prueba la conexión con Google Drive"""
    
    try:
        service = authenticate_gdrive()
        if not service:
            return False, "No se pudo autenticar"
        
        folder_id = load_gdrive_config()
        if not folder_id:
            return False, "No se pudo cargar configuración de carpetas"
        
        # Probar listado de archivos en la carpeta
        query = f"'{folder_id}' in parents"
        results = service.files().list(q=query, pageSize=5).execute()
        files = results.get('files', [])
        
        return True, f"Conexión exitosa. {len(files)} archivos en la carpeta."
        
    except Exception as e:
        return False, f"Error: {e}"

# Funciones de interfaz para Streamlit
def show_gdrive_status():
    """Muestra el estado de Google Drive en Streamlit"""
    
    st.subheader("🔗 Estado de Google Drive")
    
    success, message = test_gdrive_connection()
    
    if success:
        st.success(f"✅ {message}")
        
        # Mostrar configuración
        config = load_gdrive_config()
        if config:
            st.info(f"📁 Carpeta configurada: `{config}`")
        
        return True
    else:
        st.error(f"❌ {message}")
        
        # Botón para intentar reconectar
        if st.button("🔄 Intentar Reconectar"):
            st.rerun()
        
        return False

def show_gdrive_authentication_instructions():
    """Muestra instrucciones para autenticación manual"""
    
    st.warning("🔐 Autenticación Requerida")
    
    st.info("""
    **Para autenticar Google Drive:**
    
    1. Abre una terminal en la carpeta del proyecto
    2. Ejecuta: `python simple_gdrive_test.py`
    3. Completa el proceso de autenticación en el navegador
    4. Regresa a Streamlit y recarga la página
    """)
    
    if st.button("🔄 He completado la autenticación"):
        st.rerun()