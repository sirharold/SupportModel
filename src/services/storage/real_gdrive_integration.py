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
# drive.file: Solo archivos creados por la app
# drive.readonly: Leer todos los archivos del usuario (necesario para archivos de Colab)
# drive: Acceso completo (usar como último recurso)
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',  # Leer archivos creados por otros (Colab)
    'https://www.googleapis.com/auth/drive.file'       # Crear/editar archivos de la app
]

# Configuración de carpetas
# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GDRIVE_CONFIG_FILE = os.path.join(PROJECT_ROOT, 'src', 'config', 'gdrive_config.json')
TOKEN_FILE = os.path.join(PROJECT_ROOT, 'token.pickle')
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, 'src', 'config', 'credentials.json')

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

def upload_file_to_drive(service, folder_id, filename, file_path, mime_type='application/octet-stream'):
    """Sube un archivo específico a Google Drive"""
    
    try:
        from googleapiclient.http import MediaFileUpload
        
        # Metadatos del archivo
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        # Media upload
        media = MediaFileUpload(file_path, mimetype=mime_type)
        
        # Crear archivo
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
    
    # Subir archivo de configuración
    result = upload_json_to_drive(service, folder_id, filename, config_data)
    
    # Intentar también subir el archivo de credenciales para Colab
    credentials_uploaded = False
    try:
        if os.path.exists(CREDENTIALS_FILE):
            cred_result = upload_file_to_drive(
                service, folder_id, 'credentials.json', CREDENTIALS_FILE, 'application/json'
            )
            credentials_uploaded = cred_result['success']
            if credentials_uploaded:
                print("✅ Archivo credentials.json subido para uso en Colab")
        else:
            print("⚠️ Archivo credentials.json no encontrado, Colab requerirá autenticación manual")
    except Exception as e:
        print(f"⚠️ No se pudo subir credentials.json: {e}")
    
    if result['success']:
        # También crear/actualizar archivo de status
        status_data = {
            'status': 'config_created',
            'timestamp': datetime.now().isoformat(),
            'config_file': filename,
            'evaluation_pending': True,
            'credentials_available': credentials_uploaded
        }
        
        status_result = upload_json_to_drive(service, folder_id, 'evaluation_status.json', status_data)
        
        return {
            'success': True,
            'config_file_id': result['file_id'],
            'config_filename': filename,
            'web_link': result['web_link'],
            'status_updated': status_result['success'],
            'credentials_uploaded': credentials_uploaded
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
        # Si no hay archivo específico en el status, buscar cualquier archivo de resultados
        print("⚠️ No se especificó archivo de resultados en status, buscando automáticamente...")
        results_file = None  # Lo buscaremos más adelante
    else:
        print(f"🔍 Buscando archivo específico: {results_file}")
    
    # Buscar directamente en la carpeta acumulative (sin subcarpeta results)
    try:
        find_result = None
        found_results_files = []
        
        if results_file:
            # Buscar archivo específico directamente en acumulative
            print(f"🔍 Buscando {results_file} directamente en carpeta acumulative: {folder_id}")
            temp_result = find_file_in_drive(service, folder_id, results_file)
            if temp_result['success'] and temp_result['found']:
                find_result = temp_result
                print(f"✅ Archivo específico encontrado en acumulative: {folder_id}")
        
        # Buscar cualquier archivo de resultados en acumulative
        query = f"'{folder_id}' in parents and name contains 'cumulative_results_' and name contains '.json'"
        results = service.files().list(q=query, fields="files(id,name,modifiedTime)").execute()
        files = results.get('files', [])
        
        for file in files:
            found_results_files.append({
                'file_id': file['id'],
                'file_name': file['name'],
                'modified_time': file['modifiedTime'],
                'folder_id': folder_id
            })
        
        # Si no se encontró el archivo específico, usar el más reciente disponible
        if not find_result or not find_result['found']:
            if found_results_files:
                # Ordenar por fecha de modificación y usar el más reciente
                found_results_files.sort(key=lambda x: x['modified_time'], reverse=True)
                latest_file = found_results_files[0]
                
                print(f"⚠️ Archivo específico no encontrado, usando el más reciente: {latest_file['file_name']}")
                find_result = {
                    'success': True,
                    'found': True,
                    'file_id': latest_file['file_id'],
                    'file_name': latest_file['file_name']
                }
            else:
                # Listar archivos en acumulative para debug
                print("🔍 DEBUG: Buscando archivos de resultados en acumulative:")
                try:
                    query = f"'{folder_id}' in parents"
                    files_in_folder = service.files().list(q=query, fields="files(name,modifiedTime)").execute()
                    files = files_in_folder.get('files', [])
                    results_files_found = [f for f in files if 'cumulative_results_' in f['name'] and f['name'].endswith('.json')]
                    
                    print(f"   Total archivos en acumulative: {len(files)}")
                    print(f"   Archivos de resultados encontrados: {len(results_files_found)}")
                    
                    for file in results_files_found:
                        print(f"     - {file['name']} ({file['modifiedTime']})")
                        
                except Exception as e:
                    print(f"   Error listando acumulative: {e}")
                
                error_msg = f'No se encontró archivo de resultados'
                if results_file:
                    error_msg += f': {results_file}'
                error_msg += ' en carpeta acumulative/'
                return {'success': False, 'error': error_msg}
        
    except Exception as e:
        return {'success': False, 'error': f'Error buscando en acumulative: {str(e)}'}
    
    # Descargar resultados
    download_result = download_json_from_drive(service, find_result['file_id'])
    
    if not download_result['success']:
        return download_result
    
    return {
        'success': True,
        'results': download_result['data'],
        'status_data': status_data
    }

def cleanup_duplicate_results_folders():
    """Limpia carpetas 'results' duplicadas, conservando solo la más reciente"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    try:
        # Buscar todas las carpetas 'results'
        query = f"name='results' and '{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query, fields="files(id,name,modifiedTime)").execute()
        results_folders = results.get('files', [])
        
        if len(results_folders) <= 1:
            return {'success': True, 'message': f'Solo hay {len(results_folders)} carpeta results, no se necesita limpieza'}
        
        # Ordenar por fecha de modificación (más reciente primero)
        results_folders.sort(key=lambda x: x['modifiedTime'], reverse=True)
        
        # Conservar la primera (más reciente) y eliminar las demás
        folders_to_delete = results_folders[1:]
        
        print(f"🧹 Encontradas {len(results_folders)} carpetas 'results'")
        print(f"📁 Conservando la más reciente: {results_folders[0]['modifiedTime']}")
        print(f"🗑️ Eliminando {len(folders_to_delete)} carpetas antiguas")
        
        deleted_count = 0
        for folder in folders_to_delete:
            try:
                service.files().delete(fileId=folder['id']).execute()
                print(f"   ✅ Eliminada carpeta: {folder['id']} ({folder['modifiedTime']})")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Error eliminando carpeta {folder['id']}: {e}")
        
        return {
            'success': True,
            'message': f'Limpieza completada. Eliminadas {deleted_count} de {len(folders_to_delete)} carpetas duplicadas',
            'folders_deleted': deleted_count,
            'folders_kept': 1
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Error durante limpieza: {str(e)}'}

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
        
        # Columnas para botones adicionales
        col1, col2 = st.columns(2)
        
        with col1:
            # Botón para limpiar carpetas duplicadas
            if st.button("🧹 Limpiar Carpetas Duplicadas"):
                with st.spinner("🧹 Limpiando carpetas duplicadas..."):
                    cleanup_result = cleanup_duplicate_results_folders()
                    
                    if cleanup_result['success']:
                        if 'folders_deleted' in cleanup_result and cleanup_result['folders_deleted'] > 0:
                            st.success(f"✅ {cleanup_result['message']}")
                            st.rerun()
                        else:
                            st.info(f"ℹ️ {cleanup_result['message']}")
                    else:
                        st.error(f"❌ Error durante limpieza: {cleanup_result['error']}")
        
        with col2:
            # Placeholder para futuros botones de mantenimiento
            st.empty()
        
        return True
    else:
        st.error(f"❌ {message}")
        
        # Botón para intentar reconectar
        if st.button("🔄 Intentar Reconectar"):
            st.rerun()
        
        return False

def debug_gdrive_contents():
    """Función de debug para mostrar el contenido completo de Google Drive"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    try:
        debug_info = {
            'acumulative_folder_id': folder_id,
            'files_in_acumulative': [],
            'results_folders': [],
            'files_in_results_folders': {}
        }
        
        # 1. Listar todos los archivos en la carpeta acumulative
        query = f"'{folder_id}' in parents"
        results = service.files().list(q=query, fields="files(id,name,mimeType,modifiedTime,size)").execute()
        files = results.get('files', [])
        
        for file in files:
            debug_info['files_in_acumulative'].append({
                'name': file['name'],
                'id': file['id'],
                'type': file['mimeType'],
                'modified': file['modifiedTime'],
                'size': file.get('size', 'N/A')
            })
            
            # Si es una carpeta results, guardar su ID
            if file['name'] == 'results' and 'folder' in file['mimeType']:
                debug_info['results_folders'].append(file)
        
        # 2. Para cada carpeta results, listar su contenido
        for results_folder in debug_info['results_folders']:
            results_folder_id = results_folder['id']
            query = f"'{results_folder_id}' in parents"
            results = service.files().list(q=query, fields="files(id,name,mimeType,modifiedTime,size)").execute()
            files_in_results = results.get('files', [])
            
            debug_info['files_in_results_folders'][results_folder_id] = {
                'folder_info': results_folder,
                'files': []
            }
            
            for file in files_in_results:
                debug_info['files_in_results_folders'][results_folder_id]['files'].append({
                    'name': file['name'],
                    'id': file['id'],
                    'type': file['mimeType'],
                    'modified': file['modifiedTime'],
                    'size': file.get('size', 'N/A')
                })
        
        return {'success': True, 'debug_info': debug_info}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def show_gdrive_debug_info():
    """Muestra información de debug de Google Drive en Streamlit"""
    
    st.subheader("🔍 Debug: Contenido de Google Drive")
    
    with st.spinner("🔍 Analizando contenido de Google Drive..."):
        debug_result = debug_gdrive_contents()
    
    if not debug_result['success']:
        st.error(f"❌ Error: {debug_result['error']}")
        return
    
    debug_info = debug_result['debug_info']
    
    # Mostrar información general
    st.success(f"✅ Conexión exitosa. Carpeta ID: `{debug_info['acumulative_folder_id']}`")
    
    # Mostrar archivos en carpeta acumulative
    st.subheader("📁 Archivos en carpeta 'acumulative':")
    if debug_info['files_in_acumulative']:
        for file in debug_info['files_in_acumulative']:
            icon = "📁" if "folder" in file['type'] else "📄"
            st.write(f"{icon} **{file['name']}** ({file['type'][:20]}...) - {file['modified']}")
    else:
        st.warning("⚠️ No hay archivos en la carpeta acumulative")
    
    # Mostrar carpetas results
    st.subheader("📂 Carpetas 'results' encontradas:")
    if debug_info['results_folders']:
        for i, folder in enumerate(debug_info['results_folders']):
            st.write(f"**Carpeta {i+1}:** {folder['name']} (ID: `{folder['id']}`) - {folder['modifiedTime']}")
            
            # Mostrar contenido de cada carpeta results
            folder_id = folder['id']
            if folder_id in debug_info['files_in_results_folders']:
                files_info = debug_info['files_in_results_folders'][folder_id]
                files = files_info['files']
                
                if files:
                    st.write(f"   📄 **{len(files)} archivos:**")
                    for file in files:
                        if file['name'].startswith('cumulative_results_'):
                            st.write(f"   ✅ **{file['name']}** - {file['modified']} ({file.get('size', 'N/A')} bytes)")
                        else:
                            st.write(f"   📄 {file['name']} - {file['modified']}")
                else:
                    st.write("   ⚠️ **Carpeta vacía**")
    else:
        st.error("❌ **No se encontraron carpetas 'results'**")
        st.info("💡 Esto significa que el Colab no ha ejecutado correctamente o no ha guardado los resultados.")
    
    # Buscar archivos de resultados específicos
    st.subheader("🔍 Búsqueda de archivos de resultados:")
    all_result_files = []
    for folder_id, folder_info in debug_info['files_in_results_folders'].items():
        for file in folder_info['files']:
            if file['name'].startswith('cumulative_results_') and file['name'].endswith('.json'):
                all_result_files.append({
                    'file': file,
                    'folder_id': folder_id,
                    'folder_modified': folder_info['folder_info']['modifiedTime']
                })
    
    if all_result_files:
        st.success(f"✅ Encontrados {len(all_result_files)} archivos de resultados:")
        for item in sorted(all_result_files, key=lambda x: x['file']['modified'], reverse=True):
            st.write(f"📊 **{item['file']['name']}** - {item['file']['modified']}")
            st.write(f"   📁 En carpeta: {item['folder_id']} (modificada: {item['folder_modified']})")
    else:
        st.error("❌ **No se encontraron archivos de resultados**")
        st.info("""
        💡 **Posibles causas:**
        1. El notebook de Colab no se ha ejecutado
        2. El notebook falló durante la ejecución
        3. Los resultados se guardaron en una ubicación diferente
        4. Problemas de sincronización con Google Drive
        
        **Soluciones sugeridas:**
        1. Ejecutar el notebook en Colab nuevamente
        2. Verificar que el notebook termine sin errores
        3. Esperar unos minutos para sincronización
        4. Usar el botón 'Limpiar Carpetas Duplicadas'
        """)

def get_all_results_files_from_drive():
    """Obtiene todos los archivos de resultados disponibles en Google Drive"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    try:
        # Primero intentar búsqueda específica para archivos de resultados
        # Patrón: cumulative_results_[timestamp].json donde timestamp es número Unix
        query = f"'{folder_id}' in parents and name contains 'cumulative_results_'"
        results = service.files().list(q=query, fields="files(id,name,modifiedTime,size)").execute()
        files = results.get('files', [])
        
        # Filtrar solo archivos JSON y validar patrón
        if files:
            import re
            # Patrón: cumulative_results_ seguido de números y .json
            pattern = re.compile(r'^cumulative_results_\d+\.json$')
            
            valid_files = []
            for file in files:
                if file['name'].lower().endswith('.json') and pattern.match(file['name']):
                    valid_files.append(file)
                    print(f"✅ Archivo válido encontrado: {file['name']}")
                elif file['name'].lower().endswith('.json'):
                    print(f"🔍 Archivo cumulative_results pero formato diferente: {file['name']}")
            
            if valid_files:
                files = valid_files
                print(f"✅ Total archivos cumulative_results válidos: {len(files)}")
            else:
                print(f"⚠️ Se encontraron {len(files)} archivos con 'cumulative_results_' pero ninguno con patrón válido")
                # Incluir todos los archivos que contengan cumulative_results para ser más flexible
                files = [f for f in files if f['name'].lower().endswith('.json')]
        
        # Si no encontramos archivos específicos, hacer una búsqueda más amplia para debug
        if not files:
            print(f"🔍 DEBUG: No se encontraron archivos con 'cumulative_results_', probando búsquedas alternativas...")
            
            # Intentar diferentes variaciones de búsqueda
            alternative_queries = [
                f"'{folder_id}' in parents and name contains 'cumulative' and mimeType='application/json'",
                f"'{folder_id}' in parents and name contains 'results' and mimeType='application/json'",
                f"'{folder_id}' in parents and mimeType='application/json'"
            ]
            
            found_files = []
            for i, alt_query in enumerate(alternative_queries):
                try:
                    alt_results = service.files().list(q=alt_query, fields="files(id,name,modifiedTime,size)").execute()
                    alt_files = alt_results.get('files', [])
                    print(f"🔍 DEBUG: Búsqueda {i+1} ({alt_query}): {len(alt_files)} archivos")
                    
                    for file in alt_files:
                        if file not in found_files:
                            found_files.append(file)
                            print(f"   + {file['name']}")
                except Exception as e:
                    print(f"❌ Error en búsqueda alternativa {i+1}: {e}")
            
            # Buscar específicamente archivos con patrones de resultados
            import re
            result_patterns = [
                re.compile(r'^cumulative_results_\d+\.json$'),  # cumulative_results_1234567890.json
                re.compile(r'^results_.*\.json$'),               # results_*.json
                re.compile(r'.*cumulative.*\.json$'),            # *cumulative*.json
                re.compile(r'.*evaluation.*\.json$')             # *evaluation*.json
            ]
            
            keyword_files = []
            for file in found_files:
                filename = file['name']
                for pattern in result_patterns:
                    if pattern.match(filename):
                        keyword_files.append(file)
                        print(f"✅ Archivo candidato encontrado: {filename} (patrón: {pattern.pattern})")
                        break
            
            if keyword_files:
                print(f"🔍 DEBUG: Usando {len(keyword_files)} archivos candidatos")
                # Priorizar archivos cumulative_results_ con timestamp
                cumulative_files = [f for f in keyword_files if f['name'].startswith('cumulative_results_')]
                if cumulative_files:
                    files = cumulative_files
                    print(f"🎯 Priorizando {len(cumulative_files)} archivos cumulative_results_")
                else:
                    files = keyword_files
            else:
                # Listar todos los archivos en la carpeta para debug completo
                all_files_query = f"'{folder_id}' in parents"
                all_results = service.files().list(q=all_files_query, fields="files(id,name,modifiedTime,mimeType)").execute()
                all_files = all_results.get('files', [])
                
                print(f"🔍 DEBUG: Contenido completo de la carpeta ({len(all_files)} archivos):")
                for file in all_files:
                    print(f"   - {file['name']} ({file.get('mimeType', 'unknown type')})")
                
                return {
                    'success': False, 
                    'error': f'No se encontraron archivos de resultados en Google Drive. Se encontraron {len(all_files)} archivos en total.',
                    'debug_info': {
                        'total_files': len(all_files),
                        'json_files': len([f for f in all_files if f['name'].lower().endswith('.json')]),
                        'folder_id': folder_id,
                        'search_query': query,
                        'all_files': [f['name'] for f in all_files if f['name'].lower().endswith('.json')][:10]  # Show first 10 JSON files
                    }
                }
        
        # Procesar archivos encontrados
        results_files = []
        for file in files:
            try:
                # Extraer información del archivo
                file_name = file.get('name', 'Unknown File')
                modified_time = file.get('modifiedTime', 'N/A')
                file_info = {
                    'file_id': file.get('id', 'N/A'),
                    'file_name': file_name,
                    'modified_time': modified_time,
                    'size': file.get('size', 'N/A'),
                    'display_name': f"{file_name} ({modified_time[:19].replace('T', ' ') if modified_time != 'N/A' else 'N/A'})"
                }
                results_files.append(file_info)
                print(f"✅ Archivo de resultados encontrado: {file['name']}")
            except Exception as e:
                print(f"❌ Error procesando archivo {file.get('name', 'unknown')}: {e}")
        
        # Ordenar por fecha de modificación (más reciente primero)
        results_files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return {
            'success': True,
            'files': results_files,
            'count': len(results_files)
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Error buscando archivos de resultados: {str(e)}'}

def get_specific_results_file_from_drive(file_id):
    """Obtiene un archivo de resultados específico por su ID"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    try:
        # Descargar archivo específico
        download_result = download_json_from_drive(service, file_id)
        
        if not download_result['success']:
            return download_result
        
        return {
            'success': True,
            'results': download_result['data']
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Error descargando archivo específico: {str(e)}'}

def test_folder_access_and_list_contents():
    """Función de prueba para verificar acceso a la carpeta y listar contenido"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    try:
        # Obtener información de la carpeta misma
        folder_info = service.files().get(fileId=folder_id, fields="id,name,parents").execute()
        
        # Listar todo el contenido de la carpeta con más detalles
        query = f"'{folder_id}' in parents"
        results = service.files().list(
            q=query, 
            fields="files(id,name,mimeType,modifiedTime,size,owners,permissions)",
            pageSize=100
        ).execute()
        files = results.get('files', [])
        
        # Separar archivos por tipo para análisis
        cumulative_files = [f for f in files if 'cumulative_results_' in f['name'] and f['name'].endswith('.json')]
        config_files = [f for f in files if 'evaluation_config_' in f['name']]
        status_files = [f for f in files if f['name'] == 'evaluation_status.json']
        
        return {
            'success': True,
            'folder_info': folder_info,
            'files': files,
            'file_count': len(files),
            'cumulative_files': cumulative_files,
            'config_files': config_files,
            'status_files': status_files,
            'cumulative_count': len(cumulative_files)
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Error accediendo a la carpeta: {str(e)}'}


def test_specific_cumulative_file_access():
    """Prueba acceso específico a archivos cumulative_results"""
    
    # Autenticar
    service = authenticate_gdrive()
    if not service:
        return {'success': False, 'error': 'No se pudo autenticar con Google Drive'}
    
    # Obtener ID de carpeta
    folder_id = load_gdrive_config()
    if not folder_id:
        return {'success': False, 'error': 'No se pudo cargar configuración de carpetas'}
    
    try:
        # Buscar específicamente archivos cumulative_results
        queries = [
            f"'{folder_id}' in parents and name contains 'cumulative_results'",
            f"'{folder_id}' in parents and name = 'cumulative_results_1753056579.json'",  # Archivo específico que sabemos existe
            f"'{folder_id}' in parents"  # Todos los archivos
        ]
        
        results = {}
        for i, query in enumerate(queries):
            try:
                search_results = service.files().list(
                    q=query, 
                    fields="files(id,name,mimeType,modifiedTime,size,owners)",
                    pageSize=50
                ).execute()
                files_found = search_results.get('files', [])
                
                results[f'query_{i+1}'] = {
                    'query': query,
                    'files_found': len(files_found),
                    'files': [{'name': f['name'], 'id': f['id']} for f in files_found]
                }
                
                print(f"🔍 Query {i+1}: {query}")
                print(f"   Archivos encontrados: {len(files_found)}")
                for f in files_found:
                    print(f"   - {f['name']}")
                
            except Exception as e:
                results[f'query_{i+1}'] = {'query': query, 'error': str(e)}
                print(f"❌ Query {i+1} failed: {e}")
        
        return {'success': True, 'results': results}
        
    except Exception as e:
        return {'success': False, 'error': f'Error en test específico: {str(e)}'}


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
    
    # Botones de prueba
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Probar Acceso a Carpeta"):
            with st.spinner("🔍 Probando acceso a carpeta..."):
                result = test_folder_access_and_list_contents()
                
                if result['success']:
                    st.success("✅ Acceso a carpeta exitoso!")
                    st.write(f"📁 **Carpeta:** {result['folder_info']['name']}")
                    st.write(f"📄 **Total archivos:** {result['file_count']}")
                    st.write(f"📊 **Archivos cumulative_results:** {result['cumulative_count']}")
                    
                    if result['cumulative_files']:
                        with st.expander("📊 Archivos cumulative_results encontrados"):
                            for file in result['cumulative_files']:
                                st.write(f"- **{file['name']}** (ID: {file['id']})")
                    
                    if result['files']:
                        with st.expander("📋 Todos los archivos en la carpeta"):
                            for file in result['files']:
                                st.write(f"- **{file['name']}** ({file.get('mimeType', 'unknown')})")
                    else:
                        st.warning("📭 La carpeta está vacía")
                else:
                    st.error(f"❌ {result['error']}")
    
    with col2:
        if st.button("🎯 Test Específico Cumulative"):
            with st.spinner("🎯 Probando acceso específico a archivos cumulative..."):
                result = test_specific_cumulative_file_access()
                
                if result['success']:
                    st.success("✅ Test específico completado!")
                    
                    with st.expander("🔍 Resultados detallados del test"):
                        for key, query_result in result['results'].items():
                            if 'error' in query_result:
                                st.error(f"**{key}**: {query_result['error']}")
                            else:
                                st.write(f"**{key}**: {query_result['files_found']} archivos")
                                if query_result['files']:
                                    for file_info in query_result['files']:
                                        st.write(f"  - {file_info['name']}")
                else:
                    st.error(f"❌ {result['error']}")
    
    if st.button("🔄 He completado la autenticación"):
        st.rerun()