"""
Google Drive integration for seamless Colab workflow.
Handles authentication, file upload, and result retrieval.
"""

import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io
import zipfile

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    GDRIVE_AVAILABLE = True
    print("✅ Google Drive API dependencies loaded successfully")
except ImportError as e:
    GDRIVE_AVAILABLE = False
    print(f"⚠️ Google Drive integration not available: {e}")
    # Don't show warning in Streamlit here since this might be imported during module load


class GoogleDriveManager:
    """Manages Google Drive operations for Colab workflow."""
    
    def __init__(self):
        self.folder_name = "TesisMagister"
        self.acumulative_folder = "acumulative"
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        self.service = None
        self.folder_id = None
        self.drive_base_path = "/content/drive/MyDrive/TesisMagister/acumulative"
    
    def authenticate(self) -> bool:
        """Authenticate with Google Drive using OAuth2."""
        if not GDRIVE_AVAILABLE:
            return False
            
        try:
            # Check if we have stored credentials
            if 'gdrive_credentials' in st.session_state:
                credentials = Credentials.from_authorized_user_info(
                    st.session_state.gdrive_credentials, self.scopes
                )
                if credentials.valid:
                    self.service = build('drive', 'v3', credentials=credentials)
                    return True
            
            # Need to authenticate
            st.warning("🔐 Google Drive authentication required for Colab integration.")
            
            # Create OAuth flow
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": st.secrets.get("google_client_id", ""),
                        "client_secret": st.secrets.get("google_client_secret", ""),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost:8080"]
                    }
                },
                scopes=self.scopes
            )
            
            flow.redirect_uri = "http://localhost:8080"
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            st.markdown(f"**[Click here to authenticate with Google Drive]({auth_url})**")
            
            # Manual token input for now (can be improved with proper OAuth callback)
            auth_code = st.text_input("Paste the authorization code here:", type="password")
            
            if auth_code and st.button("Authenticate"):
                try:
                    flow.fetch_token(code=auth_code)
                    credentials = flow.credentials
                    
                    # Store credentials
                    st.session_state.gdrive_credentials = {
                        'token': credentials.token,
                        'refresh_token': credentials.refresh_token,
                        'token_uri': credentials.token_uri,
                        'client_id': credentials.client_id,
                        'client_secret': credentials.client_secret,
                        'scopes': credentials.scopes
                    }
                    
                    self.service = build('drive', 'v3', credentials=credentials)
                    st.success("✅ Google Drive authenticated successfully!")
                    st.experimental_rerun()
                    return True
                    
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
                    return False
            
            return False
            
        except Exception as e:
            st.error(f"Google Drive authentication error: {str(e)}")
            return False
    
    def create_folder_if_not_exists(self) -> Optional[str]:
        """Create project folder in Google Drive if it doesn't exist."""
        if not self.service:
            return None
            
        try:
            # Search for existing folder
            results = self.service.files().list(
                q=f"name='{self.folder_name}' and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name)"
            ).execute()
            
            if results.get('files'):
                self.folder_id = results['files'][0]['id']
                return self.folder_id
            
            # Create folder
            folder_metadata = {
                'name': self.folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            self.folder_id = folder.get('id')
            return self.folder_id
            
        except Exception as e:
            st.error(f"Error creating folder: {str(e)}")
            return None
    
    def upload_comparison_data(self, comparison_data: Dict, filename: str = None) -> Optional[str]:
        """Upload comparison data to Google Drive."""
        if not self.service or not self.folder_id:
            return None
            
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_batch_{timestamp}.json"
            
            # Convert to JSON
            json_data = json.dumps(comparison_data, indent=2, ensure_ascii=False)
            file_stream = io.BytesIO(json_data.encode('utf-8'))
            
            media = MediaIoBaseUpload(
                file_stream,
                mimetype='application/json',
                resumable=True
            )
            
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id]
            }
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink'
            ).execute()
            
            return file.get('id'), file.get('webViewLink')
            
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            return None
    
    def list_files_in_folder(self, file_pattern: str = None) -> List[Dict]:
        """List files in the project folder."""
        if not self.service or not self.folder_id:
            return []
            
        try:
            query = f"'{self.folder_id}' in parents and trashed=false"
            if file_pattern:
                query += f" and name contains '{file_pattern}'"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, createdTime, size, webViewLink)",
                orderBy="createdTime desc"
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            st.error(f"Error listing files: {str(e)}")
            return []
    
    def download_file(self, file_id: str) -> Optional[Dict]:
        """Download file content from Google Drive."""
        if not self.service:
            return None
            
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            content = file_content.read().decode('utf-8')
            
            return json.loads(content)
            
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None
    
    def get_colab_notebook_url(self) -> str:
        """Generate URL to open Colab notebook with proper setup."""
        colab_url = (
            "https://colab.research.google.com/github/yourusername/yourrepo/blob/main/"
            "E5_Comparison_Colab_Processing.ipynb"
        )
        return colab_url


def export_comparison_batch_to_drive(
    questions: List[Dict],
    comparison_config: Dict,
    embedding_models: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Export comparison batch data to Google Drive for Colab processing.
    
    Args:
        questions: List of questions with ground truth data
        comparison_config: Configuration for comparison (top_k, reranking, etc.)
        embedding_models: List of models to compare
        
    Returns:
        (success, file_url)
    """
    
    drive_manager = GoogleDriveManager()
    
    # Authenticate
    if not drive_manager.authenticate():
        return False, None
    
    # Create folder
    if not drive_manager.create_folder_if_not_exists():
        return False, None
    
    # Prepare batch data
    batch_data = {
        "timestamp": datetime.now().isoformat(),
        "config": comparison_config,
        "embedding_models": embedding_models,
        "questions": questions,
        "metadata": {
            "total_questions": len(questions),
            "models_count": len(embedding_models),
            "expected_comparisons": len(questions) * len(embedding_models)
        }
    }
    
    # Upload to Drive
    upload_result = drive_manager.upload_comparison_data(batch_data)
    
    if upload_result:
        file_id, file_url = upload_result
        st.success(f"✅ Comparison batch uploaded to Google Drive!")
        return True, file_url
    
    return False, None


def check_colab_results(drive_manager: GoogleDriveManager = None) -> Optional[Dict]:
    """Check for completed results from Colab processing."""
    
    if not drive_manager:
        drive_manager = GoogleDriveManager()
        if not drive_manager.authenticate():
            return None
        drive_manager.create_folder_if_not_exists()
    
    # Look for result files
    result_files = drive_manager.list_files_in_folder("comparison_results")
    
    if not result_files:
        return None
    
    # Get the most recent result file
    latest_file = result_files[0]  # Already ordered by creation time desc
    
    st.info(f"📥 Found Colab results: {latest_file['name']}")
    
    # Download and parse results
    results = drive_manager.download_file(latest_file['id'])
    
    if results:
        st.success("✅ Colab results loaded successfully!")
        return results
    
    return None


@st.cache_data(ttl=3600)
def get_cached_drive_results(file_id: str) -> Optional[Dict]:
    """Cache Drive results to avoid repeated downloads."""
    drive_manager = GoogleDriveManager()
    if drive_manager.authenticate():
        return drive_manager.download_file(file_id)
    return None


def show_colab_integration_ui():
    """Show the Colab integration UI in Streamlit."""
    
    st.markdown("### 🚀 Google Colab GPU Acceleration")
    
    if not GDRIVE_AVAILABLE:
        st.warning("""
        **Google Drive API disponible pero requiere configuración.**
        
        📋 **Por ahora, usa la exportación manual:**
        1. Exporta el código de evaluación
        2. Ejecuta en Google Colab con GPU
        3. Descarga resultados manualmente
        """)
        return False
    
    # Simplified version - offer manual workflow
    st.success("✅ Google Drive API disponible!")
    
    st.info("""
    **🛠️ Configuración Simplificada**
    
    **Opción 1: Workflow Manual (Recomendado)**
    - Exportar código para Colab
    - Ejecutar con GPU T4 gratis
    - Descargar resultados
    
    **Opción 2: Integración Automática**
    - Requiere configurar Google Drive API
    - Autenticación OAuth2
    - Subida/descarga automática
    """)
    
    # Show workflow options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Workflow Manual")
        st.info("Usa la exportación manual de código (más simple)")
    
    with col2:
        st.markdown("#### 🔄 Integración Automática")
        if st.button("⚙️ Configurar Drive API"):
            st.markdown("""
            **Para configurar integración automática:**
            
            1. **Crear proyecto en Google Cloud Console**
            2. **Habilitar Google Drive API**
            3. **Crear credenciales OAuth2**
            4. **Configurar archivo .env**
            
            Ver documentación completa para detalles.
            """)
    
    return True


def create_colab_instructions(config: dict = None) -> str:
    """Create step-by-step instructions for Colab processing."""
    
    if config is None:
        config = {"evaluation_type": "general"}
    
    instructions = f"""
## 🚀 Instrucciones para Google Colab

### 📋 Pasos para ejecutar en Colab:

1. **Abrir Google Colab**
   - Ve a [colab.research.google.com](https://colab.research.google.com)
   - Crea un nuevo notebook

2. **Activar GPU**
   - Ve a Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (gratuito)
   - Guardar

3. **Copiar y ejecutar código**
   - Usa el código generado automáticamente
   - Pega en una celda de código
   - Ejecuta con Ctrl+F9

4. **Descargar resultados**
   - Los archivos se generarán en `/content/`
   - Descarga manualmente los archivos JSON/CSV

### ⚡ Ventajas de usar Colab:
- 🚀 GPU T4 gratuita (10-50x más rápido)
- 💾 Sin limitaciones de memoria local  
- ☁️ Sin instalación de dependencias
- 🔄 Procesamiento paralelo optimizado

### 📊 Tipo de evaluación: {config.get('evaluation_type', 'cumulative_metrics')}
"""
    
    return instructions


def create_colab_instructions_simple() -> str:
    """Simple version of create_colab_instructions that doesn't require arguments."""
    return create_colab_instructions()
    
    colab_notebook_url = "https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID"  # Update with actual notebook
    
    instructions = f"""
    ## 🚀 Next Steps: Process in Google Colab
    
    **Data uploaded successfully!** Follow these steps:
    
    ### 1. 📖 Open Colab Notebook
    **[👉 Click here to open the processing notebook]({colab_notebook_url})**
    
    ### 2. ⚡ Enable GPU
    - Go to `Runtime > Change runtime type`
    - Select `Hardware accelerator: GPU`
    - Choose `T4` or `V100` if available
    
    ### 3. 🔄 Run Processing
    - **Cell 1**: Verify GPU and install dependencies
    - **Cell 2**: Mount Google Drive and authenticate
    - **Cell 3**: Load your comparison data automatically
    - **Cell 4**: Process all models with GPU acceleration
    - **Cell 5**: Save results back to Drive
    
    ### 4. ⏱️ Expected Time
    - **GPU T4**: ~2-5 minutes for full comparison
    - **GPU V100**: ~1-3 minutes for full comparison
    
    ### 5. ✅ Return Here
    - Once processing is complete, return to this page
    - Click "Check for Results" to load the results
    - View comprehensive comparison dashboard
    
    ---
    
    **📁 Your data file:** [View in Drive]({file_url})
    
    **💡 Tip:** Keep this browser tab open and return here after Colab processing.
    """
    
    return instructions