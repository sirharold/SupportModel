#!/usr/bin/env python3
"""
Script para diagnosticar la conexión a ChromaDB y encontrar dónde están los datos
"""

import os
import chromadb
from chromadb.config import Settings

def test_server_connection():
    """Prueba conectar al servidor ChromaDB"""
    print("=== Testing ChromaDB Server Connection ===")
    
    server_configs = [
        ("localhost", 8000),
        ("127.0.0.1", 8000),
        ("0.0.0.0", 8000),
        ("localhost", 8001),
    ]
    
    for host, port in server_configs:
        try:
            print(f"Trying server at {host}:{port}...")
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            collections = client.list_collections()
            print(f"✅ Connected to server at {host}:{port}")
            print(f"   Collections: {[c.name for c in collections]}")
            
            # Verificar si hay datos
            for collection in collections:
                count = collection.count()
                print(f"   {collection.name}: {count} items")
            
            return client
            
        except Exception as e:
            print(f"❌ Failed to connect to {host}:{port}: {e}")
    
    return None

def test_persistent_clients():
    """Prueba diferentes directorios persistentes"""
    print("\n=== Testing Persistent Client Directories ===")
    
    possible_dirs = [
        "./chroma_db",
        "chroma_db", 
        os.path.expanduser("~/chroma_db"),
        "/tmp/chroma_db",
        "./chromadb",
        "chromadb"
    ]
    
    # Agregar directorios de variables de entorno
    env_dir = os.getenv("CHROMA_PERSIST_DIR")
    if env_dir:
        possible_dirs.insert(0, env_dir)
    
    for directory in possible_dirs:
        try:
            abs_dir = os.path.abspath(directory)
            print(f"Trying persistent directory: {abs_dir}")
            
            if os.path.exists(abs_dir):
                print(f"   Directory exists, size: {get_dir_size(abs_dir)} bytes")
                files = os.listdir(abs_dir)
                print(f"   Files: {files[:10]}...")  # Show first 10 files
            else:
                print(f"   Directory does not exist")
                continue
            
            client = chromadb.PersistentClient(path=directory)
            collections = client.list_collections()
            print(f"✅ Connected to persistent client at {abs_dir}")
            print(f"   Collections: {[c.name for c in collections]}")
            
            # Verificar si hay datos
            for collection in collections:
                count = collection.count()
                print(f"   {collection.name}: {count} items")
                
                if count > 0:
                    # Mostrar una muestra de datos
                    sample = collection.get(limit=1, include=['metadatas', 'documents'])
                    if sample['metadatas']:
                        print(f"   Sample metadata keys: {list(sample['metadatas'][0].keys())}")
            
            if collections:  # Si encontramos colecciones con datos, este es probablemente el correcto
                return client, abs_dir
                
        except Exception as e:
            print(f"❌ Failed to connect to {directory}: {e}")
    
    return None, None

def get_dir_size(directory):
    """Calcula el tamaño de un directorio"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except:
        pass
    return total_size

def test_with_environment_vars():
    """Prueba con las variables de entorno actuales"""
    print("\n=== Testing with Current Environment ===")
    
    from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client
    
    try:
        config = ChromaDBConfig.from_env()
        print(f"Config loaded:")
        print(f"   chroma_host: {config.chroma_host}")
        print(f"   chroma_port: {config.chroma_port}")
        print(f"   persist_directory: {config.persist_directory}")
        
        client = get_chromadb_client(config)
        collections = client.list_collections()
        print(f"✅ Connected using environment config")
        print(f"   Collections: {[c.name for c in collections]}")
        
        for collection in collections:
            count = collection.count()
            print(f"   {collection.name}: {count} items")
        
        return client
        
    except Exception as e:
        print(f"❌ Failed with environment config: {e}")
        return None

def main():
    print("=== ChromaDB Diagnosis Tool ===\n")
    
    # Mostrar variables de entorno relevantes
    print("Environment variables:")
    env_vars = ["CHROMA_HOST", "CHROMA_PORT", "CHROMA_PERSIST_DIR"]
    for var in env_vars:
        value = os.getenv(var, "Not set")
        print(f"   {var}: {value}")
    print()
    
    # Buscar archivos de ChromaDB en el directorio actual
    print("Searching for ChromaDB files in current directory...")
    for root, dirs, files in os.walk("."):
        for file in files:
            if "chroma" in file.lower() or file.endswith(".sqlite3") or file.endswith(".db"):
                print(f"   Found: {os.path.join(root, file)}")
    print()
    
    # Probar diferentes conexiones
    server_client = test_server_connection()
    persistent_client, persistent_dir = test_persistent_clients()
    env_client = test_with_environment_vars()
    
    print("\n=== Summary ===")
    if server_client:
        print("✅ Server connection found")
    if persistent_client:
        print(f"✅ Persistent connection found at: {persistent_dir}")
    if env_client:
        print("✅ Environment config connection works")
    
    if not any([server_client, persistent_client, env_client]):
        print("❌ No working ChromaDB connections found with data")
        print("\nPossible solutions:")
        print("1. Check if ChromaDB server is running: `chroma run --host localhost --port 8000`")
        print("2. Verify the correct persist directory path")
        print("3. Check if data was migrated to the correct ChromaDB instance")

if __name__ == "__main__":
    main()