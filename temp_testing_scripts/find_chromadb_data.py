#!/usr/bin/env python3
"""
Script para encontrar todas las instancias de ChromaDB con datos
"""

import os
import chromadb

def search_chromadb_directories():
    """Busca directorios que podrían contener datos de ChromaDB"""
    print("=== Searching for ChromaDB Data ===\n")
    
    # Buscar archivos chroma.sqlite3 en el sistema
    search_paths = [
        ".",
        "..",
        "../..",
        os.path.expanduser("~"),
        "/tmp"
    ]
    
    found_instances = []
    
    for base_path in search_paths:
        try:
            for root, dirs, files in os.walk(base_path):
                # Limitar la profundidad de búsqueda
                level = root.replace(base_path, '').count(os.sep)
                if level >= 3:
                    dirs[:] = []  # No buscar más profundo
                    continue
                
                if "chroma.sqlite3" in files:
                    db_path = os.path.join(root, "chroma.sqlite3")
                    size = os.path.getsize(db_path)
                    print(f"Found ChromaDB at: {root}")
                    print(f"   Size: {size:,} bytes")
                    
                    if size > 1000:  # Solo si tiene datos significativos
                        found_instances.append(root)
                        try_connect(root)
                    else:
                        print(f"   ⚠️  Database is very small, likely empty")
                    print()
        except (PermissionError, OSError):
            continue
    
    return found_instances

def try_connect(directory):
    """Intenta conectar a una instancia de ChromaDB y verificar datos"""
    try:
        client = chromadb.PersistentClient(path=directory)
        collections = client.list_collections()
        
        print(f"   Collections: {[c.name for c in collections]}")
        
        total_items = 0
        for collection in collections:
            try:
                count = collection.count()
                total_items += count
                print(f"   {collection.name}: {count:,} items")
                
                if count > 0:
                    # Mostrar una muestra
                    sample = collection.get(limit=1, include=['metadatas'])
                    if sample['metadatas']:
                        keys = list(sample['metadatas'][0].keys())
                        print(f"      Sample keys: {keys[:5]}...")  # Primeras 5 keys
            except Exception as e:
                print(f"   Error accessing {collection.name}: {e}")
        
        if total_items > 0:
            print(f"   ✅ FOUND DATA: {total_items:,} total items")
        else:
            print(f"   ❌ No data found")
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")

def main():
    found_instances = search_chromadb_directories()
    
    print("=== Summary ===")
    if found_instances:
        print("ChromaDB instances with data found at:")
        for instance in found_instances:
            print(f"  - {os.path.abspath(instance)}")
        
        print(f"\nTo use the correct instance, update ChromaDBConfig.persist_directory to point to one of these paths.")
    else:
        print("❌ No ChromaDB instances with data found.")
        print("\nPossible next steps:")
        print("1. Check if ChromaDB server is running elsewhere")
        print("2. Verify the migration completed successfully")
        print("3. Check if data is in a different format/location")

if __name__ == "__main__":
    main()