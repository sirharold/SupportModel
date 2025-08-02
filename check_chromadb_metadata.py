#!/usr/bin/env python3
"""
Script para examinar la estructura de metadatos en ChromaDB.
"""

import chromadb
import json

def examine_chromadb_structure(persist_directory: str = "/Users/haroldgomez/chromadb2"):
    """
    Examina la estructura de datos en ChromaDB.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Listar colecciones
    collections = client.list_collections()
    print(f"📊 Colecciones disponibles: {[c.name for c in collections]}")
    
    # Examinar colección de documentos
    collection = client.get_collection(name="docs_ada")
    
    # Obtener muestra pequeña
    results = collection.get(
        limit=10,
        include=["metadatas", "documents"]
    )
    
    print(f"\n📄 ESTRUCTURA DE METADATOS (primeros 10 documentos):")
    print("=" * 60)
    
    for i, (metadata, content) in enumerate(zip(results['metadatas'], results['documents'])):
        print(f"\nDocumento {i+1}:")
        print(f"  Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        print(f"  Metadata: {metadata}")
        print(f"  Content length: {len(content)} chars")
        print(f"  Content preview: {content[:100]}...")
        
        if i >= 4:  # Solo mostrar primeros 5
            break
    
    # Contar documentos por tipo de metadata
    print(f"\n📊 ANÁLISIS DE METADATOS EN MUESTRA GRANDE:")
    print("=" * 60)
    
    large_sample = collection.get(
        limit=1000,
        include=["metadatas"]
    )
    
    source_count = 0
    empty_source_count = 0
    no_metadata_count = 0
    
    for metadata in large_sample['metadatas']:
        if not metadata:
            no_metadata_count += 1
        elif 'source' in metadata and metadata['source']:
            source_count += 1
        else:
            empty_source_count += 1
    
    print(f"  Documentos con source válido: {source_count}")
    print(f"  Documentos con source vacío: {empty_source_count}")
    print(f"  Documentos sin metadata: {no_metadata_count}")
    print(f"  Total analizado: {len(large_sample['metadatas'])}")

if __name__ == "__main__":
    examine_chromadb_structure()