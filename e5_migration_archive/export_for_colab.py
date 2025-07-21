#!/usr/bin/env python3
"""
Script para exportar datos de ChromaDB a formato compatible con Colab
"""

import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.storage.chromadb_utils import ChromaDBClientWrapper
import chromadb

def export_collection_to_json(collection_name: str, output_file: str):
    """Exportar colección completa a JSON"""
    try:
        # Usar ChromaDB client directo
        client = chromadb.PersistentClient(path="/Users/haroldgomez/chromadb2/")
        collection = client.get_collection(collection_name)
        
        print(f"📥 Loading collection '{collection_name}'...")
        
        # Obtener todos los datos incluyendo embeddings
        data = collection.get(include=['metadatas', 'documents', 'embeddings'])
        
        count = len(data['documents'])
        print(f"📊 Loaded {count:,} items")
        
        # Preparar datos para exportación
        export_data = {
            'collection_name': collection_name,
            'total_items': count,
            'export_timestamp': datetime.now().isoformat(),
            'documents': data['documents'],
            'metadatas': data['metadatas'],
            'embeddings': data['embeddings']
        }
        
        # Guardar a JSON
        print(f"💾 Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        # Verificar tamaño del archivo
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Export completed: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def export_collection_to_parquet(collection_name: str, output_file: str):
    """Exportar colección a Parquet (más eficiente para archivos grandes)"""
    try:
        # Usar ChromaDB client directo
        client = chromadb.PersistentClient(path="/Users/haroldgomez/chromadb2/")
        collection = client.get_collection(collection_name)
        
        print(f"📥 Loading collection '{collection_name}'...")
        
        # Obtener todos los datos incluyendo embeddings
        data = collection.get(include=['metadatas', 'documents', 'embeddings'])
        
        count = len(data['documents'])
        print(f"📊 Loaded {count:,} items")
        
        # Convertir a DataFrame
        records = []
        for i in range(count):
            record = {
                'document': data['documents'][i],
                'id': f"doc_{i}",  # ID secuencial
                'embedding': data['embeddings'][i] if data['embeddings'] else None,  # Vector embedding
            }
            # Agregar metadatos como columnas separadas
            if data['metadatas'][i]:
                record.update(data['metadatas'][i])
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Guardar a Parquet
        print(f"💾 Saving to {output_file}...")
        df.to_parquet(output_file, index=False, compression='snappy')
        
        # Verificar tamaño del archivo
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Export completed: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def main():
    """Exportar colecciones de documentos para procesamiento en Colab"""
    print("🚀 ChromaDB to Colab Exporter (Real Embeddings)")
    print("===============================================")
    
    # Colecciones a exportar con embeddings reales
    collections_to_export = [
        "docs_ada",      # Ada embeddings (1536 dims)
        "docs_mpnet",    # MPNet embeddings (768 dims)
        "docs_minilm",   # MiniLM embeddings (384 dims)
        "docs_e5large"   # E5-Large embeddings (1024 dims)
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"📂 Collections to export: {len(collections_to_export)}")
    for collection in collections_to_export:
        print(f"  - {collection}")
    print()
    
    export_results = {}
    
    for i, collection_name in enumerate(collections_to_export, 1):
        print(f"\n{'='*60}")
        print(f"📦 EXPORTING {collection_name.upper()} ({i}/{len(collections_to_export)})")
        print(f"{'='*60}")
        
        # Crear nombres de archivo con timestamp
        parquet_file = f"{collection_name}_export_{timestamp}.parquet"
        
        print(f"📂 Collection: {collection_name}")
        print(f"📄 Output: {parquet_file}")
        print()
        
        # Exportar solo a Parquet (más eficiente para Colab)
        print("📤 Exporting to Parquet with embeddings...")
        parquet_success = export_collection_to_parquet(collection_name, parquet_file)
        
        export_results[collection_name] = {
            'file': parquet_file if parquet_success else None,
            'success': parquet_success
        }
    
    print("\n" + "="*50)
    print("📊 EXPORT SUMMARY")
    print("="*50)
    
    total_size = 0
    successful_exports = []
    
    for collection_name, result in export_results.items():
        if result['success']:
            size_mb = os.path.getsize(result['file']) / (1024 * 1024)
            print(f"✅ {collection_name}: {result['file']} ({size_mb:.1f} MB)")
            total_size += size_mb
            successful_exports.append(result['file'])
        else:
            print(f"❌ {collection_name}: Export failed")
    
    print(f"\n📊 Total exported: {len(successful_exports)}/{len(collections_to_export)} collections ({total_size:.1f} MB)")
    
    if successful_exports:
        print("\n🚀 NEXT STEPS:")
        print("1. Upload the exported files to Google Drive")
        print("2. Open the Colab notebook")
        print("3. Mount Google Drive in Colab")
        print("4. Load real embeddings for cosine similarity calculation")
        print("5. Calculate real retrieval metrics (no simulation)")
        print("6. Generate accurate evaluation reports")
        
        print(f"\n📁 Files ready for Colab:")
        for file in successful_exports:
            print(f"  - {file}")
    
    return len(successful_exports) > 0

if __name__ == "__main__":
    main()