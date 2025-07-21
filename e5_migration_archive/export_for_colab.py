#!/usr/bin/env python3
"""
Script para exportar datos de ChromaDB a formato compatible con Colab
"""

import json
import pandas as pd
from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client
import os
from datetime import datetime

def export_collection_to_json(collection_name: str, output_file: str):
    """Exportar colección completa a JSON"""
    try:
        client = get_chromadb_client(ChromaDBConfig.from_env())
        collection = client.get_collection(collection_name)
        
        print(f"📥 Loading collection '{collection_name}'...")
        
        # Obtener todos los datos
        data = collection.get(include=['metadatas', 'documents'])
        
        count = len(data['documents'])
        print(f"📊 Loaded {count:,} items")
        
        # Preparar datos para exportación
        export_data = {
            'collection_name': collection_name,
            'total_items': count,
            'export_timestamp': datetime.now().isoformat(),
            'documents': data['documents'],
            'metadatas': data['metadatas']
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
        client = get_chromadb_client(ChromaDBConfig.from_env())
        collection = client.get_collection(collection_name)
        
        print(f"📥 Loading collection '{collection_name}'...")
        
        # Obtener todos los datos
        data = collection.get(include=['metadatas', 'documents'])
        
        count = len(data['documents'])
        print(f"📊 Loaded {count:,} items")
        
        # Convertir a DataFrame
        records = []
        for i in range(count):
            record = {
                'document': data['documents'][i],
                'id': f"doc_{i}",  # ID secuencial
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
    """Exportar docs_ada para procesamiento en Colab"""
    print("🚀 ChromaDB to Colab Exporter")
    print("=============================")
    
    collection_name = "docs_ada"
    
    # Crear nombres de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"docs_ada_export_{timestamp}.json"
    parquet_file = f"docs_ada_export_{timestamp}.parquet"
    
    print(f"📂 Exporting collection: {collection_name}")
    print()
    
    # Exportar a ambos formatos
    print("1️⃣ Exporting to JSON...")
    json_success = export_collection_to_json(collection_name, json_file)
    
    print("\n2️⃣ Exporting to Parquet...")
    parquet_success = export_collection_to_parquet(collection_name, parquet_file)
    
    print("\n" + "="*50)
    print("📊 EXPORT SUMMARY")
    print("="*50)
    
    if json_success:
        size_json = os.path.getsize(json_file) / (1024 * 1024)
        print(f"✅ JSON export: {json_file} ({size_json:.1f} MB)")
    else:
        print("❌ JSON export failed")
    
    if parquet_success:
        size_parquet = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"✅ Parquet export: {parquet_file} ({size_parquet:.1f} MB)")
        print(f"📎 Recommended for Colab: {parquet_file}")
    else:
        print("❌ Parquet export failed")
    
    if parquet_success or json_success:
        print("\n🚀 NEXT STEPS:")
        print("1. Upload the exported file to Google Drive")
        print("2. Open the Colab notebook")
        print("3. Mount Google Drive in Colab")
        print("4. Run the E5 processing script")
        print("5. Download the result file")
        print("6. Import back to ChromaDB")
    
    return parquet_success or json_success

if __name__ == "__main__":
    main()