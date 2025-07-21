#!/usr/bin/env python3
"""
Script optimizado para exportar una sola colección ChromaDB con embeddings a Parquet
"""

import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb

def export_single_collection_to_parquet(collection_name: str, batch_size: int = 1000):
    """Exportar una sola colección a Parquet con barra de progreso"""
    try:
        # Usar ChromaDB client directo
        client = chromadb.PersistentClient(path="/Users/haroldgomez/chromadb2/")
        collection = client.get_collection(collection_name)
        
        print(f"📥 Loading collection '{collection_name}'...")
        
        # Obtener información de la colección
        count = collection.count()
        print(f"📊 Total items: {count:,}")
        
        if count == 0:
            print("⚠️ Collection is empty")
            return False
        
        # Obtener todos los datos incluyendo embeddings
        print("🔄 Fetching data with embeddings...")
        data = collection.get(include=['metadatas', 'documents', 'embeddings'])
        
        print(f"✅ Loaded {len(data['documents']):,} documents")
        embedding_dims = 0
        embeddings_data = data.get('embeddings', [])
        if embeddings_data is not None and len(embeddings_data) > 0:
            if embeddings_data[0] is not None:
                embedding_dims = len(embeddings_data[0])
        print(f"🔢 Embedding dimensions: {embedding_dims}")
        
        # Crear timestamp para archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_file = f"{collection_name}_with_embeddings_{timestamp}.parquet"
        
        print(f"💾 Converting to DataFrame...")
        
        # Convertir a DataFrame con barra de progreso
        records = []
        for i in tqdm(range(count), desc="Processing documents"):
            embedding = None
            if embeddings_data is not None and len(embeddings_data) > i and embeddings_data[i] is not None:
                embedding = embeddings_data[i].tolist()  # Convert numpy array to list
            
            record = {
                'document': data['documents'][i],
                'id': f"doc_{i}",
                'embedding': embedding,
            }
            # Agregar metadatos
            if data['metadatas'][i]:
                record.update(data['metadatas'][i])
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Guardar a Parquet
        print(f"💾 Saving to {parquet_file}...")
        df.to_parquet(parquet_file, index=False, compression='snappy')
        
        # Verificar tamaño del archivo
        size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"✅ Export completed: {parquet_file} ({size_mb:.1f} MB)")
        
        return True, parquet_file, size_mb
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False, None, 0

def main():
    """Exportar una colección específica"""
    print("🚀 ChromaDB Single Collection Exporter (With Real Embeddings)")
    print("=============================================================")
    
    if len(sys.argv) < 2:
        print("Usage: python export_single_collection.py <collection_name>")
        print("\nAvailable collections:")
        print("  - docs_ada      (Ada embeddings - 1536 dims)")
        print("  - docs_mpnet    (MPNet embeddings - 768 dims)")
        print("  - docs_minilm   (MiniLM embeddings - 384 dims)")
        print("  - docs_e5large  (E5-Large embeddings - 1024 dims)")
        return False
    
    collection_name = sys.argv[1]
    
    print(f"📦 Exporting collection: {collection_name}")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    start_time = datetime.now()
    success, filename, size_mb = export_single_collection_to_parquet(collection_name)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print("📊 EXPORT SUMMARY")
    print(f"{'='*60}")
    
    if success:
        print(f"✅ Collection: {collection_name}")
        print(f"📄 File: {filename}")
        print(f"📏 Size: {size_mb:.1f} MB")
        print(f"⏱️ Duration: {duration:.1f} seconds")
        print(f"🚀 Ready for Colab with real embeddings!")
        
        print(f"\n📝 COLAB USAGE:")
        print(f"  df = pd.read_parquet('{filename}')")
        print(f"  embeddings = np.array(df['embedding'].tolist())")
        print(f"  # Now you can use real cosine similarity!")
    else:
        print(f"❌ Failed to export {collection_name}")
    
    return success

if __name__ == "__main__":
    main()