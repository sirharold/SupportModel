#!/usr/bin/env python3
"""
Script para importar embeddings E5-Large generados en Colab de vuelta a ChromaDB
"""

import json
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict
import logging
from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('colab_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ColabE5Importer:
    """Importador de embeddings E5 desde Colab"""
    
    def __init__(self):
        self.chromadb_client = get_chromadb_client(ChromaDBConfig.from_env())
        logger.info("✅ ChromaDB client initialized")
    
    def load_colab_results(self, file_path: str) -> Dict:
        """Cargar resultados desde archivo JSON/Parquet de Colab"""
        logger.info(f"📥 Loading Colab results from: {file_path}")
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"✅ Loaded JSON: {len(data.get('documents', []))} documents")
                return data
                
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                logger.info(f"✅ Loaded Parquet: {len(df)} documents")
                
                # Convertir a formato JSON estándar
                data = {
                    'export_info': {
                        'timestamp': datetime.now().isoformat(),
                        'model': 'intfloat/e5-large-v2',
                        'dimensions': 1024,
                        'total_documents': len(df),
                        'source': 'parquet_import'
                    },
                    'documents': df['document'].tolist(),
                    'embeddings': df['embedding'].tolist(),
                    'metadatas': [
                        {k: v for k, v in row.items() if k not in ['document', 'embedding', 'id']}
                        for _, row in df.iterrows()
                    ],
                    'ids': df.get('id', [f"e5_doc_{i}" for i in range(len(df))]).tolist()
                }
                return data
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Colab results: {e}")
            raise
    
    def verify_data_integrity(self, data: Dict) -> bool:
        """Verificar integridad de los datos de Colab"""
        logger.info("🔍 Verifying data integrity...")
        
        try:
            documents = data.get('documents', [])
            embeddings = data.get('embeddings', [])
            metadatas = data.get('metadatas', [])
            ids = data.get('ids', [])
            
            # Verificar conteos
            counts = [len(documents), len(embeddings), len(metadatas), len(ids)]
            if len(set(counts)) != 1:
                logger.error(f"❌ Mismatched counts: docs={counts[0]}, embeddings={counts[1]}, metadata={counts[2]}, ids={counts[3]}")
                return False
            
            total_items = counts[0]
            logger.info(f"✅ Count verification passed: {total_items} items")
            
            # Verificar dimensiones de embeddings
            if embeddings:
                sample_dims = len(embeddings[0])
                if sample_dims != 1024:
                    logger.error(f"❌ Wrong embedding dimensions: {sample_dims} (expected 1024)")
                    return False
                
                # Verificar que todos tengan las mismas dimensiones
                dims_check = all(len(emb) == 1024 for emb in embeddings[:100])  # Verificar muestra
                if not dims_check:
                    logger.error("❌ Inconsistent embedding dimensions")
                    return False
                
                logger.info(f"✅ Embedding dimensions verified: {sample_dims}")
            
            # Verificar que no haya documentos vacíos
            empty_docs = sum(1 for doc in documents if not doc or not doc.strip())
            if empty_docs > 0:
                logger.warning(f"⚠️  Found {empty_docs} empty documents")
            
            # Verificar información de exportación
            export_info = data.get('export_info', {})
            if export_info:
                logger.info(f"📊 Export info:")
                logger.info(f"   Model: {export_info.get('model', 'unknown')}")
                logger.info(f"   Timestamp: {export_info.get('timestamp', 'unknown')}")
                logger.info(f"   GPU: {export_info.get('colab_gpu', 'unknown')}")
                if 'processing_time_minutes' in export_info:
                    logger.info(f"   Processing time: {export_info['processing_time_minutes']:.1f} min")
            
            logger.info("✅ Data integrity verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data integrity check failed: {e}")
            return False
    
    def create_or_update_collection(self, collection_name: str = "docs_e5large") -> bool:
        """Crear o actualizar colección E5"""
        try:
            # Intentar obtener colección existente
            try:
                collection = self.chromadb_client.get_collection(collection_name)
                existing_count = collection.count()
                logger.info(f"📋 Existing collection '{collection_name}': {existing_count} items")
                
                # Preguntar si quiere limpiar
                response = input(f"Collection '{collection_name}' exists with {existing_count} items. Clear it? (y/N): ")
                if response.lower() == 'y':
                    self.chromadb_client.delete_collection(collection_name)
                    logger.info(f"🗑️  Deleted existing collection")
                    collection = self.chromadb_client.create_collection(
                        name=collection_name,
                        metadata={"description": "Documents with E5-Large-v2 embeddings from Colab"}
                    )
                    logger.info(f"✅ Created fresh collection '{collection_name}'")
                else:
                    logger.info("📌 Keeping existing collection (will add new items)")
                    
            except:
                # Crear nueva colección
                logger.info(f"📝 Creating new collection '{collection_name}'")
                collection = self.chromadb_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Documents with E5-Large-v2 embeddings from Colab"}
                )
                logger.info(f"✅ Created collection '{collection_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create/update collection: {e}")
            return False
    
    def import_data(self, data: Dict, collection_name: str = "docs_e5large", batch_size: int = 100) -> bool:
        """Importar datos a ChromaDB en lotes"""
        logger.info(f"🚀 Starting import to collection '{collection_name}'")
        
        try:
            collection = self.chromadb_client.get_collection(collection_name)
            
            documents = data['documents']
            embeddings = data['embeddings']
            metadatas = data['metadatas']
            ids = data['ids']
            
            total_items = len(documents)
            logger.info(f"📊 Total items to import: {total_items}")
            
            successful_batches = 0
            failed_batches = 0
            start_time = time.time()
            
            # Procesar en lotes
            for i in range(0, total_items, batch_size):
                batch_start = time.time()
                end_idx = min(i + batch_size, total_items)
                
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_metadata = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                try:
                    collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    
                    successful_batches += 1
                    batch_time = time.time() - batch_start
                    
                    # Log progreso
                    progress = end_idx / total_items * 100
                    elapsed_time = time.time() - start_time
                    items_per_sec = end_idx / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (total_items - end_idx) / items_per_sec if items_per_sec > 0 else 0
                    
                    logger.info(f"✅ Batch {i//batch_size + 1}: {end_idx}/{total_items} ({progress:.1f}%) | "
                               f"{batch_time:.1f}s | Speed: {items_per_sec:.0f}/s | ETA: {eta_seconds/60:.1f}min")
                    
                except Exception as e:
                    logger.error(f"❌ Failed batch {i//batch_size + 1}: {e}")
                    failed_batches += 1
                    continue
            
            # Resumen final
            total_time = time.time() - start_time
            final_count = collection.count()
            
            logger.info(f"\n🎉 Import completed!")
            logger.info(f"   Total time: {total_time/60:.1f} minutes")
            logger.info(f"   Successful batches: {successful_batches}")
            logger.info(f"   Failed batches: {failed_batches}")
            logger.info(f"   Final collection size: {final_count:,} items")
            logger.info(f"   Average speed: {total_items/total_time:.0f} items/second")
            
            return failed_batches == 0
            
        except Exception as e:
            logger.error(f"❌ Import failed: {e}")
            return False

def main():
    """Función principal"""
    print("🚀 Colab E5 Results Importer")
    print("============================")
    
    # Solicitar archivo de entrada
    import os
    import glob
    
    # Buscar archivos de resultados de Colab
    possible_files = glob.glob("docs_e5large_processed*.json") + glob.glob("docs_e5large_processed*.parquet")
    
    if possible_files:
        print(f"\n📁 Found potential Colab result files:")
        for i, file in enumerate(possible_files):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   {i+1}. {file} ({size_mb:.1f} MB)")
        
        try:
            choice = input(f"\nSelect file (1-{len(possible_files)}) or enter custom path: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(possible_files):
                input_file = possible_files[int(choice) - 1]
            else:
                input_file = choice
        except:
            input_file = input("Enter path to Colab results file: ").strip()
    else:
        input_file = input("Enter path to Colab results file (.json or .parquet): ").strip()
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return False
    
    # Crear importador
    try:
        importer = ColabE5Importer()
    except Exception as e:
        print(f"❌ Failed to initialize importer: {e}")
        return False
    
    # Cargar datos
    try:
        data = importer.load_colab_results(input_file)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Verificar integridad
    if not importer.verify_data_integrity(data):
        print("❌ Data integrity check failed")
        return False
    
    # Crear/actualizar colección
    if not importer.create_or_update_collection():
        print("❌ Failed to setup collection")
        return False
    
    # Importar datos
    success = importer.import_data(data, batch_size=100)
    
    if success:
        print("\n✅ Import completed successfully!")
        print("\n🔍 Next steps:")
        print("1. Run 'python verify_e5_migration.py' to verify the import")
        print("2. Update your application to use 'e5-large-v2' model")
        print("3. Enjoy the improved performance!")
        return True
    else:
        print("\n❌ Import failed. Check logs for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n⏸️  Import interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n💥 Fatal error: {e}")
        exit(1)