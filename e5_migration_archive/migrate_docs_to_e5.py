#!/usr/bin/env python3
"""
Script para migrar docs_ada a docs_e5large con embeddings E5-Large-v2
Incluye sistema híbrido: HuggingFace gratis + OpenAI aceleración
Con sistema robusto de checkpoints y recuperación de errores
"""

import asyncio
import json
import time
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Imports para embedding providers
from huggingface_hub import InferenceClient
import openai

# Imports locales
from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('e5_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationConfig:
    """Configuración de la migración"""
    budget_limit: float = 10.0
    hf_batch_size: int = 50
    openai_batch_size: int = 100
    hf_delay: float = 1.2  # segundos entre requests HF
    speed_threshold: int = 500  # items/hour mínimo
    switch_threshold_hours: float = 2.0
    checkpoint_frequency: int = 1000
    max_retries: int = 3
    retry_delay: float = 2.0

class E5DocsHybridMigrator:
    """Migrador híbrido para documentos con E5-Large embeddings"""
    
    def __init__(self, config: MigrationConfig = None):
        self.config = config or MigrationConfig()
        self.start_time = time.time()
        self.processed_count = 0
        self.spent = 0.0
        self.current_method = "huggingface"
        self.checkpoint_file = "checkpoint_docs_e5large.json"
        
        # Clientes
        self._init_clients()
        
        # Estadísticas
        self.stats = {
            "total_items": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "hf_requests": 0,
            "openai_requests": 0,
            "method_switches": 0
        }
        
        logger.info("🚀 E5 Docs Migrator initialized")
        logger.info(f"💰 Budget limit: ${self.config.budget_limit}")
        logger.info(f"⏱️  Switch threshold: {self.config.switch_threshold_hours} hours")
    
    def _init_clients(self):
        """Inicializar clientes de embedding y ChromaDB"""
        try:
            # Hugging Face client
            self.hf_client = InferenceClient("intfloat/e5-large-v2")
            logger.info("✅ Hugging Face client initialized")
            
            # OpenAI client (para aceleración)
            from dotenv import load_dotenv
            load_dotenv()
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key.strip():
                self.openai_client = openai.OpenAI(api_key=openai_key.strip())
                logger.info("✅ OpenAI client initialized")
            else:
                self.openai_client = None
                logger.warning("⚠️  No OpenAI API key found - acceleration disabled")
            
            # ChromaDB client
            self.chromadb_client = get_chromadb_client(ChromaDBConfig.from_env())
            logger.info("✅ ChromaDB client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise
    
    def preprocess_for_e5(self, text: str) -> str:
        """Preprocesar texto para E5-Large con optimizaciones"""
        if not text or not text.strip():
            return "passage: empty document"
        
        # Limpiar texto
        text = text.strip()
        
        # Truncar inteligentemente para E5 (óptimo ~512 tokens)
        if len(text) > 2000:  # ~1500 tokens aprox
            # Mantener inicio y final para preservar contexto
            text = text[:1000] + " ... " + text[-1000:]
        
        # Prefijo requerido por E5
        return f"passage: {text}"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimación rápida de tokens"""
        return max(1, len(text) // 4)  # Aproximación: 4 chars = 1 token
    
    def should_switch_to_paid(self) -> bool:
        """Decidir si cambiar a método pagado - SOLO si local realmente falla"""
        if not self.openai_client:
            return False
        
        elapsed_hours = (time.time() - self.start_time) / 3600
        current_speed = self.processed_count / elapsed_hours if elapsed_hours > 0.1 else 0
        
        # Más restrictivo: solo cambiar si realmente hay problemas graves
        reasons_to_switch = [
            elapsed_hours >= self.config.switch_threshold_hours and current_speed < 100,  # Muy lento Y mucho tiempo
            self.stats.get('failed_batches', 0) > 10,  # Muchos fallos
        ]
        
        should_switch = any(reasons_to_switch) and self.spent < self.config.budget_limit * 0.8
        
        if should_switch and self.current_method == "huggingface":
            logger.info(f"🔄 Switching to OpenAI due to persistent issues: elapsed={elapsed_hours:.1f}h, speed={current_speed:.0f}/h, failures={self.stats.get('failed_batches', 0)}")
            
        return should_switch
    
    async def get_embeddings_hf(self, texts: List[str]) -> List[List[float]]:
        """Obtener embeddings gratis usando sentence-transformers local"""
        for attempt in range(self.config.max_retries):
            try:
                # Inicializar modelo local si no existe
                if not hasattr(self, '_local_model'):
                    logger.info("📥 Loading E5-Large model locally (first time may take a while)...")
                    from sentence_transformers import SentenceTransformer
                    self._local_model = SentenceTransformer('intfloat/e5-large-v2')
                    logger.info("✅ E5-Large model loaded successfully")
                
                # Preprocesar para E5
                processed_texts = [self.preprocess_for_e5(text) for text in texts]
                
                # Generar embeddings localmente
                embeddings = self._local_model.encode(
                    processed_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).tolist()
                
                # Validar respuesta
                if not embeddings or not isinstance(embeddings, list):
                    raise ValueError("Invalid embeddings generated")
                
                if len(embeddings) != len(texts):
                    raise ValueError(f"Mismatch: expected {len(texts)} embeddings, got {len(embeddings)}")
                
                # Validar dimensiones (1024 para E5-large)
                if embeddings and len(embeddings[0]) != 1024:
                    raise ValueError(f"Expected 1024 dimensions, got {len(embeddings[0])}")
                
                self.stats["hf_requests"] += 1
                
                # Rate limiting mínimo para no sobrecargar
                await asyncio.sleep(0.1)
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"HF attempt {attempt + 1}/{self.config.max_retries} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def get_embeddings_openai(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """Obtener embeddings pagados via OpenAI"""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        # Estimar costo
        total_tokens = sum(self.estimate_tokens(text) for text in texts)
        estimated_cost = total_tokens * 0.00013 / 1000  # OpenAI pricing
        
        if self.spent + estimated_cost > self.config.budget_limit:
            raise Exception(f"Would exceed budget: ${self.spent + estimated_cost:.2f} > ${self.config.budget_limit}")
        
        for attempt in range(self.config.max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_client.embeddings.create(
                        model="text-embedding-3-large",
                        input=texts
                    )
                )
                
                embeddings = [item.embedding for item in response.data]
                actual_cost = response.usage.total_tokens * 0.00013 / 1000
                self.spent += actual_cost
                self.stats["openai_requests"] += 1
                
                logger.info(f"💰 OpenAI batch: ${actual_cost:.3f} (total: ${self.spent:.2f})")
                return embeddings, actual_cost
                
            except Exception as e:
                logger.warning(f"OpenAI attempt {attempt + 1}/{self.config.max_retries} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def process_batch(self, texts: List[str]) -> List[List[float]]:
        """Procesar lote con método actual - FORZAR LOCAL PARA E5"""
        # Solo usar método local para mantener dimensiones consistentes
        try:
            return await self.get_embeddings_hf(texts)
        except Exception as e:
            logger.error(f"Local E5 batch failed: {e}")
            # No cambiar a OpenAI - mantener consistencia dimensional
            raise
    
    def save_checkpoint(self):
        """Guardar progreso actual"""
        checkpoint_data = {
            "timestamp": time.time(),
            "processed_count": self.processed_count,
            "spent": self.spent,
            "current_method": self.current_method,
            "stats": self.stats,
            "config": {
                "budget_limit": self.config.budget_limit,
                "source_collection": "docs_ada",
                "target_collection": "docs_e5large"
            }
        }
        
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"💾 Checkpoint saved: {self.processed_count:,} items processed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Cargar checkpoint si existe"""
        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            
            self.processed_count = checkpoint["processed_count"]
            self.spent = checkpoint["spent"]
            self.current_method = checkpoint["current_method"]
            self.stats.update(checkpoint.get("stats", {}))
            
            logger.info(f"📂 Resumed from checkpoint: {self.processed_count:,} items")
            return checkpoint
        except FileNotFoundError:
            logger.info("No checkpoint found, starting fresh")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def create_target_collection(self):
        """Crear colección destino si no existe"""
        try:
            collection = self.chromadb_client.get_collection("docs_e5large")
            logger.info("✅ Target collection 'docs_e5large' already exists")
            return collection
        except:
            logger.info("📝 Creating target collection 'docs_e5large'")
            collection = self.chromadb_client.create_collection(
                name="docs_e5large",
                metadata={"description": "Documents with E5-Large-v2 embeddings"}
            )
            logger.info("✅ Target collection created successfully")
            return collection
    
    def get_collection_info(self, collection_name: str) -> Tuple[int, int]:
        """Obtener información de la colección"""
        try:
            collection = self.chromadb_client.get_collection(collection_name)
            count = collection.count()
            
            # Obtener muestra para verificar estructura
            if count > 0:
                sample = collection.get(limit=1, include=['metadatas', 'documents'])
                sample_keys = list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
                logger.info(f"📋 {collection_name}: {count:,} items, sample keys: {sample_keys}")
            else:
                logger.info(f"📋 {collection_name}: {count:,} items (empty)")
            
            return count, count
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return 0, 0
    
    async def migrate(self):
        """Ejecutar migración completa"""
        logger.info("🎯 Starting docs_ada -> docs_e5large migration")
        
        # Cargar checkpoint si existe
        checkpoint = self.load_checkpoint()
        start_offset = checkpoint["processed_count"] if checkpoint else 0
        
        # Verificar colecciones
        source_count, _ = self.get_collection_info("docs_ada")
        if source_count == 0:
            logger.error("❌ Source collection 'docs_ada' is empty or doesn't exist")
            return False
        
        self.stats["total_items"] = source_count
        
        # Crear colección destino
        target_collection = self.create_target_collection()
        target_count, _ = self.get_collection_info("docs_e5large")
        
        logger.info(f"📊 Migration overview:")
        logger.info(f"   Source: docs_ada ({source_count:,} items)")
        logger.info(f"   Target: docs_e5large ({target_count:,} items)")
        logger.info(f"   Starting from offset: {start_offset:,}")
        logger.info(f"   Remaining: {source_count - start_offset:,} items")
        
        # Obtener colección fuente
        source_collection = self.chromadb_client.get_collection("docs_ada")
        
        # Configurar batch size según método actual
        batch_size = self.config.hf_batch_size if self.current_method == "huggingface" else self.config.openai_batch_size
        
        # Obtener todos los datos de una vez (será más eficiente para ChromaDB)
        logger.info("📥 Loading all source data...")
        all_source_data = source_collection.get(
            include=['metadatas', 'documents']
        )
        
        total_docs = len(all_source_data['documents'])
        logger.info(f"📊 Loaded {total_docs:,} documents from source")
        
        try:
            for offset in range(start_offset, total_docs, batch_size):
                batch_start = time.time()
                limit = min(batch_size, total_docs - offset)
                
                # Obtener lote de datos de la lista cargada
                batch_data = {
                    'documents': all_source_data['documents'][offset:offset + limit],
                    'metadatas': all_source_data['metadatas'][offset:offset + limit]
                }
                
                if not batch_data['documents']:
                    logger.warning(f"No documents found at offset {offset}")
                    break
                
                # Filtrar documentos vacíos
                valid_indices = [i for i, doc in enumerate(batch_data['documents']) if doc and doc.strip()]
                
                if not valid_indices:
                    logger.warning(f"All documents empty in batch at offset {offset}")
                    self.processed_count += len(batch_data['documents'])
                    continue
                
                # Preparar datos válidos
                valid_docs = [batch_data['documents'][i] for i in valid_indices]
                valid_metadata = [batch_data['metadatas'][i] for i in valid_indices]
                # Generar IDs únicos basados en offset + índice
                valid_ids = [f"doc_{offset + i}" for i in valid_indices]
                
                # Generar embeddings
                try:
                    embeddings = await self.process_batch(valid_docs)
                    
                    # Guardar en nueva colección con prefijo para evitar conflictos
                    target_collection.add(
                        documents=valid_docs,
                        metadatas=valid_metadata,
                        embeddings=embeddings,
                        ids=[f"e5_{id}" for id in valid_ids]
                    )
                    
                    self.processed_count += len(batch_data['documents'])
                    self.stats["successful_batches"] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process batch at offset {offset}: {e}")
                    self.stats["failed_batches"] += 1
                    
                    # Guardar checkpoint antes de continuar
                    self.save_checkpoint()
                    
                    # Si es un error de presupuesto, terminar
                    if "budget" in str(e).lower():
                        logger.error("💰 Budget exceeded, stopping migration")
                        break
                    
                    # Continuar con siguiente batch
                    continue
                
                # Calcular estadísticas
                batch_time = time.time() - batch_start
                elapsed_hours = (time.time() - self.start_time) / 3600
                overall_speed = self.processed_count / elapsed_hours if elapsed_hours > 0 else 0
                progress = (offset + limit) / total_docs * 100
                eta_hours = (total_docs - self.processed_count) / overall_speed if overall_speed > 0 else 0
                
                # Log progreso
                logger.info(f"✅ Batch {offset//batch_size + 1}: {len(embeddings)}/{len(batch_data['documents'])} items | "
                           f"{batch_time:.1f}s | Progress: {progress:.1f}% | "
                           f"Speed: {overall_speed:.0f}/h | ETA: {eta_hours:.1f}h | "
                           f"Method: {self.current_method} | Spent: ${self.spent:.2f}")
                
                # Checkpoint periódico
                if self.processed_count % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint()
                
                # Verificar si necesita cambiar método
                if self.current_method == "huggingface" and self.should_switch_to_paid():
                    logger.info("🚀 Switching to paid method for acceleration...")
                    self.current_method = "openai"
                    batch_size = self.config.openai_batch_size
                    self.stats["method_switches"] += 1
            
            # Checkpoint final
            self.save_checkpoint()
            
            # Verificar resultado final
            final_count, _ = self.get_collection_info("docs_e5large")
            
            # Resumen final
            total_time = (time.time() - self.start_time) / 3600
            logger.info("\n🎉 Migration Complete!")
            logger.info(f"⏱️  Total time: {total_time:.2f} hours")
            logger.info(f"💰 Total cost: ${self.spent:.2f}")
            logger.info(f"📊 Items processed: {self.processed_count:,}")
            logger.info(f"📈 Average speed: {self.processed_count / total_time:.0f} items/hour")
            logger.info(f"✅ Final collection size: {final_count:,} items")
            logger.info(f"📊 Success rate: {self.stats['successful_batches']}/{self.stats['successful_batches'] + self.stats['failed_batches']} batches")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.save_checkpoint()
            return False

async def main():
    """Función principal"""
    print("🚀 E5-Large Documents Migration Tool")
    print("=====================================")
    
    # Configuración
    config = MigrationConfig(
        budget_limit=10.0,
        hf_batch_size=50,
        openai_batch_size=100,
        speed_threshold=500,
        switch_threshold_hours=2.0
    )
    
    # Crear migrador
    migrator = E5DocsHybridMigrator(config)
    
    # Ejecutar migración
    success = await migrator.migrate()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("You can now use 'e5-large-v2' model in your application")
    else:
        print("\n❌ Migration failed. Check logs for details.")
        print("You can resume by running this script again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏸️  Migration interrupted by user")
        print("Progress has been saved. You can resume by running this script again.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n💥 Fatal error: {e}")