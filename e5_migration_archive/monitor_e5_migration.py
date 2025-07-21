#!/usr/bin/env python3
"""
Monitor en tiempo real para la migración E5-Large
Muestra progreso, velocidad, costos y ETA
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

def load_checkpoint() -> Optional[Dict]:
    """Cargar checkpoint actual"""
    try:
        with open("checkpoint_docs_e5large.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def format_time(seconds: float) -> str:
    """Formatear tiempo en formato legible"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_speed(items_per_hour: float) -> str:
    """Formatear velocidad"""
    if items_per_hour < 100:
        return f"{items_per_hour:.1f}/h"
    else:
        return f"{items_per_hour:.0f}/h"

def get_collection_count(collection_name: str) -> int:
    """Obtener conteo actual de la colección"""
    try:
        from utils.chromadb_utils import ChromaDBConfig, get_chromadb_client
        client = get_chromadb_client(ChromaDBConfig.from_env())
        collection = client.get_collection(collection_name)
        return collection.count()
    except:
        return 0

def display_progress_bar(progress: float, width: int = 40) -> str:
    """Crear barra de progreso visual"""
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress:.1f}%"

def monitor_migration():
    """Monitor principal"""
    print("🔍 E5-Large Migration Monitor")
    print("=============================")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_checkpoint = None
    last_processed = 0
    last_time = time.time()
    
    # Obtener total de items de la colección fuente
    source_total = get_collection_count("docs_ada")
    if source_total == 0:
        print("❌ Source collection 'docs_ada' not found or empty")
        return
    
    while True:
        try:
            # Limpiar pantalla (funciona en la mayoría de terminales)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🔍 E5-Large Migration Monitor")
            print("=============================")
            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Cargar checkpoint actual
            checkpoint = load_checkpoint()
            
            if not checkpoint:
                print("❌ No migration in progress")
                print("   Start migration with: python migrate_docs_to_e5.py")
                time.sleep(5)
                continue
            
            # Extraer datos del checkpoint
            processed = checkpoint.get("processed_count", 0)
            spent = checkpoint.get("spent", 0.0)
            method = checkpoint.get("current_method", "unknown")
            timestamp = checkpoint.get("timestamp", time.time())
            stats = checkpoint.get("stats", {})
            
            # Calcular métricas
            elapsed_total = time.time() - (timestamp - processed * 3600 / max(1, processed))  # Estimación
            progress = (processed / source_total * 100) if source_total > 0 else 0
            
            # Velocidad actual (basada en último intervalo)
            current_time = time.time()
            if last_checkpoint and current_time - last_time > 0:
                interval_speed = (processed - last_processed) / ((current_time - last_time) / 3600)
            else:
                interval_speed = 0
            
            # Velocidad promedio
            avg_speed = processed / (elapsed_total / 3600) if elapsed_total > 0 else 0
            
            # ETA
            remaining = source_total - processed
            eta_hours = remaining / max(1, interval_speed or avg_speed)
            eta_time = datetime.now() + timedelta(hours=eta_hours)
            
            # Display principal
            print(f"📊 PROGRESS OVERVIEW")
            print(f"   {display_progress_bar(progress)}")
            print(f"   Items: {processed:,} / {source_total:,}")
            print(f"   Remaining: {remaining:,}")
            print()
            
            print(f"⏱️  TIMING")
            print(f"   Elapsed: {format_time(elapsed_total)}")
            print(f"   ETA: {eta_time.strftime('%H:%M:%S')} ({format_time(eta_hours * 3600)})")
            print()
            
            print(f"📈 PERFORMANCE")
            print(f"   Current speed: {format_speed(interval_speed)}")
            print(f"   Average speed: {format_speed(avg_speed)}")
            print(f"   Method: {method.upper()}")
            print()
            
            print(f"💰 COSTS")
            print(f"   Spent: ${spent:.3f}")
            print(f"   Budget: $10.00")
            print(f"   Remaining: ${10.0 - spent:.3f}")
            print()
            
            if stats:
                print(f"📊 STATISTICS")
                print(f"   Successful batches: {stats.get('successful_batches', 0)}")
                print(f"   Failed batches: {stats.get('failed_batches', 0)}")
                print(f"   HF requests: {stats.get('hf_requests', 0)}")
                print(f"   OpenAI requests: {stats.get('openai_requests', 0)}")
                print(f"   Method switches: {stats.get('method_switches', 0)}")
                print()
            
            # Verificar estado de la colección destino
            target_count = get_collection_count("docs_e5large")
            print(f"🎯 TARGET COLLECTION")
            print(f"   docs_e5large: {target_count:,} items")
            print()
            
            # Alertas
            if spent > 8.0:
                print("⚠️  WARNING: Approaching budget limit!")
            
            if interval_speed < 100 and method == "huggingface":
                print("🐌 SLOW: Consider switching to paid method")
            
            if stats.get('failed_batches', 0) > 10:
                print("❌ WARNING: High failure rate detected")
            
            # Estado
            age_minutes = (current_time - timestamp) / 60
            if age_minutes > 5:
                print(f"🔴 STALE: Last update {age_minutes:.1f} minutes ago")
                print("   Migration may have stopped")
            else:
                print("🟢 ACTIVE: Migration is running")
            
            print("\n" + "="*50)
            print("Press Ctrl+C to stop monitoring")
            print("Refreshing in 30 seconds...")
            
            # Guardar para próxima iteración
            last_checkpoint = checkpoint
            last_processed = processed
            last_time = current_time
            
            # Esperar antes de próxima actualización
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            time.sleep(5)

def show_quick_status():
    """Mostrar estado rápido sin loop"""
    checkpoint = load_checkpoint()
    
    if not checkpoint:
        print("❌ No migration checkpoint found")
        return
    
    processed = checkpoint.get("processed_count", 0)
    spent = checkpoint.get("spent", 0.0)
    method = checkpoint.get("current_method", "unknown")
    timestamp = checkpoint.get("timestamp", time.time())
    
    source_total = get_collection_count("docs_ada")
    target_count = get_collection_count("docs_e5large")
    
    age_minutes = (time.time() - timestamp) / 60
    progress = (processed / source_total * 100) if source_total > 0 else 0
    
    print(f"📊 Quick Status:")
    print(f"   Progress: {processed:,}/{source_total:,} ({progress:.1f}%)")
    print(f"   Target collection: {target_count:,} items")
    print(f"   Method: {method}")
    print(f"   Spent: ${spent:.3f}")
    print(f"   Last update: {age_minutes:.1f} minutes ago")
    
    if age_minutes > 5:
        print("   ⚠️  Migration may be stopped")
    else:
        print("   ✅ Migration appears active")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        show_quick_status()
    else:
        monitor_migration()