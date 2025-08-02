#!/usr/bin/env python3
"""
Script para analizar documentos de muestra y entender mejor los patrones de clasificación.
"""

import chromadb
import random
from collections import defaultdict

def analyze_sample_documents(persist_directory: str = "/Users/haroldgomez/chromadb2", 
                           sample_size: int = 50):
    """
    Analiza documentos de muestra para entender los patrones de URL y contenido.
    """
    # Conectar a ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name="docs_ada")
    
    # Obtener muestra aleatoria
    results = collection.get(
        limit=sample_size,
        offset=random.randint(0, 10000),
        include=["metadatas", "documents"]
    )
    
    print(f"📊 ANÁLISIS DE {len(results['metadatas'])} DOCUMENTOS DE MUESTRA")
    print("=" * 80)
    
    # Analizar patrones en URLs
    url_patterns = defaultdict(int)
    
    for i, (metadata, content) in enumerate(zip(results['metadatas'], results['documents'])):
        url = metadata.get('source', '')
        
        print(f"\n📄 Documento {i+1}:")
        print(f"   URL: {url}")
        print(f"   Contenido (primeros 150 chars): {content[:150]}...")
        
        # Extraer patrones de URL
        if '/azure/' in url:
            parts = url.split('/azure/')[1].split('/')[0] if '/azure/' in url else 'unknown'
            url_patterns[parts] += 1
    
    print(f"\n📈 PATRONES DE URL MÁS COMUNES:")
    print("-" * 40)
    for pattern, count in sorted(url_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"   /azure/{pattern}/: {count} documentos")

if __name__ == "__main__":
    analyze_sample_documents()