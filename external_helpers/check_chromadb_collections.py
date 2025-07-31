#!/usr/bin/env python3
"""
Script para verificar qué colecciones existen en ChromaDB y sus estadísticas básicas.
"""

import os
import chromadb
from chromadb.config import Settings

def check_chromadb_collections():
    """
    Verifica las colecciones existentes en ChromaDB
    """
    print("🔍 VERIFICANDO COLECCIONES EN CHROMADB")
    print("=" * 50)
    
    try:
        # Conectar a ChromaDB
        print("\n📋 Conectando a ChromaDB...")
        client = chromadb.PersistentClient(
            path="/Users/haroldgomez/chromadb2",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Listar todas las colecciones
        collections = client.list_collections()
        
        print(f"✅ ChromaDB conectado exitosamente")
        print(f"📊 Total de colecciones encontradas: {len(collections)}")
        
        if not collections:
            print("❌ No hay colecciones en la base de datos")
            return False
        
        # Analizar cada colección
        print(f"\n📋 DETALLE DE COLECCIONES:")
        print("-" * 60)
        
        for collection in collections:
            print(f"\n📁 Colección: {collection.name}")
            
            try:
                # Obtener estadísticas básicas
                count = collection.count()
                print(f"   📊 Número de documentos: {count:,}")
                
                # Obtener una muestra para ver la estructura
                sample = collection.get(limit=1, include=['metadatas', 'documents'])
                
                if sample['metadatas']:
                    metadata_keys = list(sample['metadatas'][0].keys())
                    print(f"   🔑 Keys en metadata: {metadata_keys}")
                    
                    # Verificar si tiene campos específicos
                    metadata = sample['metadatas'][0]
                    
                    if 'question' in metadata or 'question_content' in metadata or 'title' in metadata:
                        print(f"   🤔 TIPO: Colección de PREGUNTAS")
                        if 'accepted_answer' in metadata:
                            print(f"   ✅ Tiene campo 'accepted_answer'")
                        if 'ms_links' in metadata or 'validated_links' in metadata:
                            print(f"   🔗 Tiene campos de links validados")
                    
                    elif 'link' in metadata and 'content' in metadata:
                        print(f"   📄 TIPO: Colección de DOCUMENTOS")
                        print(f"   🔗 Link ejemplo: {metadata.get('link', 'N/A')[:50]}...")
                    
                    else:
                        print(f"   ❓ TIPO: Desconocido")
                else:
                    print(f"   ⚠️ No hay metadata disponible")
                
            except Exception as e:
                print(f"   ❌ Error accediendo a la colección: {e}")
        
        # Verificar colecciones esperadas
        print(f"\n🎯 VERIFICACIÓN DE COLECCIONES ESPERADAS:")
        print("-" * 50)
        
        collection_names = [col.name for col in collections]
        
        expected_collections = [
            ("questions_withlinks", "Preguntas con links validados"),
            ("docs_ada", "Documentos con embeddings Ada"),
            ("questions_ada", "Preguntas con embeddings Ada"),
            ("docs", "Documentos generales"),
            ("questions", "Preguntas generales")
        ]
        
        for expected_name, description in expected_collections:
            if expected_name in collection_names:
                print(f"   ✅ {expected_name}: {description}")
            else:
                print(f"   ❌ {expected_name}: {description} (FALTANTE)")
        
        # Buscar colecciones de preguntas alternativas
        question_collections = [col for col in collections if 'question' in col.name.lower()]
        doc_collections = [col for col in collections if 'doc' in col.name.lower()]
        
        print(f"\n📝 Colecciones de preguntas encontradas: {[col.name for col in question_collections]}")
        print(f"📄 Colecciones de documentos encontradas: {[col.name for col in doc_collections]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error conectando a ChromaDB: {e}")
        return False

if __name__ == "__main__":
    success = check_chromadb_collections()
    if success:
        print("\n✅ Verificación completada")
    else:
        print("\n❌ Verificación falló")