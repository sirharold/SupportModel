#!/usr/bin/env python3
"""
Script para mostrar exactamente qué línea cambiar en Google Colab
"""

print("🔧 INSTRUCCIONES PARA CORREGIR EL MODELO EN GOOGLE COLAB")
print("=" * 60)
print()
print("1. Ve a Google Colab donde tienes el notebook abierto")
print("2. Encuentra la celda que contiene 'calculate_rag_metrics_real'")
print("3. Busca esta línea INCORRECTA:")
print("   ❌ bert_model = SentenceTransformer('distilbert-base-multilingual-cased')")
print()
print("4. Reemplázala por esta línea CORRECTA:")
print("   ✅ bert_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')")
print()
print("5. Ejecuta la celda nuevamente")
print("6. Ejecuta el resto del notebook")
print()
print("💡 El modelo corregido:")
print("   - Es un modelo válido de sentence-transformers")
print("   - Soporte multilingüe (15+ idiomas)")
print("   - Genera embeddings de 512 dimensiones")
print("   - Eliminará las advertencias WARNING")
print()
print("🎯 Después del cambio, NO deberías ver más mensajes como:")
print("   'No sentence-transformers model found with name distilbert-base-multilingual-cased'")