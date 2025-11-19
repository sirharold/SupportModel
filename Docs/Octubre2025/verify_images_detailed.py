#!/usr/bin/env python3
"""
Verifica las imágenes del documento en detalle
"""

from docx import Document
from pathlib import Path

base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
words_dir = base_dir / "Words"
doc_path = words_dir / "ProyectoMagisterHaroldGomez.docx"

print("VERIFICACIÓN DETALLADA DE IMÁGENES")
print("=" * 70)

doc = Document(str(doc_path))

print(f"\n📄 Documento: {doc_path.name}\n")

# Contar imágenes embebidas
image_count = 0
image_list = []

for rel_id, rel in doc.part.rels.items():
    if "image" in rel.target_ref:
        image_count += 1
        image_list.append((rel_id, rel.target_ref))

print(f"🖼️  Total de imágenes embebidas: {image_count}\n")

if image_list:
    print("Lista de imágenes:")
    for i, (rel_id, target) in enumerate(image_list, 1):
        print(f"   {i}. {target} (ID: {rel_id})")

# Buscar referencias a figuras en el texto
print(f"\n📊 Buscando referencias a figuras en el texto...\n")

figura_mentions = []
for i, para in enumerate(doc.paragraphs):
    if 'Figura' in para.text and ':' in para.text:
        text_preview = para.text[:100] if len(para.text) > 100 else para.text
        figura_mentions.append((i, text_preview))

print(f"Referencias a figuras encontradas: {len(figura_mentions)}\n")
if figura_mentions:
    for i, (para_num, text) in enumerate(figura_mentions[:15], 1):  # Mostrar solo primeras 15
        print(f"   {i}. Párrafo {para_num}: {text}...")

print(f"\n{'='*70}")
