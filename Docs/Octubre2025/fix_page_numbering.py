#!/usr/bin/env python3
"""
Corrige la numeración de páginas:
- Sección 0 (First + preliminares): lowerRoman (i, ii, iii)
- Sección 1 (Capítulos): decimal (1, 2, 3)
"""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
words_dir = base_dir / "Words"
doc_path = words_dir / "ProyectoMagisterHaroldGomez.docx"

def set_page_number_format(section, format_type='decimal', start=1):
    """Establece el formato de numeración de páginas para una sección"""
    sectPr = section._sectPr

    # Remover elemento pgNumType existente si lo hay
    for child in list(sectPr):
        if child.tag == qn('w:pgNumType'):
            sectPr.remove(child)

    # Crear nuevo elemento pgNumType
    pgNumType = OxmlElement('w:pgNumType')
    pgNumType.set(qn('w:fmt'), format_type)
    pgNumType.set(qn('w:start'), str(start))
    sectPr.append(pgNumType)

print("CORRIGIENDO NUMERACIÓN DE PÁGINAS")
print("=" * 70)

# Cargar documento
print(f"\n1. Cargando: {doc_path.name}")
doc = Document(str(doc_path))

print(f"   Secciones encontradas: {len(doc.sections)}")

# Configurar numeración
print(f"\n2. Configurando numeración de páginas...")

# Sección 0: First + preliminares -> números romanos (i, ii, iii)
print(f"   - Sección 0: Números romanos (i, ii, iii...)")
set_page_number_format(doc.sections[0], format_type='lowerRoman', start=1)

# Sección 1: Capítulos -> números arábigos (1, 2, 3)
if len(doc.sections) > 1:
    print(f"   - Sección 1: Números arábigos (1, 2, 3...)")
    set_page_number_format(doc.sections[1], format_type='decimal', start=1)

# Guardar
print(f"\n3. Guardando documento...")
doc.save(str(doc_path))

print(f"\n{'='*70}")
print(f"✅ NUMERACIÓN CORREGIDA")
print(f"{'='*70}")
print(f"   Sección 0: Números romanos (i, ii, iii...)")
print(f"   Sección 1: Números arábigos (1, 2, 3...)")
print(f"{'='*70}")
