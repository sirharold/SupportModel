#!/usr/bin/env python3
"""
Crea dos documentos combinados:
1. Preliminares: Resumen + Abstract (numeración romana)
2. Capítulos: Cap 1-8 + Bibliografía (numeración arábiga)
"""

import subprocess
from pathlib import Path

base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
words_dir = base_dir / "Words"
template_file = base_dir / "Plantilla_Tesis_2025.dotx"

# === PRELIMINARES (Resumen + Abstract) ===
preliminares_files = [
    base_dir / "capitulo_0_resumen_espanol.md",
    base_dir / "capitulo_0_abstract_ingles.md"
]

print("1. Creando preliminares_combined.md...")
preliminares_md = words_dir / "preliminares_combined.md"

with open(preliminares_md, 'w', encoding='utf-8') as outfile:
    for i, md_file in enumerate(preliminares_files):
        print(f"  Agregando: {md_file.name}")
        with open(md_file, 'r', encoding='utf-8') as infile:
            content = infile.read()

            # Salto de página antes de cada sección
            outfile.write('\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n')
            outfile.write(content)
            outfile.write('\n\n')

print(f"  ✓ Creado: {preliminares_md.stat().st_size / 1024:.1f} KB")

# === CAPÍTULOS (Cap 1-8 + Bibliografía) ===
capitulos_files = [
    base_dir / "capitulo_1.md",
    base_dir / "capitulo_2_estado_del_arte.md",
    base_dir / "capitulo_3_marco_teorico.md",
    base_dir / "capitulo_4_analisis_exploratorio_datos.md",
    base_dir / "capitulo_5_metodologia.md",
    base_dir / "capitulo_6_implementacion.md",
    base_dir / "capitulo7_resultados.md",
    base_dir / "capitulo_8_conclusiones_y_trabajo_futuro.md",
    base_dir / "bibliografias.md"
]

print("\n2. Creando capitulos_combined.md...")
capitulos_md = words_dir / "capitulos_combined.md"

with open(capitulos_md, 'w', encoding='utf-8') as outfile:
    for i, md_file in enumerate(capitulos_files):
        print(f"  Agregando: {md_file.name}")
        with open(md_file, 'r', encoding='utf-8') as infile:
            content = infile.read()

            # Salto de página antes de cada capítulo
            outfile.write('\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n')
            outfile.write(content)
            outfile.write('\n\n')

print(f"  ✓ Creado: {capitulos_md.stat().st_size / 1024:.1f} KB")

# === CONVERTIR A DOCX ===
print("\n3. Convirtiendo preliminares a DOCX...")
cmd = [
    'pandoc',
    str(preliminares_md),
    '-o', str(words_dir / 'preliminares.docx'),
    '--reference-doc', str(template_file),
    '--resource-path', str(base_dir)
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    size = (words_dir / 'preliminares.docx').stat().st_size / (1024*1024)
    print(f"  ✓ preliminares.docx ({size:.1f} MB)")
else:
    print(f"  ✗ Error: {result.stderr}")

print("\n4. Convirtiendo capítulos a DOCX...")
cmd = [
    'pandoc',
    str(capitulos_md),
    '-o', str(words_dir / 'capitulos.docx'),
    '--reference-doc', str(template_file),
    '--resource-path', str(base_dir)
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    size = (words_dir / 'capitulos.docx').stat().st_size / (1024*1024)
    print(f"  ✓ capitulos.docx ({size:.1f} MB)")
else:
    print(f"  ✗ Error: {result.stderr}")

print("\n✓ Documentos creados. Ahora se combinarán con numeración apropiada...")
