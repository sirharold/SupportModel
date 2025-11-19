#!/usr/bin/env python3
"""
Combina todos los archivos markdown en un solo documento y lo convierte a DOCX
usando la plantilla .dotx
"""

import subprocess
from pathlib import Path

# Directorio base
base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
words_dir = base_dir / "Words"

# Plantilla
template_file = base_dir / "Plantilla_Tesis_2025.dotx"

# Archivos markdown en orden
md_files = [
    base_dir / "capitulo_0_resumen_espanol.md",
    base_dir / "capitulo_0_abstract_ingles.md",
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

# Archivo combinado temporal
combined_md = words_dir / "combined_thesis.md"
output_docx = words_dir / "ProyectoMAgisterHaroldGomez_temp.docx"

print("Combinando archivos markdown...")
with open(combined_md, 'w', encoding='utf-8') as outfile:
    for i, md_file in enumerate(md_files):
        print(f"  Agregando: {md_file.name}")
        with open(md_file, 'r', encoding='utf-8') as infile:
            content = infile.read()

            # Agregar salto de página antes de cada archivo (incluso el primero)
            # porque este markdown se combinará con First.docx
            outfile.write('\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n')

            outfile.write(content)
            outfile.write('\n\n')  # Espaciado entre secciones

print(f"\nArchivo combinado creado: {combined_md}")
print(f"Tamaño: {combined_md.stat().st_size / 1024:.1f} KB")

# Convertir el markdown combinado a DOCX usando la plantilla
print("\nConvirtiendo markdown combinado a DOCX con plantilla...")
cmd = [
    'pandoc',
    str(combined_md),
    '-o', str(output_docx),
    '--reference-doc', str(template_file),
    '--resource-path', str(base_dir)
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✓ Conversión exitosa: {output_docx}")
    print(f"  Tamaño: {output_docx.stat().st_size / (1024*1024):.1f} MB")
else:
    print(f"✗ Error en conversión:")
    print(result.stderr)

print("\nAhora necesitamos combinar este archivo con First.docx...")
