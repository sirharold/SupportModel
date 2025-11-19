#!/usr/bin/env python3
"""
Crea archivos markdown combinados en temp/ para revisión.
- preliminares_combined.md (Resumen + Abstract)
- capitulos_combined.md (Capítulos 1-8 + Bibliografía)
Agrega page breaks entre secciones usando OpenXML.
"""

from pathlib import Path

base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
temp_dir = base_dir / "temp"

# Archivos preliminares (desde temp/)
preliminares = [
    "capitulo_0_resumen_espanol.md",
    "capitulo_0_abstract_ingles.md"
]

# Capítulos principales (desde temp/)
capitulos = [
    "capitulo_1.md",
    "capitulo_2_estado_del_arte.md",
    "capitulo_3_marco_teorico.md",
    "capitulo_4_analisis_exploratorio_datos.md",
    "capitulo_5_metodologia.md",
    "capitulo_6_implementacion.md",
    "capitulo7_resultados.md",
    "capitulo_8_conclusiones_y_trabajo_futuro.md",
    "bibliografias.md"
]

def combine_files(files, output_name):
    """Combina archivos markdown con page breaks"""
    output_path = temp_dir / output_name

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, md_file in enumerate(files):
            # Leer desde el directorio base (archivos originales actualizados)
            filepath = base_dir / md_file

            if not filepath.exists():
                print(f"⚠️  Archivo no encontrado: {md_file}")
                continue

            # Agregar page break antes de cada sección (excepto la primera)
            if i > 0:
                outfile.write('\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n')

            # Leer y escribir contenido
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write('\n\n')  # Espaciado entre documentos

            print(f"✓ Agregado: {md_file}")

    print(f"📄 Creado: temp/{output_name}\n")

print("CREANDO ARCHIVOS MARKDOWN COMBINADOS EN temp/")
print("=" * 70)

print("\n1. Preliminares (Resumen + Abstract):")
combine_files(preliminares, "preliminares_combined.md")

print("2. Capítulos (1-8 + Bibliografía):")
combine_files(capitulos, "capitulos_combined.md")

print("=" * 70)
print("✅ ARCHIVOS COMBINADOS CREADOS EN temp/")
print("=" * 70)
print("\nArchivos para revisión:")
print("  - temp/preliminares_combined.md")
print("  - temp/capitulos_combined.md")
print("\nPuedes revisar estos archivos antes de generar los Word.")
print("Cada capítulo mantiene su numeración original (1., 2., 3., etc.)")
print("Las tablas tienen formato que forzará bordes en Word.")
print("=" * 70)
