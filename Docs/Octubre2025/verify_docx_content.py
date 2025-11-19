#!/usr/bin/env python3
"""
Verificar que los archivos Word contengan todas las imágenes y tablas de los archivos Markdown
"""

import re
from pathlib import Path
from docx import Document
import sys

def count_markdown_images(md_file):
    """Cuenta las imágenes en un archivo Markdown"""
    content = md_file.read_text(encoding='utf-8')
    # Buscar sintaxis de imágenes: ![alt](path)
    images = re.findall(r'!\[.*?\]\((.*?)\)', content)
    return images

def count_markdown_tables(md_file):
    """Cuenta las tablas en un archivo Markdown"""
    content = md_file.read_text(encoding='utf-8')
    lines = content.split('\n')

    # Buscar líneas que contienen tablas (líneas con |)
    table_count = 0
    in_table = False

    for line in lines:
        stripped = line.strip()
        if '|' in stripped and stripped:
            if not in_table:
                table_count += 1
                in_table = True
        else:
            in_table = False

    return table_count

def count_docx_images(docx_file):
    """Cuenta las imágenes en un archivo Word"""
    try:
        import zipfile
        # Verificar archivos media en el ZIP
        with zipfile.ZipFile(str(docx_file), 'r') as zip_ref:
            files = zip_ref.namelist()
            media_files = [f for f in files if 'word/media/' in f and f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            return len(media_files)
    except Exception as e:
        print(f"Error leyendo {docx_file}: {e}")
        return -1

def count_docx_tables(docx_file):
    """Cuenta las tablas en un archivo Word"""
    try:
        doc = Document(str(docx_file))
        return len(doc.tables)
    except Exception as e:
        print(f"Error leyendo {docx_file}: {e}")
        return -1

def verify_file_pair(md_file, docx_file):
    """Verifica que un par MD/DOCX tenga el mismo contenido"""
    print(f"\n{'='*80}")
    print(f"Verificando: {md_file.name} -> {docx_file.name}")
    print(f"{'='*80}")

    # Contar imágenes
    md_images = count_markdown_images(md_file)
    docx_images = count_docx_images(docx_file)

    print(f"\n📸 IMÁGENES:")
    print(f"  MD:   {len(md_images)} imágenes encontradas")
    if md_images:
        for img in md_images:
            print(f"    - {img}")
    print(f"  DOCX: {docx_images} imágenes encontradas")

    if len(md_images) != docx_images:
        print(f"  ⚠️  ADVERTENCIA: Diferencia en número de imágenes!")
    else:
        print(f"  ✅ OK")

    # Contar tablas
    md_tables = count_markdown_tables(md_file)
    docx_tables = count_docx_tables(docx_file)

    print(f"\n📊 TABLAS:")
    print(f"  MD:   {md_tables} tablas encontradas")
    print(f"  DOCX: {docx_tables} tablas encontradas")

    if md_tables != docx_tables:
        print(f"  ⚠️  ADVERTENCIA: Diferencia en número de tablas!")
    else:
        print(f"  ✅ OK")

    return len(md_images) == docx_images and md_tables == docx_tables

def main():
    base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
    words_dir = base_dir / "Words"

    # Pares de archivos a verificar
    file_pairs = [
        ("capitulo_0_abstract_ingles.md", "Capitulo_0_Abstract.docx"),
        ("capitulo_0_resumen_espanol.md", "Capitulo_0_Resumen.docx"),
        ("capitulo_1.md", "Capitulo_1_Introduccion.docx"),
        ("capitulo_2_estado_del_arte.md", "Capitulo_2_Estado_del_Arte.docx"),
        ("capitulo_3_marco_teorico.md", "Capitulo_3_Marco_Teorico.docx"),
        ("capitulo_4_analisis_exploratorio_datos.md", "Capitulo_4_Analisis_Exploratorio.docx"),
        ("capitulo_5_metodologia.md", "Capitulo_5_Metodologia.docx"),
        ("capitulo_6_implementacion.md", "Capitulo_6_Implementacion.docx"),
        ("capitulo7_resultados.md", "Capitulo_7_Resultados.docx"),
        ("capitulo_8_conclusiones_y_trabajo_futuro.md", "Capitulo_8_Conclusiones_y_Trabajo_Futuro.docx"),
        ("bibliografias.md", "Bibliografia.docx"),
    ]

    all_ok = True

    for md_name, docx_name in file_pairs:
        md_file = base_dir / md_name
        docx_file = words_dir / docx_name

        if not md_file.exists():
            print(f"❌ No encontrado: {md_file}")
            all_ok = False
            continue

        if not docx_file.exists():
            print(f"❌ No encontrado: {docx_file}")
            all_ok = False
            continue

        result = verify_file_pair(md_file, docx_file)
        if not result:
            all_ok = False

    print(f"\n{'='*80}")
    if all_ok:
        print("✅ TODOS LOS ARCHIVOS VERIFICADOS CORRECTAMENTE")
    else:
        print("⚠️  SE ENCONTRARON DIFERENCIAS EN ALGUNOS ARCHIVOS")
    print(f"{'='*80}\n")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
