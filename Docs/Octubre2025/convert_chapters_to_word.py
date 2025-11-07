#!/usr/bin/env python3
"""
Script para convertir Capítulos 1-6 de Markdown a Word

Convierte los capítulos de la tesis de Markdown (.md) a formato Word (.docx)
usando pandoc. Los archivos Word se guardan en la carpeta Words/

Uso:
    python convert_chapters_to_word.py

Requisitos:
    - pandoc instalado en el sistema (brew install pandoc en macOS)

Autor: Claude AI
Fecha: 2025-11-06
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

# Configuración
DOCS_DIR = Path(__file__).parent
OUTPUT_DIR = DOCS_DIR / "Words"

# Capítulos a convertir (en orden)
CHAPTERS = [
    {
        'input': 'capitulo_1.md',
        'output': 'Capitulo_1_Introduccion.docx',
        'title': 'Capítulo 1: Introducción'
    },
    {
        'input': 'capitulo_2_estado_del_arte.md',
        'output': 'Capitulo_2_Estado_del_Arte.docx',
        'title': 'Capítulo 2: Estado del Arte'
    },
    {
        'input': 'capitulo_3_marco_teorico.md',
        'output': 'Capitulo_3_Marco_Teorico.docx',
        'title': 'Capítulo 3: Marco Teórico'
    },
    {
        'input': 'capitulo_4_analisis_exploratorio_datos.md',
        'output': 'Capitulo_4_Analisis_Exploratorio.docx',
        'title': 'Capítulo 4: Análisis Exploratorio de Datos'
    },
    {
        'input': 'capitulo_5_metodologia.md',
        'output': 'Capitulo_5_Metodologia.docx',
        'title': 'Capítulo 5: Metodología'
    },
    {
        'input': 'capitulo_6_implementacion.md',
        'output': 'Capitulo_6_Implementacion.docx',
        'title': 'Capítulo 6: Implementación'
    }
]

def check_pandoc():
    """Verifica que pandoc esté instalado."""
    try:
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
        print(f"✓ Pandoc encontrado: {result.stdout.splitlines()[0]}")
        return True
    except FileNotFoundError:
        print("✗ Error: pandoc no está instalado.")
        print("  Instálalo con: brew install pandoc (macOS)")
        return False

def convert_markdown_to_docx(md_file: Path, docx_file: Path, title: str) -> bool:
    """
    Convierte un archivo Markdown a Word usando pandoc.

    Args:
        md_file: Ruta al archivo .md de entrada
        docx_file: Ruta al archivo .docx de salida
        title: Título del documento

    Returns:
        True si la conversión fue exitosa, False en caso contrario
    """
    if not md_file.exists():
        print(f"  ✗ Archivo no encontrado: {md_file.name}")
        return False

    try:
        # Comando pandoc con opciones para mejor formato
        cmd = [
            'pandoc',
            str(md_file),
            '-o', str(docx_file),
            '--from=markdown',
            '--to=docx',
            '--standalone',
            '--toc',  # Tabla de contenidos
            '--toc-depth=3',  # Profundidad del TOC
            f'--metadata=title:{title}',
            '--highlight-style=tango',  # Estilo de código
            '--reference-doc=/Applications/Microsoft Word.app/Contents/Resources/reference.docx' if Path('/Applications/Microsoft Word.app').exists() else ''
        ]

        # Remover argumento vacío si no hay Word instalado
        cmd = [arg for arg in cmd if arg]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Verificar que el archivo se creó
        if docx_file.exists():
            size_kb = docx_file.stat().st_size / 1024
            print(f"  ✓ Creado: {docx_file.name} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  ✗ Error: archivo no creado")
            return False

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error en conversión: {e.stderr}")
        return False
    except Exception as e:
        print(f"  ✗ Error inesperado: {str(e)}")
        return False

def main():
    """Función principal del script."""
    print("=" * 70)
    print("Conversión de Capítulos Markdown → Word")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Verificar pandoc
    if not check_pandoc():
        return 1

    # Crear carpeta de salida si no existe
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Carpeta de salida: {OUTPUT_DIR}")
    print()

    # Convertir cada capítulo
    print("Convirtiendo capítulos...")
    print("-" * 70)

    successful = 0
    failed = 0

    for i, chapter in enumerate(CHAPTERS, 1):
        print(f"\n{i}. {chapter['title']}")

        input_file = DOCS_DIR / chapter['input']
        output_file = OUTPUT_DIR / chapter['output']

        if convert_markdown_to_docx(input_file, output_file, chapter['title']):
            successful += 1
        else:
            failed += 1

    # Resumen
    print()
    print("-" * 70)
    print("RESUMEN")
    print("-" * 70)
    print(f"✓ Exitosos:  {successful}/{len(CHAPTERS)}")
    print(f"✗ Fallidos:  {failed}/{len(CHAPTERS)}")
    print()

    if successful > 0:
        print(f"Archivos Word guardados en: {OUTPUT_DIR}")
        print()
        print("Archivos creados:")
        for docx_file in sorted(OUTPUT_DIR.glob("*.docx")):
            size_kb = docx_file.stat().st_size / 1024
            print(f"  - {docx_file.name} ({size_kb:.1f} KB)")

    print()
    print("=" * 70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
