#!/usr/bin/env python3
"""
Script para convertir cada archivo markdown a DOCX individual usando pandoc
sin plantilla personalizada.

Autor: Harold Gómez
Fecha: 2025-08-02
"""

import os
import subprocess
from pathlib import Path
import sys

def convert_md_files():
    """Convierte todos los archivos .md a .docx individualmente"""
    
    # Rutas
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "Docs" / "Finales"
    words_dir = docs_dir / "words"
    
    # Verificar que el directorio de origen existe
    if not docs_dir.exists():
        print(f"❌ Error: El directorio {docs_dir} no existe")
        return False
    
    # Crear directorio words si no existe
    words_dir.mkdir(exist_ok=True)
    
    # Encontrar todos los archivos .md
    md_files = list(docs_dir.glob("*.md"))
    
    if not md_files:
        print(f"❌ No se encontraron archivos .md en {docs_dir}")
        return False
    
    print(f"📄 Encontrados {len(md_files)} archivos markdown para convertir")
    print("=" * 60)
    
    successful_conversions = 0
    failed_conversions = []
    
    for md_file in md_files:
        # Generar nombre del archivo de salida
        docx_name = md_file.stem + ".docx"
        docx_path = words_dir / docx_name
        
        print(f"🔄 Convirtiendo: {md_file.name}")
        
        # Comando pandoc sin template y sin numeración automática
        # Pandoc automáticamente aplica estilos Word:
        # # → Heading 1, ## → Heading 2, ### → Heading 3, #### → Heading 4
        pandoc_cmd = [
            "pandoc",
            "-s",  # standalone document
            str(md_file),  # archivo de entrada
            "-o", str(docx_path),  # archivo de salida
            "--from", "markdown",
            "--to", "docx",
            "--toc",  # tabla de contenidos
            "--toc-depth=4",  # incluir hasta heading 4 en TOC
            "--highlight-style=tango"  # estilo de sintaxis
        ]
        
        try:
            # Ejecutar pandoc
            result = subprocess.run(pandoc_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Verificar que el archivo se creó y obtener su tamaño
                if docx_path.exists():
                    file_size = docx_path.stat().st_size / 1024  # KB
                    print(f"   ✅ {docx_name} ({file_size:.1f} KB)")
                    successful_conversions += 1
                else:
                    print(f"   ❌ Error: El archivo no se creó")
                    failed_conversions.append(md_file.name)
            else:
                print(f"   ❌ Error pandoc: {result.stderr.strip()}")
                failed_conversions.append(md_file.name)
                
        except Exception as e:
            print(f"   ❌ Error inesperado: {e}")
            failed_conversions.append(md_file.name)
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"📊 RESUMEN DE CONVERSIÓN:")
    print(f"   ✅ Exitosas: {successful_conversions}")
    print(f"   ❌ Fallidas: {len(failed_conversions)}")
    print(f"   📁 Ubicación: {words_dir}")
    
    if failed_conversions:
        print(f"\n❌ Archivos que fallaron:")
        for file in failed_conversions:
            print(f"   - {file}")
    
    # Listar archivos creados
    docx_files = list(words_dir.glob("*.docx"))
    if docx_files:
        print(f"\n📄 Archivos DOCX creados:")
        for docx_file in sorted(docx_files):
            file_size = docx_file.stat().st_size / 1024  # KB
            print(f"   - {docx_file.name} ({file_size:.1f} KB)")
    
    return len(failed_conversions) == 0

def main():
    """Función principal"""
    print("🚀 Conversor Individual MD a DOCX")
    print("=" * 60)
    
    # Verificar que pandoc esté instalado
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        print("✅ pandoc está disponible")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: pandoc no está instalado")
        print("   Instálalo con: brew install pandoc")
        sys.exit(1)
    
    # Convertir archivos
    success = convert_md_files()
    
    if success:
        print("\n✨ Todas las conversiones completadas exitosamente")
    else:
        print("\n⚠️  Algunas conversiones fallaron")
        sys.exit(1)

if __name__ == "__main__":
    main()