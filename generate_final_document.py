#!/usr/bin/env python3
"""
Script para generar documento final del proyecto de título en formato DOCX
usando pandoc con numeración correlativa y copia a Google Drive.

Autor: Harold Gómez
Fecha: 2025-08-02
"""

import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import sys

def get_next_version_number(output_dir):
    """Obtiene el siguiente número de versión disponible"""
    existing_files = list(Path(output_dir).glob("ProyectoTituloHaroldGomez_ver*.docx"))
    
    if not existing_files:
        return 1
    
    # Extraer números de versión existentes
    version_numbers = []
    for file in existing_files:
        try:
            version_str = file.stem.split("_ver")[1]
            version_numbers.append(int(version_str))
        except (IndexError, ValueError):
            continue
    
    return max(version_numbers) + 1 if version_numbers else 1

def generate_document():
    """Genera el documento final con pandoc"""
    
    # Rutas base
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "Docs" / "Finales"
    output_dir = base_dir / "output"
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(exist_ok=True)
    
    # Obtener siguiente número de versión
    version_num = get_next_version_number(output_dir)
    version_str = f"{version_num:03d}"
    
    # Nombre del archivo de salida
    output_filename = f"ProyectoTituloHaroldGomez_ver{version_str}.docx"
    output_path = output_dir / output_filename
    
    # Lista ordenada de archivos markdown
    markdown_files = [
        "0_1_head.md",
        "capitulo_0_resumen.md",
        "capitulo_1.md",
        "capitulo_2_estado_del_arte.md",
        "capitulo_3_marco_teorico.md",
        "capitulo_4_analisis_exploratorio_datos.md",
        "capitulo_5_metodologia.md",
        "capitulo_6_implementacion.md",
        "capitulo_7_resultados_y_analisis.md",
        "capitulo_8_conclusiones_y_trabajo_futuro.md",
        "anexo_b_codigo_fuente.md",
        "anexo_c_configuracion_ambiente.md",
        "anexo_d_ejemplos_consultas_respuestas.md",
        "anexo_e_resultados_detallados_metricas.md",
        "anexo_f_streamlit_app.md"
    ]
    
    # Verificar que todos los archivos existen
    missing_files = []
    for file in markdown_files:
        if not (docs_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Error: Los siguientes archivos no se encontraron:")
        for file in missing_files:
            print(f"   - {file}")
        return None
    
    # Construir rutas completas
    full_paths = [str(docs_dir / file) for file in markdown_files]
    
    # Verificar si existe plantilla personalizada
    template_path = docs_dir / "ProyectoTituloMagisterHaroldGomez.dotx"
    
    # Comando pandoc base
    pandoc_cmd = [
        "pandoc",
        "-s",  # standalone document
        "-o", str(output_path),
        "--from", "markdown",
        "--to", "docx",
        "--toc",  # tabla de contenidos
        "--toc-depth=3",
        "--number-sections",  # numerar secciones
        "--highlight-style=tango",  # estilo de sintaxis
    ]
    
    # Agregar plantilla si existe
    if template_path.exists():
        pandoc_cmd.extend(["--reference-doc", str(template_path)])
        print(f"✅ Usando plantilla personalizada: {template_path.name}")
    
    # Agregar archivos markdown
    pandoc_cmd.extend(full_paths)
    
    print(f"📄 Generando documento versión {version_str}...")
    print(f"   Archivos a procesar: {len(markdown_files)}")
    
    try:
        # Ejecutar pandoc
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error al ejecutar pandoc:")
            print(result.stderr)
            return None
        
        print(f"✅ Documento generado exitosamente: {output_filename}")
        print(f"   Ubicación: {output_path}")
        
        # Copiar a Google Drive si está montado
        gdrive_path = Path.home() / "Google Drive" / "My Drive" / "Magister Data Science" / "DocFinal"
        
        if gdrive_path.exists():
            try:
                gdrive_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, gdrive_path / output_filename)
                print(f"✅ Documento copiado a Google Drive")
                print(f"   Ubicación: {gdrive_path / output_filename}")
            except Exception as e:
                print(f"⚠️  Advertencia: No se pudo copiar a Google Drive: {e}")
        else:
            print(f"ℹ️  Google Drive no está montado en la ruta esperada")
            print(f"   Ruta esperada: {gdrive_path}")
        
        # Copiar a OneDrive si está montado
        # Posibles rutas de OneDrive en macOS
        onedrive_paths = [
            Path.home() / "OneDrive - Universidad San Sebastian",
            Path.home() / "OneDrive - USS",
            Path.home() / "OneDrive",
            Path.home() / "Library" / "CloudStorage" / "OneDrive-UniversidadSanSebastian",
            Path.home() / "Library" / "CloudStorage" / "OneDrive-USS",
            Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal"
        ]
        
        onedrive_found = False
        for onedrive_base in onedrive_paths:
            if onedrive_base.exists():
                try:
                    # Intentar crear la carpeta de destino
                    onedrive_dest = onedrive_base / "Magister Data Science" / "DocFinal"
                    onedrive_dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, onedrive_dest / output_filename)
                    print(f"✅ Documento copiado a OneDrive")
                    print(f"   Ubicación: {onedrive_dest / output_filename}")
                    onedrive_found = True
                    break
                except Exception as e:
                    # Si falla, intentar la siguiente ruta
                    continue
        
        if not onedrive_found:
            print(f"ℹ️  OneDrive no está montado o no se encontró la ruta correcta")
            print(f"   Rutas verificadas:")
            for path in onedrive_paths:
                print(f"   - {path}")
        
        # Mostrar información adicional
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"\n📊 Información del documento:")
        print(f"   - Tamaño: {file_size:.2f} MB")
        print(f"   - Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   - Versión: {version_str}")
        
        return output_path
        
    except FileNotFoundError:
        print("❌ Error: pandoc no está instalado o no se encuentra en el PATH")
        print("   Instálalo con: brew install pandoc")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def main():
    """Función principal"""
    print("🚀 Generador de Documento Final - Proyecto de Título")
    print("=" * 50)
    
    # Verificar que pandoc esté instalado
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: pandoc no está instalado")
        print("   Instálalo con: brew install pandoc")
        sys.exit(1)
    
    # Generar documento
    result = generate_document()
    
    if result:
        print("\n✨ Proceso completado exitosamente")
    else:
        print("\n❌ El proceso falló")
        sys.exit(1)

if __name__ == "__main__":
    main()