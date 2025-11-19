#!/usr/bin/env python3
"""
Prepara archivos markdown asegurando que las tablas tengan bordes cuando se conviertan a Word.
Las tablas markdown pipe se envuelven en divs con estilo TableGrid para forzar bordes.
"""

import re
from pathlib import Path

base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")
temp_dir = base_dir / "temp"

# Archivos a procesar
archivos = [
    "capitulo_0_resumen_espanol.md",
    "capitulo_0_abstract_ingles.md",
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

def process_tables_with_borders(content):
    """
    Envuelve tablas markdown en divs con custom-style="Table Grid" para forzar bordes.
    Esto asegura que pandoc genere tablas con bordes visibles en Word.
    """
    lines = content.split('\n')
    result = []
    in_table = False
    table_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detectar inicio de tabla (línea que empieza con |)
        if line.strip().startswith('|') and not in_table:
            in_table = True
            table_lines = [line]
            i += 1
            continue

        # Si estamos en tabla, seguir recolectando líneas
        if in_table:
            if line.strip().startswith('|'):
                table_lines.append(line)
                i += 1
                continue
            else:
                # Fin de tabla - envolver en div
                result.append('')
                result.append('::: {custom-style="Table Grid"}')
                result.extend(table_lines)
                result.append(':::')
                result.append('')
                in_table = False
                table_lines = []
                # No incrementar i, procesar esta línea normalmente

        # Línea normal (no tabla)
        result.append(line)
        i += 1

    # Si terminamos dentro de una tabla
    if in_table and table_lines:
        result.append('')
        result.append('::: {custom-style="Table Grid"}')
        result.extend(table_lines)
        result.append(':::')
        result.append('')

    return '\n'.join(result)

print("PREPARANDO ARCHIVOS MARKDOWN CON BORDES EN TABLAS")
print("=" * 70)

stats = {
    'archivos_procesados': 0,
    'archivos_no_encontrados': 0
}

for archivo in archivos:
    filepath = base_dir / archivo

    if not filepath.exists():
        print(f"⚠️  No encontrado: {archivo}")
        stats['archivos_no_encontrados'] += 1
        continue

    # Leer contenido
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Procesar tablas
    content_with_borders = process_tables_with_borders(content)

    # Guardar en temp/
    output_path = temp_dir / archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content_with_borders)

    print(f"✓ Procesado: {archivo} → temp/{archivo}")
    stats['archivos_procesados'] += 1

print(f"\n{'='*70}")
print(f"✅ ARCHIVOS PREPARADOS EN temp/")
print(f"{'='*70}")
print(f"   Archivos procesados: {stats['archivos_procesados']}")
print(f"   No encontrados: {stats['archivos_no_encontrados']}")
print(f"\n   Las tablas están envueltas en ::: {{custom-style=\"Table Grid\"}} :::")
print(f"   Esto forzará bordes visibles cuando se conviertan a Word.")
print(f"{'='*70}")
