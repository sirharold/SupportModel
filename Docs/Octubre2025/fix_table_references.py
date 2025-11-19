#!/usr/bin/env python3
"""
Corrige las referencias a tablas en los capítulos:
1. Elimina títulos duplicados antes de las tablas (Tabla 2.1:, Tabla 7.3:, etc.)
2. Actualiza referencias en el texto a la numeración correlativa
"""

import re
from pathlib import Path

base_dir = Path("/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025")

# Mapeo de referencias antiguas a nuevas
# Capítulo 2: Tablas 2.1, 2.2, 2.3 → 1, 2, 3
# Capítulo 7: Tablas 7.1-7.11 → 8-18 (aprox, necesito verificar)
# Capítulo 8: Tablas 8.1-8.4 → números finales

table_mapping = {
    # Capítulo 2
    'Tabla 2.1': 'Tabla 1',
    'Tabla 2.2': 'Tabla 2',
    'Tabla 2.3': 'Tabla 3',
    # Capítulo 7
    'Tabla 7.1': 'Tabla 10',
    'Tabla 7.2': 'Tabla 11',
    'Tabla 7.3': 'Tabla 12',
    'Tabla 7.4': 'Tabla 13',
    'Tabla 7.5': 'Tabla 14',
    'Tabla 7.6': 'Tabla 15',
    'Tabla 7.7': 'Tabla 16',
    'Tabla 7.8': 'Tabla 17',
    'Tabla 7.9': 'Tabla 18',
    'Tabla 7.10': 'Tabla 19',
    'Tabla 7.11': 'Tabla 20',
    # Capítulo 8
    'Tabla 8.1': 'Tabla 22',
    'Tabla 8.2': 'Tabla 23',
    'Tabla 8.3': 'Tabla 24',
    'Tabla 8.4': 'Tabla 25'
}

def fix_chapter(filepath, chapter_name):
    """Corrige las referencias de tablas en un capítulo"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # 1. Eliminar títulos duplicados de tablas (líneas como **Tabla 2.1: ...**)
    # que aparecen ANTES de las tablas y que NO son el pie de tabla
    lines = content.split('\n')
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detectar títulos duplicados: **Tabla X.Y: ...**
        # Estos están ANTES de la tabla markdown (que empieza con |)
        if re.match(r'^\*\*Tabla \d+\.\d+:', line):
            # Verificar si la siguiente línea no vacía es una tabla (empieza con |)
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1

            if j < len(lines) and lines[j].strip().startswith('|'):
                # Es un título antes de tabla, NO lo incluimos
                changes.append(f"  ✗ Eliminado título duplicado: {line[:60]}...")
                i += 1
                continue

        new_lines.append(line)
        i += 1

    content = '\n'.join(new_lines)

    # 2. Reemplazar referencias en el texto
    for old_ref, new_ref in table_mapping.items():
        if old_ref in content:
            # Reemplazar solo cuando aparece en el texto (no en títulos de pie de tabla)
            pattern = re.escape(old_ref) + r'(?!:)'  # No seguido de :
            content = re.sub(pattern, new_ref, content)
            changes.append(f"  → Reemplazado: {old_ref} → {new_ref}")

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n{'='*70}")
        print(f"📄 {chapter_name}")
        print(f"{'='*70}")
        for change in changes:
            print(change)
        print(f"✓ Archivo actualizado")
        return True
    else:
        print(f"\n📄 {chapter_name}: Sin cambios necesarios")
        return False

# Procesar capítulos
print("CORRIGIENDO REFERENCIAS DE TABLAS")
print("="*70)

capitulos = [
    ("capitulo_2_estado_del_arte.md", "Capítulo 2 - Estado del Arte"),
    ("capitulo7_resultados.md", "Capítulo 7 - Resultados"),
    ("capitulo_8_conclusiones_y_trabajo_futuro.md", "Capítulo 8 - Conclusiones")
]

total_changes = 0
for filename, name in capitulos:
    filepath = base_dir / filename
    if filepath.exists():
        if fix_chapter(filepath, name):
            total_changes += 1

print("\n" + "="*70)
print(f"✅ Proceso completado")
print(f"   Capítulos modificados: {total_changes}")
print("="*70)
