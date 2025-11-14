"""
Verifica que todas las figuras mencionadas en el capítulo existan
y lista qué figuras faltan o necesitan ser creadas
"""

import re
from pathlib import Path

CHAPTER_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/capitulo7_resultados.md"
CHARTS_DIR = Path(__file__).parent / "charts"

def extract_figure_references(chapter_path: str):
    """Extrae todas las referencias a figuras del capítulo"""
    with open(chapter_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Patrón para encontrar referencias a figuras
    # Formato: ![Figura X.Y: descripción](ruta)
    pattern = r'!\[Figura\s+([\d.]+):\s*([^\]]+)\]\(([^\)]+)\)'
    matches = re.findall(pattern, content)

    figures = []
    for num, desc, path in matches:
        figures.append({
            'number': num,
            'description': desc,
            'path': path,
            'filename': Path(path).name
        })

    return figures

def check_figure_existence(figures):
    """Verifica si las figuras existen físicamente"""
    results = {
        'existing': [],
        'missing': [],
        'mentioned_not_referenced': []
    }

    for fig in figures:
        # Construir ruta completa
        full_path = CHARTS_DIR / fig['filename']

        if full_path.exists():
            results['existing'].append(fig)
        else:
            results['missing'].append(fig)

    # Verificar si hay figuras en el directorio que no se mencionan
    if CHARTS_DIR.exists():
        all_charts = set(p.name for p in CHARTS_DIR.glob('*.png'))
        referenced_charts = set(f['filename'] for f in figures)
        unreferenced = all_charts - referenced_charts

        for chart in unreferenced:
            results['mentioned_not_referenced'].append(chart)

    return results

def generate_figure_report():
    """Genera reporte de figuras"""
    print("="*80)
    print("VERIFICACIÓN DE FIGURAS - CAPÍTULO 7")
    print("="*80)
    print()

    figures = extract_figure_references(CHAPTER_FILE)
    results = check_figure_existence(figures)

    print(f"📊 Total de figuras referenciadas en el capítulo: {len(figures)}")
    print(f"✅ Figuras existentes: {len(results['existing'])}")
    print(f"❌ Figuras faltantes: {len(results['missing'])}")
    print(f"⚠️  Gráficos no referenciados: {len(results['mentioned_not_referenced'])}")
    print()

    # Reporte detallado
    report = []
    report.append("# VERIFICACIÓN DE FIGURAS - CAPÍTULO 7\n\n")
    report.append("="*80 + "\n\n")

    # Figuras existentes
    if results['existing']:
        report.append("## ✅ FIGURAS EXISTENTES Y CORRECTAS\n\n")
        for fig in results['existing']:
            report.append(f"### Figura {fig['number']}\n")
            report.append(f"**Descripción**: {fig['description']}\n")
            report.append(f"**Archivo**: `{fig['filename']}`\n")
            report.append(f"**Ruta**: `{fig['path']}`\n")
            report.append("✅ Archivo existe\n\n")

    # Figuras faltantes
    if results['missing']:
        report.append("## ❌ FIGURAS FALTANTES - CREAR ESTAS IMÁGENES\n\n")
        for fig in results['missing']:
            report.append(f"### Figura {fig['number']}: {fig['description']}\n")
            report.append(f"**Archivo esperado**: `{fig['filename']}`\n")
            report.append(f"**Ruta esperada**: `{fig['path']}`\n")
            report.append("❌ **ACCIÓN REQUERIDA**: Crear esta imagen\n\n")

            # Instrucciones específicas según el tipo de figura
            if 'comparison' in fig['filename'].lower():
                report.append("**INSTRUCCIONES PARA CREAR LA IMAGEN**:\n")
                report.append("```python\n")
                report.append("# Usar el script generate_charts.py\n")
                report.append(f"# La imagen debería mostrarse comparación antes/después para el modelo\n")
                report.append("```\n\n")
            elif 'heatmap' in fig['filename'].lower():
                report.append("**INSTRUCCIONES PARA CREAR LA IMAGEN**:\n")
                report.append("```python\n")
                report.append("# Crear heatmap de cambios porcentuales\n")
                report.append("# Filas: Modelos (Ada, MPNet, MiniLM, E5-Large)\n")
                report.append("# Columnas: Métricas (Precision, Recall, F1, NDCG, MAP, MRR)\n")
                report.append("# Valores: Cambio % antes→después del reranking\n")
                report.append("```\n\n")
            elif 'ranking' in fig['filename'].lower():
                report.append("**INSTRUCCIONES PARA CREAR LA IMAGEN**:\n")
                report.append("```python\n")
                report.append("# Crear gráfico de barras agrupadas\n")
                report.append("# Eje X: Métricas (Precision@5, Recall@5, F1@5, NDCG@5, MAP@5)\n")
                report.append("# Eje Y: Valor de la métrica\n")
                report.append("# Barras: Antes/Después para cada modelo\n")
                report.append("```\n\n")
            else:
                report.append("**INSTRUCCIONES PARA CREAR LA IMAGEN**:\n")
                report.append("Ejecutar: `python generate_charts.py` en la carpeta de análisis\n\n")

    # Gráficos no referenciados
    if results['mentioned_not_referenced']:
        report.append("## ⚠️ GRÁFICOS DISPONIBLES NO REFERENCIADOS EN EL CAPÍTULO\n\n")
        report.append("Estos gráficos existen pero no se mencionan en el capítulo:\n\n")
        for chart in sorted(results['mentioned_not_referenced']):
            report.append(f"- `{chart}`\n")
        report.append("\n")
        report.append("**RECOMENDACIÓN**: Considerar si alguno de estos gráficos debería incluirse en el capítulo.\n\n")

    # Guardar reporte
    report_path = Path(__file__).parent / "FIGURAS_VERIFICACION.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"📄 Reporte de figuras guardado en: {report_path}")
    print()

    # Imprimir resumen de acciones
    if results['missing']:
        print("🔧 ACCIONES REQUERIDAS:")
        print()
        for fig in results['missing']:
            print(f"   ❌ Crear: {fig['filename']}")
            print(f"      Descripción: {fig['description']}")
            print()

    return report_path

if __name__ == "__main__":
    generate_figure_report()
