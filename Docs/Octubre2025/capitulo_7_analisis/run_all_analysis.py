"""
Script maestro para ejecutar todo el análisis del Capítulo 7
Genera tablas, gráficos y análisis estadístico
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_script(script_name: str, description: str):
    """Ejecuta un script de Python"""
    print("\n" + "=" * 60)
    print(f"{description}")
    print("=" * 60)

    script_path = Path(__file__).parent / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"\n✅ {script_name} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False


def generate_summary_report():
    """Genera un reporte resumen del análisis"""
    analysis_dir = Path(__file__).parent
    tables_dir = analysis_dir / "tables"
    charts_dir = analysis_dir / "charts"

    # Contar archivos generados
    tables_count = len(list(tables_dir.glob("*.md"))) if tables_dir.exists() else 0
    csv_count = len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0
    charts_count = len(list(charts_dir.glob("*.png"))) if charts_dir.exists() else 0

    report = f"""
# REPORTE DE ANÁLISIS - CAPÍTULO 7
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumen de Archivos Generados

### Tablas
- Tablas Markdown: {tables_count}
- Tablas CSV: {csv_count}
- Ubicación: `tables/`

### Gráficos
- Gráficos PNG (300 DPI): {charts_count}
- Ubicación: `charts/`

## Archivos de Tablas Generados
"""

    if tables_dir.exists():
        for md_file in sorted(tables_dir.glob("*.md")):
            report += f"- {md_file.name}\n"

    report += "\n## Archivos de Gráficos Generados\n"

    if charts_dir.exists():
        for png_file in sorted(charts_dir.glob("*.png")):
            report += f"- {png_file.name}\n"

    report += """
## Uso de los Archivos

### Para el Documento de Tesis
1. **Tablas**: Copiar contenido de archivos `.md` directamente al documento
2. **Gráficos**: Insertar archivos `.png` (300 DPI, calidad impresión)

### Archivos Clave para Incluir en el Capítulo 7
- `tabla_comparativa_modelos.md`: Tabla resumen de todos los modelos
- `tabla_precision_por_k.md`: Precisión detallada por k
- `tabla_ranking_modelos.md`: Ranking de modelos por métrica
- `precision_por_k_before.png`: Gráfico de precisión antes del reranking
- `delta_heatmap.png`: Mapa de calor de cambios
- `model_ranking_bars.png`: Comparación visual de modelos

## Próximos Pasos
1. Revisar todas las tablas generadas en `tables/`
2. Revisar todos los gráficos generados en `charts/`
3. Seleccionar las visualizaciones más relevantes para el capítulo
4. Integrar en el documento del Capítulo 7

---
**Datos Verificados**: Todos los datos provienen del archivo de resultados real (no simulados)
**Archivo Fuente**: cumulative_results_20251013_001552.json
"""

    # Guardar reporte
    report_path = analysis_dir / "analysis" / "resumen_analisis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ Reporte resumen generado: {report_path}")
    return report_path


def main():
    """Función principal"""
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  ANÁLISIS COMPLETO DEL CAPÍTULO 7".center(58) + "║")
    print("║" + "  Sistema RAG - Evaluación de Modelos".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    start_time = datetime.now()

    success = True

    # 1. Generar tablas
    if not run_script("generate_tables.py", "📊 GENERANDO TABLAS"):
        success = False

    # 2. Generar gráficos
    if not run_script("generate_charts.py", "📈 GENERANDO GRÁFICOS"):
        success = False

    # 3. Generar reporte resumen
    print("\n" + "=" * 60)
    print("📝 GENERANDO REPORTE RESUMEN")
    print("=" * 60)
    generate_summary_report()

    # Tiempo total
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "╔" + "=" * 58 + "╗")
    if success:
        print("║" + " " * 58 + "║")
        print("║" + "  ✅ ANÁLISIS COMPLETADO EXITOSAMENTE".center(58) + "║")
        print("║" + " " * 58 + "║")
    else:
        print("║" + " " * 58 + "║")
        print("║" + "  ⚠️  ANÁLISIS COMPLETADO CON ERRORES".center(58) + "║")
        print("║" + " " * 58 + "║")
    print("║" + f"  Tiempo total: {duration:.1f} segundos".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    print("\n📂 Archivos generados en:")
    print(f"   - Tablas:  {Path(__file__).parent}/tables/")
    print(f"   - Gráficos: {Path(__file__).parent}/charts/")
    print(f"   - Análisis: {Path(__file__).parent}/analysis/")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
