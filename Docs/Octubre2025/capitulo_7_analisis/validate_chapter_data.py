"""
Script de Validación de Datos del Capítulo 7
Verifica que todos los valores numéricos mencionados en el capítulo
coincidan exactamente con los datos del archivo de resultados.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

# Ruta al archivo de resultados
RESULTS_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json"

class ChapterValidator:
    def __init__(self):
        self.results = self.load_results()
        self.errors = []
        self.warnings = []
        self.validations = 0

    def load_results(self) -> Dict:
        """Carga el archivo de resultados"""
        print(f"📂 Cargando resultados desde: {RESULTS_FILE}")
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verificar que son datos reales
        if not data.get('evaluation_info', {}).get('data_verification', {}).get('is_real_data', False):
            raise ValueError("⚠️ ADVERTENCIA CRÍTICA: Los datos NO están marcados como reales")

        print("✅ Datos verificados como REALES\n")
        return data

    def validate_value(self, description: str, expected: float, actual: float, tolerance: float = 0.0005):
        """Valida que un valor esperado coincida con el valor real"""
        self.validations += 1

        if abs(expected - actual) > tolerance:
            self.errors.append({
                'description': description,
                'expected': expected,
                'actual': actual,
                'difference': abs(expected - actual)
            })
            print(f"❌ ERROR: {description}")
            print(f"   Esperado: {expected:.4f}, Real: {actual:.4f}, Diferencia: {abs(expected - actual):.4f}")
            return False
        else:
            print(f"✅ OK: {description} = {actual:.4f}")
            return True

    def validate_tabla_7_1(self):
        """Valida Tabla 7.1: Métricas Principales de Ada"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.1: Métricas Principales de Ada (k=3,5,10,15)")
        print("="*80)

        ada = self.results['results']['ada']
        before = ada['avg_before_metrics']
        after = ada['avg_after_metrics']

        # Precision
        print("\n📊 Validando Precision@k...")
        self.validate_value("Precision@3 Antes", 0.111, before.get('precision@3', 0))
        self.validate_value("Precision@3 Después", 0.089, after.get('precision@3', 0))
        self.validate_value("Precision@5 Antes", 0.098, before.get('precision@5', 0))
        self.validate_value("Precision@5 Después", 0.081, after.get('precision@5', 0))
        self.validate_value("Precision@10 Antes", 0.074, before.get('precision@10', 0))
        self.validate_value("Precision@10 Después", 0.068, after.get('precision@10', 0))
        self.validate_value("Precision@15 Antes", 0.061, before.get('precision@15', 0))
        self.validate_value("Precision@15 Después", 0.061, after.get('precision@15', 0))

        # Recall
        print("\n📊 Validando Recall@k...")
        self.validate_value("Recall@3 Antes", 0.276, before.get('recall@3', 0))
        self.validate_value("Recall@3 Después", 0.219, after.get('recall@3', 0))
        self.validate_value("Recall@5 Antes", 0.398, before.get('recall@5', 0))
        self.validate_value("Recall@5 Después", 0.330, after.get('recall@5', 0))
        self.validate_value("Recall@10 Antes", 0.591, before.get('recall@10', 0))
        self.validate_value("Recall@10 Después", 0.546, after.get('recall@10', 0))
        self.validate_value("Recall@15 Antes", 0.729, before.get('recall@15', 0))
        self.validate_value("Recall@15 Después", 0.729, after.get('recall@15', 0))

        # F1
        print("\n📊 Validando F1@k...")
        self.validate_value("F1@3 Antes", 0.153, before.get('f1@3', 0))
        self.validate_value("F1@3 Después", 0.122, after.get('f1@3', 0))
        self.validate_value("F1@5 Antes", 0.152, before.get('f1@5', 0))
        self.validate_value("F1@5 Después", 0.127, after.get('f1@5', 0))
        self.validate_value("F1@10 Antes", 0.129, before.get('f1@10', 0))
        self.validate_value("F1@10 Después", 0.118, after.get('f1@10', 0))
        self.validate_value("F1@15 Antes", 0.111, before.get('f1@15', 0))
        self.validate_value("F1@15 Después", 0.111, after.get('f1@15', 0))

        # NDCG
        print("\n📊 Validando NDCG@k...")
        self.validate_value("NDCG@3 Antes", 0.209, before.get('ndcg@3', 0))
        self.validate_value("NDCG@3 Después", 0.173, after.get('ndcg@3', 0))
        self.validate_value("NDCG@5 Antes", 0.234, before.get('ndcg@5', 0))
        self.validate_value("NDCG@5 Después", 0.202, after.get('ndcg@5', 0))
        self.validate_value("NDCG@10 Antes", 0.260, before.get('ndcg@10', 0))
        self.validate_value("NDCG@10 Después", 0.234, after.get('ndcg@10', 0))
        self.validate_value("NDCG@15 Antes", 0.271, before.get('ndcg@15', 0))
        self.validate_value("NDCG@15 Después", 0.250, after.get('ndcg@15', 0))

        # MAP
        print("\n📊 Validando MAP@k...")
        self.validate_value("MAP@3 Antes", 0.211, before.get('map@3', 0))
        self.validate_value("MAP@3 Después", 0.160, after.get('map@3', 0))
        self.validate_value("MAP@5 Antes", 0.263, before.get('map@5', 0))
        self.validate_value("MAP@5 Después", 0.201, after.get('map@5', 0))
        self.validate_value("MAP@10 Antes", 0.317, before.get('map@10', 0))
        self.validate_value("MAP@10 Después", 0.256, after.get('map@10', 0))
        self.validate_value("MAP@15 Antes", 0.344, before.get('map@15', 0))
        self.validate_value("MAP@15 Después", 0.291, after.get('map@15', 0))

    def validate_tabla_7_2(self):
        """Valida Tabla 7.2: Precision@k de Ada"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.2: Precision@k de Ada (k=3,5,10,15)")
        print("="*80)

        ada = self.results['results']['ada']
        before = ada['avg_before_metrics']
        after = ada['avg_after_metrics']

        # Nota: La tabla tiene valores ligeramente diferentes a tabla 7.1 - verificar
        print("\n⚠️  ADVERTENCIA: Tabla 7.2 tiene valores que difieren de Tabla 7.1")
        print("    Validando contra los datos reales...")

        self.validate_value("Ada Precision@3 Antes (Tabla 7.2)", 0.104, before.get('precision@3', 0))
        self.validate_value("Ada Precision@5 Antes (Tabla 7.2)", 0.098, before.get('precision@5', 0))
        self.validate_value("Ada Precision@10 Antes (Tabla 7.2)", 0.079, before.get('precision@10', 0))
        self.validate_value("Ada Precision@15 Antes (Tabla 7.2)", 0.061, before.get('precision@15', 0))

        self.validate_value("Ada Precision@3 Después (Tabla 7.2)", 0.086, after.get('precision@3', 0))
        self.validate_value("Ada Precision@5 Después (Tabla 7.2)", 0.081, after.get('precision@5', 0))
        self.validate_value("Ada Precision@10 Después (Tabla 7.2)", 0.067, after.get('precision@10', 0))
        self.validate_value("Ada Precision@15 Después (Tabla 7.2)", 0.053, after.get('precision@15', 0))

    def validate_tabla_7_3(self):
        """Valida Tabla 7.3: Recall@k de Ada"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.3: Recall@k de Ada (k=3,5,10,15)")
        print("="*80)

        ada = self.results['results']['ada']
        before = ada['avg_before_metrics']
        after = ada['avg_after_metrics']

        self.validate_value("Ada Recall@3 Antes", 0.276, before.get('recall@3', 0))
        self.validate_value("Ada Recall@5 Antes", 0.398, before.get('recall@5', 0))
        self.validate_value("Ada Recall@10 Antes", 0.591, before.get('recall@10', 0))
        self.validate_value("Ada Recall@15 Antes", 0.702, before.get('recall@15', 0))

        self.validate_value("Ada Recall@3 Después", 0.228, after.get('recall@3', 0))
        self.validate_value("Ada Recall@5 Después", 0.330, after.get('recall@5', 0))
        self.validate_value("Ada Recall@10 Después", 0.539, after.get('recall@10', 0))
        self.validate_value("Ada Recall@15 Después", 0.649, after.get('recall@15', 0))

    def validate_tabla_7_4(self):
        """Valida Tabla 7.4: MPNet Métricas Principales (k=5)"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.4: Métricas Principales de MPNet (k=5)")
        print("="*80)

        mpnet = self.results['results']['mpnet']
        before = mpnet['avg_before_metrics']
        after = mpnet['avg_after_metrics']

        self.validate_value("MPNet Precision@5 Antes", 0.070, before.get('precision@5', 0))
        self.validate_value("MPNet Precision@5 Después", 0.067, after.get('precision@5', 0))

        self.validate_value("MPNet Recall@5 Antes", 0.277, before.get('recall@5', 0))
        self.validate_value("MPNet Recall@5 Después", 0.264, after.get('recall@5', 0))

        self.validate_value("MPNet F1@5 Antes", 0.108, before.get('f1@5', 0))
        self.validate_value("MPNet F1@5 Después", 0.103, after.get('f1@5', 0))

        self.validate_value("MPNet NDCG@5 Antes", 0.193, before.get('ndcg@5', 0))
        self.validate_value("MPNet NDCG@5 Después", 0.185, after.get('ndcg@5', 0))

        self.validate_value("MPNet MAP@5 Antes", 0.174, before.get('map@5', 0))
        self.validate_value("MPNet MAP@5 Después", 0.161, after.get('map@5', 0))

        self.validate_value("MPNet MRR Antes", 0.184, before.get('mrr', 0))
        self.validate_value("MPNet MRR Después", 0.177, after.get('mrr', 0))

    def validate_tabla_7_5(self):
        """Valida Tabla 7.5: Comparación Ada vs MPNet (k=5)"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.5: Comparación Ada vs MPNet (k=5, Antes Reranking)")
        print("="*80)

        ada_before = self.results['results']['ada']['avg_before_metrics']
        mpnet_before = self.results['results']['mpnet']['avg_before_metrics']

        self.validate_value("Ada Precision@5", 0.098, ada_before.get('precision@5', 0))
        self.validate_value("MPNet Precision@5", 0.070, mpnet_before.get('precision@5', 0))

        self.validate_value("Ada Recall@5", 0.398, ada_before.get('recall@5', 0))
        self.validate_value("MPNet Recall@5", 0.277, mpnet_before.get('recall@5', 0))

        self.validate_value("Ada F1@5", 0.152, ada_before.get('f1@5', 0))
        self.validate_value("MPNet F1@5", 0.108, mpnet_before.get('f1@5', 0))

        self.validate_value("Ada NDCG@5", 0.234, ada_before.get('ndcg@5', 0))
        self.validate_value("MPNet NDCG@5", 0.193, mpnet_before.get('ndcg@5', 0))

        self.validate_value("Ada MAP@5", 0.263, ada_before.get('map@5', 0))
        self.validate_value("MPNet MAP@5", 0.174, mpnet_before.get('map@5', 0))

        self.validate_value("Ada MRR", 0.222, ada_before.get('mrr', 0))
        self.validate_value("MPNet MRR", 0.184, mpnet_before.get('mrr', 0))

    def validate_tabla_7_6(self):
        """Valida Tabla 7.6: MiniLM Métricas Principales (k=5)"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.6: Métricas Principales de MiniLM (k=5)")
        print("="*80)

        minilm = self.results['results']['minilm']
        before = minilm['avg_before_metrics']
        after = minilm['avg_after_metrics']

        self.validate_value("MiniLM Precision@5 Antes", 0.053, before.get('precision@5', 0))
        self.validate_value("MiniLM Precision@5 Después", 0.060, after.get('precision@5', 0))

        self.validate_value("MiniLM Recall@5 Antes", 0.211, before.get('recall@5', 0))
        self.validate_value("MiniLM Recall@5 Después", 0.236, after.get('recall@5', 0))

        self.validate_value("MiniLM F1@5 Antes", 0.082, before.get('f1@5', 0))
        self.validate_value("MiniLM F1@5 Después", 0.093, after.get('f1@5', 0))

        self.validate_value("MiniLM NDCG@5 Antes", 0.150, before.get('ndcg@5', 0))
        self.validate_value("MiniLM NDCG@5 Después", 0.169, after.get('ndcg@5', 0))

        self.validate_value("MiniLM MAP@5 Antes", 0.132, before.get('map@5', 0))
        self.validate_value("MiniLM MAP@5 Después", 0.147, after.get('map@5', 0))

        self.validate_value("MiniLM MRR Antes", 0.145, before.get('mrr', 0))
        self.validate_value("MiniLM MRR Después", 0.159, after.get('mrr', 0))

    def validate_tabla_7_7(self):
        """Valida Tabla 7.7: Precision@k de MiniLM"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.7: Precision@k de MiniLM (k=3,5,10,15)")
        print("="*80)

        minilm = self.results['results']['minilm']
        before = minilm['avg_before_metrics']
        after = minilm['avg_after_metrics']

        self.validate_value("MiniLM Precision@3 Antes", 0.056, before.get('precision@3', 0))
        self.validate_value("MiniLM Precision@5 Antes", 0.053, before.get('precision@5', 0))
        self.validate_value("MiniLM Precision@10 Antes", 0.046, before.get('precision@10', 0))
        self.validate_value("MiniLM Precision@15 Antes", 0.040, before.get('precision@15', 0))

        self.validate_value("MiniLM Precision@3 Después", 0.063, after.get('precision@3', 0))
        self.validate_value("MiniLM Precision@5 Después", 0.060, after.get('precision@5', 0))
        self.validate_value("MiniLM Precision@10 Después", 0.052, after.get('precision@10', 0))
        self.validate_value("MiniLM Precision@15 Después", 0.045, after.get('precision@15', 0))

    def validate_tabla_7_8(self):
        """Valida Tabla 7.8: E5-Large Métricas Principales (k=5)"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.8: Métricas Principales de E5-Large (k=5)")
        print("="*80)

        e5large = self.results['results']['e5-large']
        before = e5large['avg_before_metrics']
        after = e5large['avg_after_metrics']

        self.validate_value("E5-Large Precision@5 Antes", 0.065, before.get('precision@5', 0))
        self.validate_value("E5-Large Precision@5 Después", 0.066, after.get('precision@5', 0))

        self.validate_value("E5-Large Recall@5 Antes", 0.262, before.get('recall@5', 0))
        self.validate_value("E5-Large Recall@5 Después", 0.263, after.get('recall@5', 0))

        self.validate_value("E5-Large F1@5 Antes", 0.100, before.get('f1@5', 0))
        self.validate_value("E5-Large F1@5 Después", 0.101, after.get('f1@5', 0))

        self.validate_value("E5-Large NDCG@5 Antes", 0.172, before.get('ndcg@5', 0))
        self.validate_value("E5-Large NDCG@5 Después", 0.171, after.get('ndcg@5', 0))

        self.validate_value("E5-Large MAP@5 Antes", 0.158, before.get('map@5', 0))
        self.validate_value("E5-Large MAP@5 Después", 0.164, after.get('map@5', 0))

        self.validate_value("E5-Large MRR Antes", 0.156, before.get('mrr', 0))
        self.validate_value("E5-Large MRR Después", 0.158, after.get('mrr', 0))

    def validate_tabla_7_9(self):
        """Valida Tabla 7.9: Comparación Modelos Open-Source (k=5)"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.9: Comparación Modelos Open-Source (k=5, Antes Reranking)")
        print("="*80)

        mpnet_before = self.results['results']['mpnet']['avg_before_metrics']
        e5large_before = self.results['results']['e5-large']['avg_before_metrics']
        minilm_before = self.results['results']['minilm']['avg_before_metrics']

        print("\n📊 MPNet:")
        self.validate_value("MPNet Precision@5", 0.070, mpnet_before.get('precision@5', 0))
        self.validate_value("MPNet Recall@5", 0.277, mpnet_before.get('recall@5', 0))
        self.validate_value("MPNet F1@5", 0.108, mpnet_before.get('f1@5', 0))
        self.validate_value("MPNet NDCG@5", 0.193, mpnet_before.get('ndcg@5', 0))

        print("\n📊 E5-Large:")
        self.validate_value("E5-Large Precision@5", 0.065, e5large_before.get('precision@5', 0))
        self.validate_value("E5-Large Recall@5", 0.262, e5large_before.get('recall@5', 0))
        self.validate_value("E5-Large F1@5", 0.100, e5large_before.get('f1@5', 0))
        self.validate_value("E5-Large NDCG@5", 0.172, e5large_before.get('ndcg@5', 0))

        print("\n📊 MiniLM:")
        self.validate_value("MiniLM Precision@5", 0.053, minilm_before.get('precision@5', 0))
        self.validate_value("MiniLM Recall@5", 0.211, minilm_before.get('recall@5', 0))
        self.validate_value("MiniLM F1@5", 0.082, minilm_before.get('f1@5', 0))
        self.validate_value("MiniLM NDCG@5", 0.150, minilm_before.get('ndcg@5', 0))

    def validate_metadata(self):
        """Valida metadatos mencionados en el capítulo"""
        print("\n" + "="*80)
        print("VALIDANDO METADATOS Y CONFIGURACIÓN EXPERIMENTAL")
        print("="*80)

        eval_info = self.results.get('evaluation_info', {})

        # Número de preguntas
        num_questions = eval_info.get('num_questions', 0)
        print(f"\n📊 Número de preguntas evaluadas: {num_questions}")
        if num_questions != 2067:
            self.warnings.append(f"Número de preguntas: {num_questions} (esperado: 2,067)")

        # Verificar cada modelo tiene el número correcto de preguntas
        for model in ['ada', 'mpnet', 'minilm', 'e5-large']:
            if model in self.results['results']:
                model_questions = self.results['results'][model]['num_questions_evaluated']
                print(f"   {model}: {model_questions} preguntas")
                if model_questions != 2067:
                    self.warnings.append(f"{model}: {model_questions} preguntas (esperado: 2,067)")

    def validate_tabla_7_11(self):
        """Valida Tabla 7.11: Cambio Promedio por Métrica"""
        print("\n" + "="*80)
        print("VALIDANDO TABLA 7.11: Cambio Promedio por Métrica Debido al Reranking")
        print("="*80)

        models = ['ada', 'mpnet', 'minilm', 'e5-large']
        metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']

        for model in models:
            print(f"\n📊 Validando {model.upper()}:")
            model_data = self.results['results'][model]
            before = model_data['avg_before_metrics']
            after = model_data['avg_after_metrics']

            for metric in metrics:
                if metric in before and metric in after:
                    before_val = before[metric]
                    after_val = after[metric]
                    delta = after_val - before_val
                    pct_change = (delta / before_val * 100) if before_val > 0 else 0

                    print(f"  {metric}: {pct_change:+.1f}%")

    def generate_report(self):
        """Genera reporte de validación"""
        print("\n" + "="*80)
        print("REPORTE FINAL DE VALIDACIÓN")
        print("="*80)

        print(f"\n✅ Total de validaciones realizadas: {self.validations}")
        print(f"❌ Errores encontrados: {len(self.errors)}")
        print(f"⚠️  Advertencias: {len(self.warnings)}")

        if self.errors:
            print("\n" + "="*80)
            print("ERRORES DETECTADOS:")
            print("="*80)
            for i, error in enumerate(self.errors, 1):
                print(f"\n{i}. {error['description']}")
                print(f"   Esperado: {error['expected']:.4f}")
                print(f"   Actual:   {error['actual']:.4f}")
                print(f"   Diferencia: {error['difference']:.4f}")

        if self.warnings:
            print("\n" + "="*80)
            print("ADVERTENCIAS:")
            print("="*80)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n🎉 ¡VALIDACIÓN EXITOSA! Todos los datos del capítulo son correctos y coinciden con los resultados reales.")

        # Guardar reporte
        report_path = Path(__file__).parent / "validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE VALIDACIÓN - CAPÍTULO 7\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total de validaciones: {self.validations}\n")
            f.write(f"Errores: {len(self.errors)}\n")
            f.write(f"Advertencias: {len(self.warnings)}\n\n")

            if self.errors:
                f.write("\nERRORES:\n")
                f.write("-"*80 + "\n")
                for error in self.errors:
                    f.write(f"\n{error['description']}\n")
                    f.write(f"  Esperado: {error['expected']:.4f}\n")
                    f.write(f"  Actual:   {error['actual']:.4f}\n")
                    f.write(f"  Diferencia: {error['difference']:.4f}\n")

            if self.warnings:
                f.write("\nADVERTENCIAS:\n")
                f.write("-"*80 + "\n")
                for warning in self.warnings:
                    f.write(f"- {warning}\n")

        print(f"\n📄 Reporte guardado en: {report_path}")

        return len(self.errors) == 0


def main():
    """Ejecuta todas las validaciones"""
    print("="*80)
    print("VALIDACIÓN SISTEMÁTICA DE DATOS DEL CAPÍTULO 7")
    print("="*80)
    print()

    validator = ChapterValidator()

    # Validar todas las tablas
    validator.validate_metadata()
    validator.validate_tabla_7_1()
    validator.validate_tabla_7_2()
    validator.validate_tabla_7_3()
    validator.validate_tabla_7_4()
    validator.validate_tabla_7_5()
    validator.validate_tabla_7_6()
    validator.validate_tabla_7_7()
    validator.validate_tabla_7_8()
    validator.validate_tabla_7_9()
    validator.validate_tabla_7_11()

    # Generar reporte final
    success = validator.generate_report()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
