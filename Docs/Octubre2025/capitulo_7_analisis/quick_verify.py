"""
Script de verificación rápida de valores del Capítulo 7
Permite verificar cualquier valor específico de forma interactiva
"""

import json
from pathlib import Path

RESULTS_FILE = "/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/Docs/Octubre2025/cumulative_results_20251013_001552.json"

class QuickVerifier:
    def __init__(self):
        print("Cargando datos...")
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print("✅ Datos cargados\n")

    def show_menu(self):
        """Muestra menú de opciones"""
        print("\n" + "="*80)
        print("VERIFICACIÓN RÁPIDA DE VALORES - CAPÍTULO 7")
        print("="*80)
        print("\nOpciones:")
        print("  1. Ver Precision@k de un modelo")
        print("  2. Ver Recall@k de un modelo")
        print("  3. Ver F1@k de un modelo")
        print("  4. Ver NDCG@k de un modelo")
        print("  5. Ver MAP@k de un modelo")
        print("  6. Ver MRR de un modelo")
        print("  7. Ver TODAS las métricas de un modelo (k=5)")
        print("  8. Comparar dos modelos")
        print("  9. Ver metadatos de evaluación")
        print("  0. Salir")
        print()

    def select_model(self):
        """Permite seleccionar un modelo"""
        print("\nModelos disponibles:")
        print("  1. Ada (text-embedding-ada-002)")
        print("  2. MPNet (multi-qa-mpnet-base-dot-v1)")
        print("  3. MiniLM (all-MiniLM-L6-v2)")
        print("  4. E5-Large (intfloat/e5-large-v2)")

        choice = input("\nSelecciona modelo (1-4): ").strip()
        model_map = {'1': 'ada', '2': 'mpnet', '3': 'minilm', '4': 'e5-large'}
        return model_map.get(choice)

    def select_k(self):
        """Permite seleccionar valor de k"""
        k = input("Ingresa valor de k (1-15): ").strip()
        try:
            k_val = int(k)
            if 1 <= k_val <= 15:
                return k_val
        except:
            pass
        return None

    def show_metric_by_k(self, metric_name: str):
        """Muestra una métrica específica por k para un modelo"""
        model = self.select_model()
        if not model:
            print("❌ Modelo inválido")
            return

        k = self.select_k()
        if not k:
            print("❌ Valor de k inválido")
            return

        model_data = self.data['results'][model]
        before = model_data['avg_before_metrics']
        after = model_data['avg_after_metrics']

        metric_key = f"{metric_name}@{k}"

        print("\n" + "="*80)
        print(f"{metric_name.upper()}@{k} para {model.upper()}")
        print("="*80)

        if metric_key in before:
            before_val = before[metric_key]
            after_val = after.get(metric_key, 0)
            delta = after_val - before_val
            pct = (delta / before_val * 100) if before_val > 0 else 0

            print(f"\n✅ Antes del reranking:   {before_val:.4f}")
            print(f"✅ Después del reranking: {after_val:.4f}")
            print(f"📊 Cambio absoluto:       {delta:+.4f}")
            print(f"📊 Cambio porcentual:     {pct:+.2f}%")

            if abs(delta) < 0.001:
                print("\n💡 El reranking prácticamente NO afecta esta métrica")
            elif delta > 0:
                print("\n📈 El reranking MEJORA esta métrica")
            else:
                print("\n📉 El reranking DEGRADA esta métrica")
        else:
            print(f"\n❌ {metric_key} no encontrado para {model}")

    def show_all_metrics_k5(self):
        """Muestra todas las métricas en k=5 para un modelo"""
        model = self.select_model()
        if not model:
            print("❌ Modelo inválido")
            return

        model_data = self.data['results'][model]
        before = model_data['avg_before_metrics']
        after = model_data['avg_after_metrics']

        print("\n" + "="*80)
        print(f"TODAS LAS MÉTRICAS (k=5) para {model.upper()}")
        print("="*80)

        metrics = ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5', 'mrr']

        print(f"\n{'Métrica':<15} {'Antes':<10} {'Después':<10} {'Cambio':<12} {'% Cambio':<10}")
        print("-" * 80)

        for metric in metrics:
            if metric in before:
                before_val = before[metric]
                after_val = after.get(metric, 0)
                delta = after_val - before_val
                pct = (delta / before_val * 100) if before_val > 0 else 0

                symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                print(f"{symbol} {metric.upper():<12} {before_val:<10.4f} {after_val:<10.4f} {delta:+<12.4f} {pct:+<10.2f}%")

    def compare_models(self):
        """Compara dos modelos"""
        print("\nPrimer modelo:")
        model1 = self.select_model()
        if not model1:
            print("❌ Modelo inválido")
            return

        print("\nSegundo modelo:")
        model2 = self.select_model()
        if not model2:
            print("❌ Modelo inválido")
            return

        k = self.select_k()
        if not k:
            print("❌ Valor de k inválido")
            return

        data1 = self.data['results'][model1]['avg_before_metrics']
        data2 = self.data['results'][model2]['avg_before_metrics']

        print("\n" + "="*80)
        print(f"COMPARACIÓN: {model1.upper()} vs {model2.upper()} (k={k})")
        print("="*80)

        metrics = [f'precision@{k}', f'recall@{k}', f'f1@{k}', f'ndcg@{k}', f'map@{k}']

        print(f"\n{'Métrica':<15} {model1.upper():<10} {model2.upper():<10} {'Diferencia':<12} {'% Diff':<10}")
        print("-" * 80)

        for metric in metrics:
            if metric in data1 and metric in data2:
                val1 = data1[metric]
                val2 = data2[metric]
                diff = val2 - val1
                pct_diff = (diff / val1 * 100) if val1 > 0 else 0

                symbol = "✅" if val1 > val2 else "❌" if val1 < val2 else "➡️"
                print(f"{symbol} {metric.upper():<12} {val1:<10.4f} {val2:<10.4f} {diff:+<12.4f} {pct_diff:+<10.2f}%")

        # Resumen
        print("\n" + "-"*80)
        if val1 > val2:
            print(f"🏆 GANADOR: {model1.upper()}")
        else:
            print(f"🏆 GANADOR: {model2.upper()}")

    def show_metadata(self):
        """Muestra metadatos de la evaluación"""
        eval_info = self.data.get('evaluation_info', {})

        print("\n" + "="*80)
        print("METADATOS DE EVALUACIÓN")
        print("="*80)

        print(f"\n📊 Preguntas evaluadas: {eval_info.get('num_questions', 'N/A')}")
        print(f"📁 Archivo de resultados: cumulative_results_20251013_001552.json")
        print(f"✅ Datos reales: {eval_info.get('data_verification', {}).get('is_real_data', False)}")

        print("\n🤖 Modelos evaluados:")
        for model_key in self.data['results'].keys():
            model_data = self.data['results'][model_key]
            num_q = model_data.get('num_questions_evaluated', 0)
            print(f"   - {model_key}: {num_q} preguntas")

    def run(self):
        """Ejecuta el verificador interactivo"""
        while True:
            self.show_menu()
            choice = input("Selecciona opción: ").strip()

            if choice == '0':
                print("\n👋 ¡Hasta luego!")
                break
            elif choice == '1':
                self.show_metric_by_k('precision')
            elif choice == '2':
                self.show_metric_by_k('recall')
            elif choice == '3':
                self.show_metric_by_k('f1')
            elif choice == '4':
                self.show_metric_by_k('ndcg')
            elif choice == '5':
                self.show_metric_by_k('map')
            elif choice == '6':
                model = self.select_model()
                if model:
                    data = self.data['results'][model]
                    before = data['avg_before_metrics'].get('mrr', 0)
                    after = data['avg_after_metrics'].get('mrr', 0)
                    print(f"\nMRR antes: {before:.4f}")
                    print(f"MRR después: {after:.4f}")
                    print(f"Cambio: {after-before:+.4f}")
            elif choice == '7':
                self.show_all_metrics_k5()
            elif choice == '8':
                self.compare_models()
            elif choice == '9':
                self.show_metadata()
            else:
                print("❌ Opción inválida")

            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    verifier = QuickVerifier()
    verifier.run()
