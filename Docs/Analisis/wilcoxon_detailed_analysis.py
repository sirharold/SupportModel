import json
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd

def load_and_analyze_detailed(filepath):
    """Análisis detallado de Wilcoxon con visualización de datos"""
    
    # Cargar datos
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extraer métricas
    models = list(data['results'].keys())
    metric_name = 'precision@5'
    
    print(f"📊 ANÁLISIS DETALLADO PARA {metric_name}")
    print("=" * 80)
    
    # Mostrar valores por modelo
    for model in models:
        values = []
        if 'all_after_metrics' in data['results'][model]:
            for metrics in data['results'][model]['all_after_metrics']:
                if metric_name in metrics:
                    values.append(metrics[metric_name])
        
        print(f"\n📈 {model}:")
        print(f"   Valores: {values}")
        print(f"   Media: {np.mean(values):.4f}")
        print(f"   Mediana: {np.median(values):.4f}")
        print(f"   Std: {np.std(values):.4f}")
        print(f"   Min/Max: {np.min(values):.4f} / {np.max(values):.4f}")
    
    # Comparación detallada ada vs mpnet
    print("\n\n🔬 COMPARACIÓN DETALLADA: ada vs mpnet")
    print("=" * 60)
    
    ada_values = []
    mpnet_values = []
    
    for i, metrics_ada in enumerate(data['results']['ada']['all_after_metrics']):
        metrics_mpnet = data['results']['mpnet']['all_after_metrics'][i]
        
        if metric_name in metrics_ada and metric_name in metrics_mpnet:
            ada_val = metrics_ada[metric_name]
            mpnet_val = metrics_mpnet[metric_name]
            ada_values.append(ada_val)
            mpnet_values.append(mpnet_val)
            
            print(f"Pregunta {i+1}: ada={ada_val:.3f}, mpnet={mpnet_val:.3f}, diff={ada_val-mpnet_val:+.3f}")
    
    # Test de Wilcoxon
    if len(ada_values) >= 5:
        try:
            stat, p_val = wilcoxon(ada_values, mpnet_values, alternative='two-sided')
            print(f"\n📊 Resultado Wilcoxon:")
            print(f"   Estadístico: {stat}")
            print(f"   p-value: {p_val:.4f}")
            print(f"   Significativo (p<0.05): {'SÍ' if p_val < 0.05 else 'NO'}")
            
            # Información adicional
            print(f"\n📌 Interpretación:")
            print(f"   - N muestras: {len(ada_values)}")
            print(f"   - Diferencias positivas: {sum(1 for a, m in zip(ada_values, mpnet_values) if a > m)}")
            print(f"   - Diferencias negativas: {sum(1 for a, m in zip(ada_values, mpnet_values) if a < m)}")
            print(f"   - Sin diferencia: {sum(1 for a, m in zip(ada_values, mpnet_values) if a == m)}")
            
        except Exception as e:
            print(f"\nError: {e}")

    # Análisis de poder estadístico
    print("\n\n🎯 ANÁLISIS DE PODER ESTADÍSTICO")
    print("=" * 60)
    print(f"Con {len(ada_values)} muestras:")
    print("- Para detectar diferencias pequeñas (d=0.2): Necesitas ~50 muestras")
    print("- Para detectar diferencias medianas (d=0.5): Necesitas ~20 muestras")
    print("- Para detectar diferencias grandes (d=0.8): Necesitas ~12 muestras")
    
    effect_size = abs(np.mean(ada_values) - np.mean(mpnet_values)) / np.std(ada_values + mpnet_values)
    print(f"\nTamaño del efecto observado (Cohen's d): {effect_size:.3f}")
    
    if effect_size < 0.2:
        print("→ Efecto muy pequeño")
    elif effect_size < 0.5:
        print("→ Efecto pequeño")
    elif effect_size < 0.8:
        print("→ Efecto mediano")
    else:
        print("→ Efecto grande")

if __name__ == "__main__":
    filepath = '/Users/haroldgomez/Downloads/cumulative_results_20250731_140825.json'
    load_and_analyze_detailed(filepath)