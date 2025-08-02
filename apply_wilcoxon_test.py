import json
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd

def load_results(filepath):
    """Carga los resultados del archivo JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_question_metrics(results):
    """Extrae las métricas por pregunta para cada modelo"""
    question_metrics = {}
    
    for model in results['results']:
        model_data = results['results'][model]
        
        # Los datos están en all_after_metrics (o all_before_metrics si no hay after)
        if 'all_after_metrics' in model_data:
            question_metrics[model] = model_data['all_after_metrics']
        elif 'all_before_metrics' in model_data:
            question_metrics[model] = model_data['all_before_metrics']
    
    return question_metrics

def perform_wilcoxon_tests(question_metrics, metric_name='precision@5'):
    """
    Realiza tests de Wilcoxon para comparar todos los pares de modelos
    para una métrica específica
    """
    models = list(question_metrics.keys())
    results = []
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            
            # all_after_metrics es una lista de dicts, cada dict corresponde a una pregunta
            metrics_list1 = question_metrics[model1]
            metrics_list2 = question_metrics[model2]
            
            # Verificar que tenemos la misma cantidad de preguntas
            if len(metrics_list1) != len(metrics_list2):
                print(f"⚠️  Diferente número de preguntas entre {model1} y {model2}")
                continue
            
            # Extraer valores de la métrica para cada pregunta
            values1 = []
            values2 = []
            
            for idx in range(len(metrics_list1)):
                if metric_name in metrics_list1[idx] and metric_name in metrics_list2[idx]:
                    values1.append(metrics_list1[idx][metric_name])
                    values2.append(metrics_list2[idx][metric_name])
            
            if len(values1) < 5:
                print(f"⚠️  Insuficientes datos para {metric_name} entre {model1} y {model2}")
                continue
            
            # Realizar test de Wilcoxon
            try:
                # Si todas las diferencias son cero, el test fallará
                if all(v1 == v2 for v1, v2 in zip(values1, values2)):
                    print(f"⚠️  Todos los valores son idénticos entre {model1} y {model2} para {metric_name}")
                    continue
                
                statistic, p_value = wilcoxon(values1, values2, 
                                            alternative='two-sided',
                                            zero_method='wilcox')
                
                results.append({
                    'model1': model1,
                    'model2': model2,
                    'metric': metric_name,
                    'n_samples': len(values1),
                    'mean_model1': np.mean(values1),
                    'mean_model2': np.mean(values2),
                    'median_model1': np.median(values1),
                    'median_model2': np.median(values2),
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
                
            except Exception as e:
                print(f"Error en test {model1} vs {model2}: {e}")
    
    return results

def display_results(results):
    """Muestra los resultados de forma legible"""
    df = pd.DataFrame(results)
    
    print("\n📊 RESULTADOS DE TESTS DE WILCOXON")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\n🔍 {row['model1']} vs {row['model2']}")
        print(f"   Métrica: {row['metric']}")
        print(f"   N muestras: {row['n_samples']}")
        print(f"   Media {row['model1']}: {row['mean_model1']:.4f}")
        print(f"   Media {row['model2']}: {row['mean_model2']:.4f}")
        print(f"   Mediana {row['model1']}: {row['median_model1']:.4f}")
        print(f"   Mediana {row['model2']}: {row['median_model2']:.4f}")
        print(f"   p-value: {row['p_value']:.4f}")
        
        if row['significant']:
            winner = row['model1'] if row['mean_model1'] > row['mean_model2'] else row['model2']
            print(f"   ✅ Diferencia significativa - {winner} es mejor")
        else:
            print(f"   ❌ No hay diferencia significativa")

def analyze_multiple_metrics(question_metrics, metrics_to_test):
    """Analiza múltiples métricas y crea un resumen"""
    all_results = []
    
    for metric in metrics_to_test:
        print(f"\n🎯 Analizando métrica: {metric}")
        results = perform_wilcoxon_tests(question_metrics, metric)
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar resultados
    filepath = '/Users/haroldgomez/Downloads/cumulative_results_20250731_140825.json'
    results = load_results(filepath)
    
    # Extraer métricas por pregunta
    question_metrics = extract_question_metrics(results)
    
    # Definir métricas a evaluar
    metrics_to_test = [
        'precision@5', 
        'recall@5', 
        'f1@5',
        'ndcg@5',
        'map@5',
        'mrr'
    ]
    
    # Realizar análisis
    df_results = analyze_multiple_metrics(question_metrics, metrics_to_test)
    
    # Crear tabla resumen si hay resultados
    if not df_results.empty:
        print("\n\n📋 RESUMEN DE SIGNIFICANCIA ESTADÍSTICA")
        print("=" * 80)
        
        pivot = df_results.pivot_table(
            index=['model1', 'model2'],
            columns='metric',
            values='significant',
            aggfunc='first'
        )
        
        print(pivot)
    else:
        print("\n⚠️  No se encontraron suficientes datos para realizar tests de Wilcoxon")
    
    # Guardar resultados
    df_results.to_csv('wilcoxon_test_results.csv', index=False)
    print("\n💾 Resultados guardados en 'wilcoxon_test_results.csv'")