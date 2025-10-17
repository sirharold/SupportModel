

import json
import sys

# Cargar el archivo JSON
file_path = '/Users/haroldgomez/Documents/ProyectoTituloMAgister/SupportModel/data/cumulative_results_20251010_131215.json'
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: El archivo no se encontró en la ruta {file_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: El archivo {file_path} no es un JSON válido.")
    sys.exit(1)

# Inspeccionar la estructura de un resultado
model_to_inspect = 'ada'
if model_to_inspect in data.get('results', {}):
    model_data = data['results'][model_to_inspect]
    ragas_bert_evals = model_data.get('ragas_bert_evaluations', [])
    
    if ragas_bert_evals:
        print("--- Contenido del primer resultado de Ragas/BERT ---")
        first_result = ragas_bert_evals[0]
        print(json.dumps(first_result, indent=2))
        
        print("\n--- Claves disponibles en cada resultado ---")
        print(list(first_result.keys()))
    else:
        print(f"No se encontró la clave 'ragas_bert_evaluations' o está vacía para el modelo '{model_to_inspect}'.")
else:
    print(f"No se encontró el modelo '{model_to_inspect}' en los resultados.")

