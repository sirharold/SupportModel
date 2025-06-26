from utils.weaviate_utils import cargar_credenciales, conectar, WeaviateClientWrapper
from utils.embedding import EmbeddingClient
from utils.dataset_builder import build_and_save_dataset, generate_classification_examples, generate_embedding_dataset
from utils.train_model import train_model

# Paso 1: Conectar
config = cargar_credenciales()
client = conectar(config)
weaviate_wrapper = WeaviateClientWrapper(client)
embedding_client = EmbeddingClient(api_key=config["OPENAI_API_KEY"], model="text-embedding-ada-002")

# Paso 2: Construir datasets de preguntas (si no existen)
build_and_save_dataset(weaviate_wrapper, output_dir="data")

# Paso 3: Generar ejemplos de clasificación binaria
import json
with open("data/train_set.json") as f:
    train_questions = json.load(f)
with open("data/val_set.json") as f:
    val_questions = json.load(f)

train_examples = generate_classification_examples(train_questions, weaviate_wrapper, embedding_client)
val_examples = generate_classification_examples(val_questions, weaviate_wrapper, embedding_client)

# Paso 4: Generar embeddings para entrenamiento
generate_embedding_dataset(train_examples, embedding_client, "data/embedding_train.json")
generate_embedding_dataset(val_examples, embedding_client, "data/embedding_val.json")

# Paso 5: Entrenar el modelo
train_model("data/embedding_train.json", "data/embedding_val.json", epochs=5)

if client:
    client.close()