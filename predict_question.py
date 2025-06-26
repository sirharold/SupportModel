import torch
import numpy as np
from utils.weaviate_utils import cargar_credenciales, conectar, WeaviateClientWrapper
from utils.embedding import EmbeddingClient
from utils.train_model import load_model

def predict_question(question: str, top_k: int = 10):
    # Configurar entorno
    config = cargar_credenciales()
    client = conectar(config)
    wrapper = WeaviateClientWrapper(client)
    embedding_client = EmbeddingClient(api_key=config["OPENAI_API_KEY"], model="text-embedding-ada-002")
    model = load_model("data/mlp_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Obtener vector de pregunta
        q_vector = embedding_client.generate_embedding(question)

        # Buscar documentos similares
        docs = wrapper.search_docs_by_vector(q_vector, top_k=top_k)

        # Predecir con modelo entrenado
        inputs = []
        doc_infos = []
        for doc in docs:
            doc_vec = embedding_client.generate_embedding(doc.get("content", ""))
            if doc_vec:
                inputs.append(doc_vec)
                doc_infos.append({
                    "title": doc.get("title", "No Title"),
                    "link": doc.get("link", ""),
                    "content": doc.get("content", "")[:200]  # preview
                })

        if not inputs:
            print("⚠️ No valid document embeddings were generated.")
            return

        X = torch.tensor(np.array(inputs), dtype=torch.float).to(device)
        with torch.no_grad():
            logits = model(X)
            scores = torch.sigmoid(logits).cpu().numpy()

        ranked = sorted(zip(doc_infos, scores), key=lambda x: x[1], reverse=True)

        print("\n🔎 Predicted Relevant Documents:\n")
        for i, (doc, score) in enumerate(ranked, 1):
            print(f"{i}. [{doc['title']}]({doc['link']})")
            print(f"   🔹 Score: {score:.4f}")
            #print(f"   📝 {doc['content']}...\n")
            print("\n")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    question = "How can I enable and use Managed Identity in an Azure Function to securely connect to Azure SQL, Key Vault, or Service Bus without using connection strings or secrets?"
    predict_question(question)
