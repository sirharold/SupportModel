from utils.train_model import train_model
from utils.train_ranking_model import train_ranking_model

# Modo de entrenamiento: "binary" o "ranking"
MODE = "ranking"

if MODE == "binary":
    train_model(
        train_path="data/embedding_train.json",
        val_path="data/embedding_val.json",
        epochs=15,
        batch_size=32,
        lr=1e-4,
        save_path="data/mlp_model.pt",
        only_eval=False
    )

elif MODE == "ranking":
    train_ranking_model(
        train_path="data/embedding_train_triplets.json",
        val_path="data/embedding_val_triplets.json",
        epochs=15,
        batch_size=32,
        lr=1e-4,
        save_path="data/ranking_model.pt"
    )
else:
    raise ValueError("Unsupported mode. Use 'binary' or 'ranking'.")
