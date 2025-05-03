import torch
from torch.utils.data import DataLoader
from embeddings_model import RenewableEnergyEmbedder
from datasets import load_dataset

def train_embeddings(data_path="data/training_texts.txt", save_path="models/embedder.pth"):
    # 1. Cargar datos de ejemplo (o usar tus documentos)
    dataset = load_dataset("text", data_files=data_path)["train"]
    
    # 2. Crear dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Inicializar modelo y optimizador
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RenewableEnergyEmbedder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. Entrenamiento
    for epoch in range(5):  # Épocas de entrenamiento
        for batch in loader:
            inputs = model.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            embeddings = model(inputs["input_ids"])
            loss = embeddings.norm(dim=1).mean()  # Loss de ejemplo (¡mejorable!)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Época {epoch+1}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_embeddings()

