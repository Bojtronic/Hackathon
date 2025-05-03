import torch
from transformers import AutoTokenizer
from generation_model import RenewableEnergyGenerator
from datasets import load_dataset

def train_generator(data_path="data/qa_pairs.csv", save_path="models/generator.pth"):
    # 1. Cargar datos (formato: pregunta, respuesta)
    dataset = load_dataset("csv", data_files=data_path)["train"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Preparar datos
    def encode(examples):
        inputs = tokenizer(examples["Pregunta"], examples["Respuesta"], 
                          truncation=True, padding="max_length", max_length=512)
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    
    dataset = dataset.map(encode, batched=True)
    
    # 3. Entrenamiento
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RenewableEnergyGenerator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(3):  # Épocas
        for batch in DataLoader(dataset, batch_size=8):
            outputs = model(batch["input_ids"].to(device))
            loss_fn = nn.CrossEntropyLoss()
            logits = outputs.view(-1, outputs.size(-1))
            loss = loss_fn(logits, inputs["labels"].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Época {epoch+1}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_generator()
