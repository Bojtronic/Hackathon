from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


dataset = load_dataset("wikitext", "wikitext-2-v1")
train_loader = DataLoader(dataset["train"], batch_size=32)

model = RenewableEnergyEmbedder().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):
    for batch in train_loader:
        inputs = tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        outputs = model(inputs["input_ids"])
        loss = outputs.norm(dim=1).mean()  # Loss de ejemplo
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.save(model.state_dict(), f"embedding_model_epoch_{epoch}.pt")


