import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class RenewableEnergyEmbedder(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )
        
    def forward(self, input_ids):
        return self.encoder(self.embedding(input_ids))
    
    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self(inputs['input_ids'])
        return embeddings.numpy()


