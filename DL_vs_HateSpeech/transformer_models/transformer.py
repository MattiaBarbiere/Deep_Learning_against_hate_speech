import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, dropout=0.3, output_dim=3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, token_embeddings, attention_mask=None):
        """
        token_embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
        attention_mask: Optional mask to ignore padded tokens of shape (batch_size, seq_len)
        """
        # Simple mean pooling (ignoring padding if attention_mask is provided)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = token_embeddings.mean(dim=1)

        # Classification head
        pooled = self.norm(pooled)
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)
    