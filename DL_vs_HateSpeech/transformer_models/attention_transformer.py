import torch.nn as nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3, output_dim=3):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Important for (batch, seq_len, dim) inputs
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, attention_mask=None):
        """
        token_embeddings: Tensor (batch_size, seq_len, embedding_dim)
        attention_mask: Optional Tensor (batch_size, seq_len), 1 for valid tokens, 0 for pad
        """
        token_embeddings = self.norm(token_embeddings)

        if attention_mask is not None:
            # Transformer expects a bool mask with True for *masked out* positions
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(token_embeddings, src_key_padding_mask=src_key_padding_mask)

        # Use CLS token (assume it's the first token)
        cls_token = x[:, 0]  # (batch_size, embedding_dim)

        # Classifier
        x = self.dropout(cls_token)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
