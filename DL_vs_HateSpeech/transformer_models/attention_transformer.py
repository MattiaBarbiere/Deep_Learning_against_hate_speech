import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self attention part
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        self._last_attention_weights = attn_weights.detach()
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward part
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def get_attention_weights(self):
        return self._last_attention_weights


class AttentionClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3, output_dim=3):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

        self.encoder_layer = CustomTransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
        # Use our custom encoder layer
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, attention_mask=None):
        token_embeddings = self.norm(token_embeddings)

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(token_embeddings, src_key_padding_mask=src_key_padding_mask)

        cls_token = x[:, 0]
        x = self.dropout(cls_token)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_attention_weights(self, layer_idx=None):
        """Get attention weights from all layers or a specific layer.
        
        Args:
            layer_idx (int, optional): If specified, returns weights only from this layer.
                                      If None, returns list of weights from all layers.
        Returns:
            List of attention weights (batch_size, num_heads, seq_len, seq_len)
            or single tensor if layer_idx is specified.
        """
        if layer_idx is not None:
            return self.transformer.layers[layer_idx].get_attention_weights()
        return [layer.get_attention_weights() for layer in self.transformer.layers]
    
    def set_attention_weights(self, weights):
        self._last_attentions = weights