import torch.nn as nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3, output_dim=3):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self._last_attentions = []  # buffer for attention weights
        self._hooks_registered = False

    def _register_hooks(self):
        # Only register hooks once
        if self._hooks_registered:
            return

        def hook_fn(module, input, output):
            # self_attn returns attention weights as a tuple: (attn_output, attn_weights)
            # But TransformerEncoderLayer hides this unless you re-implement it, so this will NOT work directly
            # Instead, we attach to the `self_attn` submodules
            def inner_hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]  # (batch, heads, seq_len, seq_len)
                    self._last_attentions.append(attn_weights.detach())
            module.self_attn.register_forward_hook(inner_hook)

        # Register a hook on every TransformerEncoderLayer
        for layer in self.transformer.layers:
            hook_fn(layer)

        self._hooks_registered = True

    def forward(self, token_embeddings, attention_mask=None):
        token_embeddings = self.norm(token_embeddings)
        self._last_attentions = []
        self._register_hooks()

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

    def get_attention_weights(self):
        return self._last_attentions

    def set_attention_weights(self, weights):
        self._last_attentions = weights