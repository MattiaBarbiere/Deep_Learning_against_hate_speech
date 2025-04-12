import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.linear(x)
        return x