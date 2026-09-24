import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionGate(nn.Module):
    def __init__(self, input_size, num_experts, hidden_size=32):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_proj = nn.Linear(1, hidden_size)
        self.value_proj = nn.Linear(1, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        x = x.unsqueeze(-1)  # [batch, seq, 1]
        keys = self.key_proj(x)  # [batch, seq, hidden]
        values = self.value_proj(x)  # [batch, seq, hidden]

        # Compute attention scores
        scores = torch.matmul(keys, self.query) / math.sqrt(keys.size(-1))
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq]

        # Weighted sum of values
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)
        return self.out_proj(context)
