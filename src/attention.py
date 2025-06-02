import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, embed_dim)
        Q = self.query(x)  # (batch_size, embed_dim)
        K = self.key(x)  # (batch_size, embed_dim)
        V = self.value(x)  # (batch_size, embed_dim)

        # Scaled dot-product attention
        scores = torch.bmm(
            Q.unsqueeze(1), K.unsqueeze(2)
        ).squeeze()  # (batch_size, 1, 1) -> (batch_size)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size)
        output = attention_weights.unsqueeze(-1) * V  # (batch_size, embed_dim)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        qkv = self.qkv(x).chunk(3, dim=-1)

        # Split into heads
        q, k, v = [
            t.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return self.out(out)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = AttentionLayer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # FFN with residual
        x = x + self.linear(self.norm2(x))
        return x
