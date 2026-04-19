import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('/mnt/agents/output', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scale = embed_dim ** -0.5  # 缩放因子 1/√d_k
        
        self.W_q = nn.Linear(embed_dim, embed_dim)  # Query 投影
        self.W_k = nn.Linear(embed_dim, embed_dim)  # Key 投影  
        self.W_v = nn.Linear(embed_dim, embed_dim)  # Value 投影
        
    def forward(self, x, return_attention=False):
        Q = self.W_q(x)  # [B, seq, dim]
        print(self.W_q.weight)
        K = self.W_k(x)  # [B, seq, dim]
        V = self.W_v(x)  # [B, seq, dim]
        
        # QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  
        # scores: [B, seq, seq] - 每个位置对其他位置的关注成都
        
        # Softmax 归一化: 权重高的关注, 权重低的忽略
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, V)  # [B, seq, dim]
        # 输出是 V 的加权组合,权重由 QK 相似度决定
        if return_attention:
            return out, attn_weights
        return out

torch.manual_seed(42)
seq_len, embed_dim = 4, 8
world_states = torch.randn(1, seq_len, embed_dim)
attn_layer = SelfAttention(embed_dim)
output, attention_map = attn_layer(world_states, return_attention=True)

print(f"Input shape: {world_states.shape}")
print(f"Attention weights:\n{attention_map[0].detach().numpy().round(3)}")

fig, ax = plt.subplots(figsize=(6, 5))
labels = ['Ball Pos', 'Ball Vel', 'Wall', 'Target']
im = ax.imshow(attention_map[0].detach().numpy(), cmap='viridis', aspect='auto')
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_title('Attention Weight Matrix: What the model focuses on')
plt.colorbar(im, ax=ax, label='Attention Weight')
plt.tight_layout()
plt.savefig('/mnt/agents/output/attention_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: /mnt/agents/output/attention_map.png")