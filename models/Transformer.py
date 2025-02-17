import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, input_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        # self.seq_len = seq_len
        self.output_dim = input_dim

        # 映射 C 维度到 Transformer 的嵌入维度
        self.embedding = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

        # 位置编码
        # self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim))

        # 用于将 Transformer 输出映射回原始通道数 C
        self.output_layer = nn.Conv2d(embed_dim, self.output_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape  # 输入为 (B, C, H, W)
        # 映射到嵌入维度 (B, C, H, W) -> (B, embed_dim, H, W)
        x = self.embedding(x)
        # 转换为序列 (B, embed_dim, H, W) -> (B, N, embed_dim)
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, embed_dim)，其中 N = H * W

        # 添加位置编码
        # x += self.positional_encoding  (尝试不使用位置编码)

        # Transformer 编码
        x = x.permute(1, 0, 2)  # 转换为 (N, B, embed_dim)
        x = self.transformer_encoder(x)  # 输出 (N, B, embed_dim)

        # 映射回 (B, C, H, W)
        x = x.permute(1, 2, 0).reshape(B, self.embed_dim, H, W)  # (B, embed_dim, H, W)
        
        # 用输出层将嵌入维度映射回 C
        x = self.output_layer(x)  # 输出 (B, C, H, W)
        return x
