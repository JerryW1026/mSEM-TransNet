import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange


# === 复用 TransNet 的基础组件 ===
def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim ** .5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn


class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []
        for i in range(out_shape):
            index = i * self.stride
            input = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)
        out = torch.cat(out, dim=-1)
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        out, _ = attention(q, k, v)
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)
        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output


# === 新增：单分区特征提取块 ===
class RegionExtractor(nn.Module):
    """
    针对单个脑区（左/中/右）进行 TCNN + Spatial CNN 处理
    """

    def __init__(self, num_channels, embed_dim):
        super().__init__()
        # 1. 多尺度时序卷积 (复用 TransNet 的 4 分支结构)
        # 注意：这里我们让每个分区都拥有独立的时序卷积核，以适应不同脑区的频率特性
        self.temp_conv1 = nn.Conv2d(1, embed_dim // 4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, embed_dim // 4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, embed_dim // 4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, embed_dim // 4, (1, 65), padding=(0, 32))

        self.bn1 = nn.BatchNorm2d(embed_dim)

        # 2. 局部空间卷积 (只压缩当前分区的通道)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        # x: (Batch, 1, Channels, Time)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)

        # 拼接 4 个时序分支
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, embed_dim, C, T)
        x = self.bn1(x)

        # 空间卷积压缩通道
        x = self.spatial_conv(x)  # (B, embed_dim, 1, T)
        x = self.bn2(x)
        x = self.elu(x)
        return x


# === 核心：分区结合 TransNet ===
class PartitionTransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()

        # 1. 定义分区索引
        self.idx_left = [1, 2, 6, 7, 8, 13, 14, 18]  # 8 channels
        self.idx_mid = [0, 3, 9, 15, 19, 21]  # 6 channels
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]  # 8 channels

        # 2. 三个独立的提取器
        self.extractor_L = RegionExtractor(len(self.idx_left), embed_dim)
        self.extractor_M = RegionExtractor(len(self.idx_mid), embed_dim)
        self.extractor_R = RegionExtractor(len(self.idx_right), embed_dim)

        # 3. 后处理与 Transformer (输入维度变大为 3 * embed_dim)
        # 因为我们拼接了左、中、右三个分区的特征
        combined_dim = embed_dim * 3

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1

        self.dropout = nn.Dropout()

        # 注意：Transformer 的输入维度现在是 combined_dim
        # 为了保持 Transformer 参数量不过大，可以先用 1x1 卷积降维，或者直接让 Transformer 处理高维
        # 这里为了保留信息量，让 Transformer 处理 combined_dim (96)
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(combined_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        )

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
            nn.BatchNorm2d(temp_embedding_dim),
            nn.ELU()
        )

        # 分类层
        self.classify = nn.Linear(combined_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        # x: (Batch, 22, Time) or (Batch, 1, 22, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Ensure (B, 1, 22, T)

        device = x.device

        # 1. 物理分区切片
        # Lazy loading tensor indices
        if not hasattr(self, 'idx_L_tensor'):
            self.idx_L_tensor = torch.tensor(self.idx_left, device=device)
            self.idx_M_tensor = torch.tensor(self.idx_mid, device=device)
            self.idx_R_tensor = torch.tensor(self.idx_right, device=device)

        if self.idx_L_tensor.device != device:
            self.idx_L_tensor = self.idx_L_tensor.to(device)
            self.idx_M_tensor = self.idx_M_tensor.to(device)
            self.idx_R_tensor = self.idx_R_tensor.to(device)

        x_L = torch.index_select(x, 2, self.idx_L_tensor)  # (B, 1, 8, T)
        x_M = torch.index_select(x, 2, self.idx_M_tensor)  # (B, 1, 6, T)
        x_R = torch.index_select(x, 2, self.idx_R_tensor)  # (B, 1, 8, T)

        # 2. 独立特征提取
        feat_L = self.extractor_L(x_L)  # (B, 32, 1, T)
        feat_M = self.extractor_M(x_M)  # (B, 32, 1, T)
        feat_R = self.extractor_R(x_R)  # (B, 32, 1, T)

        # 3. 特征融合 (在 Channel 维度拼接)
        # 拼接后 shape: (B, 32+32+32, 1, T) = (B, 96, 1, T)
        x = torch.cat([feat_L, feat_M, feat_R], dim=1)
        x = x.squeeze(dim=2)  # (B, 96, T)

        # 4. 双流池化
        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)  # 注意：CSP特征在这里是针对 96 个通道计算的

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')

        # 5. Transformer 编码
        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)

        # 6. 最终融合与分类
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)

        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)

        x = x.reshape(x.size(0), -1)
        out = self.classify(x)

        return out