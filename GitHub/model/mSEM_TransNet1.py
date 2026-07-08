import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange


# ==========================================
# 1. 基础组件 (保持不变)
# ==========================================

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


# ==========================================
# 2. 核心模块: mSEM (Multilevel Spatial Feature Extraction Module)
# ==========================================
class mSEM(nn.Module):
    def __init__(self, F1, D, num_channels, region_indices):
        """
        F1: 输入特征图的通道数 (Temporal Filters)
        D:  Depthwise 参数 (通常为 2 或 1)
        num_channels: 原始 EEG 通道数 (22)
        region_indices: 脑区分组索引列表 [[idx_left], [idx_mid], [idx_right]]
        """
        super(mSEM, self).__init__()
        self.region_indices = region_indices
        self.F1 = F1

        # 1. Local Spatial Convolutions (针对每个脑区)
        # 论文中提到：对每个分区的特征做 Depthwise Conv
        # 输出大小保持 (1, 1)，即每个脑区被压缩成一个“超级通道”
        self.local_convs = nn.ModuleList([
            nn.Conv2d(F1, F1, (len(idx), 1), groups=F1, bias=False)
            for idx in region_indices
        ])

        # 2. Global Spatial Convolution
        # 输入通道数 = 原始通道数 (C) + 脑区数 (k)
        # 输出通道数 = F1 * D (即 embed_dim)
        total_spatial_channels = num_channels + len(region_indices)
        self.global_conv = nn.Conv2d(F1, F1 * D, (total_spatial_channels, 1), groups=F1, bias=False)

        # BN 层
        self.bn = nn.BatchNorm2d(F1 * D)

    def forward(self, x):
        # x shape: (Batch, F1, Channels, Time)

        local_features = []

        # Step A: 提取局部脑区特征
        for i, idx in enumerate(self.region_indices):
            # 1. 根据索引切片 (Slice)
            # x[:, :, idx, :] -> (Batch, F1, num_electrodes_in_region, Time)
            region_x = x[:, :, idx, :]

            # 2. 局部卷积
            # out -> (Batch, F1, 1, Time)
            feat = self.local_convs[i](region_x)
            local_features.append(feat)

        # Step B: 拼接 (Concatenate)
        # 将原始输入 x 和 提取出的 local_features 在空间(Channel)维度拼接
        # x: (B, F1, 22, T)
        # local_features[i]: (B, F1, 1, T)
        # concat_x: (B, F1, 22 + k, T)
        local_features_tensor = torch.cat(local_features, dim=2)
        concat_x = torch.cat([x, local_features_tensor], dim=2)

        # Step C: 全局卷积 (Global Convolution)
        # out -> (B, F1*D, 1, T)
        out = self.global_conv(concat_x)
        out = self.bn(out)

        return out


# ==========================================
# 3. mSEM_TransNet 模型
# ==========================================
class mSEM_TransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()

        # --- 1. 定义脑区策略 (Strategy 3: Trichotomy 左中右) ---
        # 参考论文 Fig 3(b) 和 BCI-IV-2a 的 montage
        self.idx_left = [0, 1, 2, 6, 7, 8, 13, 14, 18]  # 左侧
        self.idx_mid = [3, 9, 15, 19, 21]  # 中线 (Fz, Cz, Pz 等)
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]  # 右侧

        region_indices = [self.idx_left, self.idx_mid, self.idx_right]

        # --- 2. Temporal Convolution (第一层) ---
        # 论文中：F1 kernels of size (1, 64)
        # TransNet原版用了4个尺度的卷积，这里我们保留 TransNet 的多尺度优势，
        # 但为了适配 mSEM，我们需要统一输出通道。
        # 这里我们将 embed_dim 分给 4 个分支，每个分支 F1 = embed_dim // 4
        self.F1_per_branch = embed_dim // 4

        self.temp_conv1 = nn.Conv2d(1, self.F1_per_branch, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, self.F1_per_branch, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, self.F1_per_branch, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, self.F1_per_branch, (1, 65), padding=(0, 32), bias=False)

        self.bn_temp = nn.BatchNorm2d(embed_dim)  # embed_dim = F1_per_branch * 4

        # --- 3. mSEM 模块 (替换原来的 Spatial Conv) ---
        # Input channels to mSEM = embed_dim (这是Temporal卷积后的 feature map 数量)
        # D = 1 (因为我们希望输出保持 embed_dim，或者设为 D=2 增加深度)
        # 这里设 D=1 以匹配 TransNet 后续维度
        self.msem = mSEM(F1=embed_dim, D=1, num_channels=num_channels, region_indices=region_indices)

        self.elu = nn.ELU()

        # --- 4. TransNet 后端 (Pooling + Transformer) ---
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1

        self.dropout = nn.Dropout(attn_drop)  # 使用传入的 dropout 率

        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        )

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
            nn.BatchNorm2d(temp_embedding_dim),
            nn.ELU()
        )
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        # x: (Batch, 22, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, 22, T)

        # 1. Temporal Convolution (Multi-scale)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)

        # 拼接 4 个尺度的特征 -> (B, embed_dim, 22, T)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn_temp(x)

        # 2. mSEM Spatial Extraction
        # 输入: (B, embed_dim, 22, T)
        # 输出: (B, embed_dim, 1, T)
        x = self.msem(x)

        x = self.elu(x)
        x = x.squeeze(2)  # (B, embed_dim, T)

        # 3. Dual Pooling (Mean & Var)
        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')

        # 4. Transformer
        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)

        # 5. Fusion & Classify
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)
        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)

        x = x.reshape(x.size(0), -1)
        out = self.classify(x)

        return out