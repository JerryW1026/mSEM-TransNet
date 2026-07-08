import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange


# === 基础组件 (保持不变) ===
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


# === 提取器模块 (可复用) ===
class FeatureExtractor(nn.Module):
    def __init__(self, num_channels, embed_dim):
        super().__init__()
        # 减少每个分支的 filter 数量，防止 Global+Local 结合后总参数爆炸
        # 原版是 embed_dim//4，这里为了轻量化，如果显存够可以用原版配置
        f = embed_dim // 4
        self.temp_conv1 = nn.Conv2d(1, f, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, f, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, f, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, f, (1, 65), padding=(0, 32))

        self.bn1 = nn.BatchNorm2d(f * 4)  # 4个分支拼接
        self.spatial_conv = nn.Conv2d(f * 4, embed_dim, (num_channels, 1))
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        return x


# === 终极模型：Global-Local TransNet ===
class GL_TransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()

        # 1. 索引定义
        self.idx_left = [1, 2, 6, 7, 8, 13, 14, 18]
        self.idx_mid = [0, 3, 9, 15, 19, 21]
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]

        # 2. 四路提取器 (全独立，不共享权重)
        # Global: 处理所有 22 通道 (保底 Baseline)
        self.global_extractor = FeatureExtractor(num_channels, embed_dim)

        # Local: 处理分区
        self.left_extractor = FeatureExtractor(len(self.idx_left), embed_dim)
        self.mid_extractor = FeatureExtractor(len(self.idx_mid), embed_dim)
        self.right_extractor = FeatureExtractor(len(self.idx_right), embed_dim)

        # 3. 后处理配置
        # 特征融合后的维度：Global(1) + Left(1) + Mid(1) + Right(1) = 4 * embed_dim
        self.fusion_dim = embed_dim * 4

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1

        self.dropout = nn.Dropout(0.5)  # 稍微增加Dropout防止过拟合

        # 4. Transformer 处理融合特征
        # 注意：这里输入维度变大了，Transformer 将学习 Global 和 Local 之间的关系
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(self.fusion_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        )

        # 5. 最终降维与分类
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),  # 融合 mean 和 var 流
            nn.BatchNorm2d(temp_embedding_dim),
            nn.ELU()
        )
        self.classify = nn.Linear(self.fusion_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        device = x.device

        # 1. 准备数据
        if not hasattr(self, 'idx_L_tensor'):
            self.idx_L_tensor = torch.tensor(self.idx_left, device=device)
            self.idx_M_tensor = torch.tensor(self.idx_mid, device=device)
            self.idx_R_tensor = torch.tensor(self.idx_right, device=device)

        if self.idx_L_tensor.device != device:
            self.idx_L_tensor = self.idx_L_tensor.to(device)
            self.idx_M_tensor = self.idx_M_tensor.to(device)
            self.idx_R_tensor = self.idx_R_tensor.to(device)

        x_G = x  # Global
        x_L = torch.index_select(x, 2, self.idx_L_tensor)
        x_M = torch.index_select(x, 2, self.idx_M_tensor)
        x_R = torch.index_select(x, 2, self.idx_R_tensor)

        # 2. 并行提取特征
        feat_G = self.global_extractor(x_G)  # (B, 32, 1, T)
        feat_L = self.left_extractor(x_L)  # (B, 32, 1, T)
        feat_M = self.mid_extractor(x_M)  # (B, 32, 1, T)
        feat_R = self.right_extractor(x_R)  # (B, 32, 1, T)

        # 3. 强力融合 (Concatenate)
        # 将全局视野和局部细节拼在一起
        x = torch.cat([feat_G, feat_L, feat_M, feat_R], dim=1)  # (B, 128, 1, T)
        x = x.squeeze(dim=2)

        # 4. 双流池化
        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')

        # 5. Transformer 全局建模
        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)

        # 6. 分类
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)
        x = torch.cat((x1, x2), dim=2)

        x = self.conv_encoder(x)
        x = x.reshape(x.size(0), -1)
        out = self.classify(x)

        return out