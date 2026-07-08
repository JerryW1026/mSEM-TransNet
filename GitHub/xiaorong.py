import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from einops import rearrange


# ==========================================
# 1. 基础组件 (保留原架构)
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
            input_window = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input_window.var(dim=-1, keepdim=True), 1e-6, 1e6))
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


class mSEM(nn.Module):
    def __init__(self, F1, D, num_channels, region_indices):
        super(mSEM, self).__init__()
        self.region_indices = region_indices
        self.F1 = F1
        self.local_convs = nn.ModuleList([
            nn.Conv2d(F1, F1, (len(idx), 1), groups=F1, bias=False)
            for idx in region_indices
        ])
        total_spatial_channels = num_channels + len(region_indices)
        self.global_conv = nn.Conv2d(F1, F1 * D, (total_spatial_channels, 1), groups=F1, bias=False)
        self.bn = nn.BatchNorm2d(F1 * D)

    def forward(self, x):
        local_features = []
        for i, idx in enumerate(self.region_indices):
            region_x = x[:, :, idx, :]
            feat = self.local_convs[i](region_x)
            local_features.append(feat)
        local_features_tensor = torch.cat(local_features, dim=2)
        concat_x = torch.cat([x, local_features_tensor], dim=2)
        out = self.global_conv(concat_x)
        out = self.bn(out)
        return out


# ==========================================
# 2. 升级版消融模型定义 (新增频段消融)
# ==========================================

# 🟢 变体 A: 完整模型 (4频段提取 + mSEM + Transformer)
class mSEM_TransNet_Full(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.idx_left = [0, 1, 2, 6, 7, 8, 13, 14, 18]
        self.idx_mid = [3, 9, 15, 19, 21]
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]
        region_indices = [self.idx_left, self.idx_mid, self.idx_right]

        self.F1_per_branch = embed_dim // 4
        self.temp_conv1 = nn.Conv2d(1, self.F1_per_branch, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, self.F1_per_branch, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, self.F1_per_branch, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, self.F1_per_branch, (1, 65), padding=(0, 32), bias=False)

        self.bn_temp = nn.BatchNorm2d(embed_dim)
        self.msem = mSEM(F1=embed_dim, D=1, num_channels=num_channels, region_indices=region_indices)
        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)

        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)])
        self.conv_encoder = nn.Sequential(nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
                                          nn.BatchNorm2d(temp_embedding_dim), nn.ELU())
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = torch.cat((self.temp_conv1(x), self.temp_conv2(x), self.temp_conv3(x), self.temp_conv4(x)), dim=1)
        x = self.elu(self.msem(self.bn_temp(x))).squeeze(2)
        x1, x2 = rearrange(self.dropout(self.avg_pool(x)), 'b d n -> b n d'), rearrange(self.dropout(self.var_pool(x)),
                                                                                        'b d n -> b n d')
        for encoder in self.transformer_encoders: x1, x2 = encoder(x1), encoder(x2)
        return self.classify(
            self.conv_encoder(torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), dim=2)).reshape(x.size(0), -1))


# 🔴 变体 B: 移除多频段提取 (单一卷积尺度 + mSEM + Transformer)
class Single_Freq_mSEM_TransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.idx_left = [0, 1, 2, 6, 7, 8, 13, 14, 18]
        self.idx_mid = [3, 9, 15, 19, 21]
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]
        region_indices = [self.idx_left, self.idx_mid, self.idx_right]

        # 核心修改：使用单一尺度的卷积核(以 25 为例)替代 4 个分支，直接输出 embed_dim 保证参数公平
        self.single_temp_conv = nn.Conv2d(1, embed_dim, (1, 25), padding=(0, 12), bias=False)

        self.bn_temp = nn.BatchNorm2d(embed_dim)
        self.msem = mSEM(F1=embed_dim, D=1, num_channels=num_channels, region_indices=region_indices)
        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)

        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)])
        self.conv_encoder = nn.Sequential(nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
                                          nn.BatchNorm2d(temp_embedding_dim), nn.ELU())
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        # 单频段提取
        x = self.single_temp_conv(x)
        x = self.elu(self.msem(self.bn_temp(x))).squeeze(2)
        x1, x2 = rearrange(self.dropout(self.avg_pool(x)), 'b d n -> b n d'), rearrange(self.dropout(self.var_pool(x)),
                                                                                        'b d n -> b n d')
        for encoder in self.transformer_encoders: x1, x2 = encoder(x1), encoder(x2)
        return self.classify(
            self.conv_encoder(torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), dim=2)).reshape(x.size(0), -1))


# 🔵 变体 C: 移除 mSEM 模块 (4频段提取 + 全局空间卷积 + Transformer)
class TransNet_Only(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.F1_per_branch = embed_dim // 4
        self.temp_conv1 = nn.Conv2d(1, self.F1_per_branch, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, self.F1_per_branch, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, self.F1_per_branch, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, self.F1_per_branch, (1, 65), padding=(0, 32), bias=False)

        self.bn_temp = nn.BatchNorm2d(embed_dim)

        # 核心修改：使用跨越所有通道的全局空间卷积，不进行脑区划分
        self.global_spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), bias=False)
        self.bn_spatial = nn.BatchNorm2d(embed_dim)

        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)

        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)])
        self.conv_encoder = nn.Sequential(nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
                                          nn.BatchNorm2d(temp_embedding_dim), nn.ELU())
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = torch.cat((self.temp_conv1(x), self.temp_conv2(x), self.temp_conv3(x), self.temp_conv4(x)), dim=1)
        x = self.bn_spatial(self.global_spatial_conv(self.bn_temp(x)))
        x = self.elu(x).squeeze(2)
        x1, x2 = rearrange(self.dropout(self.avg_pool(x)), 'b d n -> b n d'), rearrange(self.dropout(self.var_pool(x)),
                                                                                        'b d n -> b n d')
        for encoder in self.transformer_encoders: x1, x2 = encoder(x1), encoder(x2)
        return self.classify(
            self.conv_encoder(torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), dim=2)).reshape(x.size(0), -1))


# 🟡 变体 D: 移除 Transformer (4频段提取 + mSEM + 1D CNN)
class mSEM_CNN_Only(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, attn_drop=0.5):
        super().__init__()
        self.idx_left = [0, 1, 2, 6, 7, 8, 13, 14, 18]
        self.idx_mid = [3, 9, 15, 19, 21]
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]
        region_indices = [self.idx_left, self.idx_mid, self.idx_right]

        self.F1_per_branch = embed_dim // 4
        self.temp_conv1 = nn.Conv2d(1, self.F1_per_branch, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, self.F1_per_branch, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, self.F1_per_branch, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, self.F1_per_branch, (1, 65), padding=(0, 32), bias=False)

        self.bn_temp = nn.BatchNorm2d(embed_dim)
        self.msem = mSEM(F1=embed_dim, D=1, num_channels=num_channels, region_indices=region_indices)
        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)

        # 核心修改：使用 1D 卷积网络替代 Transformer
        self.seq_encoder = nn.Sequential(
            nn.Conv1d(temp_embedding_dim, temp_embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(temp_embedding_dim), nn.GELU(),
            nn.Conv1d(temp_embedding_dim, temp_embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(temp_embedding_dim), nn.GELU()
        )
        self.conv_encoder = nn.Sequential(nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
                                          nn.BatchNorm2d(temp_embedding_dim), nn.ELU())
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = torch.cat((self.temp_conv1(x), self.temp_conv2(x), self.temp_conv3(x), self.temp_conv4(x)), dim=1)
        x = self.elu(self.msem(self.bn_temp(x))).squeeze(2)
        x1, x2 = rearrange(self.dropout(self.avg_pool(x)), 'b d n -> b n d'), rearrange(self.dropout(self.var_pool(x)),
                                                                                        'b d n -> b n d')

        x1, x2 = self.seq_encoder(x1), self.seq_encoder(x2)

        return self.classify(
            self.conv_encoder(torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), dim=2)).reshape(x.size(0), -1))


# ==========================================
# 3. 自动化消融测试框架
# ==========================================
def evaluate_ablation_models(X_data, y_labels, device, num_classes=4, num_samples=1000, epochs=100, batch_size=64):
    model_variants = {
        "1. Full Model (mSEM_TransNet)": mSEM_TransNet_Full,
        "2. w/o Multi-Freq (Single Scale)": Single_Freq_mSEM_TransNet,  # 新增
        "3. w/o mSEM (Global Spatial Conv)": TransNet_Only,
        "4. w/o Transformer (CNN Seq)": mSEM_CNN_Only
    }

    results = {name: [] for name in model_variants.keys()}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_np, y_np = X_data.numpy(), y_labels.numpy()

    print("-" * 65)
    print("🚀 开始执行 4维结构消融实验 (5-Fold Cross Validation)")
    print(f"数据总维度: {X_data.shape}, 类别数: {num_classes}")
    print("-" * 65)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_np)):
        print(f"\n[Fold {fold + 1}/5]")
        X_train, y_train = torch.tensor(X_np[train_idx]), torch.tensor(y_np[train_idx])
        X_test, y_test = torch.tensor(X_np[test_idx]), torch.tensor(y_np[test_idx])

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        for name, ModelClass in model_variants.items():
            model = ModelClass(num_classes=num_classes, num_samples=num_samples).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

            best_acc = 0.0
            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc

            results[name].append(best_acc)
            print(f"  {name:<35}: 最佳准确率 {best_acc:.2f}%")

    print("\n" + "=" * 65)
    print("🏆 消融实验最终结果汇总 (平均准确率 ± 标准差):")
    for name, acc_list in results.items():
        print(f" - {name:<35}: {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")
    print("=" * 65)
    return results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试代码框架可用性
    print("正在初始化评估环境...")
    dummy_X = torch.randn(500, 22, 1000)
    dummy_y = torch.randint(0, 4, (500,))

    evaluate_ablation_models(dummy_X, dummy_y, device, num_classes=4, num_samples=1000, epochs=30)