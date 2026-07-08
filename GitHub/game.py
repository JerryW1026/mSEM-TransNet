import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import mne
from einops import rearrange
import warnings

# 忽略 MNE 的一些无关警告，保持控制台清爽
warnings.filterwarnings('ignore')


# ==========================================
# 1. 基础组件 (完全保留你原有的 mSEM_TransNet 架构)
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


class mSEM_TransNet_EDF(nn.Module):
    def __init__(self, num_classes=3, num_samples=200, num_channels=12, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()

        # EDF 数据映射好的左脑与右脑前 6 个通道索引
        self.idx_left = [0, 1, 2, 3, 4, 5]
        self.idx_right = [6, 7, 8, 9, 10, 11]
        region_indices = [self.idx_left, self.idx_right]

        self.F1_per_branch = embed_dim // 4
        self.temp_conv1 = nn.Conv2d(1, self.F1_per_branch, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, self.F1_per_branch, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, self.F1_per_branch, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, self.F1_per_branch, (1, 65), padding=(0, 32), bias=False)

        self.bn_temp = nn.BatchNorm2d(embed_dim)
        self.msem = mSEM(F1=embed_dim, D=1, num_channels=num_channels, region_indices=region_indices)
        self.elu = nn.ELU()

        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        self.var_pool = VarPoold(pool_size, pool_stride)

        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)

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
        if x.dim() == 3: x = x.unsqueeze(1)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn_temp(x)
        x = self.msem(x)
        x = self.elu(x)
        x = x.squeeze(2)

        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')

        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)

        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)
        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)

        x = x.reshape(x.size(0), -1)
        out = self.classify(x)
        return out


# ==========================================
# 2. 核心：EDF 数据解析与清洗管道
# ==========================================
def load_and_preprocess_subject_edf(subject_files, window_size=200, stride=200, target_sfreq=200):
    """
    加载单个被试的所有 EDF 文件，降采样，切片，打标签，返回完整的张量。
    """
    # 挑选 12 个对称的最佳脑电通道
    # 额极(Fp), 额叶(F), 中央(C), 颞叶(T), 顶叶(P)
    target_channels = [
        'Fp1', 'F3', 'F7', 'C3', 'T7', 'P3',  # 6个左脑通道 (索引 0-5)
        'Fp2', 'F4', 'F8', 'C4', 'T8', 'P4'  # 6个右脑通道 (索引 6-11)
    ]

    all_epochs, all_labels = [], []

    for filepath in subject_files:
        filename = os.path.basename(filepath).lower()

        # 标签映射：Game(高认知负荷)->2, Neutral(基线)->1, Rest(低唤醒)->0
        if 'game' in filename:
            label = 2
        elif 'neutral' in filename:
            label = 1
        elif 'rest' in filename:
            label = 0
        else:
            continue  # 跳过无法识别文件名的文件

        try:
            # 1. 加载 EDF
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

            # 2. 降采样至 200 Hz
            if raw.info['sfreq'] != target_sfreq:
                raw.resample(target_sfreq, npad="auto")

            # 3. 精准抽取 12 通道数据并保持左右脑顺序
            data = raw.get_data(picks=target_channels)  # Shape: (12, TimePoints)

            # 4. Z-score 归一化 (抗阻抗漂移)
            mean_val = np.mean(data, axis=1, keepdims=True)
            std_val = np.std(data, axis=1, keepdims=True)
            data = (data - mean_val) / (std_val + 1e-7)

            # 5. 滑动窗口切片 (1 秒切片)
            total_samples = data.shape[1]
            num_epochs = (total_samples - window_size) // stride + 1

            for i in range(num_epochs):
                start = i * stride
                end = start + window_size
                all_epochs.append(data[:, start:end])
                all_labels.append(label)

        except Exception as e:
            print(f"      [警告] 读取文件 {filename} 时出错: {e}")

    if len(all_epochs) == 0:
        return None, None

    X_tensor = torch.tensor(np.array(all_epochs), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
    return X_tensor, y_tensor


# ==========================================
# 3. 被试内 5 折交叉验证逻辑
# ==========================================
def run_5fold_cv_for_subject(X, y, device, subject_name, epochs_per_fold=50, batch_size=64, lr=5e-4):
    print(
        f"      [数据分布] 总样本数: {len(y)} | Game: {(y == 2).sum().item()} | Neutral: {(y == 1).sum().item()} | Rest: {(y == 0).sum().item()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    # 将 Tensor 转为 numpy 索引切分更方便
    X_np, y_np = X.numpy(), y.numpy()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_np)):
        X_train, y_train = torch.tensor(X_np[train_idx]), torch.tensor(y_np[train_idx])
        X_test, y_test = torch.tensor(X_np[test_idx]), torch.tensor(y_np[test_idx])

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = mSEM_TransNet_EDF(attn_drop=0.6, fc_drop=0.7).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

        best_fold_acc = 0.0

        for epoch in range(epochs_per_fold):
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

            test_acc = 100 * correct / total
            if test_acc > best_fold_acc: best_fold_acc = test_acc

        print(f"      ✅ Fold {fold + 1}/5 最佳准确率: {best_fold_acc:.2f}%")
        fold_accuracies.append(best_fold_acc)

    mean_cv_acc = np.mean(fold_accuracies)
    return mean_cv_acc


# ==========================================
# 4. 自动化搜寻被试与主循环
# ==========================================
if __name__ == '__main__':
    # 📌 设置你的独立数据文件夹路径
    data_dir = r"C:\Users\WJR\Desktop\分区+多特征融合实验\EEG-TransNet-main\EEG-TransNet-main -+分区\data-xinan"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 自动检索所有以 .edf 结尾的文件
    all_edf_files = glob.glob(os.path.join(data_dir, "*.edf"))

    # 提取所有被试的前缀 (例如：从 '2246lifachuan_game.edf' 中提取 '2246lifachuan')
    subject_prefixes = set()
    for f in all_edf_files:
        basename = os.path.basename(f)
        prefix = basename.split('_')[0]
        subject_prefixes.add(prefix)

    subject_prefixes = sorted(list(subject_prefixes))
    print(f"🔍 检测到 {len(subject_prefixes)} 个独立被试: {subject_prefixes}")

    all_subject_acc = []

    for subject_prefix in subject_prefixes:
        print(f"\n{'-' * 70}")
        print(f" 🚀 正在处理被试: {subject_prefix}")

        # 寻找该被试所有的 EDF 文件 (理论上是 3 个)
        subject_files = [f for f in all_edf_files if os.path.basename(f).startswith(subject_prefix)]
        print(f"      找到 {len(subject_files)} 个关联数据文件。正在读取并重采样 (500Hz -> 200Hz)...")

        # 加载并预处理数据
        X, y = load_and_preprocess_subject_edf(subject_files)

        if X is None or len(X) < 100:
            print(f"      [警告] 被试 {subject_prefix} 数据提取失败或样本极少，已跳过。")
            continue

        # 运行 5折交叉验证 (设置 50 轮训练，因数据量不大，通常极快收敛)
        mean_cv_acc = run_5fold_cv_for_subject(X, y, device, subject_prefix, epochs_per_fold=50)

        all_subject_acc.append(mean_cv_acc)
        print(f" 🏆 被试 {subject_prefix} 5折交叉验证平均准确率: {mean_cv_acc:.2f}%")

    if len(all_subject_acc) > 0:
        final_mean = np.mean(all_subject_acc)
        final_std = np.std(all_subject_acc)
        print(f"\n{'=' * 70}")
        print(f" 🎉 拓展实验 (Data-Xinan) 认知负荷解码全部完成！")
        print(f" 总体平均准确率 (Mean ACC): {final_mean:.2f}%")
        print(f" 总体标准差 (Std Dev): {final_std:.2f}%")
        print(f" 各被试结果阵列: {[round(acc, 2) for acc in all_subject_acc]}")
        print(f"{'=' * 70}")