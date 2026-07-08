import os
import glob
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import re


# ==========================================
# 1. 基础组件
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
# 2. mSEM_TransNet (SEED 12通道)
# ==========================================
class mSEM_TransNet_SEED12(nn.Module):
    def __init__(self, num_classes=3, num_samples=200, num_channels=12, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()

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
# 3. 数据加载模块
# ==========================================
class SEEDDataset12Ch(Dataset):
    def __init__(self, file_paths, window_size=200, stride=200):
        self.window_size = window_size
        self.stride = stride
        self.selected_channels = [14, 23, 24, 32, 33, 41, 22, 31, 30, 40, 39, 49]
        self.data, self.labels = [], []

        for filepath in file_paths:
            if os.path.exists(filepath):
                d, l = self._load_and_epoch(filepath)
                if d is not None and l is not None:
                    self.data.append(d)
                    self.labels.append(l)
            else:
                print(f"Warning: File {filepath} not found!")

        if len(self.data) > 0:
            self.data = torch.cat(self.data, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            raise ValueError("未加载任何有效数据，请检查数据格式。")

    def _load_and_epoch(self, filepath):
        mat_data = sio.loadmat(filepath)
        keys = list(mat_data.keys())
        trial_keys = [k for k in keys if 'eeg' in k.lower()]

        if len(trial_keys) == 0: return None, None
        trial_keys.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
        raw_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        mapped_labels = [label + 1 for label in raw_labels]

        all_epochs, all_epoch_labels = [], []

        for trial_idx, key in enumerate(trial_keys):
            if trial_idx >= 15: break

            trial_data = mat_data[key]
            trial_data = trial_data[self.selected_channels, :]

            mean_val = np.mean(trial_data, axis=1, keepdims=True)
            std_val = np.std(trial_data, axis=1, keepdims=True)
            trial_data = (trial_data - mean_val) / (std_val + 1e-7)

            current_label = mapped_labels[trial_idx]
            if trial_data.shape[1] < self.window_size: continue

            num_epochs = (trial_data.shape[1] - self.window_size) // self.stride + 1
            for i in range(num_epochs):
                start = i * self.stride
                end = start + self.window_size
                all_epochs.append(trial_data[:, start:end])
                all_epoch_labels.append(current_label)

        if len(all_epochs) == 0: return None, None

        epochs_tensor = torch.tensor(np.array(all_epochs), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(all_epoch_labels), dtype=torch.long)
        return epochs_tensor, labels_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==========================================
# 4. 评估逻辑
# ==========================================
def train_and_evaluate_cross_session(train_files, test_files, device, epochs=100, batch_size=64, lr=5e-4):
    train_dataset = SEEDDataset12Ch(train_files)
    test_dataset = SEEDDataset12Ch(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = mSEM_TransNet_SEED12(attn_drop=0.6, fc_drop=0.7).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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
        if test_acc > best_acc: best_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"      Epoch [{epoch + 1:02d}/{epochs}] | Loss: {train_loss / len(train_loader):.4f} | Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)")

    return best_acc


# ==========================================
# 5. 自动化实验流: 变更逻辑 [Train: S1, S3 | Test: S2]
# ==========================================
if __name__ == '__main__':
    data_dir = r"C:\Users\WJR\Desktop\分区+多特征融合实验\EEG-TransNet-main\EEG-TransNet-main -+分区\SEED_Data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_subject_acc = []

    for subject_id in range(1, 16):
        print(f"\n{'-' * 70}")
        print(f" 🧪 开始探索评估被试 (Subject) {subject_id}/15")
        print(f" ⚠️ 注意：当前采用非标准切分逻辑 [训练: Session 1 & 3 | 验证: Session 2]")

        search_pattern = os.path.join(data_dir, f"{subject_id}_*.mat")
        subject_files = glob.glob(search_pattern)
        subject_files.sort()

        if len(subject_files) != 3:
            print(f"[警告] 被试 {subject_id} 未找到 3 个完整 session 文件，已跳过。")
            continue

        # 【核心修改逻辑】：调整训练集和测试集的划分
        train_file_list = [subject_files[0], subject_files[2]]  # 使用第 1 和第 3 个文件训练
        test_file_list = [subject_files[1]]  # 使用第 2 个文件测试

        acc = train_and_evaluate_cross_session(train_file_list, test_file_list, device, epochs=500)

        all_subject_acc.append(acc)
        print(f" 🏆 被试 {subject_id} [新逻辑] 跨会话最佳准确率: {acc:.2f}%")

    if len(all_subject_acc) > 0:
        final_mean_acc = np.mean(all_subject_acc)
        final_std_acc = np.std(all_subject_acc)

        print(f"\n{'=' * 70}")
        print(" 15 名被试 [Train: S1+S3 / Test: S2] 探索性实验全部完成！")
        print(f" 总体平均准确率 (Mean ACC): {final_mean_acc:.2f}%")
        print(f" 总体标准差 (Std Dev): {final_std_acc:.2f}%")
        print(f" 各被试结果阵列: {[round(acc, 2) for acc in all_subject_acc]}")
        print(f"{'=' * 70}")