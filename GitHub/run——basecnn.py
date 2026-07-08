import os
import glob
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re


# ==========================================
# 1. 核心对比基线: CNN-BiLSTM (代表 RNN 序列建模流派)
# 适配 12通道 200Hz SEED 数据集
# ==========================================
class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=3, channels=12, samples=200, hidden_size=64, num_layers=2):
        super(CNN_BiLSTM, self).__init__()

        # 1. 空间-时间卷积前端 (CNN 特征提取)
        self.conv_block = nn.Sequential(
            # 时间维卷积 (提取局部频率特征)
            nn.Conv2d(1, 16, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # 空间维卷积 (融合 12 个通道)
            nn.Conv2d(16, 32, kernel_size=(channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # 时域池化，压缩序列长度
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.5)
        )

        # 2. 序列建模后端 (BiLSTM)
        # 经过池化后，200个采样点变成了 200/4 = 50 步的时序特征
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=32,  # CNN 输出的特征维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 双向 LSTM
        )

        self.dropout = nn.Dropout(0.5)
        # 双向 LSTM 输出维度为 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 输入维度: (Batch, 1, Channels, Time) -> (B, 1, 12, 200)
        if x.dim() == 3: x = x.unsqueeze(1)

        # 1. CNN 提取局部时空特征
        x = self.conv_block(x)  # 输出: (B, 32, 1, 50)

        # 2. 维度重排，适配 LSTM 输入格式: (Batch, Seq_Len, Feature_Dim)
        x = x.squeeze(2)  # 移除高度维度变为 (B, 32, 50)
        x = x.permute(0, 2, 1)  # 换轴变为 (B, 50, 32)

        # 3. BiLSTM 捕捉长程时序依赖
        # out: (Batch, Seq_Len, Hidden_Size*2)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 提取序列的全局特征：取整个序列的时间平均，对情绪这种连续状态更鲁棒
        seq_features = torch.mean(lstm_out, dim=1)

        # 4. 分类器
        out = self.dropout(seq_features)
        out = self.fc(out)

        return out


# ==========================================
# 2. 数据加载模块 (维持 12通道 SEED 原版逻辑)
# ==========================================
class SEEDDataset12Ch(Dataset):
    def __init__(self, file_paths, window_size=200, stride=200):
        self.window_size = window_size
        self.stride = stride

        # 提取 12 关键通道
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

        # 正则匹配 eeg 并排序
        trial_keys = [k for k in keys if 'eeg' in k.lower()]
        if len(trial_keys) == 0: return None, None
        trial_keys.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

        # SEED 3分类官方标签映射
        raw_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        mapped_labels = [label + 1 for label in raw_labels]

        all_epochs, all_epoch_labels = [], []

        for trial_idx, key in enumerate(trial_keys):
            if trial_idx >= 15: break

            trial_data = mat_data[key]
            trial_data = trial_data[self.selected_channels, :]

            # Z-score 归一化
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
# 3. 跨会话评估逻辑 (调用 CNN-BiLSTM)
# ==========================================
def train_and_evaluate_cross_session(train_files, test_files, device, epochs=100, batch_size=64, lr=1e-3):
    train_dataset = SEEDDataset12Ch(train_files)
    test_dataset = SEEDDataset12Ch(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化 RNN 流派代表：CNN-BiLSTM
    model = CNN_BiLSTM(num_classes=3, channels=12, samples=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
# 4. 15人自动化实验流: 2 Train / 1 Test
# ==========================================
if __name__ == '__main__':
    # 📌 确保此路径与跑 mSEM_TransNet 时完全一致
    data_dir = r"C:\Users\WJR\Desktop\分区+多特征融合实验\EEG-TransNet-main\EEG-TransNet-main -+分区\SEED_Data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_subject_acc = []
    # CNN-BiLSTM 包含 RNN 结构，收敛可能稍慢，100轮是比较稳妥的设置
    run_epochs = 100

    for subject_id in range(1, 16):
        print(f"\n{'-' * 70}")
        print(f" 🥊 开始对比评估被试 (Subject) {subject_id}/15 [模型: CNN-BiLSTM Baseline]")

        search_pattern = os.path.join(data_dir, f"{subject_id}_*.mat")
        subject_files = glob.glob(search_pattern)
        subject_files.sort()

        if len(subject_files) != 3:
            print(f"[警告] 被试 {subject_id} 未找到 3 个完整 session 文件，已跳过。")
            continue

        train_file_list = [subject_files[0], subject_files[1]]
        test_file_list = [subject_files[2]]

        acc = train_and_evaluate_cross_session(train_file_list, test_file_list, device, epochs=run_epochs)

        all_subject_acc.append(acc)
        print(f" 🏆 被试 {subject_id} CNN-BiLSTM 跨会话最佳准确率: {acc:.2f}%")

    if len(all_subject_acc) > 0:
        final_mean_acc = np.mean(all_subject_acc)
        final_std_acc = np.std(all_subject_acc)

        print(f"\n{'=' * 70}")
        print(" 🎉 15 名被试 [CNN-BiLSTM 对比实验] 全部完成！")
        print(f" 🎯 采用配置: RNN 流派代表 CNN-BiLSTM (12 通道, SEED 3分类)")
        print(f" 总体平均准确率 (Mean ACC): {final_mean_acc:.2f}%")
        print(f" 总体标准差 (Std Dev): {final_std_acc:.2f}%")
        print(f" 各被试结果阵列: {[round(acc, 2) for acc in all_subject_acc]}")
        print(f"{'=' * 70}")