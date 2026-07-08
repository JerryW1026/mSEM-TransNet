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
# 1. 核心对比基线: DeepConvNet (深层重型 CNN 代表)
# 适配 12通道 200Hz SEED 数据集 (输入长度 200)
# ==========================================
class DeepConvNet(nn.Module):
    def __init__(self, num_classes=3, channels=12, samples=200):
        super(DeepConvNet, self).__init__()

        # Block 1: 时空特征提取
        # 时间维卷积 (T=200 -> 191)
        self.conv1 = nn.Conv2d(1, 25, (1, 10))
        # 空间维卷积 跨越所有12个通道
        self.conv2 = nn.Conv2d(25, 25, (channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        # T=191 -> 95
        self.pool1 = nn.MaxPool2d((1, 2))

        # Block 2: 深层特征提取
        # T=95 -> 86
        self.conv3 = nn.Conv2d(25, 50, (1, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        # T=86 -> 43
        self.pool2 = nn.MaxPool2d((1, 2))

        # Block 3: 更深层特征提取
        # T=43 -> 34
        self.conv4 = nn.Conv2d(50, 100, (1, 10), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        # T=34 -> 17
        self.pool3 = nn.MaxPool2d((1, 2))

        # Block 4: 顶层特征提取
        # T=17 -> 8
        self.conv5 = nn.Conv2d(100, 200, (1, 10), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        # T=8 -> 4
        self.pool4 = nn.MaxPool2d((1, 2))

        self.dropout = nn.Dropout(0.5)

        # Classifier: 200 个通道，时间维度剩 4
        out_features = 200 * 4
        self.fc = nn.Linear(out_features, num_classes)

    def forward(self, x):
        # 维度对齐: (Batch, 1, Channels, Time)
        if x.dim() == 3: x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.dropout(x)

        # 展平与分类
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
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
# 3. 跨会话评估逻辑 (调用 DeepConvNet)
# ==========================================
def train_and_evaluate_cross_session(train_files, test_files, device, epochs=100, batch_size=64, lr=1e-3):
    train_dataset = SEEDDataset12Ch(train_files)
    test_dataset = SEEDDataset12Ch(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化深层 CNN 代表：DeepConvNet
    model = DeepConvNet(num_classes=3, channels=12, samples=200).to(device)
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
    # 📌 确保此路径与跑之前实验时完全一致
    data_dir = r"C:\Users\WJR\Desktop\分区+多特征融合实验\EEG-TransNet-main\EEG-TransNet-main -+分区\SEED_Data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_subject_acc = []
    # DeepConvNet 参数量庞大，设置 100 轮让其充分收敛
    run_epochs = 100

    for subject_id in range(1, 16):
        print(f"\n{'-' * 70}")
        print(f" 🥊 开始对比评估被试 (Subject) {subject_id}/15 [模型: DeepConvNet Baseline]")

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
        print(f" 🏆 被试 {subject_id} DeepConvNet 跨会话最佳准确率: {acc:.2f}%")

    if len(all_subject_acc) > 0:
        final_mean_acc = np.mean(all_subject_acc)
        final_std_acc = np.std(all_subject_acc)

        print(f"\n{'=' * 70}")
        print(" 🎉 15 名被试 [DeepConvNet 对比实验] 全部完成！")
        print(f" 🎯 采用配置: 深层 CNN 代表 DeepConvNet (12 通道, SEED 3分类)")
        print(f" 总体平均准确率 (Mean ACC): {final_mean_acc:.2f}%")
        print(f" 总体标准差 (Std Dev): {final_std_acc:.2f}%")
        print(f" 各被试结果阵列: {[round(acc, 2) for acc in all_subject_acc]}")
        print(f"{'=' * 70}")