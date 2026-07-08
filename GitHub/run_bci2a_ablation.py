import os
import sys
import time
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data.data_utils import load_BCI42_data
from einops import rearrange
from sklearn.metrics import cohen_kappa_score  # <--- 新增：导入 Kappa 计算函数
import warnings

warnings.filterwarnings('ignore')
torch.set_num_threads(10)


# =========================================================
# 🌟 新增：双路日志记录器 (同时输出到控制台和 txt 文件)
# =========================================================
class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 实时刷新，防止程序中断丢失日志

    def flush(self):
        pass


# =========================================================
# 0. 全局设置
# =========================================================
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_BCI_data_standalone(data_path, filename):
    filepath = os.path.join(data_path, filename + '.mat')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据文件: {filepath}")
    mat_data = sio.loadmat(filepath)
    keys = list(mat_data.keys())
    data_key = next((k for k in keys if 'data' in k.lower() or k.lower() == 'x'), None)
    label_key = next((k for k in keys if 'label' in k.lower() or k.lower() == 'y'), None)
    if data_key is None or label_key is None:
        valid_keys = [k for k in keys if not k.startswith('__')]
        data_key, label_key = valid_keys[0], valid_keys[1]
    X = mat_data[data_key]
    y = mat_data[label_key].squeeze()
    return X, y


# =========================================================
# 1. 基础依赖组件 (Transformer & mSEM)
# =========================================================
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
        return self.dropout(self.w_o(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.act(self.w_1(x)))))


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
        return out + self.feed_forward(res)


class mSEM(nn.Module):
    def __init__(self, F1, D, num_channels, region_indices):
        super(mSEM, self).__init__()
        self.region_indices = region_indices
        self.F1 = F1
        self.local_convs = nn.ModuleList([
            nn.Conv2d(F1, F1, (len(idx), 1), groups=F1, bias=False) for idx in region_indices
        ])
        self.global_conv = nn.Conv2d(F1, F1 * D, (num_channels + len(region_indices), 1), groups=F1, bias=False)
        self.bn = nn.BatchNorm2d(F1 * D)

    def forward(self, x):
        local_features = [self.local_convs[i](x[:, :, idx, :]) for i, idx in enumerate(self.region_indices)]
        concat_x = torch.cat([x, torch.cat(local_features, dim=2)], dim=2)
        return self.bn(self.global_conv(concat_x))


# =========================================================
# 2. 消融变体模型库 (BCI-2a 22通道专用)
# =========================================================

# 🟢 变体 1: 完整版 (Full Model)
class mSEM_TransNet_Full(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, pool_stride=15,
                 num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
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
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)])
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


# 🔴 变体 2: 无多频段提取 (Single-Scale Time Conv)
class Ablation_wo_MultiFreq(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, pool_stride=15,
                 num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.idx_left = [0, 1, 2, 6, 7, 8, 13, 14, 18]
        self.idx_mid = [3, 9, 15, 19, 21]
        self.idx_right = [4, 5, 10, 11, 12, 16, 17, 20]
        region_indices = [self.idx_left, self.idx_mid, self.idx_right]

        self.single_temp_conv = nn.Conv2d(1, embed_dim, (1, 25), padding=(0, 12), bias=False)
        self.bn_temp = nn.BatchNorm2d(embed_dim)
        self.msem = mSEM(F1=embed_dim, D=1, num_channels=num_channels, region_indices=region_indices)
        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)])
        self.conv_encoder = nn.Sequential(nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
                                          nn.BatchNorm2d(temp_embedding_dim), nn.ELU())
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.single_temp_conv(x)
        x = self.elu(self.msem(self.bn_temp(x))).squeeze(2)
        x1, x2 = rearrange(self.dropout(self.avg_pool(x)), 'b d n -> b n d'), rearrange(self.dropout(self.var_pool(x)),
                                                                                        'b d n -> b n d')
        for encoder in self.transformer_encoders: x1, x2 = encoder(x1), encoder(x2)
        return self.classify(
            self.conv_encoder(torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), dim=2)).reshape(x.size(0), -1))


# 🔵 变体 3: 无分区空间卷积 (Global Spatial Conv Only)
class Ablation_wo_mSEM(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, pool_stride=15,
                 num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.F1_per_branch = embed_dim // 4
        self.temp_conv1 = nn.Conv2d(1, self.F1_per_branch, (1, 15), padding=(0, 7), bias=False)
        self.temp_conv2 = nn.Conv2d(1, self.F1_per_branch, (1, 25), padding=(0, 12), bias=False)
        self.temp_conv3 = nn.Conv2d(1, self.F1_per_branch, (1, 51), padding=(0, 25), bias=False)
        self.temp_conv4 = nn.Conv2d(1, self.F1_per_branch, (1, 65), padding=(0, 32), bias=False)
        self.bn_temp = nn.BatchNorm2d(embed_dim)

        self.global_spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), bias=False)
        self.bn_spatial = nn.BatchNorm2d(embed_dim)

        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout(attn_drop)
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)])
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


# 🟡 变体 4: 无注意力机制 (CNN Sequence Encoder)
class Ablation_wo_Transformer(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, pool_stride=15,
                 attn_drop=0.5):
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


# =========================================================
# 3. 独立且标准的训练/测试引擎 (增加 Kappa 计算)
# =========================================================
def run_model_training(model, train_loader, test_loader, device, epochs=150, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_test_acc = 0.0
    best_test_kappa = 0.0  # 新增：记录与最佳准确率对应的 Kappa 值

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Test ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # 收集用于计算指标的真实值和预测值
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算准确率和 Kappa
        correct = np.sum(np.array(all_preds) == np.array(all_labels))
        test_acc = 100 * correct / len(all_labels)
        test_kappa = cohen_kappa_score(all_labels, all_preds)

        # 同步更新最佳的 Acc 和 Kappa
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_kappa = test_kappa

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(
                f"      Epoch [{epoch + 1:03d}/{epochs}] Loss: {train_loss / len(train_loader):.4f} | Test Acc: {test_acc:.2f}% (Kappa: {test_kappa:.4f}) | Best Acc: {best_test_acc:.2f}% (Best Kappa: {best_test_kappa:.4f})")

    return best_test_acc, best_test_kappa


# =========================================================
# 4. 主控程序 (执行 9 个被试的 4 维消融)
# =========================================================
if __name__ == '__main__':
    # 📌 数据集路径
    data_path = r"C:\Users\WJR\Desktop\分区+多特征融合实验\EEG-TransNet-main\EEG-TransNet-main -+分区\data\bci_iv_2a1"

    # 📌 创建 output 文件夹并配置日志
    out_dir = r"C:\Users\WJR\Desktop\分区+多特征融合实验\EEG-TransNet-main\EEG-TransNet-main -+分区\output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(out_dir, f"Ablation_Log_{timestamp}.txt")

    # 将标准输出定向到 Logger，实现控制台和文件的双路写入
    sys.stdout = Logger(log_file_path)

    print(f"✅ 执行开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ 运行日志将实时保存至: {log_file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 执行设备: {device}")

    ablation_models = {
        '1_Full_mSEM_TransNet': mSEM_TransNet_Full,
        '2_wo_MultiFreq': Ablation_wo_MultiFreq,
        '3_wo_mSEM': Ablation_wo_mSEM,
        '4_wo_Transformer': Ablation_wo_Transformer
    }

    # 存储最终结果的字典：分别记录 acc 和 kappa
    final_results = {name: {'acc': [], 'kappa': []} for name in ablation_models.keys()}

    for model_name, ModelClass in ablation_models.items():
        print(f"\n{'=' * 70}")
        print(f"🚀 开始评估模型变体: [{model_name}]")
        print(f"{'=' * 70}")

        for subId in range(1, 10):
            print(f"\n---> 正在处理 被试 Subject A0{subId} ...")

            train_filename = f'A0{subId}T'
            test_filename = f'A0{subId}E'

            try:
                X_train, y_train = load_BCI42_data(data_path, train_filename)
                X_test, y_test = load_BCI42_data(data_path, test_filename)
            except Exception as e:
                print(f"❌ 数据加载失败: {e}。")
                continue

            if np.max(y_train) == 4:
                y_train = y_train - 1
            elif np.min(y_train) == -1:
                y_train = y_train + 1

            if np.max(y_test) == 4:
                y_test = y_test - 1
            elif np.min(y_test) == -1:
                y_test = y_test + 1

            batch_size = 64
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                                      batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size,
                                     shuffle=False)

            set_random_seed(42)
            model = ModelClass(num_classes=4, num_samples=1000, num_channels=22).to(device)

            # 接收返回的 acc 和 kappa
            best_acc, best_kappa = run_model_training(model, train_loader, test_loader, device, epochs=150, lr=1e-3)

            final_results[model_name]['acc'].append(best_acc)
            final_results[model_name]['kappa'].append(best_kappa)
            print(f"🏆 {model_name} @ A0{subId} 最终最佳 -> ACC: {best_acc:.2f}% | Kappa: {best_kappa:.4f}")

    # 打印并保存最终对比表格
    if len(final_results['1_Full_mSEM_TransNet']['acc']) > 0:
        print("\n" + "🌟" * 40)
        print(" BCI Competition IV 2a 消融实验最终结果汇总 ")
        print("🌟" * 40)

        for name, metrics in final_results.items():
            acc_list = metrics['acc']
            kappa_list = metrics['kappa']

            mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
            mean_kappa, std_kappa = np.mean(kappa_list), np.std(kappa_list)

            print(f" 🔹 {name:<25}")
            print(f"    - Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
            print(f"    - Kappa    : {mean_kappa:.4f} ± {std_kappa:.4f}")
            print(f"    - 各被试 ACC: {[round(a, 2) for a in acc_list]}")
            print(f"    - 各被试 KPA: {[round(k, 4) for k in kappa_list]}")
            print("-" * 50)

        print("🌟" * 40 + "\n")
        print(f"✅ 本次实验的所有信息已成功保存至: {log_file_path}")