import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from visdom import Visdom
from model.TransNet import TransNet
from model.baseModel import baseModel
from model.GL_TransNet1 import GL_TransNet
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset
from utils import *
import time
from model.PartitionTransNet import PartitionTransNet
from model.mSEM_TransNet import mSEM_TransNet
torch.set_num_threads(10)
def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def main(config):
    data_path = config['data_path']
    out_folder = config['out_folder']
    random_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    
    lr = config['lr']

    for subId in range(1,10):
        train_datafile = 'A0' + str(subId) + 'T'
        test_datafile = 'A0' + str(subId) + 'E'

        out_path = os.path.join(out_folder, config['network'], 'sub'+str(subId), random_folder)
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        print("Results will be saved in folder: " + out_path)

        dictToYaml(os.path.join(out_path, 'config.yaml'), config)

        setRandom(config['random_seed'])

        train_data, train_labels = load_BCI42_data(data_path, train_datafile)
        test_data, test_labels = load_BCI42_data(data_path, test_datafile)

        # =========================================================
        # === 修改开始: 标签智能修正 (v2.0) ===
        # =========================================================

        print(f"原始训练标签范围: Min={np.min(train_labels)}, Max={np.max(train_labels)}")
        print(f"原始测试标签范围: Min={np.min(test_labels)}, Max={np.max(test_labels)}")

        # --- 修正训练集 ---
        if np.max(train_labels) == 4:
            print(f"Warning: 检测到 Subject {subId} 训练集标签为 1-4，修正为 0-3 (-1)")
            train_labels = train_labels - 1
        elif np.min(train_labels) == -1:
            print(f"Warning: 检测到 Subject {subId} 训练集标签为 -1~2，修正为 0-3 (+1)")
            train_labels = train_labels + 1

        # --- 修正测试集 (针对你现在的 -1, 2 情况) ---
        if np.max(test_labels) == 4:
            print(f"Warning: 检测到 Subject {subId} 测试集标签为 1-4，修正为 0-3 (-1)")
            test_labels = test_labels - 1
        elif np.min(test_labels) == -1:
            # 这就是解决你当前报错的关键！
            print(f"Warning: 检测到 Subject {subId} 测试集标签为 -1~2，修正为 0-3 (+1)")
            test_labels = test_labels + 1

        # --- 最终安全检查 ---
        # 必须确保标签都在 [0, 1, 2, 3] 范围内
        if np.min(train_labels) < 0 or np.max(train_labels) > 3:
            raise ValueError(f"训练标签修正失败！范围: {np.min(train_labels)} ~ {np.max(train_labels)}")

        if np.min(test_labels) < 0 or np.max(test_labels) > 3:
            raise ValueError(f"测试标签修正失败！范围: {np.min(test_labels)} ~ {np.max(test_labels)}")

        print(">>> 标签检查通过，范围正常 [0-3]")

        # =========================================================
        # === 修改结束 =============================================
        # =========================================================


        train_dataset = eegDataset(train_data, train_labels)
        test_dataset = eegDataset(test_data, test_labels)

        net_args = config['network_args']
        net = eval(config['network'])(**net_args)
        print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        model = baseModel(net, config, optimizer, loss_func, result_savepath=out_path)

        model.train_test(train_dataset, test_dataset)

if __name__ == '__main__':
    configFile = 'config/bciiv2a_transnet.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)

