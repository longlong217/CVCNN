import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random


def TrainDataset(num, random_num):
    x = np.load(f"Dataset_ADS-B/X_train_20Class.npy")
    y = np.load(f"Dataset_ADS-B/Y_train_20Class.npy")
    y = y.astype(np.uint8)

    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.3, random_state=random_num)

    return X_train, X_val, Y_train, Y_val


# 加载测试数据集
def TestDataset(num):
    x = np.load(f"Dataset_ADS-B/X_test_20Class.npy")
    y = np.load(f"Dataset_ADS-B/Y_test_20Class.npy")
    y = y.astype(np.uint8)
    return x, y

if __name__ == "main":
    c=TestDataset(20)
    print(c)

