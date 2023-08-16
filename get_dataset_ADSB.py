import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random


def TrainDataset(num, random_num):  # 导入了 Python 的内置 random 模块，用于生成随机数#num：表示数据集的类别数和random_num表示划分数据时使用的随机数
    x = np.load(f"Dataset_ADS-B/X_train_20Class.npy")
    y = np.load(f"Dataset_ADS-B/Y_train_20Class.npy")  # 加载训练数据集的特征和标签
    y = y.astype(np.uint8)  # 将数组 y 的数据类型转换为无符号整型(uint8)

    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.3, random_state=random_num)

    return X_train, X_val, Y_train, Y_val


# 加载测试数据集
def TestDataset(num):  # num表示要加载哪种类型的测试数据
    x = np.load(f"Dataset_ADS-B/X_test_20Class.npy")
    y = np.load(f"Dataset_ADS-B/Y_test_20Class.npy")
    # np.load() 函数可以按照给定的文件名读取指定的数组数据文件，并返回其中保存的 Numpy 数组类型的数据
    y = y.astype(np.uint8)#以便进行图像的显示、处理等操作
    return x, y  # 返回测试集的特征矩阵 x 和标签向量 y

if __name__ == "main":
    c=TestDataset(20)
    print(c)

