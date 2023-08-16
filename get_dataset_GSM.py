import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
from scipy.io import loadmat, savemat


def TrainDataset(num, random_num):  # 导入了 Python 的内置 random 模块，用于生成随机数#num：表示数据集的类别数和random_num表示划分数据时使用的随机数
    # data = loadmat('datasets\GSMData.mat')
    # GSM_data = data['arraydata_GSM']
    # GSM_data_trans =GSM_data.transpose(2, 0, 1)
    # GSM_data_down_trans= GSM_data_trans.reshape(-1, GSM_data_trans.shape[2])
    #
    # GSM_real=GSM_data_down_trans.real
    # GSM_imag = GSM_data_down_trans.imag
    #
    # GSM_real_3d= np.expand_dims(GSM_real, axis=1)
    # GSM_imag_3d = np.expand_dims(GSM_imag, axis=1)
    # GSM_3d = np.concatenate((GSM_real_3d, GSM_imag_3d), axis=1)
    #
    # GSM_data_trans_shape=GSM_data_trans.shape
    # label=[]
    # for i in range(12):
    #     label=label+[i]*GSM_data_trans_shape[1]

    # x = np.array(GSM_3d)
    # y = np.array(label) # 加载训练数据集的特征和标签
    x = np.load(f"Dataset_GSM/X_train.npy")
    y = np.load(f"Dataset_GSM/Y_train.npy")
    y = y.astype(np.uint8)  # 将数组 y 的数据类型转换为无符号整型(uint8)

    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.3, random_state=random_num)

    return X_train, X_val, Y_train, Y_val


# 加载测试数据集
def TestDataset(num):  # num表示要加载哪种类型的测试数据
    x = np.load(f"Dataset_GSM/X_test.npy")
    y = np.load(f"Dataset_GSM/Y_test.npy")
    # np.load() 函数可以按照给定的文件名读取指定的数组数据文件，并返回其中保存的 Numpy 数组类型的数据
    y = y.astype(np.uint8)#以便进行图像的显示、处理等操作
    return x, y  # 返回测试集的特征矩阵 x 和标签向量 y

if __name__ == "main":
    c=TestDataset(12)
    print(c)

