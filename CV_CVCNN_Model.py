import torch
from torch import nn
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear, Dropout
from torch.nn import ReLU, Softmax
from complexcnn import ComplexConv#导入了名为 ComplexConv 的类，它来自 complexcnn 模块
import torch.nn.functional as F

class base_complex_model(nn.Module):
    def __init__(self):#初始化#定义
        super(base_complex_model, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=3)#复数卷积操作，用于处理输入数据中的实部和虚部
        self.batchnorm1 = BatchNorm1d(num_features=128) #批量标准化层：加速训练
        self.maxpool1 = MaxPool1d(kernel_size=2) #一维最大池化层：减小特征图的维度

        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=3)
        self.batchnorm2 = BatchNorm1d(num_features=128)
        self.maxpool2 = MaxPool1d(kernel_size=2)

        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = BatchNorm1d(num_features=128)
        self.maxpool3 = MaxPool1d(kernel_size=2)

        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = BatchNorm1d(num_features=128)
        self.maxpool4 = MaxPool1d(kernel_size=2)

        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = BatchNorm1d(num_features=128)
        self.maxpool5 = MaxPool1d(kernel_size=2)

        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = BatchNorm1d(num_features=128)
        self.maxpool6 = MaxPool1d(kernel_size=2)

        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = BatchNorm1d(num_features=128)
        self.maxpool7 = MaxPool1d(kernel_size=2)

        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = BatchNorm1d(num_features=128)
        self.maxpool8 = MaxPool1d(kernel_size=2)

        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = BatchNorm1d(num_features=128)
        self.maxpool9 = MaxPool1d(kernel_size=2)



#多层感知机---全连接层：将前面提取到的特征映射到模型输出的类别空间中
        self.flatten = Flatten()#全连接层的输入需要是一维向量。所以使用Flatten层将多维特征图转换为一维向量，以进行后续的全连接操作
        self.linear1 = LazyLinear(512)
        self.linear2 = LazyLinear(128)
        self.linear3 = LazyLinear(12)

    def forward(self,x):
        x = self.conv1(x)#调用 self.conv1 对象，将输入数据 x 处理为卷积操作后的输出张量 x
        x = F.relu(x)#对输出张量 x 进行 ReLU 激活函数操作对张量进行ReLU激活函数操作可以将负数变为0，而将正数保持不变，使其具有非线性特性
        x = self.batchnorm1(x)#进行标准化处理，加速神经网络的训练和提高模型精度
        x = self.maxpool1(x)#将标准化后的张量输入到最大池化层 self.maxpool1 中进行下采样操作，减小特征图的大小，并提取主要特征

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        embedding = F.relu(x)# ReLU激活函数操作
        output = self.linear3(embedding)#实现了一个线性修正单元 (ReLU)，用于对输入特征进行非线性变换，提高预测能力
        return output

if __name__ == "__main__":#表示如果当前模块作为主程序运行，则执行下面的代码，否则不执行
    input = torch.randn((32,2,2600))#创建一个张量对象input:表示有 10 个样本，每个样本有 2 个通道（IQ信号），每个通道有 4800 个时间或位置信息（可以理解为信号长度为 4800（））。
    model = base_complex_model()#通过调用 base_complex_model 方法创建了一个基于复杂卷积神经网络的模型对象 model
    output = model(input)#将随机张量对象 input 传递给模型对象 model 进行前向传播，即计算输入数据的输出结果
    #便模块的可重用性和跨平台性，同时也能使代码更具可读性和易于维护
    print(output.shape)
