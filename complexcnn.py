#这段代码定义了一个用于复杂数卷积的PyTorch模型。
# 使用了一个 ComplexConv 类，它包含两个 1-D 的卷积层，其中一个是处理实部（x_real），另一个则是处理虚部 (x_img)
#当需要实现处理带虚数数据的卷积任务时，这种基于实部和虚部独立卷积的复数卷积实现方法是十分高效和可行的
# -*- coding: utf-8 -*-# Python 源文件中的注释语句，为了适应不同的python版本

import torch
import torch.nn as nn
import numpy as np  # numpy 主要对象是具有相同数据类型的多维数组

class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        # 在初始化参数中输入各种参数来构造一个复数卷积神经网络模型
        super(ComplexConv, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 用于检测当前系统是否使用了可用的 GPU，并动态选择在 CPU 或 GPU 上执行代码
        self.padding = padding  # 设置数据在各个维度上的 padding 大小，padding是在输入数据周围添加额外的0元素，以保证输出大小等于输入大小

        ## Model components 模型的构造组件
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)  # 定义一个 nn.Conv1d 类型的实例 conv_re
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)  # 定义另一个 nn.Conv1d 类型的实例 conv_im
        # 因为是complexCNN，故而复数存在实部和虚部，因此需要分两层来处理输入数据，再生成新的复数数据输出

    def forward(self, x):
        x_real = x[:, 0:x.shape[1] // 2, :]#以，为分界来看！
        x_img = x[:, x.shape[1] // 2: x.shape[1], :]

         #以下进行的是复数卷积层的计算
        real = self.conv_re(x_real) - self.conv_im(x_img) #卷积后的实部输出
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)#卷积后的虚部输出
        output = torch.cat((real, imaginary), dim=1)#torch.cat()：指定维度（dim=几）上的多个张量的拼接，dim=需要拼在一起的变量个数，如dim=1表示两个拼在一起（0表示第一个）
        return output
