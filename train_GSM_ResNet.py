import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from model_complexcnn_onlycnn import *#导入了名为model_complexcnn_onlycnn的模块
from CV_ResNet_Model import *
from get_dataset_GSM import TrainDataset, TestDataset #导入了名为 TrainDataset 和 TestDataset 的类，它们来自 get2 模块
import random
import os
from sklearn.model_selection import train_test_split


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#设置随机种子(seed)以确保实验的可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化，哈希值是一种将任意长度的数据映射为固定长度值的算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def train(model, loss, train_dataloader, optimizer, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:" + str(device_num))
    correct = 0
    classifier_loss = 0
    for data_nnl in train_dataloader:
        data, target = data_nnl
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        output = model(data)#更新模型
        classifier_output = F.log_softmax(output, dim=1)
        dim = 1
        #参数表示在第1个维度上进行softmax运算
        #NLLLoss代表负对数似然损失，前向传播（预测值）——概率分布（softmax操作），用于后面的NLLLoss计算来预测概率分布和真实标签概率分布之间的差异
        classifier_loss_batch = loss(classifier_output, target)
        result_loss_batch = classifier_loss_batch#将分类器损失值存储到变量result_loss_batch中，用于后续的处理或记录。
        result_loss_batch.backward()#自动计算损失函数关于每个可学习参数的梯度，并将梯度存储在参数的.grad属性中。然后，我们可以使用这些梯度来更新参数，从而最小化损失函数。
        optimizer.step()#通过调用optimizer.step()，模型的参数根据梯度大小和学习率等因素进行更新

        classifier_loss += result_loss_batch.item()#每个batch做类和并存储在classifier_loss中
        pred = classifier_output.argmax(dim=1, keepdim=True)#根据分类器的输出classifier_output,使用argmax()函数找到具有最高概率的类别，并存储在pred中。
        correct += pred.eq(target.view_as(pred)).sum().item()
    classifier_loss /= len(train_dataloader)
    print('Train Epoch: {} \tClassifier_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        classifier_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * (correct / len(train_dataloader.dataset)), epoch)
    writer.add_scalar('Classifier_Loss/train', classifier_loss, epoch)


def test(model, loss, test_dataloader, epoch, writer, device_num):#验证集
    model.eval()#准备模型
    val_loss = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():#用于在执行代码块时禁用梯度计算
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += loss(output, target).item()#计算每个batch的损失值，并做类和
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(test_dataloader)#len(test_dataloader)：测试数据集的总批次数量#将累计的验证损失值除以总批次数量，得到验证集的平均损失
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * (correct / len(test_dataloader.dataset)),#len(test_dataloader.dataset)：测试数据集中的总样本数量
        )
    )

    writer.add_scalar('Accuracy/validation', 100.0 * (correct / len(test_dataloader.dataset)), epoch)
    writer.add_scalar('Classifier_Loss/validation', val_loss, epoch)

    return val_loss


def train_and_test(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer, save_path,
                   device_num):
    current_min_test_loss = 100#根据验证集的loss大小来设置，所设置的要保证大于验证集的loss值
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch, writer, device_num)#函数调用
        val_loss = test(model, loss_function, val_dataloader, epoch, writer, device_num)#函数调用
        if val_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, val_loss))
            current_min_test_loss = val_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

#获得训练集和验证集数据两个中的最大值和最小值，便于后续归一化数据、确定模型输入范围、预处理步步骤。
def Data_prepared(n_classes, rand_num):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

    min_value = X_train.min()#假设训练集的最小值为最小值
    min_in_val = X_val.min()#假设验证集的最小值为最小值
    if min_in_val < min_value:
        min_value = min_in_val#比较大小后，取最小的那个为最小

    max_value = X_train.max()#假设验证集的最大值为最大值
    max_in_val = X_val.max()#假设验证集的最大值为最大值
    if max_in_val > max_value:
        max_value = max_in_val#比较大小后，取最大的那个为最大

    return max_value, min_value


def TrainDataset_prepared(n_classes, rand_num):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_train = (X_train - min_value) / (max_value - min_value)#对训练集的输入特征进行归一化操作
    X_val = (X_val - min_value) / (max_value - min_value)#对验证集的输入特征进行归一化操作

    # X_train = X_train.transpose(0, 2, 1)#对训练集的输入特征进行转置操作
    # X_val = X_val.transpose(0, 2, 1)#对验证集进行转置操作（）

    return X_train, X_val, value_Y_train, value_Y_val

#config类用于存储和管理模型训练的各种参数
class Config:
    def __init__(
            self,
            batch_size: int = 32,#训练时每个批次的样本数量
            test_batch_size: int = 32,#在测试或验证阶段每个批次的样本数量
            epochs: int = 300,#训练轮数
            lr: float = 0.001,
            n_classes: int = 12,#涉及的分类数
            save_path: str = 'model_weight/train_GSM_ResNet_lunwen.pth',
            device_num: int = 0,#指定使用的计算设备编号
            rand_num: int = 50,#随机数，使用时需要根据实际情况进行解释和使用，默认为 50。
    ):#以下在定义 Config 类的构造函数中，用于将传入的参数值赋给类的成员变量
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_classes = n_classes
        self.save_path = save_path
        self.device_num = device_num
        self.rand_num = rand_num


def main():
    conf = Config()
    writer = SummaryWriter("logs_train_GSM_ResNet_lunwen_rand" + str(conf.rand_num))#这个路径将用于存储TensorBoard日志文件
    device = torch.device("cuda:" + str(conf.device_num))

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train, X_val, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train))#TensorDataset用于封装张量数据的类，它能够同时将多个张量按照相同的索引进行组合。
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)#训练数据加载器，并按批次加载数据并进行训练，
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)#shuffle=Tru，每个epoch开始时打乱数据顺序的目的是为了增加训练的随机性和泛化能力
    model = Network(2, 128, 10, 12)
    if torch.cuda.is_available():
        model = model.to(device)
    print(model)
    loss = nn.NLLLoss()
    if torch.cuda.is_available():#判定可将损失函数转移到GPU上计算，否的话就跳过该步骤。
        loss = loss.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)#更新模型参数，weight_decay表示L2正则化的权重衰减系数它 它=0表示不进行权重衰减
    train_and_test(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                   optimizer=optim, epochs=conf.epochs, writer=writer, save_path=conf.save_path,
                   device_num=conf.device_num)#用来训练和更新模型

if __name__ == '__main__':
    main()
