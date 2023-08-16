from torch.utils.data import DataLoader, TensorDataset
from get_dataset_ADSB import *#从名为get2的模块中导入所有的可用对象
# from ResNet_Model import *
from ResNet_model import *
from confusion import confusion

def test(model, test_dataloader, rand_num):
    model.eval()
    correct = 0
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []#为了后续的数据存储或处理而创建这些空列表
    with torch.no_grad():#禁止梯度计算，在优化器更新参数时节省内存和计算资源。
        for data, target in test_dataloader:
            target = target.long()#将 target 转换为长整型可能是为了确保数据类型的一致性
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target) - 1] = pred.tolist()#切片赋值的方式允许我们替换现有列表中的特定范围，以及将其他列表的元素添加到已有列表中的特定位置
            target_real[len(target_real):len(target) - 1] = target.tolist()#target是真实标签的列表，target_pred是预测结果的列表

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'#板中的三个占位符分别是 {}, {} 和 {:.6f}%，分别对应准确度的计数、总样本数和百分比形式的准确度
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )  # 打印当前模型在测试集上的准确性
    return target_pred, target_real


def Data_prepared(n_classes, rand_num):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)
#准备训练数据集和验证数据集的特征(X_train和X_val)以及相应的标签(value_Y_train和value_Y_val)
    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value


# 准备测试数据集
def TestDataset_prepared(n_classes, rand_num):
    X_test, Y_test = TestDataset(n_classes)  # 调用了 TestDataset() 函数来获得测试集

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)  # 归一化测试集中的数据，每个特征值都减去最小值，并除以特征值的范围（最大值与最小值之差）

    X_test = X_test.transpose(0, 2, 1)  # 张量维度进行转换

    return X_test, Y_test


def main():
    rand_num = 50  # 生成随机数
    X_test, Y_test = TestDataset_prepared(20, rand_num)  # 调用准备好的数据集

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))  # 使用 torch.Tensor() 函数将测试数据转换为张量（Tensor），并使用 TensorDataset() 函数将测试集发包装为数据集
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # 使用 DataLoader() 函数创建一个数据加载器（dataloader），用于对测试数据进行迭代访问，shuffle=True 表示打乱数据顺序，以确保每次迭代访问的数据不同

    model = torch.load('model_weight/train_ADSb_ResNet.pth')
    print(model)
    pred, real = test(model, test_dataloader, rand_num)
    # 是使用给定的模型model在测试数据集test_dataloader上进行预测，并返回预测结果pred和真实标签real
    confusion(pred, real, range(20))


if __name__ == '__main__':#这段代码的作用是判断当前脚本是否作为主程序直接执行
    main()
