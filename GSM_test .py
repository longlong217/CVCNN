from torch.utils.data import DataLoader, TensorDataset
from get_dataset_GSM import *
# from ResNet_Model import *
from CV_CVCNN_Model import *
from confusion import confusion

def test(model, test_dataloader, rand_num):
    model.eval()
    correct = 0
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            target_real[len(target_real):len(target) - 1] = target.tolist()

        target_pred = np.array(target_pred)
        target_real = np.array(target_real)

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return target_pred, target_real


def Data_prepared(n_classes, rand_num):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

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
    X_test, Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)

    #X_test = X_test.transpose(0, 2, 1)  # 张量维度进行转换

    return X_test, Y_test


def main():
    rand_num = 50
    X_test, Y_test = TestDataset_prepared(12, rand_num)

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = torch.load('model_weight/train_GSM_CVCNN_lunwen.pth')
    print(model)
    pred, real = test(model, test_dataloader, rand_num)

    confusion(pred, real, range(12))


if __name__ == '__main__':
    main()
