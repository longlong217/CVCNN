import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from CV_CVCNN_Model import *
from get_dataset_GSM import TrainDataset, TestDataset
import random
import os
from sklearn.model_selection import train_test_split

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        output = model(data)
        classifier_output = F.log_softmax(output, dim=1)
        dim = 1

        classifier_loss_batch = loss(classifier_output, target)
        result_loss_batch = classifier_loss_batch
        result_loss_batch.backward()
        optimizer.step()

        classifier_loss += result_loss_batch.item()
        pred = classifier_output.argmax(dim=1, keepdim=True)
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


def test(model, loss, test_dataloader, epoch, writer, device_num):
    model.eval()
    val_loss = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(test_dataloader)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * (correct / len(test_dataloader.dataset)),
        )
    )

    writer.add_scalar('Accuracy/validation', 100.0 * (correct / len(test_dataloader.dataset)), epoch)
    writer.add_scalar('Classifier_Loss/validation', val_loss, epoch)
    return val_loss


def train_and_test(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer, save_path,
                   device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch, writer, device_num)
        val_loss = test(model, loss_function, val_dataloader, epoch, writer, device_num)
        if val_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, val_loss))
            current_min_test_loss = val_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

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


def TrainDataset_prepared(n_classes, rand_num):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    # X_train = X_train.transpose(0, 2, 1)#对训练集的输入特征进行转置操作
    # X_val = X_val.transpose(0, 2, 1)#对验证集进行转置操作（）

    return X_train, X_val, value_Y_train, value_Y_val

class Config:
    def __init__(
            self,
            batch_size: int = 32,
            test_batch_size: int = 32,
            epochs: int = 300,
            lr: float = 0.001,
            n_classes: int = 12,
            save_path: str = 'model_weight/train_GSM_CVCNN_lunwen.pth',
            device_num: int = 0,
            rand_num: int = 50,
    ):
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
    writer = SummaryWriter("logs_train_GSM_CVCNN_rand_lunwen" + str(conf.rand_num))
    device = torch.device("cuda:" + str(conf.device_num))

    RANDOM_SEED = 300
    set_seed(RANDOM_SEED)

    X_train, X_val, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)
    model = base_complex_model()
    if torch.cuda.is_available():
        model = model.to(device)
    print(model)
    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)
    train_and_test(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                   optimizer=optim, epochs=conf.epochs, writer=writer, save_path=conf.save_path,
                   device_num=conf.device_num)

if __name__ == '__main__':
    main()
