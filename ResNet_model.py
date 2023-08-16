import torch
from thop import profile
from torch import nn
import torch.nn.functional as F


# Residual Block with SE-Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        path = x + path
        return F.relu(path)


# Network Module
class Network(nn.Module):
    def __init__(self, in_channel, filters, blocks, num_classes):
        super(Network, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks - 1)])
        self.out_conv = nn.Sequential(
            nn.Conv1d(filters, 128, 1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)

        x = self.out_conv(x)
        x = F.adaptive_avg_pool1d(x, 1)

        x = x.view(x.data.size(0), -1)
        x = self.fc(x)

        return x


# if __name__ == "__main__":
#     a=torch.load('model_weight/train_ADSb_ResNet.pth')
#     input = torch.randn((32, 2, 4800))
#     model = Network(2, 128, 10, 20)
#     output = model(input)
#     print(output.shape)
