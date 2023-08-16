import torch
from thop import profile
from torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv

# Residual Block with SEBlock
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            ComplexConv(int(channels/2), int(channels/2), 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            ComplexConv(int(channels/2), int(channels/2), 3, padding=1, bias=False),
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
            ComplexConv(int(in_channel/2), int(filters/2), 3, padding=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        # self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks - 1)])
        a=[]
        for i in range(blocks-1):
            a.append(ResBlock(filters))
        self.res_blocks = nn.Sequential(*a)

        self.out_conv = nn.Sequential(
            ComplexConv(int(filters/2), 64, 1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        #x = self.conv_block(x)
        x = self.conv_block.forward(x)
        x = self.res_blocks(x)

        x = self.out_conv(x)
        x = F.adaptive_avg_pool1d(x, 1)

        x = x.view(x.data.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    input = torch.randn((32, 2, 4800))
    model = Network(2, 128, 10, 12)
    output = model.forward(input)  # torch.Size([32, 2, 4800]) --> torch.Size([32, 10])
    print(output.shape)
