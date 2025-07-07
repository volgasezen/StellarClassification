import torch
from torch import nn
import torch.nn.functional as F

class ConvBackbone(nn.Module):
    def __init__(self, filter_sizes):
        super().__init__()
        self.n_filt = len(filter_sizes)-1
        self.filters = [filter_sizes[i:i+self.n_filt] for i in range(2)]

        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 3)
                                    for i,j in zip(*self.filters)])

        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 5)
                                    for i,j in zip(*self.filters)])

        self.convs3 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 7)
                                    for i,j in zip(*self.filters)])
        self.merge = nn.Conv2d(self.filters[-1][-1],1,kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        out1, out2, out3 = x, x, x

        for conv in self.convs1:
            out1 = F.relu(conv(out1))

        for conv in self.convs2:
            out2 = F.relu(conv(out2))

        for conv in self.convs3:
            out3 = F.relu(conv(out3))
        
        return [out1, out2, out3]

class StarClassifier2(nn.Module):
    def __init__(self, filter_sizes, output_dim, hidden_dim, dropout, final_set=True):
        super().__init__()
        self.backbone = ConvBackbone(filter_sizes)
        
        if final_set:
            self.linear1 = nn.Linear(6855, hidden_dim)
        else:
            self.linear1 = nn.Linear(6867, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear3 = nn.Linear(hidden_dim//2, output_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch = x.shape[0]
        outs = self.backbone(x)
        pooled = [F.max_pool2d(i, (self.backbone.filters[1][-1],1)).squeeze(1) for i in outs]

        cat = torch.cat(pooled, dim = 1)
        cat = self.drop(F.relu(self.linear1(cat)))
        cat = self.drop(F.relu(self.linear2(cat)))
        cat = self.linear3(cat)
        return cat


class StarClassifier3(nn.Module):
    def __init__(self, filter_sizes, output_dim, hidden_dim, dropout, final_set=True):
        super().__init__()
        self.backbone = ConvBackbone(filter_sizes)

        self.linear1 = nn.Linear(6855, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        outs = self.backbone(x)
        pooled = [F.max_pool2d(i, (self.filters[1][-1],1)).squeeze(1) for i in outs]

        cat = torch.cat(pooled, dim = 1)
        cat = self.drop(F.relu(self.linear1(cat)))
        cat = self.drop(F.relu(self.linear2(cat)))
        cat = self.linear3(cat)
        return cat

class StarClassifier2_old(nn.Module):
    def __init__(self, filter_sizes, output_dim, hidden_dim, dropout):
        super().__init__()
        self.n_filt = len(filter_sizes)-1
        self.filters = [filter_sizes[i:i+self.n_filt] for i in range(2)]

        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 3)
                                    for i,j in zip(*self.filters)])

        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 5)
                                    for i,j in zip(*self.filters)])

        self.convs3 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 7)
                                    for i,j in zip(*self.filters)])

        self.linear1 = nn.Linear(6867, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear3 = nn.Linear(hidden_dim//2, output_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        out1, out2, out3 = x, x, x

        for conv in self.convs1:
            out1 = F.relu(conv(out1))

        for conv in self.convs2:
            out2 = F.relu(conv(out2))

        for conv in self.convs3:
            out3 = F.relu(conv(out3))

        pooled = [F.max_pool2d(i, (self.filters[1][-1],1)).squeeze(1) for i in [out1, out2, out3]]

        cat = torch.cat(pooled, dim = 1)
        cat = self.drop(F.relu(self.linear1(cat)))
        cat = self.drop(F.relu(self.linear2(cat)))
        cat = self.linear3(cat)
        return cat

class StarClassifier2_new(nn.Module):
    def __init__(self, filter_sizes, output_dim, hidden_dim, dropout, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.n_filt = len(filter_sizes)-1
        self.filters = [filter_sizes[i:i+self.n_filt] for i in range(2)]

        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 3)
                                    for i,j in zip(*self.filters)])

        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 5)
                                    for i,j in zip(*self.filters)])

        self.convs3 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = 7, stride=2)
                                    for i,j in zip(*self.filters)])

        self.flattened_size = self._get_flattened_size_()

        self.linear1 = nn.Linear(self.flattened_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear3 = nn.Linear(hidden_dim//2, output_dim)

        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights_)

    def _get_flattened_size_(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_shape)
            out1, out2, out3 = x, x, x

            for conv in self.convs1:
                out1 = F.relu(conv(out1))

            for conv in self.convs2:
                out2 = F.relu(conv(out2))

            for conv in self.convs3:
                out3 = F.relu(conv(out3))

            pooled = [F.max_pool2d(i, (self.filters[1][-1],1)).squeeze(1) for i in [out1, out2, out3]]
            cat = torch.cat(pooled, dim = 1)
            return cat.view(1, -1).size(1)

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        out1, out2, out3 = x, x, x

        for conv in self.convs1:
            out1 = F.relu(conv(out1))

        for conv in self.convs2:
            out2 = F.relu(conv(out2))

        for conv in self.convs3:
            out3 = F.relu(conv(out3))

        pooled = [F.max_pool2d(i, (self.filters[1][-1],1)).squeeze(1) for i in [out1, out2, out3]]

        cat = torch.cat(pooled, dim = 1)
        cat = self.drop(F.relu(self.linear1(cat)))
        cat = self.drop(F.relu(self.linear2(cat)))
        cat = self.linear3(cat)
        return cat
        

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, downsample, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x shape: (batch, in_channels, signal_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # shape: (batch, channels, 1)
        x = torch.flatten(x, 1)  # shape: (batch, channels)
        x = self.fc(x)

        return x

def ResNet1D50(num_classes=1000, in_channels=1):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], num_classes, in_channels)

class StarClassifier4(nn.Module):
    def __init__(self, filter_sizes, kernels, strides, output_dim, hidden_dim, dropout, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.n_filt = len(filter_sizes)-1
        self.filters = [filter_sizes[i:i+self.n_filt] for i in range(2)]

        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = kernels[0], stride=n)
                                    for i,j,n in zip(*self.filters, strides)])

        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = kernels[1], stride=n)
                                    for i,j,n in zip(*self.filters, strides)])

        self.convs3 = nn.ModuleList([nn.Conv1d(in_channels = i,
                                              out_channels = j,
                                              kernel_size = kernels[2], stride=n)
                                    for i,j,n in zip(*self.filters, strides)])
        
        dummy = torch.zeros(1, self.input_shape)
        self.flattened_size = self.forward(dummy, True)

        self.linear1 = nn.Linear(self.flattened_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights_)

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x, get_size=False):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        out1, out2, out3 = x, x, x

        for conv in self.convs1:
            out1 = F.relu(conv(out1))

        for conv in self.convs2:
            out2 = F.relu(conv(out2))

        for conv in self.convs3:
            out3 = F.relu(conv(out3))

        pooled = [i.view(batch,-1) for i in [out1, out2, out3]]
        cat = torch.cat(pooled, dim = 1)
        if get_size:
            return cat.size(1)
        else:
            cat = self.drop(F.relu(self.linear1(cat)))
            cat = self.linear2(cat)
            return cat
