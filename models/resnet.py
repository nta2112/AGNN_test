
import torch.nn as nn
import torch
import torch.nn.functional as F
from .models import register

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_layers = 1 # Not used but often expected in blocks

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out

@register('resnet12')
class ResNet12(nn.Module):
    def __init__(self, emb_size=128, drop_rate=0.0):
        super(ResNet12, self).__init__()
        self.inplanes = 3
        # ResNet12 usually has channels [64, 128, 256, 512]
        self.layer1 = self._make_layer(BasicBlock, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(BasicBlock, 128, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(BasicBlock, 256, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(BasicBlock, 512, stride=2, drop_rate=drop_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * BasicBlock.expansion, emb_size) # We usually don't use FC layer for embeddings directly in AGNN backbone context often, but let is match
        # Actually AGNN expects output shape (N, Emb_Size) ??
        # Let's check GNN model. GNN expects 'input_features' to be flexible or hardcoded?
        # gnn.py: input_features=128 + n_way...
        # So the output of encoder MUST be 128 dimensions if gnn.py expects 128.
        
        self.out_dim = emb_size
        # However, looking at convnet4, it has an FC layer at the end to project to 'z_dim'. 
        # ConvNet4_128: fc -> z_dim=128.
        # So we MUST project to 128 for compatibility with current GNN config.
        self.fc = nn.Linear(512, emb_size)
        
    def _make_layer(self, block, planes, stride=1, drop_rate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate))
        self.inplanes = planes * block.expansion # Update inplanes for next layer
        # Typically ResNet12 has 1 block per layer that contains 3 convs.
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

@register('resnet50_pretrain')
class ResNet50Pretrain(nn.Module):
    def __init__(self, emb_size=128, pretrained=True):
        super(ResNet50, self).__init__()
        import torchvision.models as models_tv
        # Use weights argument for newer torchvision, or pretrained=True for older.
        # To be safe across versions, we can check or try/except, but usually 'weights' is the way now.
        # Fallback for older torch in colab?
        # Let's try 'weights' first, if it fails, user might see error, effectively standard now.
        # Actually, simpler: define generic load.
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models_tv.resnet50(weights=weights)
        except ImportError:
            self.model = models_tv.resnet50(pretrained=pretrained)

        # Replace the final fully connected layer
        # ResNet50 fc input features is 2048
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, emb_size)
        self.out_dim = emb_size

    def forward(self, x):
        return self.model(x)
