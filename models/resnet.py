
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

@register('resnet50')
@register('resnet50_pretrain')
class ResNet50Pretrain(nn.Module):
    def __init__(self, emb_size=128, pretrained=True, **kwargs):
        super(ResNet50Pretrain, self).__init__()
        import torchvision.models as models_tv
        # Use weights argument for newer torchvision, or pretrained=True for older.
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models_tv.resnet50(weights=weights)
        except ImportError:
            self.model = models_tv.resnet50(pretrained=pretrained)
        except Exception as e:
            # Fallback for any other unexpected error (e.g. connection issue even with weights=None?)
            # Usually weights=None is safe. If downloading fails for pretrained=True, this catch helps?
            if pretrained:
                print(f"Warning: Failed to load pretrained weights: {e}. Loading random weights.")
                self.model = models_tv.resnet50(pretrained=False)
            else:
                raise e

        # Replace the final fully connected layer
        # ResNet50 fc input features is 2048
        num_ftrs = self.model.fc.in_features
        # Use Identity to remove the FC layer effect inside the model
        self.model.fc = nn.Identity()
        # Add projection layer
        self.projection = nn.Linear(num_ftrs, emb_size)
        self.out_dim = emb_size

    def forward(self, x):
        # Extract features from resnet
        x = self.model(x)
        # Apply projection
        return self.projection(x)

@register('resnet18')
@register('resnet18_pretrain')
class ResNet18Pretrain(nn.Module):
    def __init__(self, emb_size=128, pretrained=True, **kwargs):
        super(ResNet18Pretrain, self).__init__()
        import torchvision.models as models_tv
        # Use weights argument for newer torchvision, or pretrained=True for older.
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models_tv.resnet18(weights=weights)
        except ImportError:
            self.model = models_tv.resnet18(pretrained=pretrained)
        except Exception as e:
            if pretrained:
                print(f"Warning: Failed to load pretrained weights for ResNet18: {e}. Loading random weights.")
                self.model = models_tv.resnet18(pretrained=False)
            else:
                raise e

        # Replace the final fully connected layer
        # ResNet18 fc input features is 512
        num_ftrs = self.model.fc.in_features
        # Use Identity to remove the FC layer effect inside the model
        self.model.fc = nn.Identity()
        # Add projection layer
        self.projection = nn.Linear(num_ftrs, emb_size)
        self.out_dim = emb_size

    def forward(self, x):
        # Extract features from resnet
        x = self.model(x)
        # Apply projection
        return self.projection(x)


class AGNNResNet12Block(nn.Module):
    def __init__(self, inplanes, planes):
        super(AGNNResNet12Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out


@register('agnn-resnet12')
class AGNNResNet12(nn.Module):
    def __init__(self, emb_size, cifar_flag=False, **kwargs):
        super(AGNNResNet12, self).__init__()
        cfg = [64, 128, 256, 512]
        iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU()
        self.emb_size = emb_size
        
        self.layer1 = self._make_layer(AGNNResNet12Block, cfg[0], cfg[0])
        self.layer2 = self._make_layer(AGNNResNet12Block, cfg[0], cfg[1])
        self.layer3 = self._make_layer(AGNNResNet12Block, cfg[1], cfg[2])
        self.layer4 = self._make_layer(AGNNResNet12Block, cfg[2], cfg[3])
        
        self.avgpool = nn.AvgPool2d(6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        self.layer_second = nn.Sequential(nn.Linear(in_features=layer_second_in_feat,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))
        
        self.out_dim = emb_size

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        inter = self.layer3(x)
        x = self.layer4(inter)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_embed = self.layer_last(x)
        
        inter = self.maxpool(inter)
        inter = inter.view(inter.size(0), -1)
        inter_embed = self.layer_second(inter)
        
        # Returns list to match Extension/backbone.py exactly
        return [x_embed, inter_embed]
