import torchvision
import torch
import os
import torch.nn as nn

__all__ = ["ResNet50"]


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, weight_path='weights_models', pretrained=True):
        super(ResNet50, self).__init__()

        self.model = torchvision.models.resnet50()
        if pretrained:
            state_dict = torch.load(os.path.join(weight_path, 'resnet50-19c8e357.pth'))
            self.model.load_state_dict(state_dict)

        del self.model.fc
        self.model.avgpool = GMP_and_GAP()

    def forward(self, x):

        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1) 

        return x


class GMP_and_GAP(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        
        return self.gap(x) + self.gmp(x)
