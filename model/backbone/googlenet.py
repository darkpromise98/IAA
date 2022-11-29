import torch
import torch.nn as nn
import os
import torchvision


class GoogLeNet(nn.Module):

    output_size = 1024

    def __init__(self, weight_path='weights_models', pretrained=True):
        super().__init__()

        self.model = torchvision.models.GoogLeNet(init_weights=True)

        if pretrained:
            state_dict = torch.load(os.path.join(weight_path, 'googlenet-1378be20.pth'))
            self.model.load_state_dict(state_dict)

        del self.model.fc 

    def forward(self, x):

        if self.model.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        x = self.model.conv1(x)
        x = self.model.maxpool1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.maxpool2(x)

        x = self.model.inception3a(x)
        x = self.model.inception3b(x)
        x = self.model.maxpool3(x)
        x = self.model.inception4a(x)

        x = self.model.inception4b(x)
        x = self.model.inception4c(x)
        x = self.model.inception4d(x)

        x = self.model.inception4e(x)
        x = self.model.maxpool4(x)
        x = self.model.inception5a(x)
        x = self.model.inception5b(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


if __name__ == '__main__':

    model = GoogLeNet()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(y.shape)
