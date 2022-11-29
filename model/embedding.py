import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, base, feature_size=512, embedding_size=128):
        super().__init__()

        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)
        self.linear.apply(weights_init_kaiming)

    def forward(self, x):
        feat = self.base(x)
        embedding = self.linear(feat)

        return embedding


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)  
