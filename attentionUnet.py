import torch
import torch.nn as nn
from modules.modules import *



class Attention_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, middle_layers= [64,128,256,512,1024] , bilinear=False):

        if len(middle_layers) != 5:
            raise ValueError("middle_layers must be a list of length 5")

        super(Attention_UNet, self).__init__()
        self.n_channels = n_channels
        self.middle_layers = middle_layers
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, middle_layers[0]))
        self.down1 = (Down(middle_layers[0], middle_layers[1]))
        self.down2 = (Down(middle_layers[1], middle_layers[2]))
        self.down3 = (Down(middle_layers[2], middle_layers[3]))
        self.attention_1 = SelfAttention(middle_layers[0])
        self.attention_2 = SelfAttention(middle_layers[1])
        self.attention_3 = SelfAttention(middle_layers[2])
        self.attention_4 = SelfAttention(middle_layers[3])

        factor = 2 if bilinear else 1
        self.down4 = (Down(middle_layers[3], middle_layers[4] // factor))
        self.up1 = (Up(middle_layers[4], middle_layers[3] // factor, bilinear))
        self.up2 = (Up(middle_layers[3], middle_layers[2] // factor, bilinear))
        self.up3 = (Up(middle_layers[2], middle_layers[1] // factor, bilinear))
        self.up4 = (Up(middle_layers[1], middle_layers[0], bilinear))
        self.outc = (OutConv(middle_layers[0], n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        v1, _ = self.attention_4(x4)
        v2, _ = self.attention_3(x3)
        v3, _ = self.attention_2(x2)
        v4, _ = self.attention_1(x1)
        x = self.up1(x5, v1)
        x = self.up2(x, v2)
        x = self.up3(x, v3)
        x = self.up4(x, v4)
        logits = self.outc(x)
        return logits
