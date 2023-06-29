import torch
import torch.nn as nn
from typing import Callable, Optional
import math

class SEBlock(nn.Module):# Squeeze and Excitation Block
    def __init__(self, in_ch, reduction_ratio = 16,
                activation: Callable[..., nn.Module] = nn.ReLU):
        super().__init__()
        out_ch = max(1, in_ch // reduction_ratio)
        
        self.layers = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_ch, out_ch, 1),
        activation(),
        nn.Conv2d(out_ch, in_ch, 1),
        nn.Sigmoid()
        )
    
    def forward(self, input):
        return input * self.layers(input)

class ConvBNActivation(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1,
                activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch)
        ]
        if activation is not None:
            layers += [activation()]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        return self.layers(input)

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_ratio):
        super().__init__()
        med_ch = int(in_ch * expansion_ratio)
        
        self.layers = nn.Sequential(
            ConvBNActivation(in_ch, med_ch, 1, activation=nn.Mish),#point_wise1
            ConvBNActivation(med_ch, med_ch, kernel_size, stride, med_ch, activation=nn.Mish),#depth_wise
            SEBlock(med_ch, activation=nn.Mish),#se block
            ConvBNActivation(med_ch, out_ch, 1, activation=None)#point_wise2
        )
        if(stride != 1 or in_ch != out_ch):
            self.shortcut = ConvBNActivation(in_ch, out_ch, 1, stride, activation=None)
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self, input):
        return self.shortcut(input) + self.layers(input)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, last_ch=128, dropout_rate=0.2,
                width_mult=1.0, depth_mult=1.0):
        super(EfficientNet, self).__init__()
        # expand_ratio, channel, repeats, stride, kernel_size                   
        self.cfg = [
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112                   
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56                   
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28                   
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14                   
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14                   
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7                   
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7]                  
        ]

        self.last_ch = last_ch
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        
        self.features = self._make_features()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(math.ceil(self.last_ch * self.width_mult)), num_classes)
        )

    def _make_features(self, in_ch=3):
        out_ch = int(math.ceil(32 * self.width_mult))
        features = []
        features += [ConvBNActivation(in_ch, out_ch, 3, 2, activation=nn.Mish)]
        in_ch = out_ch
        
        for expansion, ch, repeats, stride, kernel in self.cfg:
            out_ch = int(math.ceil(ch * self.width_mult))
            repeats = int(math.ceil(repeats * self.depth_mult))
            for i in range(repeats):
                if i != 0:
                    stride = 1 
                features += [MBConvBlock(in_ch, out_ch, kernel, stride, expansion)]
                in_ch = out_ch
        
        out_ch = int(math.ceil(self.last_ch * self.width_mult))
        features += [ConvBNActivation(in_ch, out_ch, 1, activation=nn.Mish)]
        
        return nn.Sequential(*features)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
