import os, sys
from libs import *

class Classifier(nn.Module):
    def __init__(self, 
        num_classes = 10, 
    ):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(
                512, num_classes, 
            ), 
        )

    def forward(self, 
        input, 
    ):
        output = self.classifier(input)

        return output

class MultiGateSMoE(nn.Module):
    def __init__(self, 
        num_classes = 10, 
    ):
        super(MultiGateSMoE, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels =  1, out_channels = 32, kernel_size = 5, stride = 1, 
            ), 
            nn.ReLU(), 
            nn.Conv2d(
                in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, 
            ), 
            nn.ReLU(), 
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size = 2, 
        )

        # self.moe = moe.moe_layer()

    def forward(self, 
        input, 
    ):
        input = self.backbone(input)
        input = self.max_pool(input)
        input = input.view(input.shape[0], -1)

        return input