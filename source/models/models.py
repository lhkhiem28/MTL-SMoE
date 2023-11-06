import os, sys
from libs import *

class MultiGateSMoE(nn.Module):
    def __init__(self, 
        num_tasks = 2, num_classes = 10, 
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
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(
                kernel_size = 2, 
            ), 
            nn.Dropout(0.2), 
        )

        self.moe = moe.moe_layer()

    def forward(self, 
        input, 
    ):
        output = self.backbone(input)
        output = self.max_pool(output)
        output = output.view(output.shape[0], -1)

        return output