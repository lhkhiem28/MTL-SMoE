import os, sys
from libs import *

class Classifier(nn.Module):
    def __init__(self, 
        num_classes = 10, 
    ):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(
                128, num_classes, 
            ), 
        )

    def forward(self, 
        input, 
    ):
        out = self.clf(input)

        return out

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

        self.moe_layer = moe.moe_layer(
            model_dim = 12544, 
            experts = {
                "type":"ffn", "count_per_node":8, "hidden_size_per_expert":128, 
                "output_dim":128, 
            }, scan_expert_func = lambda _, parameter: setattr(parameter, "skip_allreduce", True), 
            gate_type = [{"type":"top", "k":4, "capacity_factor":0}, {"type":"top", "k":4, "capacity_factor":0}]
        )
        self.clf0, self.clf1 = Classifier(num_classes), Classifier(num_classes)

    def forward(self, 
        input, 
    ):
        input = self.backbone(input)
        input = self.max_pool(input)
        input = input.view(input.shape[0], -1)

        moe0, moe1 = self.moe_layer(input, gate_index = 0), self.moe_layer(input, gate_index = 1)
        out0, out1 = self.clf0(moe0), self.clf1(moe1)

        return out0, out1