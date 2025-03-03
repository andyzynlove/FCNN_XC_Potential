import torch
import torch.nn as nn
import os
from DNN import *

class LoadModel(nn.Module):
    def __init__(self):
        super(LoadModel, self).__init__()
        self.model = DNN()
        self.load_model()

    def load_model(self):
        self.model = nn.DataParallel(self.model)
        model_path = os.path.dirname(__file__) + '/fcnn_vxc'
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):
        x = self.model.forward(x)
        return x
