import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.rho_type = "mGGA"
        self.ln = nn.LayerNorm(256, eps= 1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(39, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = 0.1*torch.log(x + 1e-7)
        x = F.tanh(self.fc1(x))
        x = F.elu(self.fc2(self.ln(x)))
        x = F.elu(self.fc3(self.ln(x)))
        x = F.elu(self.fc4(self.ln(x)))
        x = F.elu(self.fc5(self.ln(x)))
        x = F.elu(self.fc6(self.ln(x)))
        x = F.elu(self.fc7(self.ln(x)))
        x = self.fc8(self.ln(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features