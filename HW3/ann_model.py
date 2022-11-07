import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylab import mpl

mpl.use('macosx')


class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):

        hl_size = np.linspace(num_features[1], num_features[0], 5, dtype=int)
        super(Autoencoder, self).__init__()

        self.enc1 = nn.Linear(in_features=num_features[0], out_features=hl_size[3])
        self.enc2 = nn.Linear(in_features=hl_size[3], out_features=hl_size[2])
        self.enc3 = nn.Linear(in_features=hl_size[2], out_features=hl_size[1])
        self.enc4 = nn.Linear(in_features=hl_size[1], out_features=num_features[1])

        # self.enc5 = nn.Linear(in_features=300, out_features=200)
        # self.dec1 = nn.Linear(in_features=200, out_features=300)

        self.dec2 = nn.Linear(in_features=num_features[1], out_features=hl_size[1])
        self.dec3 = nn.Linear(in_features=hl_size[1], out_features=hl_size[2])
        self.dec4 = nn.Linear(in_features=hl_size[2], out_features=hl_size[3])
        self.dec5 = nn.Linear(in_features=hl_size[3], out_features=num_features[0])

    def forward(self, x):

        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        x = torch.tanh(self.enc4(x))

        # x = F.leaky_relu(self.enc5(x))
        # x = F.leaky_relu(self.dec1(x))

        x = F.leaky_relu(self.dec2(x))
        x = F.leaky_relu(self.dec3(x))
        x = F.leaky_relu(self.dec4(x))
        decoded = torch.tanh((self.dec5(x)))

        return decoded


if __name__ == "__main__":
    pass
