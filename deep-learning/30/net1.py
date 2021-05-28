"""Defines the neural network and the loss function"""


import torch
from torch import nn, optim
import module

class VolAutoEncoder(nn.Module):
    """
       This is the standard way to define a network in PyTorch. The components
       (layers) of the network are defined in the __init__ function.
       Then, in the forward function it is defined how to apply these layers on the input step-by-step.
    """

    def __init__(self):
        super(VolAutoEncoder, self).__init__()

        self.encoder=nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.Conv3d(1, 32, (3, 3, 3), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, (3, 3, 3), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 256, (4, 4, 4), stride=1),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Conv3d(256, 256, (1, 1, 1), stride=1),   
            nn.ReLU(inplace=True),
            module.SwitchNorm3d(256),
            nn.Dropout(p=0.5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2, 2, 2), stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, (3, 3, 3), stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, (3, 3, 3), stride=3)       
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        This function defines how to use the components of the network to operate on an input batch.
        """
        #print("First size: ", x.shape)
        x = self.encoder(x)
        #print("Third size: ", x.shape)
        x = self.linear(x)
        #print("Fifth size: ", x.shape)
        x = self.decoder(x)
        #print("Eighth size: ", x.shape)
        x = x.view(27000)  #30*30*30=27000
        #print("Ninth size: ", x.shape)
        x = self.sigmoid(x)
        #print("Tenth size: ", x.shape)

        return x

def loss_fn(outputs, targets):
    """
    Computes the cross entropy loss given outputs and labels
    """
    loss = nn.BCELoss()

    return loss(outputs, targets)