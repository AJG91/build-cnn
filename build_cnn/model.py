
import torch as tc
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A convolutional neural network (CNN) for image classification.
    Inherits functionality from nn.Module.

    Architecture:
        - Conv2d -> ReLU -> MaxPool2d
        - Conv2d -> ReLU -> MaxPool2d
        - Flatten
        - Linear -> ReLU
        - Linear (output)
        
    Attributes
    ----------
    conv : nn.Sequential
        Convolutional neural network.
    fc : nn.Sequential
        Fully connected feed-forward multi-layer perceptron.

    Parameters
    ----------
    input_size : int, optional (default=1)
        Number of input channels.
        Default is for grayscale images.
    output_size : int, optional (default=10)
        Number of output classes.
    hidden_size : int, optional (default=128)
        Number of neurons in the hidden fully connected layer.
    """
    def __init__(self, input_size=1, output_size=10, hidden_size=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Parameters
        ----------
        x : tc.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        tc.Tensor
            Output tensor after applying convolutional neural network and fully connected layers.
        """
        x = self.conv(x)
        return self.fc(x)

