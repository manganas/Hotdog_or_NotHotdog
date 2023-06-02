from torch import nn, Tensor
import torch


class CNN_model(nn.Module):
    """
    Initial approach to a CNN for classification.
    """

    def __init__(self, in_channels: int, n_classes: int, h: int, w: int) -> None:
        super(CNN_model, self).__init__()

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels, 8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        x = torch.rand(1, in_channels, h, w)
        lin_in_dim = self.get_output_conv_shape(x)

        self.fully_connected = nn.Sequential(
            nn.Linear(lin_in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Do not forget to assert and raise exception based on shape!
        pass

    def get_output_conv_shape(self, x: Tensor) -> int:
        pass
