from torch import nn, Tensor
import torch
import torch.nn.functional as F


class CNN_model(nn.Module):
    """
    Initial approach to a CNN for classification.
    """

    def __init__(self, in_channels: int, n_classes: int, h: int, w: int) -> None:
        super(CNN_model, self).__init__()

        # Should we also add Recurrent layers?

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels, 8, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
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
        x = self.convolution_part(x)
        x = self.fully_connected(x)
        x = F.log_softmax(x, dim=1)

        assert len(x.shape) == 2, "Not a tensor of shape N,2"
        assert x.shape[-1] == 2, "Not two class probs returned"

        return x

    def get_output_conv_shape(self, x: Tensor) -> int:
        out = self.convolution_part(x)
        return out.shape[1] * out.shape[2] * out.shape[3]
