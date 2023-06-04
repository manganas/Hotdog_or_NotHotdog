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

        kernel_size = 3

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size, padding="same"),
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
            nn.LogSoftmax(dim=1),  # I am using NLLLoss for training!
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.convolution_part(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        assert len(x.shape) == 2, "Not an array type of tensor"
        assert x.shape[-1] == 2, "Not two class probs returned"

        return x

    def get_output_conv_shape(self, x: Tensor) -> int:
        out = self.convolution_part(x)
        return out.shape[1] * out.shape[2] * out.shape[3]


class PretrainedModel(nn.Module):
    """
    Instantiates a pretrained model of choice, keeps the first layers_keep layers
    and appends a fully connected network, that is trainable,
    for classification in 2 classes: hotdog or nothotdog
    """

    def __init__(self, model_type: nn.Module, layers_keep: int) -> None:
        super(PretrainedModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# class VGG_n_model(nn.Module):
#     """
#     VGG based pretrained model.
#     n is indicative of the pretrained model layers used,
#     should probably be changed to reflect that.
#     """

#     from torchvision.models.vgg import vgg

#     def __init__(self) -> None:
#         super(VGG_n_model, self).__init__()

#         # From https://pytorch.org/hub/pytorch_vision_vgg/
#         mean_ = [0.485, 0.456, 0.406]
#         std_ = [0.229, 0.224, 0.225]

#     def forward(self, x):
#         return x
