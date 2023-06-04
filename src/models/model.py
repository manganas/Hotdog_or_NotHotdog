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


class VGG_n_model(nn.Module):
    """
    VGG based pretrained model.
    n is indicative of the pretrained model layers used,
    should probably be changed to reflect that.
    """

    def __init__(self) -> None:
        super(VGG_n_model, self).__init__()

        # From https://pytorch.org/hub/pytorch_vision_vgg/
        # *_bn stands for batch normalized version
        # from torchvision.models import vgg19_bn, VGG19_BN_Weights
        from torchvision.models.vgg import vgg19_bn, VGG19_BN_Weights

        # Step 1: Initialize model with the best available weights
        weights = VGG19_BN_Weights.DEFAULT
        model = vgg19_bn(weights=weights)

        children = list(model.children())

        self.feature_extractor = nn.Sequential(*children[0])
        self.adataptive_average_pooling = children[1]

        # per 3. Next layer is -4
        self.fully_connected = nn.Sequential(*children[2])

        # Freeze the pretrained layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.fully_connected.parameters():
            param.requires_grad = False

        # The last layer of VGG19_bn has 4096 -> 1000 features. changed to 2 features, since 2 classes
        self.custom_fc_layer = nn.Sequential(
            nn.Linear(in_features=1000, out_features=2, bias=True), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        x = self.adataptive_average_pooling(x)

        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)

        x = self.custom_fc_layer(x)

        return x


def main():
    model = VGG_n_model()
    print(model.feature_extractor)
    print("-" * 10)
    print(model.adataptive_average_pooling)
    print("-" * 10)
    print(model.fully_connected)
    print("-" * 10)
    print(model.custom_fc_layer)


if __name__ == "__main__":
    main()
