from torch import nn, Tensor
import torch
import torch.nn.functional as F

class BottleNeckResNetBlock(nn.Module):
    def __init__(self, n_features:int, n_bottleneck:int, kernel_size:int=3)->None:
        '''
        Creates a bottlenck resnet block
        n_features are the number of channels coming into the block
        n_bottleneck are the reduced channels
        n_bottleneck < n_features: bottleneck
        n_bottleneck > n_features: inverse bottleneck
        '''
        super(BottleNeckResNetBlock, self).__init__()
        p = 0.1

        self.bottleneck__block = nn.Sequential(
            nn.Conv2d(n_features, n_bottleneck, 1, padding='same' ),
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.Conv2d(n_bottleneck, n_bottleneck, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.Conv2d(n_bottleneck, n_features, 1, padding='same')
        )
        
        
    def forward(self, x):
        
        x_conv = self.bottleneck__block(x)
        
        out = F.relu(x_conv + x)
        
        return out


class ResNetBlock(nn.Module):
    '''
    Simple ResNet block that can be used for building models
    '''
    def __init__(self, n_features:int, kernel_size: int=3):
        super(ResNetBlock, self).__init__()
                
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size, padding='same' ),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, kernel_size, padding='same')
        )
        
        
    def forward(self, x):
        
        x_conv = self.conv_block(x)
        
        out = F.relu(x_conv + x)
        
        return out

class CustomModelIma(nn.Module):
    '''
    Create a model from scratch
    '''
    def __init__(self,in_channels:int, n_classes:int, height:int, width:int,  bn:bool=True)->None:
        super(CustomModelIma, self).__init__()
        self.bn = bn

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            self.BN_layer_2d(16),
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            self.BN_layer_2d(32),
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.LeakyReLU()
        )

        self.residual_part = nn.Sequential(
            BottleNeckResNetBlock(32, 64, 3),
            self.BN_layer_2d(32),
            BottleNeckResNetBlock(32, 64, 3),
            self.BN_layer_2d(32),
            BottleNeckResNetBlock(32, 16, 3),
            self.BN_layer_2d(32)
        )

        x = torch.rand(1,3,height, width)
        in_shape = self.get_output_conv_shape(x)

        self.fully_connected_part = nn.Sequential(
            nn.Linear(in_shape, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_classes),
            nn.LogSoftmax(dim=1)
        )


    
    def BN_layer_2d(self, n_features:int):
        return nn.BatchNorm2d(n_features) if self.bn else nn.Identity()
    
    def get_output_conv_shape(self, x: Tensor) -> int:
        x = self.convolution_part(x)
        # x = self.residual_part(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x: Tensor)-> Tensor:
        x = self.convolution_part(x)
        # x = self.residual_part(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_part(x)
        return x

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
    
    
class CNN_model3(nn.Module):
    """
    Initial approach to a CNN for classification.
    """

    def __init__(self, in_channels: int, n_classes: int, h: int, w: int) -> None:
        super(CNN_model3, self).__init__()

        kernel_size = 3
        dropout_rate = 0.4  # Dropout rate can be tuned as per requirement

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #nn.Dropout(dropout_rate)  # Adding dropout after conv layers
        )

        x = torch.rand(1, in_channels, h, w)
        lin_in_dim = self.get_output_conv_shape(x)

        self.fully_connected = nn.Sequential(
            nn.Linear(lin_in_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  # Adding dropout in between fully connected layers
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  # Adding dropout in between fully connected layers
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=1)
        )
    ## No LogSoftmax or Sigmoid, cause i use crossentropy   
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




class VGG_19_model(nn.Module):
    """
    VGG based pretrained model.
    n is indicative of the pretrained model layers used,
    should probably be changed to reflect that.
    """

    def __init__(self, use_pretrained:bool=True) -> None:
        super(VGG_19_model, self).__init__()

        # From https://pytorch.org/hub/pytorch_vision_vgg/
        # *_bn stands for batch normalized version
        # from torchvision.models import vgg19_bn, VGG19_BN_Weights
        from torchvision.models.vgg import vgg19_bn, VGG19_BN_Weights

        # Initialize model with the best available weights
        if use_pretrained:
            weights = VGG19_BN_Weights.DEFAULT
            model = vgg19_bn(weights=weights)
        else:
            model = vgg19_bn()

        children = list(model.children())

        self.feature_extractor = nn.Sequential(*children[0])
        self.adaptive_average_pooling = children[1]

        # per 3. If I remove the last linear layer (4096->1000), then I have [-1]. Next layer is -4
        self.fully_connected = nn.Sequential(*children[2])

        if use_pretrained:
            # Freeze the pretrained layers
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            for param in self.adaptive_average_pooling.parameters():
                param.requires_grad = False

            for param in self.fully_connected.parameters():
                param.requires_grad = False

        # The last layer of VGG19_bn has output 1000 features. Added a layer from 1000 to 2 features, since 2 classes
        self.custom_fc_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=2, bias=True), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        x = self.adaptive_average_pooling(x)

        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)

        x = self.custom_fc_layer(x)

        return x


class ResNet_model(nn.Module):
    """
    ResNet based pretrained model.
    """

    def __init__(self, use_pretrained:bool=True) -> None:
        super(ResNet_model, self).__init__()

        # From https://pytorch.org/hub/pytorch_vision_resnet/
        # *_bn stands for batch normalized version
        # from torchvision.models import vgg19_bn, VGG19_BN_Weights
        from torchvision.models.resnet import resnet152, ResNet152_Weights

        # Initialize model with the best available weights
        if use_pretrained:
            weights = ResNet152_Weights.DEFAULT
            model = resnet152(weights=weights)
        else:
            model = resnet152()

        children = list(model.children())

        self.resnet = nn.Sequential(*children)[:-1]
        self.fully_connected = nn.Sequential(*children)[-1]

        if use_pretrained:
            # Freeze the pretrained layers
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.fully_connected.parameters():
                param.requires_grad = False

            # The last layer of ResNet152 has 1000 features. added a layer to 2 features, since 2 classes
            self.custom_fc_layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=1000, out_features=2, bias=True), nn.LogSoftmax(dim=1)
            )

    def forward(self, x):
        x = self.resnet(x)

        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)

        x = self.custom_fc_layer(x)

        return x


def main():
    # h = 224
    # w = 224
    # model = CustomModelIma(3, 2, h, w, bn=True)
    
    # x = torch.rand(1, 3, h, w)
    # print(model(x))
    pass

    


if __name__ == "__main__":
    main()
