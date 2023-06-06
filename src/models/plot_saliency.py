import hydra
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.data.dataset import HotDogDataset
from src.models.model import VGG_19_model, ResNet_model, CNN_model3

from omegaconf import OmegaConf


def get_smooth_grad(image, model, n_samples=25, std_dev=0.1):
    """
    Computes the SmoothGrad saliency map of a given image and model.

    Parameters:
        image (torch.Tensor): the input image
        model (torch.nn.Module): the model
        n_samples (int): the number of noisy samples to generate
        std_dev (float): the standard deviation of the noise

    Returns:
        saliency_map (torch.Tensor): the computed saliency map
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Runs on: {device}')

    model.to(device)

    # Make sure the model is in evaluation mode
    model.eval()

    # Make sure the image requires gradient
    # image = Variable(image.data, requires_grad=True)
    image.to(device)
    image.requires_grad=True
    # image.requires_grad_()
    
    # Keep a running total of gradients
    total_gradients = 0

    # Create noisy samples and compute gradients
    for _ in range(n_samples):
        # Create a noisy image
        noisy_image = image + std_dev * torch.randn(*image.shape).to(image.device)
        
        noisy_image.to(device)


        print(noisy_image.shape)
        # Forward pass through the model
        # output = model(noisy_image)
        output = model(image)

        # Get the index of the max log-probability
        output_max_index = output.argmax()

        # Zero gradients everywhere
        model.zero_grad()

        # Backward pass from the maximum output
        output_max_index.backward()

        # Add up the gradients
        total_gradients += noisy_image.grad.data.abs()

    # Compute the saliency map as the average of the gradients
    saliency_map = total_gradients / n_samples

    return saliency_map

def get_model(hparams):
    model_name = hparams['model'].lower().strip()
    load_path = hparams['weights_path']

    if model_name=='vgg':
        model = VGG_19_model(use_pretrained=hparams['use_pretrained'])
    elif model_name=='resnet':
        model = ResNet_model(use_pretrained=hparams['use_pretrained'])
    else:
        model = CNN_model3(3, 2, 224, 224)    

    model.load_state_dict(torch.load(load_path)['model'])
    model.eval()
    return model

@hydra.main(config_path="../conf", config_name="default_config.yaml")
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment

    # Load a model
    model = get_model(hparams)
    
    # Get a set of test images
    test_transformation = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    testset_to_be_split = HotDogDataset(hparams['dataset_path'], train=False, transform=test_transformation)

    generator1 = torch.Generator().manual_seed(hparams['seed'])
    _, testset =  random_split(testset_to_be_split, [0.2, 0.8], generator=generator1)

    test_loader = DataLoader(testset, batch_size=hparams['batch_size'], shuffle=False)
    
    ###### Change this! only to test
    
    for i, (inputs, _) in enumerate(test_loader):
        image = inputs
        break

    image = image.data[0]

    # image = next(iter(test_loader))[0]

    image = torch.unsqueeze(image, 0)  # Adds a batch dimension

    print(image.shape)

    saliency_map = get_smooth_grad(image, model)


if __name__=='__main__':
    main()
