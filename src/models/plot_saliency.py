import hydra
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.data.dataset import HotDogDataset
from src.models.model import VGG_19_model, ResNet_model, CNN_model3

from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt

def save_saliency(saliency):
    
    img_sal = torch.permute(torch.from_numpy(saliency), (1,2,0)).cpu().numpy()
    img_nomr = np.zeros_like(img_sal)

    for i in range(3):
        img_nomr[i,:,:] = 255/(img_sal[i,:,:].max()-img_sal[i,:,:].min())*(img_sal[i,:,:]-img_sal[i,:,:].min())
        img_nomr[i, :, :] = img_nomr[i, :,:].astype(int)

    fig = plt.figure()
    plt.imshow(img_nomr)


    plt.savefig('saliency.pdf')
    plt.show()


    


def get_smooth_grad(image, model, n_samples=25, noise_level=0.1):
    """
    Computes the SmoothGrad saliency map of a given image and model.

    Parameters:
        image (torch.Tensor): the input image
        model (torch.nn.Module): the model
        n_samples (int): the number of noisy samples to generate
        noise level (float): Percentage, so < 1 and >0

    Returns:
        saliency_map (torch.Tensor): the computed saliency map
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Runs on: {device}')

    model.to(device)

    # Make sure the model is in evaluation mode
    model.eval()

    # Make sure the image requires gradient    
    # Keep a running total of gradients
    total_gradients = 0
    

    image = image.cpu().numpy()
    total_gradients = np.zeros_like(image)

    std_dev = noise_level*(image.max()-image.min())
    # Create noisy samples and compute gradients
    for _ in range(n_samples):
        # Create a noisy image
        # noisy_image = image + std_dev * torch.randn(*image.shape).to(image.device)
        noisy_image = image + np.random.normal(0, std_dev, image.shape).astype(np.float32)
        noisy_image = Variable(torch.from_numpy(noisy_image).to(device), requires_grad=True)

        # Forward pass through the model
        
        output = model(noisy_image)

        # Get the index of the max log-probability
        index = np.argmax(output.data.cpu().numpy())

        print(output)
        print(index)
        return

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1

        one_hot = Variable(torch.from_numpy(one_hot).to(device), requires_grad=True)

        one_hot = torch.sum(one_hot * output)

        if noisy_image.grad is not None:
                noisy_image.grad.data.zero_()

        one_hot.backward()

        grad = noisy_image.grad.data.cpu().numpy()

        total_gradients += grad

    avg_gradients = total_gradients[0, :, :, :] / n_samples

    return avg_gradients


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
    model_path = '/zhome/39/c/174709/git/Hotdog_or_NotHotodog/models/testing_save_model.pt'

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


    saliency_map = get_smooth_grad(image, model)

    print(saliency_map.min(), saliency_map.max())

    save_saliency(saliency_map)


if __name__=='__main__':
    main()
