import hydra
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.data.dataset import HotDogDataset
from src.models.model import VGG_19_model, ResNet_model, CNN_model

from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt

# def save_saliency(saliency, image, noise_lvl, n_samples, label):
    
#     img_sal = torch.permute(torch.from_numpy(saliency), (1,2,0)).cpu().numpy()
#     img_sal = np.max(img_sal, axis=-1)
#     img_nomr = (img_sal - img_sal.min()) / (img_sal.max() / img_sal.min()) 


#     image = torch.permute(image[0], (1,2,0)).cpu().numpy()
    
#     fig = plt.figure()
#     fig.tight_layout()
#     plt.tight_layout()

#     plt.subplot(1,2,1)
#     plt.imshow(image)

#     plt.axis(False)

#     plt.subplot(1,2,2)
#     plt.imshow(img_nomr, cmap='gray')

#     plt.axis(False)
    
#     plt.suptitle(f"TL: {label}\nNoise_level: {noise_lvl}\n N samples: {n_samples}")

#     plt.savefig(f'saliency_{noise_lvl}_{n_samples}.pdf')
#     plt.show()


    


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
    # print(f'Runs on: {device}')

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
        noisy_image = image + np.random.normal(0, std_dev, image.shape).astype(np.float32)
        noisy_image = Variable(torch.from_numpy(noisy_image).to(device), requires_grad=True)

        # Forward pass through the model
        
        output = model(noisy_image)

        # Get the index of the max log-probability
        index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1

        one_hot = Variable(torch.from_numpy(one_hot).to(device), requires_grad=True)

        one_hot = torch.sum(one_hot * output)

        if noisy_image.grad is not None:
                noisy_image.grad.data.zero_()

        one_hot.backward()

        grad = noisy_image.grad.data.cpu().numpy()

        total_gradients += grad

    # Average per gradient feature dim, I have n_samples depth
    avg_gradients = total_gradients[0, :, :, :] / n_samples

    return avg_gradients

def place_image(img_tensor_3, axis):
    axis.imshow(torch.permute(img_tensor_3, (1,2,0)).cpu().numpy())
    axis.set_axis_off()

def place_saliency_map(saliency, axis):
    img_sal = torch.permute(torch.from_numpy(saliency), (1,2,0)).cpu().numpy()
    img_sal = np.max(img_sal, axis=-1)
    img_norm = (img_sal - img_sal.min()) / (img_sal.max() / img_sal.min())
    axis.imshow(img_norm, cmap='hot')
    axis.set_axis_off()


def plot_saliency_matrix(images, model, noise_levels, n_samples, path='.', figsize=(13, 6)):
    fig, ax = plt.subplots(images.shape[0], len(noise_levels)+1, figsize=figsize)
    
    # Set the titles for the image grid
    for ax_, col in zip(ax[0], ['Noise\nlevel:']+noise_levels):
        ax_.set_title(f"{int(col*100)}%" if type(col)==float else 'Noise level:')

    
    for i, image in enumerate(images):
        # Plot the original image
        place_image(image, ax[i,0])
        
        image = torch.unsqueeze(image, 0)
        # For each image, plot the saliency maps for each noise level
        for j, noise_lvl in enumerate(noise_levels):
            saliency_map = get_smooth_grad(image, model ,n_samples=n_samples, noise_level=noise_lvl)
            place_saliency_map(saliency_map, ax[i, j+1])

              
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{path}/saliencies.pdf")
    plt.show()
    

def main():
    
    # Hardcoded model weights dictionary
    saved_weights_path = '/zhome/39/c/174709/git/Hotdog_or_NotHotodog/models/testing_save_model.pt'
    saliency_images_path = '/zhome/39/c/174709/git/Hotdog_or_NotHotodog/data/saliency'
    
    seed = 7

    batch_size = 64
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Runs on: {device}")


    # Load a model
    model = CNN_model(3, 2, 224, 224)
    # model = VGG_19_model(use_pretrained=True)
    model.load_state_dict(torch.load(saved_weights_path)['model'])
    model.to(device)
    model.eval()

    
    # Get a set of test images
    test_transformation = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    dataset = HotDogDataset(saliency_images_path, train=True, transform=test_transformation)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    ###### Change this! only to test
    

    images, _ = next(iter(loader))

    noise_levels = [0., 0.05, 0.1, 0.2, 0.3, 0.5]
    n_samples = 25
    
    plot_saliency_matrix(images, model, noise_levels, n_samples, path='.')


if __name__=='__main__':
    main()
