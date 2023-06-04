import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.data.dataset import HotDogDataset
from src.models.model import CNN_model

from tqdm import tqdm
import numpy as np

import os
import logging
import hydra
from omegaconf import OmegaConf

# Initialize the logger
log = logging.getLogger(__name__)


def loss_fun(output, target):
    return F.nll_loss(output, target)


@hydra.main(config_path="../conf", config_name="default_config.yaml")
def main(config) -> None:
    ## Hydra config parameters
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment
    seed = hparams["seed"]

    # Transformation parameters
    img_size = hparams["img_size"]
    rotation_deg = hparams["rotation_deg"]

    # Paths
    raw_data_path = hparams["dataset_path"]

    # Training parameters
    n_epochs = hparams["n_epochs"]
    lr = hparams["lr"]
    batch_size = hparams["batch_size"]

    ## Training loop init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training is running on: {device}")

    torch.manual_seed(seed)

    # Transformations for training and test sets
    train_images_mean = [0.5] * 3
    train_images_std = [1.0] * 3
    resize_dims = [img_size] * 2

    # Maybe add some blurring
    train_transformation = transforms.Compose(
        [
            transforms.Resize(
                resize_dims, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(rotation_deg),
            transforms.ToTensor(),
            transforms.Normalize(train_images_mean, train_images_std),
        ]
    )

    # Need not have the same transformations as for training, other than resizing and tensorizing. Maybe normalize with train data
    test_transformation = transforms.Compose(
        [
            transforms.Resize(
                resize_dims, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            # transforms.Normalize(train_images_mean, train_images_std)
        ]
    )

    # Create the datasets and dataloaders
    trainset = HotDogDataset(raw_data_path, train=True, transform=train_transformation)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = HotDogDataset(raw_data_path, train=False, transform=test_transformation)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Define model
    in_channels = 3  # RGB image
    n_classes = 2  # Hotdog or not hotdog
    model = CNN_model(in_channels, n_classes, img_size, img_size)
    model.to(device)

    # Define optimizer and loss function
    weight_decay = 0  # Similar to L2 regularization. With dropout not really needed
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Begin training loop
    out_dict = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for epoch in tqdm(range(n_epochs), unit="epoch"):
        model.train()
        # For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=None
        ):
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)

            # Compute the loss
            loss = loss_fun(output, target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
        # Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target == predicted).sum().cpu().item()
        out_dict["train_acc"].append(train_correct / len(trainset))
        out_dict["test_acc"].append(test_correct / len(testset))
        out_dict["train_loss"].append(np.mean(train_loss))
        out_dict["test_loss"].append(np.mean(test_loss))
        print(
            f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
            f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%",
        )


if __name__ == "__main__":
    main()
