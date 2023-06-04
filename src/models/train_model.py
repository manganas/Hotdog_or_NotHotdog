import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.data.dataset import HotDogDataset
from src.models.model import CNN_model, VGG_19_model, ResNet_model

from tqdm import tqdm
import numpy as np
import pickle

import os
import logging
import hydra
from omegaconf import OmegaConf

import wandb

# Initialize wandb
wandb.init()

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
    # img_size = hparams["img_size"]
    rotation_deg = hparams["rotation_deg"]

    # Model and optimizer
    model_name = hparams["model"].lower().strip()
    optim_name = hparams["optimizer"].lower().strip()

    # Paths
    raw_data_path = hparams["dataset_path"]
    saved_models_path = hparams["saved_models_path"]

    # Training parameters
    n_epochs = hparams["n_epochs"]
    lr = hparams["lr"]
    batch_size = hparams["batch_size"]
    save_per_n_epochs = hparams["save_after_n_epochs"]

    ## Training loop init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training is running on: {device}")

    torch.manual_seed(seed)

    # Standard preprocessing for ResNet and VGG
    # https://pytorch.org/hub/pytorch_vision_resnet/
    # https://pytorch.org/hub/pytorch_vision_vgg/

    # Maybe add some blurring
    train_transformation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(rotation_deg),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Need not have the same transformations as for training, other than resizing and tensorizing. Maybe normalize with train data
    test_transformation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    img_size = 224  # from vgg and resnet cropped, see transformations above

    # Improve implementation later!
    if model_name == "vgg":
        model = VGG_19_model()
    elif model_name == "resnet":
        model = ResNet_model()
    else:
        model = CNN_model(in_channels, n_classes, img_size, img_size)

    # Magic
    wandb.watch(model, log_freq=50)
    model.to(device)

    # Define optimizer

    weight_decay = 0  # Similar to L2 regularization. With dropout not really needed
    if optim_name == "sgd":
        momentum = hparams["momentum"]
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    # elif optim_name=='adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    print(model)
    print(optimizer)

    # Begin training loop
    out_dict = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    log.info("Start training...")
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
        for data, target in tqdm(test_loader, desc="Test", leave=None):
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

        wandb.log(out_dict)

        # Save the model weights
        if epoch % save_per_n_epochs == 0:
            # save weights
            # torch.save(model, os.join(saved_models_path,model_name+'.pt' ))
            # torch.save(
            #     {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            #     os.path.join(saved_models_path, model_name + "_" + optim_name + ".pt"),
            # )
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                f"{saved_models_path}{model_name}_{optim_name}.pt",
            )

    with open(
        saved_models_path + f"{model_name}_{optim_name}_{lr}_{n_epochs}.pkl", "wb"
    ) as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # After training is done, we should use the test images or another,
    # never seen test image set and generate a confusion matrix, as well as the images that are classified wrong.


if __name__ == "__main__":
    main()
