from torch import Tensor
from typing import Tuple

import os
import click

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
from tqdm import tqdm


class HotDogDataset(Dataset):
    def __init__(self, data_folder_path: str, train: bool = True) -> None:
        super(HotDogDataset, self).__init__()

        data_suffix = "train"
        if not train:
            data_suffix = "test"

        self.data_path = os.path.join(data_folder_path, data_suffix)

        self.images = ImageFolder(self.data_path)

        self.labels = self.images.classes

        # Transform labels from hotdog and nothotdog to 0 and 1 ids respectively.
        # Maybe hotdog should be 1?
        self.id_to_label = {i: lbl for (i, lbl) in enumerate(self.labels)}
        self.label_to_id = {lbl: i for (i, lbl) in enumerate(self.labels)}

        blur_kernel = 5
        normalize = transforms.Normalize([0.0] * 3, [1.0] * 3)

        # If we crop the images the moment they are loaded and aded to the batch,
        # then we need to decide upon the crop dimensions, whether the crop should be random
        # and changed every epoch when the image is reploaded, or static in the middle eg.
        # However, the latter imposes the danger that the hotdog is not in the center.
        # The former might not catch the hotdog, but maybe the next time it will :)

        crop_size = self.get_min_resolutions(print_stats=False)

        crop_size = [int(min(crop_size))] * 2

        crop = transforms.RandomCrop(crop_size)

        self.transforms_ = transforms.Compose(
            [
                crop,
                transforms.RandomGrayscale(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(blur_kernel),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __getitem__(self, index: int) -> Tensor:
        img = self.images[index]
        return (
            self.transforms_(img),
            torch.Tensor(self.label_to_id[self.labels[index]]),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def get_min_resolutions(self, print_stats: bool = False) -> Tuple[int]:
        """
        Method to get the min and max resolution of the images in the train folder.
        If print_stats is True, then it will print the statistics of the resolutions in height and width.
        """

        heights = np.zeros(len(self.images))
        widths = np.zeros(len(self.images))

        for i, image in tqdm(enumerate(self.images.imgs)):
            img = Image.open(image[0])
            heights[i] = img.height
            widths[i] = img.width

        if print_stats:
            print(f"{self.data_path}")
            print(
                f"Mean height: {np.round(heights.mean(),2)}, Std heights: {np.round(heights.std(),2)}"
            )
            print(
                f"Max height: {np.round(heights.max(),2)}, Min height: {np.round(heights.min(),2)}"
            )
            print(
                f"Mean width: {np.round(widths.mean(),2)}, Std widths: {np.round(widths.std(),2)}\n"
            )
            print(
                f"Max width: {np.round(widths.max(),2)}, Min width: {np.round(widths.min(),2)}"
            )

        return heights.min(), widths.min()

    def preprocess_save(self, output_path: str) -> None:
        """
        Method used to save the preprocessed images from the raw, to the processed directory
        """

        pass


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def main(input_filepath: str):
    train_data_set = HotDogDataset(data_folder_path=input_filepath, train=True)
    test_data_set = HotDogDataset(data_folder_path=input_filepath, train=False)

    print()
    print(f"Train images:\n{train_data_set.images}")
    print(
        f"Train number of labels:\n{train_data_set.__len__()}, {train_data_set.labels}"
    )
    print()
    print(f"Test images:\n{test_data_set.images}")
    print(f"Test number of labels:\n{test_data_set.__len__()}, {test_data_set.labels}")
    print()

    # Here we can preprocess and save to the processed folder tensors (.pth) or images (cropped etc.)


if __name__ == "__main__":
    main()
