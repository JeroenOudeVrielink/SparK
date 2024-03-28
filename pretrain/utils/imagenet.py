# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
import torch
import numpy as np
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode

    interpolation = InterpolationMode.BICUBIC
except:
    import PIL

    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def extract_patches(images, patch_size=32, num_channels=3):
    """
    Extracts patches from a batch of images.

    Args:
        images: A PyTorch tensor of images with shape (batch_size, num_channels, height, width).
        patch_size: The size of the patches to extract.
        num_channels: The number of channels in the images.

    Returns:
        A PyTorch tensor of patches with shape (batch_size, num_patches, num_channels, patch_size, patch_size).
    """

    batch_size, _, height, width = images.shape

    # Ensure that the image dimensions are divisible by the patch size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by the patch size.")

    # Use torch.nn.functional.unfold to extract patches
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)

    # Reshape patches to the desired format
    patches = patches.transpose(1, 2).reshape(
        batch_size, -1, num_channels, patch_size, patch_size
    )

    return patches


def get_binary_weights(image_tensor):
    """
    Applies Otsu thresholding to a PyTorch image tensor.

    Args:
        image_tensor: A PyTorch tensor of shape (channels, height, width) representing an image.

    Returns:
        A numpy array of the same shape with the Otsu threshold applied.
    """

    # Check if grayscale
    if image_tensor.shape[0] != 1:
        raise ValueError("Otsu thresholding typically expects a grayscale image.")
    # Convert to NumPy array
    image_array = image_tensor.numpy()
    # Calculate Otsu threshold
    otsu_threshold = threshold_otsu(image_array)
    # Apply threshold
    binary_image = image_array > otsu_threshold
    # Convert back to PyTorch tensor
    binary_tensor = torch.from_numpy(
        binary_image.astype(np.float32)
    )  # Assuming you want a float representation
    patches = extract_patches(binary_tensor.unsqueeze(0), patch_size=32, num_channels=1)
    weights = patches.squeeze(0).sum(dim=(2, 3))
    return weights.squeeze(1).numpy()


def get_weighted_random_mask(image, masking_ratio=0.6):
    weights = get_binary_weights(image)
    n_masked = round(masking_ratio * np.count_nonzero(weights))
    weights = weights + 1e-6
    weights = weights / weights.sum()
    indices = np.random.choice(49, n_masked, replace=False, p=weights)
    indices = torch.tensor(indices)
    mask = (
        torch.ones(7 * 7, dtype=torch.bool)
        .scatter_(dim=0, index=indices, value=False)
        .view(7, 7)
    )
    return mask


class AIMLDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_path: str,
        transform=None,
        weighted_masking=False,
        masking_ratio=0.6,
    ):
        self.img_paths_labels = pd.read_csv(os.path.join(data_path, annotations_file))
        self.data_path = data_path
        self.transform = transform
        self.weighted_masking = weighted_masking
        self.masking_ratio = masking_ratio

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_paths_labels.iloc[idx, -1])
        image = Image.open(img_path)

        mask = 0
        if self.transform:
            image = self.transform(image)
            if self.weighted_masking:
                mask = get_weighted_random_mask(
                    image[0].unsqueeze(0), self.masking_ratio
                )
        return image, mask


class ImageNetDataset(DatasetFolder):
    def __init__(
        self,
        imagenet_folder: str,
        train: bool,
        transform: Callable,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, "train" if train else "val")
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None,
            is_valid_file=is_valid_file,
        )

        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None  # this is self-supervised learning so we don't need labels

    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(
    annotations_file, dataset_path, input_size, weighted_masking, masking_ratio
) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow.

    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    # trans_train = transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(
    #             input_size, scale=(0.67, 1.0), interpolation=interpolation
    #         ),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    #     ]
    # )
    mod_trans_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size, scale=(0.67, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

    # dataset_path = os.path.abspath(dataset_path)
    # for postfix in ("train", "val"):
    #     if dataset_path.endswith(postfix):
    #         dataset_path = dataset_path[: -len(postfix)]

    # dataset_train = ImageNetDataset(
    #     imagenet_folder=dataset_path, transform=trans_train, train=True
    # )
    dataset_train = AIMLDataset(
        annotations_file,
        dataset_path,
        mod_trans_train,
        weighted_masking=weighted_masking,
        masking_ratio=masking_ratio,
    )
    print_transform(mod_trans_train, "[pre-train]")
    if weighted_masking:
        print("Weighted masking is ENABELED!")

    return dataset_train


def print_transform(transform, s):
    print(f"Transform {s} = ")
    for t in transform.transforms:
        print(t)
    print("---------------------------\n")
