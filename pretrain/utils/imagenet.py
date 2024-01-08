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


class AIMLDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_path: str,
        transform=None,
    ):
        self.img_paths_labels = pd.read_pickle(
            os.path.join(data_path, annotations_file)
        )
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_paths_labels.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image


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
        img_file_path = self.samples[index, -1]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
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
    annotations_file = os.path.join(dataset_path, "annotations/img_paths_mini.csv")
    dataset_train = AIMLDataset(annotations_file, dataset_path, mod_trans_train)
    print_transform(mod_trans_train, "[pre-train]")
    return dataset_train


def print_transform(transform, s):
    print(f"Transform {s} = ")
    for t in transform.transforms:
        print(t)
    print("---------------------------\n")
