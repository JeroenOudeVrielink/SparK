import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms
from skimage.filters import threshold_otsu
import numpy as np
import scipy
from pathlib import Path

IMG_PATHS = "test_imgs"
SAVE_DIR = "average"

mod_trans_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.67, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)


def apply_otsu_thresholding(image_tensor):
    """
    Applies Otsu thresholding to a PyTorch image tensor.

    Args:
        image_tensor: A PyTorch tensor of shape (channels, height, width) representing an image.

    Returns:
        A PyTorch tensor of the same shape with the Otsu threshold applied.
    """

    # Check if grayscale
    if image_tensor.shape[0] != 1:
        raise ValueError("Otsu thresholding typically expects a grayscale image.")

    # Convert to NumPy array
    image_array = image_tensor.squeeze(0).numpy()  # Remove channel dimension

    # Calculate Otsu threshold
    otsu_threshold = threshold_otsu(image_array)

    # Apply threshold
    binary_image = image_array > otsu_threshold

    # Convert back to PyTorch tensor
    binary_tensor = torch.from_numpy(
        binary_image.astype(np.float32)
    )  # Assuming you want a float representation
    return binary_tensor.unsqueeze(0)  # Re-add channel dimension


def plot_patches(patches, labels=None, save_path=None):
    """
    Plots a batch of patches in a grid and optionally displays a label in the middle of each patch.

    Args:
        patches: A PyTorch tensor of images with shape (batch_size, num_patches, num_channels, patch_size, patch_size).
        labels:  A list of labels (numbers) to display on the patches. If None, no labels are displayed.
    """
    num_patches = patches.shape[1]
    sqrt_num_patches = int(num_patches**0.5)  # Assuming a square grid

    fig, axes = plt.subplots(sqrt_num_patches, sqrt_num_patches, figsize=(10, 10))
    axes = axes.flatten()
    custom_cmap = plt.cm.get_cmap("RdBu")  # Red-blue colormap

    for patch, label, ax in zip(patches[0], labels, axes):
        if patch.shape[0] == 1:  # Grayscale
            if label == 1024:
                patch[0][0][0] = 0
            ax.imshow(patch[0].numpy(), cmap=custom_cmap)
        else:  # RGB
            ax.imshow(patch.permute(1, 2, 0).numpy())
        ax.axis("off")

        if labels:
            patch_center = patch.shape[2] // 2  # Assumes square patches
            ax.text(
                patch_center,
                patch_center,
                str(label),
                ha="center",
                va="center",
                color="white",
                fontsize=12,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def calculate_patch_entropies(patches):
    """
    Calculates the entropies of image patches. Assumes no batch dimension.

    Args:
        patches: A PyTorch tensor of shape (num_patches, num_channels, patch_size, patch_size).

    Returns:
        A NumPy array containing the entropy of each patch.
    """

    patch_entropies = []
    for patch in patches:  # Iterate over each patch

        # Flatten the patch
        patch_flattened = patch.reshape(
            patch.shape[0] * patch.shape[1] * patch.shape[2]
        )

        # Calculate histogram
        hist, _ = np.histogram(patch_flattened, bins=100, range=(0, 1))

        # Normalize to probabilities
        prob_dist = hist / hist.sum()

        # Calculate entropy (remove zeros to avoid log issues)
        entropy = scipy.stats.entropy(prob_dist[prob_dist > 0], base=2)

        patch_entropies.append(entropy)

    return patch_entropies


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


def binary_mask(img_path, save_path_og=None, save_path=None):
    pil_img = Image.open(img_path)
    # filt_img = pil_img.filter(ImageFilter.FIND_EDGES)
    # filt_img.show()

    pt_img = mod_trans_train(pil_img)

    # Add a batch dimension
    pt_img = pt_img.unsqueeze(0)

    patches = extract_patches(pt_img, patch_size=32, num_channels=3)
    # print(patches.shape)  # Output: (2, 49, 1, 32, 32)

    labels = list(range(patches.shape[1]))  # Example labels: 0, 1, 2, ...
    plot_patches(patches, labels=labels, save_path=save_path_og)

    # Apply Otsu thresholding to the patches

    bin_img = apply_otsu_thresholding(pt_img.squeeze(0)[0].unsqueeze(0))
    patches = extract_patches(bin_img.unsqueeze(0), patch_size=32, num_channels=1)
    labels = patches.squeeze(0).sum((1, 2, 3)).tolist()  # Example labels: 0, 1, 2, ...
    plot_patches(patches, labels=labels, save_path=save_path)


def edge_detection(img_path, save_path_og=None, save_path=None):
    pil_img = Image.open(img_path)
    # filt_img = pil_img.filter(ImageFilter.FIND_EDGES)
    # filt_img.show()

    pt_img = mod_trans_train(pil_img)

    # Add a batch dimension
    pt_img = pt_img.unsqueeze(0)

    patches = extract_patches(pt_img, patch_size=32, num_channels=3)
    # print(patches.shape)  # Output: (2, 49, 1, 32, 32)

    labels = list(range(patches.shape[1]))  # Example labels: 0, 1, 2, ...
    plot_patches(patches, labels=labels, save_path=save_path_og)

    new_pil_img = transforms.ToPILImage()(pt_img.squeeze(0))
    filt_img = new_pil_img.filter(ImageFilter.FIND_EDGES)
    new_pt_img = transforms.ToTensor()(filt_img).unsqueeze(0)
    patches = extract_patches(new_pt_img, patch_size=32, num_channels=3)
    labels = patches.squeeze(0).sum((1, 2, 3)).tolist()  # Example labels: 0, 1, 2, ...
    plot_patches(patches, labels=[round(num) for num in labels], save_path=save_path)


def entropy(img_path, save_path_og=None, save_path=None):
    pil_img = Image.open(img_path)

    pt_img = mod_trans_train(pil_img)

    # Add a batch dimension
    pt_img = pt_img.unsqueeze(0)

    patches = extract_patches(pt_img, patch_size=32, num_channels=3)
    # print(patches.shape)  # Output: (2, 49, 1, 32, 32)

    labels = list(range(patches.shape[1]))  # Example labels: 0, 1, 2, ...
    plot_patches(patches, labels=labels, save_path=save_path_og)

    labels = calculate_patch_entropies(patches.squeeze(0))
    plot_patches(patches, labels=[round(num, 2) for num in labels], save_path=save_path)


def pixel_avg(img_path, save_path_og=None, save_path=None):
    pil_img = Image.open(img_path)

    pt_img = mod_trans_train(pil_img)

    # Add a batch dimension
    pt_img = pt_img.unsqueeze(0)

    patches = extract_patches(pt_img, patch_size=32, num_channels=3)
    # print(patches.shape)  # Output: (2, 49, 1, 32, 32)

    labels = list(range(patches.shape[1]))  # Example labels: 0, 1, 2, ...
    plot_patches(patches, labels=labels, save_path=save_path_og)

    labels = patches.squeeze(0).mean((1, 2, 3)).tolist()
    plot_patches(patches, labels=[round(num, 2) for num in labels], save_path=save_path)


if __name__ == "__main__":
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        img_path = Path(IMG_PATHS) / f"{i}.png"
        save_path_og = save_dir / f"{i}_og.png"
        save_path = save_dir / f"{i}_bin.png"
        pixel_avg(img_path, save_path_og, save_path)
