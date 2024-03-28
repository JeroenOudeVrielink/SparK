import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


IMG_PATHS = "test_imgs"
SAVE_DIR = "mask"

mod_trans_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.67, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)


def plot_patches(patches, labels=None, save_path=None):
    """
    Plots a batch of patches in a grid and optionally displays a label in the middle of each patch.

    Args:
        patches: A PyTorch tensor of images with shape (batch_size, num_patches, num_channels, patch_size, patch_size).
        labels:  A list of labels (numbers) to display on the patches. If None, no labels are displayed.
    """
    num_patches = patches.shape[0]
    sqrt_num_patches = int(num_patches**0.5)  # Assuming a square grid

    fig, axes = plt.subplots(sqrt_num_patches, sqrt_num_patches, figsize=(10, 10))
    axes = axes.flatten()
    custom_cmap = plt.cm.get_cmap("RdBu")  # Red-blue colormap

    for patch, label, ax in zip(patches, labels, axes):
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


def get_weighted_random_weights(image, masking_ratio=0.6):
    weights = get_binary_weights(image)
    weights = weights / weights.sum()
    n_masked = round(masking_ratio * np.count_nonzero(weights))
    indices = np.random.choice(49, n_masked, replace=False, p=weights)
    indices = torch.tensor(indices)
    mask = (
        torch.ones(7 * 7, dtype=torch.bool)
        .scatter_(dim=0, index=indices, value=False)
        .view(7, 7)
    )
    return mask


def save_mask(img_path, save_path_og, save_path):
    img = Image.open(img_path)
    img = mod_trans_train(img)

    labels = get_binary_weights(img[0].unsqueeze(0))
    labels = labels / labels.sum()
    labels = labels.tolist()
    labels_round = [round(l, 3) for l in labels]
    mask = get_weighted_random_weights(img[0].unsqueeze(0))
    mask = (
        mask.unsqueeze(0).unsqueeze(0).repeat_interleave(32, 2).repeat_interleave(32, 3)
    )
    x = img * mask
    patches = extract_patches(x, patch_size=32, num_channels=3)
    plot_patches(patches.squeeze(0), labels=labels_round, save_path=save_path)


if __name__ == "__main__":
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        img_path = Path(IMG_PATHS) / f"{i}.png"
        save_path_og = save_dir / f"{i}_og.png"
        save_path = save_dir / f"{i}_masked.png"
        save_mask(img_path, save_path_og, save_path)
