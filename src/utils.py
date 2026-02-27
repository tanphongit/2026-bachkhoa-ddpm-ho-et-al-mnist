import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    """
    images: (B, C, H, W) in [-1, 1]
    """
    imgs = (images.clamp(-1, 1) + 1) / 2  # -> [0,1]
    grid = make_grid(imgs, nrow=nrow, padding=2)
    save_image(grid, path)


def plot_loss_curve(losses, path: str):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title("DDPM Training Loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()