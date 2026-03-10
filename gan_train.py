import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.gan import Generator, Discriminator
from src.utils import seed_all, ensure_dir


def save_gan_samples(generator, device, path, n=64):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, generator.z_dim, device=device)
        samples = generator(z).cpu()
        grid = make_grid(samples, nrow=int(math.sqrt(n)), pad_value=1.0)
        save_image(grid, path)


def plot_curve(values, path, title, ylabel):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    seed_all(42)

    # ===== Device =====
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # ===== Hyperparameters =====
    batch_size = 128
    epochs = 5
    lr = 2e-4
    z_dim = 64

    # ===== Create folders =====
    ensure_dir("data")
    ensure_dir("checkpoints/gan")
    ensure_dir("outputs/gan")
    ensure_dir("outputs/gan/samples")

    # ===== Dataset =====
    tfm = transforms.ToTensor()  # keep in [0,1]
    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=tfm
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # ===== Models =====
    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    # ===== Training =====
    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch}/{epochs}")

        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            # ==========================================
            # 1) Train Discriminator
            # ==========================================
            z = torch.randn(bsz, z_dim, device=device)
            fake_imgs = generator(z).detach()

            real_logits = discriminator(real_imgs)
            fake_logits = discriminator(fake_imgs)

            d_loss_real = F.binary_cross_entropy_with_logits(
                real_logits,
                torch.ones_like(real_logits)
            )
            d_loss_fake = F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.zeros_like(fake_logits)
            )
            d_loss = d_loss_real + d_loss_fake

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # ==========================================
            # 2) Train Generator
            # ==========================================
            z = torch.randn(bsz, z_dim, device=device)
            fake_imgs = generator(z)
            fake_logits = discriminator(fake_imgs)

            g_loss = F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.ones_like(fake_logits)
            )

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            pbar.set_postfix(
                d_loss=f"{d_loss.item():.4f}",
                g_loss=f"{g_loss.item():.4f}"
            )

        # ===== Save generated samples after each epoch =====
        save_gan_samples(
            generator,
            device,
            f"outputs/gan/samples/samples_epoch_{epoch:02d}.png",
            n=64
        )

    # ===== Save checkpoint =====
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "z_dim": z_dim
        },
        "checkpoints/gan/gan_mnist.pt"
    )

    # ===== Save loss curves =====
    plot_curve(g_losses, "outputs/gan/loss_curve_G.png", "GAN Generator Loss", "Loss")
    plot_curve(d_losses, "outputs/gan/loss_curve_D.png", "GAN Discriminator Loss", "Loss")

    print("GAN training complete.")
    print("Check outputs/gan/ and checkpoints/gan/ folders.")


if __name__ == "__main__":
    main()