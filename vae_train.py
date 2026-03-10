import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.vae import VAE, vae_loss
from src.utils import seed_all, ensure_dir


def save_vae_samples(model, device, path, n=64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        samples = model.decode(z).cpu()
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
    lr = 1e-3
    latent_dim = 32

    # ===== Create folders =====
    ensure_dir("data")
    ensure_dir("checkpoints/vae")
    ensure_dir("outputs/vae")
    ensure_dir("outputs/vae/samples")

    # ===== Dataset =====
    tfm = transforms.ToTensor()  # keep in [0,1] for BCE
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
        num_workers=0
    )

    # ===== Model =====
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    # ===== Training =====
    model.train()
    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"VAE Epoch {epoch}/{epochs}")

        for x, _ in pbar:
            x = x.to(device)

            x_hat, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_per_sample = loss.item() / x.size(0)
            losses.append(loss_per_sample)

            pbar.set_postfix(
                loss=f"{loss_per_sample:.4f}",
                recon=f"{recon_loss.item() / x.size(0):.4f}",
                kl=f"{kl_loss.item() / x.size(0):.4f}"
            )

        # ===== Save samples after each epoch =====
        save_vae_samples(
            model,
            device,
            f"outputs/vae/samples/samples_epoch_{epoch:02d}.png",
            n=64
        )

    # ===== Save checkpoint =====
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": latent_dim
        },
        "checkpoints/vae/vae_mnist.pt"
    )

    # ===== Save loss curve =====
    plot_curve(losses, "outputs/vae/loss_curve.png", "VAE Training Loss", "Loss per sample")

    print("VAE training complete.")
    print("Check outputs/vae/ and checkpoints/vae/ folders.")


if __name__ == "__main__":
    main()