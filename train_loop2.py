import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import UNetSmall
from src.diffusion import Diffusion
from src.utils import seed_all, ensure_dir, save_image_grid, plot_loss_curve


def main():
    seed_all(42)

    # ===== Device =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ===== Hyperparameters (nhẹ cho Colab) =====
    T = 50
    batch_size = 64
    epochs = 2
    lr = 1e-3

    # ===== Create folders =====
    ensure_dir("data")
    ensure_dir("checkpoints/diffusion")
    ensure_dir("outputs/diffusion")
    ensure_dir("outputs/diffusion/forward_noise")
    ensure_dir("outputs/diffusion/samples")

    # ===== Dataset =====
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

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

    # ===== Model =====
    model = UNetSmall(in_ch=1, base_ch=64, time_dim=128).to(device)

    diffusion = Diffusion(
        T=T,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    # =====================================================
    # Forward noise demo (ít ảnh để chạy nhanh)
    # =====================================================
    x0_demo, _ = next(iter(train_loader))
    x0_demo = x0_demo[:8].to(device)

    save_image_grid(
        x0_demo,
        "outputs/diffusion/forward_noise/x0.png",
        nrow=4
    )

    for tval in [0, 25, 49]:
        t = torch.full(
            (x0_demo.size(0),),
            tval,
            device=device,
            dtype=torch.long
        )

        xt, _ = diffusion.q_sample(x0_demo, t)

        save_image_grid(
            xt,
            f"outputs/diffusion/forward_noise/xt_t{tval}.png",
            nrow=4
        )

    # =====================================================
    # Training
    # =====================================================
    model.train()

    for epoch in range(1, epochs + 1):

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for x0, _ in pbar:

            x0 = x0.to(device)

            t = torch.randint(
                0,
                T,
                (x0.size(0),),
                device=device,
                dtype=torch.long
            )

            loss = diffusion.p_losses(model, x0, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== Save checkpoint =====
        torch.save(
            {
                "model": model.state_dict(),
                "T": T
            },
            f"checkpoints/diffusion/ddpm_mnist_ep{epoch}.pt"
        )

        # ===== Sample preview (ít ảnh) =====
        samples = diffusion.sample(model, (8, 1, 28, 28))

        save_image_grid(
            samples,
            f"outputs/diffusion/samples/sample_ep{epoch}.png",
            nrow=4
        )

    # =====================================================
    # Save loss curve
    # =====================================================
    plot_loss_curve(losses, "outputs/diffusion/loss_curve.png")

    print("Training complete.")


if __name__ == "__main__":
    main()