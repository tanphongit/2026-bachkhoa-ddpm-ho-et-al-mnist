import torch
from torch.utils.data import DataLoader, Subset
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

    # ===== Hyperparameters (siêu nhẹ cho Colab) =====
    T = 20
    batch_size = 32
    epochs = 1
    lr = 1e-3
    max_train_samples = 5000   # chỉ lấy 5000 ảnh train để chạy nhanh

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

    # Chỉ lấy 1 phần dữ liệu để train nhanh
    train_ds = Subset(train_ds, range(max_train_samples))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # ===== Model + Diffusion =====
    model = UNetSmall(in_ch=1, base_ch=32, time_dim=64).to(device)
    diffusion = Diffusion(
        T=T,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    # =====================================================
    # 1) Forward noise demo
    # =====================================================
    x0_demo, _ = next(iter(train_loader))
    x0_demo = x0_demo[:8].to(device)

    save_image_grid(
        x0_demo,
        "outputs/diffusion/forward_noise/x0.png",
        nrow=4
    )

    for tval in [0, 10, 19]:
        t = torch.full((x0_demo.size(0),), tval, device=device, dtype=torch.long)
        xt, _ = diffusion.q_sample(x0_demo, t)
        save_image_grid(
            xt,
            f"outputs/diffusion/forward_noise/xt_t{tval}.png",
            nrow=4
        )

    # =====================================================
    # 2) Training
    # =====================================================
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for x0, _ in pbar:
            x0 = x0.to(device)

            t = torch.randint(
                0, T, (x0.size(0),),
                device=device,
                dtype=torch.long
            )

            loss = diffusion.p_losses(model, x0, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "T": T
            },
            f"checkpoints/diffusion/ddpm_mnist_ep{epoch}.pt"
        )

    # =====================================================
    # 3) Sample finnal
    # =====================================================
    samples = diffusion.sample(model, (8, 1, 28, 28))
    save_image_grid(
        samples,
        "outputs/diffusion/samples/sample_final.png",
        nrow=4
    )

    # =====================================================
    # 4) Save loss curve
    # =====================================================
    plot_loss_curve(losses, "outputs/diffusion/loss_curve.png")

    print("Training complete.")
    print("Checkpoint: checkpoints/diffusion/ddpm_mnist_ep1.pt")
    print("Sample: outputs/diffusion/samples/sample_final.png")


if __name__ == "__main__":
    main()