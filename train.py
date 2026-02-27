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
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # ===== Hyperparameters (nhẹ cho đồ án 1 tín chỉ) =====
    T = 200
    batch_size = 128
    epochs = 10
    lr = 1e-3

    # ===== Create folders =====
    ensure_dir("data")
    ensure_dir("checkpoints")
    ensure_dir("outputs")
    ensure_dir("outputs/forward_noise")
    ensure_dir("outputs/samples")

    # ===== Dataset (Normalize về [-1,1]) =====
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # map [0,1] -> [-1,1]
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
        num_workers=0,  # ✅ tránh lỗi pickle Python 3.14
        drop_last=True
    )

    # ===== Model + Diffusion =====
    model = UNetSmall(in_ch=1, base_ch=64, time_dim=128).to(device)
    diffusion = Diffusion(T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    global_step = 0

    # =====================================================
    # 1) Forward noise demo (để đưa vào báo cáo)
    # =====================================================
    x0_demo, _ = next(iter(train_loader))
    x0_demo = x0_demo[:16].to(device)

    save_image_grid(x0_demo, "outputs/forward_noise/x0.png", nrow=4)

    for tval in [0, 50, 100, 150, 199]:
        t = torch.full((x0_demo.size(0),), tval, device=device, dtype=torch.long)
        xt, _ = diffusion.q_sample(x0_demo, t)
        save_image_grid(
            xt,
            f"outputs/forward_noise/xt_t{tval}.png",
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

            # Random timestep
            t = torch.randint(0, T, (x0.size(0),), device=device, dtype=torch.long)

            loss = diffusion.p_losses(model, x0, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            global_step += 1

            if global_step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== Save checkpoint =====
        torch.save(
            {
                "model": model.state_dict(),
                "T": T
            },
            f"checkpoints/ddpm_mnist_ep{epoch}.pt"
        )

        # ===== Sample preview mỗi epoch =====
        samples = diffusion.sample(model, (16, 1, 28, 28))
        save_image_grid(
            samples,
            f"outputs/samples/sample_ep{epoch}.png",
            nrow=4
        )

    # =====================================================
    # 3) Save loss curve
    # =====================================================
    plot_loss_curve(losses, "outputs/loss_curve.png")

    print("Training complete.")
    print("Check outputs/ and checkpoints/ folders.")


if __name__ == "__main__":
    main()