import argparse
import torch

from src.model import UNetSmall
from src.diffusion import Diffusion
from src.utils import ensure_dir, save_image_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ensure_dir("outputs/progressive")

    # load ckpt
    state = torch.load(args.ckpt, map_location=device)
    T = state.get("T", 200)

    model = UNetSmall(in_ch=1, base_ch=64, time_dim=128).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    diff = Diffusion(T=T, beta_start=1e-4, beta_end=0.02, device=device)

    # final samples
    samples = diff.sample(model, (args.n, 1, 28, 28))
    save_image_grid(samples, "outputs/final_samples.png", nrow=8)

    # progressive samples (noise -> rõ dần)
    snaps = diff.progressive_samples(model, (16, 1, 28, 28), save_steps=(T-1, int(T*0.75), int(T*0.5), int(T*0.25), 0))
    for i, img in enumerate(snaps):
        save_image_grid(img, f"outputs/progressive/step_{i}.png", nrow=4)

    print("Saved: outputs/final_samples.png and outputs/progressive/*.png")


if __name__ == "__main__":
    main()