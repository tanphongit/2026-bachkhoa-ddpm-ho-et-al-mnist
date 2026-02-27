# DDPM (Ho et al.) - MNIST (PyTorch)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pip install --upgrade pip

python train.py

python sample.py --ckpt checkpoints/ddpm_mnist_ep10.pt