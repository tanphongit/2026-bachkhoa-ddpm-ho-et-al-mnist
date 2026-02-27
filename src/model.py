import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


class UNetSmall(nn.Module):
    """
    UNet nhỏ cho MNIST 28x28.
    Output: epsilon_hat cùng shape với input.
    """
    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down path
        self.rb1 = ResBlock(base_ch, base_ch, time_dim)
        self.down1 = Down(base_ch)                # 28 -> 14

        self.rb2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = Down(base_ch * 2)            # 14 -> 7

        self.rb3 = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        # Middle
        self.mid1 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.mid2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        # Up path
        self.up1 = Up(base_ch * 2)                # 7 -> 14
        self.rb4 = ResBlock(base_ch * 4, base_ch, time_dim)  # concat with skip2

        self.up2 = Up(base_ch)                    # 14 -> 28
        self.rb5 = ResBlock(base_ch * 2, base_ch, time_dim)  # concat with skip1

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        x0 = self.in_conv(x)
        s1 = self.rb1(x0, t_emb)     # skip1
        d1 = self.down1(s1)

        s2 = self.rb2(d1, t_emb)     # skip2
        d2 = self.down2(s2)

        h = self.rb3(d2, t_emb)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        h = self.up1(h)
        h = self.rb4(torch.cat([h, s2], dim=1), t_emb)

        h = self.up2(h)
        h = self.rb5(torch.cat([h, s1], dim=1), t_emb)

        out = self.out_conv(F.silu(self.out_norm(h)))
        return out