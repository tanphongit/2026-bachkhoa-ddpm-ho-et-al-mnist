import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)  # (T,)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bar[:-1]])
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    def p_losses(self, model, x0, t):
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, xt, t):
        betat = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)

        eps_theta = model(xt, t)
        mean = sqrt_recip_alpha * (xt - (betat / sqrt_one_minus_ab) * eps_theta)

        if (t == 0).all():
            return mean

        var = self.posterior_variance[t].view(-1, 1, 1, 1).clamp(min=1e-20)
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, shape):
        model.eval()
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def progressive_samples(self, model, shape, save_steps=(199, 150, 100, 50, 0)):
        model.eval()
        x = torch.randn(shape, device=self.device)
        snaps = {}
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
            if step in save_steps:
                snaps[step] = x.clone()
        return [snaps[s] for s in save_steps if s in snaps]