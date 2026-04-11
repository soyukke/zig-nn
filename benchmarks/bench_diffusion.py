"""Diffusion (DDPM) benchmark: matches examples/diffusion/main.zig exactly
Simple 2D DDPM with 4 Gaussian clusters, T=200, batch=256
Architecture: 9 Linear layers with SiLU, time injection via tp1/tp2/tp3
Noise schedule: linear (1e-4 to 0.02)
"""
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

T_STEPS = 200
DATA_DIM = 2
TIME_EMBED_DIM = 64   # matches DIFF_TIME_DIM
HIDDEN_DIM = 128      # matches DIFF_HIDDEN
NUM_POINTS = 1024     # matches DIFF_NUM_SAMPLES
BATCH_SIZE = 256      # matches DIFF_BATCH

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule matching diffusion.NoiseSchedule.initLinear"""
    betas = torch.zeros(T, dtype=torch.float32)
    for t in range(T):
        betas[t] = beta_start + (t / (T - 1)) * (beta_end - beta_start)
    return betas

class DdpmModel(nn.Module):
    """Exact replica of zig-nn DdpmModel with 9 Linear layers.

    time_fc1: Linear(64, 128)   time MLP layer 1
    time_fc2: Linear(128, 128)  time MLP layer 2
    fc1:  Linear(2, 128)        input projection
    fc2:  Linear(128, 128)      hidden layer 2
    fc3:  Linear(128, 128)      hidden layer 3
    fc_out: Linear(128, 2)      output projection
    tp1:  Linear(128, 128)      time projection for layer 1
    tp2:  Linear(128, 128)      time projection for layer 2
    tp3:  Linear(128, 128)      time projection for layer 3

    Forward:
        t_h1 = SiLU(time_fc1(time_emb))
        t_hidden = time_fc2(t_h1)
        h1 = SiLU(fc1(x_t) + tp1(t_hidden))
        h2 = SiLU(fc2(h1) + tp2(t_hidden))
        h3 = SiLU(fc3(h2) + tp3(t_hidden))
        output = fc_out(h3)
    """
    def __init__(self):
        super().__init__()
        # Time MLP
        self.time_fc1 = nn.Linear(TIME_EMBED_DIM, HIDDEN_DIM)
        self.time_fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # Main MLP
        self.fc1 = nn.Linear(DATA_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc_out = nn.Linear(HIDDEN_DIM, DATA_DIM)
        # Time projection layers (one per hidden layer)
        self.tp1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.tp2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.tp3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x_t, time_emb):
        # Time MLP
        t_h1 = F.silu(self.time_fc1(time_emb))
        t_hidden = self.time_fc2(t_h1)
        # MLP with time injection
        h1 = F.silu(self.fc1(x_t) + self.tp1(t_hidden))
        h2 = F.silu(self.fc2(h1) + self.tp2(t_hidden))
        h3 = F.silu(self.fc3(h2) + self.tp3(t_hidden))
        return self.fc_out(h3)

def sinusoidal_embedding(t, dim):
    """Matches diffusion.sinusoidalEmbedding: [sin..., cos...]"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def generate_data(n, device):
    """Matches generateGaussianMixture: 4 clusters, sigma=0.3"""
    rng = np.random.RandomState(42)
    centers = np.array([[2,0],[0,2],[-2,0],[0,-2]], dtype=np.float32)
    data = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        c = centers[i % 4]
        data[i] = c + rng.randn(2).astype(np.float32) * 0.3
    return torch.tensor(data, device=device)

def run(device):
    torch.manual_seed(42)

    # Linear noise schedule (matching zig-nn)
    betas = linear_beta_schedule(T_STEPS).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, 0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    data = generate_data(NUM_POINTS, device)
    print(f"  Dataset: {NUM_POINTS} points, 4 Gaussian clusters")

    model = DdpmModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params} (~{total_params*4//1024}KB)")
    print(f"  Diffusion steps: T={T_STEPS}, batch={BATCH_SIZE}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1000

    start = time.perf_counter()
    for epoch in range(num_epochs):
        idx = torch.randint(0, NUM_POINTS, (BATCH_SIZE,), device=device)
        x0 = data[idx]
        t = torch.randint(0, T_STEPS, (BATCH_SIZE,), device=device)
        noise = torch.randn_like(x0)

        sab = sqrt_alpha_bar[t].unsqueeze(-1)
        soma = sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sab * x0 + soma * noise

        time_emb = sinusoidal_embedding(t.float(), TIME_EMBED_DIM)

        optimizer.zero_grad()
        pred_noise = model(x_t, time_emb)
        loss = nn.functional.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:>4}: loss = {loss.item():.6f}")

    elapsed = time.perf_counter() - start
    print(f"\n  Training time: {elapsed*1000:.0f}ms")

    # Sampling
    print(f"\n  Sampling 64 points via DDPM reverse process...")
    model.eval()
    n_samples = 64
    x = torch.randn(n_samples, DATA_DIM, device=device)
    with torch.no_grad():
        for step in reversed(range(T_STEPS)):
            t_batch = torch.full((n_samples,), step, dtype=torch.float32, device=device)
            time_emb = sinusoidal_embedding(t_batch, TIME_EMBED_DIM)
            pred = model(x, time_emb)
            beta = betas[step]
            alpha = alphas[step]
            ab = alpha_bar[step]
            x = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - ab).sqrt()) * pred)
            if step > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

    samples = x.cpu().numpy()
    print(f"\n  Generated samples (x, y):")
    for i in range(0, n_samples, 4):
        row = "    "
        for j in range(4):
            if i+j < n_samples:
                row += f"({samples[i+j,0]:>7.3f}, {samples[i+j,1]:>7.3f})    "
        print(row)

    centers = np.array([[2,0],[0,2],[-2,0],[0,-2]])
    counts = [0]*4
    for s in samples:
        dists = np.sum((centers - s)**2, axis=1)
        counts[np.argmin(dists)] += 1
    print(f"\n  Cluster analysis:")
    for i, (c, cnt) in enumerate(zip(centers, counts)):
        print(f"    Cluster {i} ({c[0]:>4.1f}, {c[1]:>4.1f}): {cnt} points")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    device = "cuda" if mode == "cuda" else "cpu"
    print(f"=== DDPM Diffusion (PyTorch {mode.upper()}) ===\n")
    run(device)
