"""DDPM 2D Gaussian Mixture - PyTorch benchmark
同一条件: batch=256, hidden=128, time_dim=64, T=200, 1000 steps, Adam lr=1e-3
"""
import torch
import torch.nn as nn
import time
import math
import argparse

import os
BATCH = int(os.environ.get("BENCH_BATCH", "256"))
HIDDEN = int(os.environ.get("BENCH_HIDDEN", "128"))
TIME_DIM = 64
T = 200
NUM_STEPS = 1000

class DDPMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Time MLP
        self.time_fc1 = nn.Linear(TIME_DIM, HIDDEN)
        self.time_fc2 = nn.Linear(HIDDEN, HIDDEN)
        # Main MLP
        self.fc1 = nn.Linear(2, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)
        self.fc_out = nn.Linear(HIDDEN, 2)
        # Time projections
        self.time_proj1 = nn.Linear(HIDDEN, HIDDEN)
        self.time_proj2 = nn.Linear(HIDDEN, HIDDEN)
        self.time_proj3 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x_t, time_emb):
        # Time MLP
        t = torch.nn.functional.silu(self.time_fc1(time_emb))
        t = self.time_fc2(t)
        # MLP with time injection
        h = torch.nn.functional.silu(self.fc1(x_t) + self.time_proj1(t))
        h = torch.nn.functional.silu(self.fc2(h) + self.time_proj2(t))
        h = torch.nn.functional.silu(self.fc3(h) + self.time_proj3(t))
        return self.fc_out(h)

def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

def linear_schedule(T):
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def generate_gaussian_mixture(n, device):
    centers = torch.tensor([[2,0],[0,2],[-2,0],[0,-2]], dtype=torch.float32, device=device)
    idx = torch.arange(n, device=device) % 4
    pts = centers[idx] + 0.3 * torch.randn(n, 2, device=device)
    return pts

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def bench(device_name):
    device = torch.device(device_name)
    model = DDPMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    betas, alphas, alpha_bars = linear_schedule(T)
    sqrt_ab = torch.sqrt(alpha_bars).to(device)
    sqrt_omab = torch.sqrt(1.0 - alpha_bars).to(device)

    dataset = generate_gaussian_mixture(1024, device)

    print(f"  Device: {device_name}")
    print(f"  Parameters: {count_params(model)}")

    # Warmup
    for _ in range(10):
        idx = torch.randint(0, 1024, (BATCH,), device=device)
        x0 = dataset[idx]
        t = torch.randint(0, T, (BATCH,), device=device)
        eps = torch.randn_like(x0)
        x_t = sqrt_ab[t].unsqueeze(1) * x0 + sqrt_omab[t].unsqueeze(1) * eps
        time_emb = sinusoidal_embedding(t, TIME_DIM)
        eps_pred = model(x_t, time_emb)
        loss = nn.functional.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Benchmark
    if device_name == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()

    for step in range(NUM_STEPS):
        idx = torch.randint(0, 1024, (BATCH,), device=device)
        x0 = dataset[idx]
        t = torch.randint(0, T, (BATCH,), device=device)
        eps = torch.randn_like(x0)
        x_t = sqrt_ab[t].unsqueeze(1) * x0 + sqrt_omab[t].unsqueeze(1) * eps
        time_emb = sinusoidal_embedding(t, TIME_DIM)
        eps_pred = model(x_t, time_emb)
        loss = nn.functional.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device_name == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  {NUM_STEPS} steps in {elapsed*1000:.0f}ms ({elapsed/NUM_STEPS*1000:.2f} ms/step)")
    print(f"  Throughput: {BATCH * NUM_STEPS / elapsed:.0f} samples/sec")
    print(f"  Final loss: {loss.item():.6f}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    print(f"=== DDPM Benchmark (batch={BATCH}, hidden={HIDDEN}, steps={NUM_STEPS}) ===\n")
    bench(args.device)
