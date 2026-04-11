"""Spiral benchmark: matches examples/spiral/main.zig"""
import time
import torch
import torch.nn as nn
import numpy as np

N_PER_CLASS = 50
N_CLASSES = 3
TOTAL = N_PER_CLASS * N_CLASSES

def generate_spiral_data():
    rng = np.random.RandomState(12345)
    X = np.zeros((TOTAL, 2), dtype=np.float32)
    Y = np.zeros(TOTAL, dtype=np.int64)
    for cls in range(N_CLASSES):
        for i in range(N_PER_CLASS):
            idx = cls * N_PER_CLASS + i
            r = i / N_PER_CLASS
            base_angle = cls * 4.0 + r * 4.0
            noise = (rng.rand() - 0.5) * 0.3
            angle = base_angle + noise
            X[idx, 0] = r * np.cos(angle)
            X[idx, 1] = r * np.sin(angle)
            Y[idx] = cls
    return X, Y

def run(device):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    ).to(device)

    X_np, Y_np = generate_spiral_data()
    X = torch.tensor(X_np, device=device)
    Y = torch.tensor(Y_np, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 500

    start = time.perf_counter()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            pred = logits.argmax(dim=1)
            acc = (pred == Y).float().mean().item()
            print(f"  Epoch {epoch:>3}: loss = {loss.item():.4f}, accuracy = {acc*100:.1f}%")
    elapsed = time.perf_counter() - start

    print(f"\n  Training time: {elapsed*1000:.0f}ms")
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        acc = (pred == Y).float().mean().item()
        print(f"  Final accuracy: {acc*100:.1f}%")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    device = "cuda" if mode == "cuda" else "cpu"
    print(f"=== Spiral Classification (PyTorch {mode.upper()}: CrossEntropy + Adam) ===\n")
    print(f"  Data: {TOTAL} samples, {N_CLASSES} classes")
    run(device)
