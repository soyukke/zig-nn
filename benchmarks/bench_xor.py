"""XOR benchmark: matches examples/xor/main.zig"""
import time
import torch
import torch.nn as nn

def run(device):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    ).to(device)

    inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32, device=device)
    targets = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 2000

    start = time.perf_counter()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = nn.functional.mse_loss(output, targets)
        loss.backward()
        optimizer.step()

        if epoch % 400 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:>4}: loss = {loss.item():.6f}")
    elapsed = time.perf_counter() - start

    print(f"\n  Training time: {elapsed*1000:.0f}ms")
    with torch.no_grad():
        pred = model(inputs).squeeze()
        for i, ((a,b), t) in enumerate(zip(inputs, targets.squeeze())):
            p = pred[i].item()
            mark = "OK" if abs(p - t.item()) < 0.3 else "NG"
            print(f"    [{a:.0f}, {b:.0f}] -> {p:.4f} (target: {t.item():.0f}) {mark}")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    device = "cuda" if mode == "cuda" else "cpu"
    print(f"=== XOR Training (PyTorch {mode.upper()}: MSE + Adam) ===\n")
    run(device)
