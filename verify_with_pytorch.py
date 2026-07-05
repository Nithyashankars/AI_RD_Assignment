"""
verify_with_pytorch.py

This is not the main solution - estimate.py is. This script exists purely
as a second, independent check on the answer, using a completely different
optimization engine (gradient descent via PyTorch autograd instead of
scipy's least_squares/brentq).

The reasoning behind the model (rotation + translation of a simpler base
curve) is identical to estimate.py - only the tool used to search for the
parameters is different. If both scripts agree, that's good evidence the
answer isn't an artifact of one particular optimizer.

Run:
    python verify_with_pytorch.py
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/xy_data.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    x = torch.tensor(df["x"].values, dtype=torch.float64)
    y = torch.tensor(df["y"].values, dtype=torch.float64)
    return x, y


def bounded_params(raw_theta, raw_M, raw_X):
    """
    Squash unconstrained parameters into the assignment's allowed ranges
    with a sigmoid, the same way you'd bound a weight in a constrained
    neural net, so gradient descent can never wander outside the legal
    search space.
    """
    theta_deg = 50.0 * torch.sigmoid(raw_theta)
    M = -0.05 + 0.10 * torch.sigmoid(raw_M)
    X = 100.0 * torch.sigmoid(raw_X)
    return theta_deg, M, X


def loss_fn(raw_theta, raw_M, raw_X, x, y):
    theta_deg, M, X = bounded_params(raw_theta, raw_M, raw_X)
    theta = theta_deg * np.pi / 180.0
    c, s = torch.cos(theta), torch.sin(theta)
    xs, ys = x - X, y - 42.0
    u = xs * c + ys * s
    v = -xs * s + ys * c
    v_model = torch.exp(M * torch.abs(u)) * torch.sin(0.3 * u)
    residual = v - v_model
    return torch.mean(residual ** 2)


def main(n_steps=3000, lr=0.05, seed_theta=10.0):
    x, y = load_data()

    # deliberately start far from the expected answer to show it
    # still converges reliably
    raw_theta = torch.nn.Parameter(torch.tensor(seed_theta, dtype=torch.float64))
    raw_M = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    raw_X = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    optimizer = torch.optim.Adam([raw_theta, raw_M, raw_X], lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(raw_theta, raw_M, raw_X, x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 500 == 0:
            td, M_, X_ = bounded_params(raw_theta, raw_M, raw_X)
            print(f"step {step:5d}  loss={loss.item():.4e}  "
                  f"theta={td.item():.4f}  M={M_.item():.5f}  X={X_.item():.4f}")

    theta_deg, M, X = bounded_params(raw_theta, raw_M, raw_X)
    theta_deg, M, X = theta_deg.item(), M.item(), X.item()

    print("\n--- PyTorch cross-check result ---")
    print(f"theta = {theta_deg:.6f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("gradient descent step")
    plt.ylabel("loss (log scale)")
    plt.title("PyTorch training curve")
    plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=150)
    print("Saved results/training_curve.png")

    return theta_deg, M, X


if __name__ == "__main__":
    main()
