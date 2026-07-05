"""
estimate.py

Parameter estimation for the AI R&D assignment curve:

    x(t) = t*cos(theta) - e^(M|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M|t|) * sin(0.3t) * cos(theta)

Unknowns: theta (0-50 deg), M (-0.05 to 0.05), X (0-100), for t in [6, 60].
We only have unordered (x, y) samples in data/xy_data.csv - no t values,
no correspondence between rows and curve positions.

--------------------------------------------------------------------------
Method summary (full reasoning is in README.md, this is just the short
version so the code stands on its own):

1. The equations describe a rotation + translation of a simpler curve
   (t, r(t)) where r(t) = e^(M|t|)*sin(0.3t). That means for the correct
   theta, if we rotate the data BACKWARDS by theta, we recover the exact
   t and r(t) values that generated every point - no correspondence
   search needed.

2. Instead of blindly searching all three unknowns at once, we use a
   fact given in the assignment itself: t always spans exactly
   60 - 6 = 54 units. Rotating the raw data by a candidate theta and
   checking whether the resulting spread equals 54 turns the search for
   theta into a 1-D root-finding problem instead of a 3-D one.

3. This 1-D check can have more than one root (a wrong angle can
   sometimes fake the right spread), so every candidate is scored by how
   well it satisfies the full model, and the best one is kept.

4. The winning candidate is refined with a standard local least-squares
   solve to squeeze out the last bit of numerical error.

Run:
    python estimate.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares

DATA_PATH = "data/xy_data.csv"
RESULTS_DIR = "results"

THETA_BOUNDS = (0.0, 50.0)   # degrees
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.0, 100.0)
T_LOW, T_HIGH = 6.0, 60.0
T_SPAN = T_HIGH - T_LOW      # = 54, this is the fact we exploit


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df["x"].values, df["y"].values


def span_error(theta_deg, x, y):
    """
    Rotate the raw data by -theta_deg (ignoring X, which only shifts
    values and doesn't change their spread) and compare the resulting
    spread to the known span of t, which is exactly 54.
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    u_raw = x * c + (y - 42.0) * s
    return (u_raw.max() - u_raw.min()) - T_SPAN


def find_theta_candidates(x, y, n_scan=5000):
    """
    Scan theta across its allowed range and return every angle where
    span_error crosses zero. There is usually more than one crossing.
    """
    thetas = np.linspace(THETA_BOUNDS[0] + 0.01, THETA_BOUNDS[1] - 0.01, n_scan)
    errs = np.array([span_error(t, x, y) for t in thetas])
    crossings = np.where(np.diff(np.sign(errs)) != 0)[0]

    roots = []
    for idx in crossings:
        root = brentq(span_error, thetas[idx], thetas[idx + 1], args=(x, y))
        roots.append(root)
    return roots


def score_candidate(theta_deg, x, y):
    """
    Given a candidate theta, solve for X directly (assuming the data
    reaches down to t=6 at its minimum, which holds for a dense enough
    sample), fit M with a quick 1-parameter least squares, and report
    how well the full model matches - this is the score used to pick
    the correct root when there's more than one.
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)

    u_raw = x * c + (y - 42.0) * s
    X_guess = (u_raw.min() - T_LOW) / c
    v_raw = -x * s + (y - 42.0) * c

    u = u_raw - X_guess * c
    v = v_raw + X_guess * s

    def resid_M(params):
        (M,) = params
        return v - np.exp(M * np.abs(u)) * np.sin(0.3 * u)

    res = least_squares(resid_M, x0=[0.0], bounds=([M_BOUNDS[0]], [M_BOUNDS[1]]))
    M_guess = res.x[0]
    total_residual = float(np.sum(res.fun ** 2))
    return X_guess, M_guess, total_residual


def full_residuals(params, x, y):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    xs, ys = x - X, y - 42.0
    u = xs * c + ys * s
    v = -xs * s + ys * c
    v_model = np.exp(M * np.abs(u)) * np.sin(0.3 * u)
    return v - v_model


def estimate_parameters(x, y, verbose=True):
    roots = find_theta_candidates(x, y)
    if verbose:
        print(f"Found {len(roots)} candidate angle(s) satisfying the span constraint:")

    best = None
    for r in roots:
        X_g, M_g, resid = score_candidate(r, x, y)
        if verbose:
            print(f"  theta={r:8.4f}  X={X_g:8.4f}  M={M_g:8.5f}  residual={resid:.3e}")
        if best is None or resid < best[3]:
            best = (r, X_g, M_g, resid)

    theta0, X0, M0, _ = best

    result = least_squares(
        full_residuals,
        x0=[theta0, M0, X0],
        args=(x, y),
        bounds=([THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]],
                [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]),
        xtol=1e-15, ftol=1e-15, gtol=1e-15,
    )
    theta_deg, M, X = result.x
    max_residual = float(np.max(np.abs(result.fun)))
    return theta_deg, M, X, max_residual


def l1_score(theta_deg, M, X, x, y, n_curve_samples=3000):
    """
    Assessment-style score: uniformly sample the fitted curve and compute
    the L1 distance from each observed point to the nearest sampled point
    on the curve, matching the metric described in the assignment.
    """
    theta = np.deg2rad(theta_deg)
    t_dense = np.linspace(T_LOW, T_HIGH, n_curve_samples)
    r_dense = np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    x_curve = t_dense * np.cos(theta) - r_dense * np.sin(theta) + X
    y_curve = 42.0 + t_dense * np.sin(theta) + r_dense * np.cos(theta)

    curve_pts = np.column_stack([x_curve, y_curve])
    data_pts = np.column_stack([x, y])

    # nearest neighbour in L1, done in chunks to keep memory sane
    total = 0.0
    for i in range(0, len(data_pts), 200):
        chunk = data_pts[i:i + 200]
        d = np.abs(chunk[:, None, :] - curve_pts[None, :, :]).sum(axis=2)
        total += d.min(axis=1).sum()
    return total


def make_plot(theta_deg, M, X, x, y, out_path):
    theta = np.deg2rad(theta_deg)
    t_dense = np.linspace(T_LOW, T_HIGH, 3000)
    r_dense = np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    x_curve = t_dense * np.cos(theta) - r_dense * np.sin(theta) + X
    y_curve = 42.0 + t_dense * np.sin(theta) + r_dense * np.cos(theta)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=6, alpha=0.4, color="tab:blue", label="observed data")
    plt.plot(x_curve, y_curve, color="tab:red", lw=1.5, label="fitted curve")
    plt.title(f"theta={theta_deg:.4f} deg, M={M:.5f}, X={X:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    x, y = load_data()

    t0 = time.time()
    theta_deg, M, X, max_residual = estimate_parameters(x, y)
    elapsed = time.time() - t0

    l1 = l1_score(theta_deg, M, X, x, y)

    print("\n--- Final estimate ---")
    print(f"theta = {theta_deg:.6f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"max algebraic residual = {max_residual:.3e}")
    print(f"L1 score (assignment metric) = {l1:.4f}")
    print(f"total time = {elapsed:.4f} s")

    theta_rad = np.deg2rad(theta_deg)
    latex = (
        r"\left(t\cos\left(" + f"{theta_rad:.10f}" + r"\right)-e^{" + f"{M:.6f}" +
        r"\left|t\right|}\cdot\sin\left(0.3t\right)\sin\left(" + f"{theta_rad:.10f}" +
        r"\right)+" + f"{X:.6f}" + r"," +
        f"{42}" + r"+t\sin\left(" + f"{theta_rad:.10f}" + r"\right)+e^{" + f"{M:.6f}" +
        r"\left|t\right|}\cdot\sin\left(0.3t\right)\cos\left(" + f"{theta_rad:.10f}" +
        r"\right)\right)"
    )

    make_plot(theta_deg, M, X, x, y, f"{RESULTS_DIR}/plot.png")

    with open(f"{RESULTS_DIR}/result.txt", "w") as f:
        f.write("AI R&D Assignment - Parameter Estimation Result\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"theta (degrees) : {theta_deg:.6f}\n")
        f.write(f"theta (radians) : {theta_rad:.10f}\n")
        f.write(f"M               : {M:.6f}\n")
        f.write(f"X               : {X:.6f}\n\n")
        f.write(f"max algebraic residual : {max_residual:.6e}\n")
        f.write(f"L1 score (assignment metric, {3000} curve samples) : {l1:.6f}\n")
        f.write(f"runtime : {elapsed:.4f} s\n\n")
        f.write("Desmos-ready LaTeX:\n")
        f.write(latex + "\n")

    print(f"\nSaved plot to {RESULTS_DIR}/plot.png")
    print(f"Saved result.txt to {RESULTS_DIR}/result.txt")
    print("\nDesmos LaTeX:")
    print(latex)


if __name__ == "__main__":
    main()
