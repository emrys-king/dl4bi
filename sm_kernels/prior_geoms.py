"""
Posterior Scatter Plots — Betancourt Style
==========================================
Reads saved MCMC samples and divergence flags from a results directory
and produces pairwise scatter plots of posterior hyperparameters in
log-space, with divergent transitions highlighted in green.

Mirrors the diagnostic figures in Betancourt's GP case study (Section 3),
adapted for the SM kernel parameterisation used in simulation.py:
  - log(ells[q,p])   — log lengthscale per component / dimension
  - log(means[q,p])  — log frequency per component / dimension
  - log(sigma)       — log noise standard deviation
  - log(raw_w[q])    — log unnormalised mixture weight

Each panel plots two parameters against each other.  Dark red points are
non-divergent samples; bright green points are divergent transitions.
Divergences appearing in a particular region of parameter space indicate
that the posterior geometry is pathological there.

Usage
-----
    # Plot SM-DeepRV diagnostics for the 'slow' config
    python posterior_scatter.py --results_dir results/sim_YYYYMMDD_HHMMSS/slow

    # Plot Exact GP diagnostics
    python posterior_scatter.py --results_dir results/.../slow --method egp

    # Custom parameter pairs
    python posterior_scatter.py --results_dir results/.../slow \\
        --pairs "log_ells_0_0,log_sigma" "log_ells_0_1,log_means_0_0"

    # Save to a specific file
    python posterior_scatter.py --results_dir results/.../slow \\
        --out my_scatter.pdf

Dependencies
------------
    pip install numpy matplotlib
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Colours matching Betancourt's palette
# ---------------------------------------------------------------------------
C_NORMAL   = "#8B1A1A"    # dark red — non-divergent
C_DIVERGE  = "#00C853"    # bright green — divergent
ALPHA_NORM = 0.25
ALPHA_DIV  = 0.90
SIZE_NORM  = 8
SIZE_DIV   = 18


# ---------------------------------------------------------------------------
# Load samples from results directory
# ---------------------------------------------------------------------------

def load_samples(results_dir: Path, method: str) -> dict:
    """Load all saved sample arrays for a given method prefix (drv or egp).

    Returns a dict mapping short parameter names to 1-D numpy arrays
    (one value per posterior sample), plus a boolean 'diverging' array.

    Files expected (written by simulation.py):
        {method}_samples_ells.npy       shape (S, Q, D)
        {method}_samples_means.npy      shape (S, Q, D)
        {method}_samples_raw_w.npy      shape (S, Q)
        {method}_samples_log_sigma.npy  shape (S,)   [or sigma if HalfNormal]
        {method}_energy.npy             shape (S,)
        {method}_diverging.npy          shape (S,)   bool
    """
    prefix  = results_dir / method
    samples = {}

    def try_load(path):
        p = Path(str(path) + ".npy")
        if p.exists():
            return np.array(np.load(p))
        return None

    # Divergences
    div = try_load(f"{prefix}_diverging")
    samples["diverging"] = div.astype(bool).ravel() if div is not None \
                           else None

    # Energy
    energy = try_load(f"{prefix}_energy")
    if energy is not None:
        samples["energy"] = energy.ravel()

    # ells — shape (S, Q, D) → flatten to (S, Q*D) with named keys
    ells = try_load(f"{prefix}_samples_ells")
    if ells is not None:
        S, Q, D = ells.shape
        for q in range(Q):
            for d in range(D):
                samples[f"log_ells_{q}_{d}"] = np.log(ells[:, q, d] + 1e-12)
    else:
        print(f"  [warn] {prefix}_samples_ells.npy not found — "
              f"run simulation.py with the patched version first.")

    # means — shape (S, Q, D)
    means = try_load(f"{prefix}_samples_means")
    if means is not None:
        S, Q, D = means.shape
        for q in range(Q):
            for d in range(D):
                samples[f"log_means_{q}_{d}"] = np.log(means[:, q, d] + 1e-12)

    # raw_w — shape (S, Q)
    raw_w = try_load(f"{prefix}_samples_raw_w")
    if raw_w is not None:
        S, Q = raw_w.shape
        for q in range(Q):
            samples[f"log_raw_w_{q}"] = np.log(raw_w[:, q] + 1e-12)

    # log_sigma (log-parameterised) or sigma (HalfNormal)
    log_sigma = try_load(f"{prefix}_samples_log_sigma")
    if log_sigma is not None:
        samples["log_sigma"] = log_sigma.ravel()
    else:
        sigma = try_load(f"{prefix}_samples_sigma")
        if sigma is not None:
            samples["log_sigma"] = np.log(sigma.ravel() + 1e-12)

    return samples


# ---------------------------------------------------------------------------
# Auto-select informative parameter pairs
# ---------------------------------------------------------------------------

def default_pairs(samples: dict, max_pairs: int = 9):
    """Return a list of (x_key, y_key) pairs to plot.

    Priority:
      1. All pairs within {log_ells, log_sigma}  — most diagnostic for prior
      2. log_ells vs log_means for component 0   — frequency/lengthscale joint
      3. log_sigma vs log_ells component 0       — noise/lengthscale joint
    """
    ell_keys   = sorted(k for k in samples if k.startswith("log_ells"))
    mean_keys  = sorted(k for k in samples if k.startswith("log_means"))
    w_keys     = sorted(k for k in samples if k.startswith("log_raw_w"))
    sigma_keys = [k for k in samples if k == "log_sigma"]

    pairs = []

    # All pairwise within ells + sigma (most Betancourt-like)
    core = ell_keys[:4] + sigma_keys   # cap ells at 4 for readability
    for a, b in itertools.combinations(core, 2):
        pairs.append((a, b))
        if len(pairs) >= max_pairs:
            return pairs[:max_pairs]

    # ells vs means, component 0
    for ek, mk in zip(ell_keys[:2], mean_keys[:2]):
        pairs.append((ek, mk))
        if len(pairs) >= max_pairs:
            return pairs[:max_pairs]

    # log_sigma vs raw_w
    for sk in sigma_keys:
        for wk in w_keys[:2]:
            pairs.append((wk, sk))
            if len(pairs) >= max_pairs:
                return pairs[:max_pairs]

    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Axis labels — readable human names
# ---------------------------------------------------------------------------

def axis_label(key: str) -> str:
    """Convert internal key to a readable axis label."""
    m = re.match(r"log_ells_(\d+)_(\d+)", key)
    if m:
        return f"log ℓ[{m.group(1)},{m.group(2)}]"
    m = re.match(r"log_means_(\d+)_(\d+)", key)
    if m:
        return f"log μ[{m.group(1)},{m.group(2)}]"
    m = re.match(r"log_raw_w_(\d+)", key)
    if m:
        return f"log w̃[{m.group(1)}]"
    if key == "log_sigma":
        return "log σ"
    return key


# ---------------------------------------------------------------------------
# E-BFMI
# ---------------------------------------------------------------------------

def compute_ebfmi(samples: dict) -> float | None:
    if "energy" not in samples:
        return None
    H = samples["energy"]
    return float(np.mean(np.diff(H)**2) / (np.var(H) + 1e-12))


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def make_scatter_figure(samples: dict, pairs: list,
                        title: str = "") -> plt.Figure:
    """
    Produce a grid of pairwise scatter plots.

    samples  : dict from load_samples()
    pairs    : list of (x_key, y_key) tuples
    title    : figure suptitle
    """
    n      = len(pairs)
    ncols  = min(3, n)
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    div = samples.get("diverging")
    if div is None:
        div = np.zeros(next(
            len(v) for k, v in samples.items()
            if k not in ("diverging", "energy")
        ), dtype=bool)

    n_div  = int(div.sum())
    n_total = len(div)

    for ax, (xk, yk) in zip(axes_flat, pairs):
        if xk not in samples or yk not in samples:
            ax.set_visible(False)
            continue

        x = samples[xk]
        y = samples[yk]

        # Non-divergent
        mask_ok  = ~div
        ax.scatter(x[mask_ok],  y[mask_ok],
                   c=C_NORMAL,  alpha=ALPHA_NORM,
                   s=SIZE_NORM, linewidths=0, rasterized=True)

        # Divergent — plotted on top
        if n_div > 0:
            ax.scatter(x[div], y[div],
                       c=C_DIVERGE, alpha=ALPHA_DIV,
                       s=SIZE_DIV,  linewidths=0.3,
                       edgecolors="white", zorder=5)

        ax.set_xlabel(axis_label(xk), fontsize=9)
        ax.set_ylabel(axis_label(yk), fontsize=9)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.5")
        ax.set_facecolor("white")

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=C_NORMAL, markersize=7, alpha=0.7,
               label=f"Non-divergent  (n={n_total - n_div:,})"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=C_DIVERGE, markersize=9,
               label=f"Divergent  (n={n_div:,})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.01))

    # E-BFMI in title
    ebfmi = compute_ebfmi(samples)
    ebfmi_str = f"   |   E-BFMI = {ebfmi:.3f}" if ebfmi is not None else ""
    if ebfmi is not None and ebfmi < 0.3:
        ebfmi_str += "  ⚠ < 0.3"

    fig.suptitle(
        f"{title}\n"
        f"Divergences: {n_div}/{n_total} "
        f"({100*n_div/max(n_total,1):.1f}%){ebfmi_str}",
        fontsize=11, y=0.995
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Betancourt-style posterior scatter plots from saved samples"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to config-level results directory "
                             "(e.g. results/sim_YYYYMMDD/slow)")
    parser.add_argument("--method", type=str, default="drv",
                        choices=["drv", "egp"],
                        help="Which method's samples to plot "
                             "(drv=SM-DeepRV, egp=Exact GP; default: drv)")
    parser.add_argument("--pairs", type=str, nargs="+", default=None,
                        help="Custom parameter pairs as 'x_key,y_key' strings. "
                             "Keys: log_ells_Q_D, log_means_Q_D, "
                             "log_raw_w_Q, log_sigma. "
                             "Default: auto-selected most diagnostic pairs.")
    parser.add_argument("--max_pairs", type=int, default=9,
                        help="Maximum number of panels (default: 9)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: posterior_scatter_METHOD.pdf "
                             "in results_dir)")
    parser.add_argument("--show", action="store_true",
                        help="Display figure instead of saving")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Loading {args.method.upper()} samples from {results_dir} ...")
    samples = load_samples(results_dir, args.method)

    available = [k for k in samples
                 if k not in ("diverging", "energy")]
    print(f"Available parameters: {available}")

    div = samples.get("diverging")
    n_div = int(div.sum()) if div is not None else "unknown"
    print(f"Divergences: {n_div}")

    ebfmi = compute_ebfmi(samples)
    if ebfmi is not None:
        flag = "  ⚠ < 0.3 — may indicate restricted energy exploration" \
               if ebfmi < 0.3 else "  ✓"
        print(f"E-BFMI: {ebfmi:.3f}{flag}")
    else:
        print("E-BFMI: not available (energy field not saved)")

    # Parse custom pairs if given
    if args.pairs:
        pairs = []
        for p in args.pairs:
            parts = p.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid pair '{p}' — expected 'x_key,y_key'"
                )
            pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        pairs = default_pairs(samples, max_pairs=args.max_pairs)

    print(f"Plotting {len(pairs)} parameter pairs ...")

    method_label = {"drv": "SM-DeepRV", "egp": "Exact GP"}[args.method]
    config_name  = results_dir.name
    title        = f"{method_label}  —  config: {config_name}"

    fig = make_scatter_figure(samples, pairs, title=title)

    if args.show:
        plt.show()
    else:
        out = args.out or str(
            results_dir / f"posterior_scatter_{args.method}.pdf"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()