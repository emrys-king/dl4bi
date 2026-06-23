"""
plot_kde_from_saved.py
=======================
Regenerate the hyperparameter posterior KDE plot from already-saved
basic_inference.py output, without rerunning any NUTS sampling.

Useful when a run was killed before reaching its own automatic plotting
step (e.g. killed during the Exact GP phase, after SM-DeepRV's samples
were already saved to disk).

If egp_samples_means.npy / egp_samples_variances.npy exist in the config
directory, the Exact GP posterior is overlaid too. If not (e.g. Exact GP
never finished), the plot just shows SM-DeepRV vs the true value.

Usage
-----
    python plot_kde_from_saved.py \\
        results/inferbasic_20260620_212031_grid64/slow --config slow
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same base (Q=1) truth values as in simulation_100.py / basic_inference.py's
# SM_CONFIG_BASE. At Q=1, build_sm_config(name, 1) reduces exactly to these,
# so it's safe to hardcode them here for reconstructing the "true value" line.
SM_CONFIG_BASE = {
    "slow": {
        "means": [[0.003, 0.002]], "variances": [[8e-5, 8e-5]],
        "label": "SM slow (low-frequency)",
    },
    "medium": {
        "means": [[0.010, 0.008]], "variances": [[3e-4, 3e-4]],
        "label": "SM medium (mid-frequency)",
    },
    "fast": {
        "means": [[0.020, 0.015]], "variances": [[1e-3, 8e-4]],
        "label": "SM fast (high-frequency)",
    },
}


def plot_hyperparam_kde(cfg_dir, config_name, out_path=None):
    cfg_dir   = Path(cfg_dir)
    sm_config = SM_CONFIG_BASE[config_name]
    true_means = np.array(sm_config["means"])      # (Q, D)
    true_vars  = np.array(sm_config["variances"])   # (Q, D)
    Q, D = true_means.shape

    drv_samples = {
        "means":     np.load(cfg_dir / "drv_samples_means.npy"),
        "variances": np.load(cfg_dir / "drv_samples_variances.npy"),
    }

    egp_samples = None
    if (cfg_dir / "egp_samples_means.npy").exists():
        egp_samples = {
            "means":     np.load(cfg_dir / "egp_samples_means.npy"),
            "variances": np.load(cfg_dir / "egp_samples_variances.npy"),
        }
        print("Found Exact GP samples — overlaying both methods.")
    else:
        print("No Exact GP samples found (expected if that phase was "
              "killed before finishing) — plotting SM-DeepRV only.")

    n_cols = 2 * D
    n_rows = Q
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    colours = {"SM-DeepRV": "#2196F3", "Exact GP": "#FF5722"}

    for q in range(Q):
        for d in range(D):
            for j, (name, true_all) in enumerate([("means", true_means),
                                                   ("variances", true_vars)]):
                ax = axes[q][d + j * D]
                true_val = float(true_all[q, d])

                series = [("SM-DeepRV", drv_samples)]
                if egp_samples is not None:
                    series.append(("Exact GP", egp_samples))

                all_draws = []
                for label, samp in series:
                    draws = np.array(samp[name])[:, q, d]
                    all_draws.append(draws)

                lo = min(d_.min() for d_ in all_draws + [np.array([true_val])])
                hi = max(d_.max() for d_ in all_draws + [np.array([true_val])])
                pad = 0.1 * (hi - lo + 1e-8)
                xs  = np.linspace(lo - pad, hi + pad, 400)

                for (label, _), draws in zip(series, all_draws):
                    try:
                        kde = gaussian_kde(draws)
                        ax.plot(xs, kde(xs), color=colours[label],
                               linewidth=1.6, label=label)
                        ax.fill_between(xs, kde(xs), alpha=0.15,
                                        color=colours[label])
                    except np.linalg.LinAlgError:
                        # degenerate (near-zero variance) draws
                        ax.axvline(float(np.mean(draws)), color=colours[label],
                                  linewidth=1.6, label=label)

                ax.axvline(true_val, color="k", linestyle="--",
                          linewidth=1.5, label=f"True={true_val:.4g}")
                ax.set_title(f"{name}[{q},{d}]", fontsize=10)
                ax.set_xlabel("Value", fontsize=8)
                ax.set_ylabel("Density", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)

    plt.suptitle(f"Posterior KDE — SM hyperparameters ({sm_config['label']})",
                fontsize=12)
    plt.tight_layout()
    out_path = out_path or (cfg_dir / "hyperparam_kde_regenerated.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cfg_dir", type=str,
                   help="Path to the saved config directory, e.g. "
                        "results/inferbasic_.../slow")
    p.add_argument("--config", type=str, required=True,
                   choices=list(SM_CONFIG_BASE.keys()))
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG path (default: <cfg_dir>/hyperparam_kde_regenerated.png)")
    args = p.parse_args()
    plot_hyperparam_kde(args.cfg_dir, args.config, args.out)