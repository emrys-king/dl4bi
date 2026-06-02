"""
SM-DeepRV Simulation Experiment
================================
Evaluates SM-DeepRV using the same experimental structure as the DeepRV paper
(Section 6.1 / Appendix B.1), with a Spectral Mixture kernel and a Gaussian
likelihood appropriate for continuous air quality measurements.

Compares two methods:
  1. SM-DeepRV (NUTS)  — surrogate-accelerated inference (this work)
  2. Exact GP  (NUTS)  — full O(N³) GP inference with SM kernel (gold standard)

Likelihood:
  y_obs ~ N(f[obs_idx], σ²),  σ ~ HalfNormal(1.0)

Structural elements matched to the DeepRV paper:
  - ~50% of grid masked in a contiguous rectangular region
  - NUTS: 4000 warmup, 6000 posterior draws
  - 2 chains for grid ≤ 32², 1 chain for larger grids
  - Training: 200K steps (≤32²) or 300K steps (≥48²)
  - AdamW + cosine LR: 1e-3 (≤32²) or 2e-3 (≥48²), gradient clip ≤ 3

The three SM configurations sweep from slow/smooth to fast/rough spatial
variation, analogous to the paper's l ∈ {10, 30, 50} sweep.

Usage
-----
    # Full run — all three SM configurations
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 16

    # Single SM configuration
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 16 --sm_config medium

    # Quick sanity check
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 8 --sm_config medium \\
        --train_steps 5000 --num_warmup 100 --num_samples 200

    # Skip exact GP baseline (much faster, for development)
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 16 --no_exact_gp
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
from jax import jit, vmap, random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import optax
import orbax.checkpoint as ocp
import numpy as np
import pandas as pd
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dl4bi_sps.utils import build_grid
from dl4bi.core.model_output import VAEOutput
from dl4bi.train import Callback, TrainState, train, cosine_annealing_lr
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

MASK_FRACTION = 0.5
BATCH_SIZE    = 32
JITTER        = 5e-4

SM_CONFIGS = {
    "slow": {
        "weights":   [0.5, 0.3, 0.2],
        "means":     [[0.5, 0.4], [1.0, 0.8], [1.5, 1.2]],
        "variances": [[0.5, 0.5], [0.8, 0.6], [1.0, 0.8]],
        "label":     "SM slow (low-frequency)",
    },
    "medium": {
        "weights":   [0.5, 0.3, 0.2],
        "means":     [[1.0, 0.8], [2.5, 1.5], [4.0, 3.0]],
        "variances": [[0.5, 0.5], [1.0, 0.8], [1.5, 1.2]],
        "label":     "SM medium (mid-frequency)",
    },
    "fast": {
        "weights":   [0.5, 0.3, 0.2],
        "means":     [[2.0, 1.5], [5.0, 3.5], [8.0, 6.0]],
        "variances": [[0.8, 0.6], [1.2, 1.0], [2.0, 1.5]],
        "label":     "SM fast (high-frequency)",
    },
}

# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SM-DeepRV simulation vs exact GP baseline"
    )
    p.add_argument("--grid_size",    type=int,   default=16)
    p.add_argument("--sm_config",    type=str,   default=None,
                   choices=["slow", "medium", "fast"])
    p.add_argument("--train_steps",  type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--num_warmup",   type=int,   default=4000)
    p.add_argument("--num_samples",  type=int,   default=6000)
    p.add_argument("--q",            type=int,   default=3)
    p.add_argument("--noise_std",    type=float, default=0.05)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--results_dir",  type=str,   default="results")
    p.add_argument("--ckpt_dir",     type=str,   default=None)
    p.add_argument("--no_exact_gp",  action="store_true",
                   help="Skip the exact GP NUTS baseline (faster development runs)")

    # Training mode control
    train_mode = p.add_mutually_exclusive_group()
    train_mode.add_argument("--fresh_run", action="store_true",
                   help="Delete existing checkpoints and train from scratch. "
                        "Use when you want a clean run with new hyperparameters.")
    train_mode.add_argument("--load_ckpt", type=str, default=None,
                   metavar="CKPT_DIR",
                   help="Load a trained network from this checkpoint directory "
                        "and skip training entirely. Use when the surrogate is "
                        "already trained and you only want to run inference. "
                        "Example: --load_ckpt results/sim_checkpoints_grid16_q3")
    return p.parse_args()


def paper_train_steps(grid_size, override=None):
    return override if override else (200_000 if grid_size <= 32 else 300_000)

def paper_lr(grid_size, override=None):
    return override if override else (1e-3 if grid_size <= 32 else 2e-3)

def paper_num_chains(grid_size):
    return 2 if grid_size <= 32 else 1


# =============================================================================
# Spectral mixture kernel
# =============================================================================

@jit
def spectral_mixture(x, y, weights, means, variances):
    x    = x.reshape(-1, x.shape[-1])
    y    = y.reshape(-1, y.shape[-1])
    diff = x[:, None, :] - y[None, :, :]

    def component(w, mu, v):
        env  = jnp.exp(-2 * jnp.pi**2 * jnp.sum(v * diff**2, axis=-1))
        freq = jnp.cos(2 * jnp.pi * jnp.sum(mu * diff, axis=-1))
        return w * env * freq

    return jnp.sum(vmap(component)(weights, means, variances), axis=0)


# =============================================================================
# SM dataloader
# =============================================================================

def sm_dataloader(s, Q, batch_size=BATCH_SIZE):
    N, D   = s.shape
    jitter = JITTER * jnp.eye(N)

    def dataloader(rng):
        while True:
            rng, rng_w, rng_mu, rng_v, rng_z = random.split(rng, 5)
            raw     = random.gamma(rng_w, a=1.0, shape=(Q,))
            weights = raw / raw.sum()
            means   = jnp.exp(random.normal(rng_mu, shape=(Q, D)))
            vars_   = 1.0 / random.gamma(rng_v, a=2.0, shape=(Q, D))
            K = spectral_mixture(s, s, weights, means, vars_) + jitter
            L = jnp.linalg.cholesky(K)
            z = random.normal(rng_z, shape=(batch_size, N))
            f = jnp.einsum("ij,bj->bi", L, z)
            conditionals = jnp.concatenate(
                [weights, means.ravel(), vars_.ravel()]
            )
            yield {"s": s, "z": z, "conditionals": conditionals, "f": f}

    return dataloader


# =============================================================================
# Contiguous masking
# =============================================================================

def contiguous_mask(grid_size, rng_key=None):
    N        = grid_size ** 2
    n_masked = int(N * MASK_FRACTION)
    rect_h   = min(int(np.sqrt(n_masked)), grid_size)
    rect_w   = min(int(np.ceil(n_masked / rect_h)), grid_size)

    seed   = int(jax.random.bits(rng_key)) if rng_key is not None else 0
    rng_np = np.random.default_rng(seed)
    row0   = rng_np.integers(0, grid_size - rect_h + 1)
    col0   = rng_np.integers(0, grid_size - rect_w + 1)

    mask            = np.zeros((grid_size, grid_size), dtype=bool)
    mask[row0:row0 + rect_h, col0:col0 + rect_w] = True
    flat_mask       = mask.ravel()
    tst_idx         = jnp.array(np.where( flat_mask)[0])
    obs_idx         = jnp.array(np.where(~flat_mask)[0])

    log.info(f"Mask: {rect_h}×{rect_w} at ({row0},{col0}) — "
             f"{len(tst_idx)}/{N} held out ({100*len(tst_idx)/N:.1f}%)")
    return obs_idx, tst_idx


# =============================================================================
# Synthetic data generation
# =============================================================================

def generate_observations(s_all, grid_size, sm_config, noise_std, rng):
    """Sample GP from true SM kernel, add Gaussian noise, standardise."""
    N_all = s_all.shape[0]
    rng, rng_f, rng_y, rng_mask = random.split(rng, 4)

    weights   = jnp.array(sm_config["weights"])
    means     = jnp.array(sm_config["means"])
    variances = jnp.array(sm_config["variances"])

    K     = spectral_mixture(s_all, s_all, weights, means, variances) \
            + JITTER * jnp.eye(N_all)
    L     = jnp.linalg.cholesky(K)
    f_all = L @ random.normal(rng_f, shape=(N_all,))

    if jnp.any(jnp.isnan(f_all)):
        raise ValueError("f_all contains NaN — kernel near-singular.")

    obs_idx, tst_idx = contiguous_mask(grid_size, rng_key=rng_mask)
    y_obs  = f_all[obs_idx] + noise_std * random.normal(rng_y, shape=(len(obs_idx),))

    y_mean     = float(jnp.mean(y_obs))
    y_std      = float(jnp.std(y_obs))  + 1e-6
    y_obs_norm = (y_obs - y_mean) / y_std

    true_cond = jnp.concatenate([weights, means.ravel(), variances.ravel()])
    return y_obs_norm, y_mean, y_std, f_all, true_cond, obs_idx, tst_idx


# =============================================================================
# SM-DeepRV NUTS inference  (surrogate likelihood)
# =============================================================================

def deeprv_nuts_inference(decoder, y_obs_norm, y_mean, y_std,
                          s_all, obs_idx, tst_idx,
                          Q, num_warmup, num_samples, num_chains, rng):
    """NUTS over z and SM hyperparameters using the surrogate decoder."""
    N_all, D = s_all.shape

    def model():
        raw     = numpyro.sample("raw_w", dist.Gamma(jnp.ones(Q), jnp.ones(Q)))
        weights = raw / raw.sum()
        numpyro.deterministic("weights", weights)
        means   = numpyro.sample("means",
                      dist.LogNormal(jnp.zeros((Q, D)), jnp.ones((Q, D))))
        vars_   = numpyro.sample("variances",
                      dist.InverseGamma(2.0 * jnp.ones((Q, D)),
                                        jnp.ones((Q, D))))
        cond    = jnp.concatenate([weights, means.ravel(), vars_.ravel()])
        z       = numpyro.sample("z",
                      dist.Normal(jnp.zeros(N_all), jnp.ones(N_all)))
        f       = decoder(z[None], cond, s=s_all).squeeze()
        sigma   = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y_obs_norm)
        numpyro.deterministic("f_all", f)

    t0      = time.time()
    rng_key = jax.random.fold_in(rng, 0)
    mcmc    = MCMC(NUTS(model, target_accept_prob=0.8),
                   num_warmup=num_warmup, num_samples=num_samples,
                   num_chains=num_chains,
                   chain_method="parallel" if num_chains > 1 else "sequential",
                   progress_bar=True)
    mcmc.run(rng_key)
    elapsed = time.time() - t0
    log.info(f"SM-DeepRV NUTS: {elapsed:.1f}s")
    mcmc.print_summary()

    samples = mcmc.get_samples()
    f_samps = samples["f_all"][:, tst_idx] * y_std + y_mean
    return f_samps.mean(0), f_samps.std(0), samples, elapsed


# =============================================================================
# Exact GP NUTS inference  (O(N³) per step — gold standard)
# =============================================================================

def exact_gp_nuts_inference(y_obs_norm, y_mean, y_std,
                             s_obs, s_test,
                             Q, num_warmup, num_samples, num_chains, rng):
    """NUTS with exact SM GP likelihood — the gold-standard comparison.

    Uses the MultivariateNormal likelihood with the full Cholesky at each step,
    scaling as O(N_obs³). Predictions at test locations use the standard GP
    conditional formulae applied to posterior samples of the hyperparameters.
    """
    N_obs, D = s_obs.shape
    N_test   = s_test.shape[0]

    def model():
        raw     = numpyro.sample("raw_w", dist.Gamma(jnp.ones(Q), jnp.ones(Q)))
        weights = raw / raw.sum()
        numpyro.deterministic("weights", weights)
        means   = numpyro.sample("means",
                      dist.LogNormal(jnp.zeros((Q, D)), jnp.ones((Q, D))))
        vars_   = numpyro.sample("variances",
                      dist.InverseGamma(2.0 * jnp.ones((Q, D)),
                                        jnp.ones((Q, D))))
        sigma   = numpyro.sample("sigma", dist.HalfNormal(1.0))

        K_obs = spectral_mixture(s_obs, s_obs, weights, means, vars_) \
                + (sigma**2 + JITTER) * jnp.eye(N_obs)
        numpyro.sample("y",
                       dist.MultivariateNormal(jnp.zeros(N_obs), K_obs),
                       obs=y_obs_norm)
        # Store hyperparameters for post-hoc predictive
        numpyro.deterministic("means_det",     means)
        numpyro.deterministic("variances_det", vars_)
        numpyro.deterministic("sigma_det",     sigma)

    t0      = time.time()
    rng_key = jax.random.fold_in(rng, 1)
    mcmc    = MCMC(NUTS(model, target_accept_prob=0.8),
                   num_warmup=num_warmup, num_samples=num_samples,
                   num_chains=num_chains,
                   chain_method="parallel" if num_chains > 1 else "sequential",
                   progress_bar=True)
    mcmc.run(rng_key)
    elapsed = time.time() - t0
    log.info(f"Exact GP NUTS: {elapsed:.1f}s")
    mcmc.print_summary()

    samples  = mcmc.get_samples()
    weights_s= samples["weights"]           # [S, Q]
    means_s  = samples["means_det"]         # [S, Q, D]
    vars_s   = samples["variances_det"]     # [S, Q, D]
    sigma_s  = samples["sigma_det"]         # [S]

    # GP predictive for each posterior sample
    def predict_one(w, mu, v, sig):
        K_obs  = spectral_mixture(s_obs, s_obs, w, mu, v) \
                 + (sig**2 + JITTER) * jnp.eye(N_obs)
        K_star = spectral_mixture(s_test, s_obs, w, mu, v)
        K_ss   = jnp.diag(spectral_mixture(s_test, s_test, w, mu, v))
        alpha  = jnp.linalg.solve(K_obs, y_obs_norm)
        mu_pred= K_star @ alpha
        L      = jnp.linalg.cholesky(K_obs)
        v_pred = K_ss - jnp.sum(jnp.linalg.solve(L, K_star.T)**2, axis=0)
        return mu_pred, jnp.sqrt(jnp.clip(v_pred, 0.0))

    mu_preds, std_preds = vmap(predict_one)(
        weights_s, means_s, vars_s, sigma_s
    )   # [S, N_test]

    # Rescale to original units
    mu_preds  = mu_preds  * y_std + y_mean
    std_preds = std_preds * y_std

    return mu_preds.mean(0), std_preds.mean(0), samples, elapsed


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(mean_pred, std_pred, f_true):
    errors   = mean_pred - f_true
    rmse     = float(jnp.sqrt(jnp.mean(errors**2)))
    mae      = float(jnp.mean(jnp.abs(errors)))
    z_scores = jnp.abs(errors) / (std_pred + 1e-8)
    coverage = float(jnp.mean(z_scores < 1.96))
    mean_std = float(jnp.mean(std_pred))
    return {"RMSE": rmse, "MAE": mae,
            "Coverage_95": coverage, "Mean_std": mean_std}


# =============================================================================
# Plotting
# =============================================================================

def plot_spatial(s_all, f_all, y_obs_norm, y_mean, y_std,
                 obs_idx, tst_idx,
                 drv_mean, drv_std,
                 egp_mean, egp_std,
                 config_label, grid_size, out_dir, tag):
    """Two-row spatial plot.

    Row 1: True f | Observations | SM-DeepRV mean | SM-DeepRV std
    Row 2: Error (DeepRV) | Error (Exact GP) | Exact GP mean | Exact GP std
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    def scatter(ax, s, vals, title, cmap="RdYlBu_r", vmin=None, vmax=None, s_=14):
        sc = ax.scatter(s[:, 0], s[:, 1], c=np.array(vals),
                        cmap=cmap, vmin=vmin, vmax=vmax, s=s_)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        return sc

    s_obs  = s_all[obs_idx]
    s_test = s_all[tst_idx]
    f_test = f_all[tst_idx]
    fv     = (float(f_all.min()), float(f_all.max()))
    ev     = float(max(np.abs(drv_mean - f_test).max(),
                        np.abs(egp_mean - f_test).max())) if egp_mean is not None \
             else float(np.abs(drv_mean - f_test).max())
    sv     = float(max(drv_std.max(),
                        egp_std.max() if egp_std is not None else 0))

    # Row 1
    scatter(axes[0,0], s_all, f_all, f"True f  ({config_label})",
            vmin=fv[0], vmax=fv[1])
    axes[0,0].scatter(s_obs[:,0], s_obs[:,1], c="k", s=5,
                      alpha=0.3, label="Observed")
    axes[0,0].legend(fontsize=7)

    y_orig = np.array(y_obs_norm) * y_std + y_mean
    scatter(axes[0,1], s_obs, y_orig, "Observed y",
            vmin=fv[0], vmax=fv[1])

    scatter(axes[0,2], s_test, drv_mean, "SM-DeepRV  E[f|y]",
            vmin=fv[0], vmax=fv[1])

    scatter(axes[0,3], s_test, drv_std, "SM-DeepRV  Std[f|y]",
            cmap="Blues", vmin=0, vmax=sv)

    # Row 2
    drv_err = np.array(drv_mean) - np.array(f_test)
    scatter(axes[1,0], s_test, drv_err, "SM-DeepRV error",
            cmap="RdBu_r", vmin=-ev, vmax=ev)

    if egp_mean is not None:
        egp_err = np.array(egp_mean) - np.array(f_test)
        scatter(axes[1,1], s_test, egp_err, "Exact GP error",
                cmap="RdBu_r", vmin=-ev, vmax=ev)
        scatter(axes[1,2], s_test, egp_mean, "Exact GP  E[f|y]",
                vmin=fv[0], vmax=fv[1])
        scatter(axes[1,3], s_test, egp_std, "Exact GP  Std[f|y]",
                cmap="Blues", vmin=0, vmax=sv)
    else:
        for ax in axes[1, 1:]:
            ax.set_visible(False)
        axes[1,1].text(0.5, 0.5, "Exact GP\nnot run",
                       ha="center", va="center", transform=axes[1,1].transAxes)

    plt.suptitle(
        f"SM-DeepRV vs Exact GP — {config_label}  |  grid={grid_size}²",
        fontsize=13,
    )
    plt.tight_layout()
    path = Path(out_dir) / f"spatial_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Spatial plot saved → {path}")


def plot_metrics_comparison(drv_metrics, egp_metrics, config_label,
                             drv_time, egp_time, out_dir, tag):
    """Bar chart comparing SM-DeepRV and Exact GP on key metrics."""
    metric_names = ["RMSE", "MAE", "Coverage_95", "Mean_std"]
    labels       = ["SM-DeepRV", "Exact GP"]
    colours      = ["#2196F3", "#FF5722"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    for i, metric in enumerate(metric_names):
        vals = [drv_metrics[metric],
                egp_metrics[metric] if egp_metrics else float("nan")]
        bars = axes[i].bar(labels, vals, color=colours, alpha=0.8, edgecolor="k",
                           linewidth=0.5)
        axes[i].set_title(metric, fontsize=11)
        axes[i].set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.3 + 1e-6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.01 * axes[i].get_ylim()[1],
                             f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        # Draw nominal 95% coverage line
        if metric == "Coverage_95":
            axes[i].axhline(0.95, color="green", linestyle="--",
                            linewidth=1, label="Nominal 95%")
            axes[i].legend(fontsize=8)

    # Timing panel
    times  = [drv_time, egp_time if egp_time else float("nan")]
    bars   = axes[4].bar(labels, times, color=colours, alpha=0.8,
                         edgecolor="k", linewidth=0.5)
    axes[4].set_title("Inference time (s)", fontsize=11)
    for bar, val in zip(bars, times):
        if not np.isnan(val):
            axes[4].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.01 * (max(t for t in times if not np.isnan(t)) or 1),
                         f"{val:.0f}s", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        f"Metrics comparison — {config_label}", fontsize=12
    )
    plt.tight_layout()
    path = Path(out_dir) / f"metrics_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Metrics plot saved → {path}")


def plot_posterior_hyperparams(drv_samples, egp_samples, Q,
                                sm_config, out_dir, tag):
    """Histogram of posterior SM weight samples for both methods."""
    fig, axes = plt.subplots(1, Q, figsize=(5 * Q, 4))
    if Q == 1:
        axes = [axes]

    true_weights = sm_config["weights"]
    colours      = ["#2196F3", "#FF5722"]

    for q in range(Q):
        ax = axes[q]
        drv_w = np.array(drv_samples["weights"])[:, q]
        ax.hist(drv_w, bins=40, alpha=0.6, color=colours[0],
                label="SM-DeepRV", density=True)
        if egp_samples is not None:
            egp_w = np.array(egp_samples["weights"])[:, q]
            ax.hist(egp_w, bins=40, alpha=0.6, color=colours[1],
                    label="Exact GP", density=True)
        ax.axvline(true_weights[q], color="k", linestyle="--",
                   linewidth=1.5, label=f"True = {true_weights[q]}")
        ax.set_title(f"Weight w_{q+1}", fontsize=11)
        ax.set_xlabel("Value"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("Posterior SM mixture weights", fontsize=12)
    plt.tight_layout()
    path = Path(out_dir) / f"hyperparams_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Hyperparameter plot saved → {path}")


def plot_scatter_pred_vs_true(drv_mean, egp_mean, f_test,
                               config_label, out_dir, tag):
    """Scatter plot of predicted vs true f values for both methods."""
    n_cols = 2 if egp_mean is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    f_np   = np.array(f_test)
    lims   = (f_np.min() - 0.1, f_np.max() + 0.1)
    titles = ["SM-DeepRV", "Exact GP"]
    means  = [drv_mean, egp_mean] if egp_mean is not None else [drv_mean]
    colours= ["#2196F3", "#FF5722"]

    for ax, mean, title, col in zip(axes, means, titles, colours):
        m_np = np.array(mean)
        ax.scatter(f_np, m_np, alpha=0.5, s=15, color=col)
        ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
        rmse = float(np.sqrt(np.mean((m_np - f_np)**2)))
        ax.set_title(f"{title}  (RMSE={rmse:.3f})", fontsize=11)
        ax.set_xlabel("True f"); ax.set_ylabel("Predicted f")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    plt.suptitle(f"Predicted vs True — {config_label}", fontsize=12)
    plt.tight_layout()
    path = Path(out_dir) / f"pred_vs_true_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Pred-vs-true plot saved → {path}")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def make_save_callback(ckpt_dir, interval):
    def save_fn(step, rng, state, batch, extra):
        path = Path(ckpt_dir) / f"step_{step:07d}"
        path.mkdir(parents=True, exist_ok=True)
        ocp.PyTreeCheckpointer().save(
            str(path.absolute()),
            {"params": state.params, "kwargs": state.kwargs, "step": step},
            force=True,
        )
        log.info(f"Checkpoint saved at step {step}")
    return Callback(fn=save_fn, interval=interval)


def load_latest_checkpoint(ckpt_dir, model, optimizer, loader):
    ckpt_dir    = Path(ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("step_*"),
                         key=lambda p: int(p.name.split("_")[1]))
    if not checkpoints:
        return None, 0
    latest     = checkpoints[-1]
    start_step = int(latest.name.split("_")[1])
    ckpt       = ocp.PyTreeCheckpointer().restore(str(latest.absolute()))
    dummy_rng  = random.key(0)
    dummy_batch= next(loader(dummy_rng))
    init_vars  = model.init({"params": dummy_rng, "extra": dummy_rng},
                            **dummy_batch)
    init_vars.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=ckpt["params"],
        kwargs=ckpt["kwargs"],
        tx=optimizer,
    )
    log.info(f"Resumed from step {start_step} ({latest.name})")
    return state, start_step


# =============================================================================
# Single-configuration experiment
# =============================================================================

def run_single_config(args, s_all, decoder, config_name, sm_config,
                      out_dir, num_chains, rng):
    log.info(f"\n--- SM config: {config_name} ({sm_config['label']}) ---")
    rng, rng_data, rng_drv, rng_egp = random.split(rng, 4)

    # Generate data
    (y_obs_norm, y_mean, y_std,
     f_all, true_cond, obs_idx, tst_idx) = generate_observations(
        s_all, args.grid_size, sm_config, args.noise_std, rng_data
    )
    s_obs  = s_all[obs_idx]
    s_test = s_all[tst_idx]
    f_test = f_all[tst_idx]
    log.info(f"Observed: {len(obs_idx)}  Held-out: {len(tst_idx)}")

    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [("y_obs_norm", y_obs_norm), ("f_all", f_all),
                      ("true_cond", true_cond),
                      ("obs_idx", obs_idx), ("tst_idx", tst_idx)]:
        np.save(cfg_dir / f"{name}.npy", np.array(arr))
    np.save(cfg_dir / "y_normalisation.npy", np.array([y_mean, y_std]))

    # --- SM-DeepRV NUTS ---
    log.info(f"Running SM-DeepRV NUTS ({num_chains} chain(s)) …")
    drv_mean, drv_std, drv_samples, drv_time = deeprv_nuts_inference(
        decoder, y_obs_norm, y_mean, y_std,
        s_all, obs_idx, tst_idx,
        args.q, args.num_warmup, args.num_samples, num_chains, rng_drv,
    )
    for name, arr in [("drv_mean", drv_mean), ("drv_std", drv_std),
                      ("drv_weights", drv_samples["weights"]),
                      ("drv_sigma",   drv_samples["sigma"])]:
        np.save(cfg_dir / f"{name}.npy", np.array(arr))

    drv_metrics = evaluate(drv_mean, drv_std, f_test)
    log.info(f"SM-DeepRV — RMSE={drv_metrics['RMSE']:.4f}  "
             f"Coverage={drv_metrics['Coverage_95']:.3f}  "
             f"Time={drv_time:.1f}s")

    # --- Exact GP NUTS ---
    egp_mean = egp_std = egp_samples = egp_time = None
    egp_metrics = {}

    if not args.no_exact_gp:
        log.info(f"Running Exact GP NUTS ({num_chains} chain(s)) …")
        egp_mean, egp_std, egp_samples, egp_time = exact_gp_nuts_inference(
            y_obs_norm, y_mean, y_std,
            s_obs, s_test,
            args.q, args.num_warmup, args.num_samples, num_chains, rng_egp,
        )
        for name, arr in [("egp_mean", egp_mean), ("egp_std", egp_std),
                          ("egp_weights", egp_samples["weights"])]:
            np.save(cfg_dir / f"{name}.npy", np.array(arr))

        egp_metrics = evaluate(egp_mean, egp_std, f_test)
        log.info(f"Exact GP  — RMSE={egp_metrics['RMSE']:.4f}  "
                 f"Coverage={egp_metrics['Coverage_95']:.3f}  "
                 f"Time={egp_time:.1f}s")

    # --- Plots ---
    plot_spatial(s_all, f_all, y_obs_norm, y_mean, y_std,
                 obs_idx, tst_idx,
                 drv_mean, drv_std, egp_mean, egp_std,
                 sm_config["label"], args.grid_size, cfg_dir, config_name)

    plot_metrics_comparison(drv_metrics, egp_metrics if egp_metrics else None,
                            sm_config["label"],
                            drv_time, egp_time,
                            cfg_dir, config_name)

    plot_posterior_hyperparams(drv_samples,
                               egp_samples if egp_samples else None,
                               args.q, sm_config, cfg_dir, config_name)

    plot_scatter_pred_vs_true(drv_mean, egp_mean, f_test,
                              sm_config["label"], cfg_dir, config_name)

    # --- Combined metrics row ---
    row = {"config": config_name, "grid_size": args.grid_size,
           "drv_time_s": drv_time, "egp_time_s": egp_time}
    for k, v in drv_metrics.items():
        row[f"drv_{k}"] = v
    for k, v in egp_metrics.items():
        row[f"egp_{k}"] = v
    return row


# =============================================================================
# Main
# =============================================================================

def main():
    args        = parse_args()
    train_steps = paper_train_steps(args.grid_size, args.train_steps)
    lr          = paper_lr(args.grid_size, args.lr)
    num_chains  = paper_num_chains(args.grid_size)
    configs     = ({args.sm_config: SM_CONFIGS[args.sm_config]}
                   if args.sm_config else SM_CONFIGS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"sim_{timestamp}_grid{args.grid_size}"
    out_dir   = Path(args.results_dir) / run_name
    ckpt_dir  = args.ckpt_dir or str(
        Path(args.results_dir) /
        f"sim_checkpoints_grid{args.grid_size}_q{args.q}"
    )

    # Override ckpt_dir if loading from a specific checkpoint
    if args.load_ckpt:
        ckpt_dir = args.load_ckpt

    # Delete checkpoints if fresh run requested
    if args.fresh_run and Path(ckpt_dir).exists():
        import shutil
        log.info(f"--fresh_run: deleting existing checkpoints at {ckpt_dir}")
        shutil.rmtree(ckpt_dir)
        log.info("Checkpoints deleted — training from scratch.")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== SM-DeepRV Simulation ===")
    log.info(f"Grid: {args.grid_size}²  Q={args.q}  "
             f"Steps={train_steps}  LR={lr}")
    log.info(f"NUTS: {args.num_warmup} warmup / {args.num_samples} samples / "
             f"{num_chains} chain(s)")
    log.info(f"Exact GP baseline: {'OFF' if args.no_exact_gp else 'ON'}")
    log.info(f"Training mode: "
             f"{'fresh (checkpoints cleared)' if args.fresh_run else 'load only (no training)' if args.load_ckpt else 'resume / train'}")
    log.info(f"Checkpoint dir: {ckpt_dir}")
    log.info(f"Configs: {list(configs.keys())}")
    log.info(f"Results → {out_dir}")

    rng = random.key(args.seed)
    rng, rng_train, rng_exp = random.split(rng, 3)

    # Spatial grid
    s_all  = build_grid(
        [{"start": 0.0, "stop": 1.0, "num": args.grid_size}] * 2
    ).reshape(-1, 2)
    loader = sm_dataloader(s_all, args.q, BATCH_SIZE)

    # Train SM-DeepRV
    log.info("\n=== B. Training SM-DeepRV ===")
    wandb.init(project="sm_deeprv", name=run_name, config={
        "grid_size": args.grid_size, "train_steps": train_steps,
        "lr": lr, "Q": args.q, "kernel": "spectral_mixture",
        "likelihood": "gaussian", "noise_std": args.noise_std,
        "exact_gp_baseline": not args.no_exact_gp,
    })

    nn_model  = gMLPDeepRV(num_blks=2)
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(train_steps, lr), weight_decay=1e-2),
    )

    @jit
    def valid_step(rng, state, batch):
        output: VAEOutput = state.apply_fn(
            {"params": state.params, **state.kwargs},
            **batch, rngs={"extra": rng},
        )
        return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}

    saved_state, start_step = load_latest_checkpoint(
        ckpt_dir, nn_model, optimizer, loader
    )
    remaining = train_steps - start_step

    if args.load_ckpt:
        # Skip training entirely — use the loaded checkpoint as-is
        if saved_state is None:
            raise FileNotFoundError(
                f"--load_ckpt: no checkpoint found at {ckpt_dir}. "
                "Check the path and try again."
            )
        state = saved_state
        log.info(f"Loaded trained network from {ckpt_dir} (step {start_step}) "
                 "— skipping training.")
    elif remaining > 0:
        state = train(
            rng_train, nn_model, optimizer, deep_rv_train_step,
            train_num_steps      = remaining,
            train_dataloader     = loader,
            valid_step           = valid_step,
            valid_interval       = 10_000,
            valid_num_steps      = 200,
            valid_dataloader     = loader,
            valid_monitor_metric = "norm MSE",
            callbacks            = [make_save_callback(ckpt_dir, 10_000)],
            state                = saved_state,
            return_state         = "best",
        )
    else:
        state = saved_state
        log.info("Training already complete.")

    decoder = generate_surrogate_decoder(state, nn_model)

    # Inference across SM configurations
    log.info("\n=== C. Inference across SM configurations ===")
    all_rows = []
    for config_name, sm_config in configs.items():
        rng_exp, rng_cfg = random.split(rng_exp)
        row = run_single_config(
            args, s_all, decoder, config_name, sm_config,
            out_dir, num_chains, rng_cfg,
        )
        all_rows.append(row)
        wandb.log({
            f"{config_name}/drv_RMSE":     row.get("drv_RMSE", float("nan")),
            f"{config_name}/drv_Coverage": row.get("drv_Coverage_95", float("nan")),
            f"{config_name}/egp_RMSE":     row.get("egp_RMSE", float("nan")),
            f"{config_name}/egp_Coverage": row.get("egp_Coverage_95", float("nan")),
            f"{config_name}/speedup":      (row.get("egp_time_s") or 1) /
                                           max(row.get("drv_time_s", 1), 1e-6),
        })

    # Summary table
    log.info("\n=== D. Summary ===")
    summary = pd.DataFrame(all_rows).set_index("config")
    print("\n" + summary.to_string())
    summary.to_csv(out_dir / "summary.csv")

    wandb.finish()
    log.info(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()