"""
SM-DeepRV Simulation Experiment
================================
Validates the SM-DeepRV framework on synthetic data where the ground truth
kernel hyperparameters are known. Uses NumPyro NUTS for posterior inference
in the latent z-space.

Usage
-----
    python simulation.py [--grid_size N] [--train_steps T] [--q Q]
                         [--num_warmup W] [--num_samples S] [--device D]

Examples
--------
    # Default 16x16 grid, 100k training steps
    CUDA_VISIBLE_DEVICES=0 python simulation.py

    # 32x32 grid, 200k steps
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 32 --train_steps 200000

    # Quick test run
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 8 --train_steps 5000 \
                                                --num_warmup 100 --num_samples 200
"""

# =============================================================================
# Imports
# =============================================================================

import os
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import optax
import orbax.checkpoint as ocp
import flax.linen as nn
import numpy as np
import pandas as pd
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SM-DeepRV simulation experiment")
    p.add_argument("--grid_size",   type=int,   default=16,
                   help="Grid is grid_size × grid_size locations (default: 16)")
    p.add_argument("--q",           type=int,   default=3,
                   help="Number of SM mixture components (default: 3)")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--train_steps", type=int,   default=100_000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--ckpt_interval",type=int,  default=10_000)
    p.add_argument("--n_obs",       type=int,   default=None,
                   help="Number of observed locations. Defaults to ~60%% of grid.")
    p.add_argument("--noise_std",   type=float, default=0.05)
    p.add_argument("--num_warmup",  type=int,   default=500,
                   help="NUTS warmup steps (default: 500)")
    p.add_argument("--num_samples", type=int,   default=1000,
                   help="NUTS posterior samples (default: 1000)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--results_dir", type=str,   default="results")
    p.add_argument("--ckpt_dir",    type=str,   default=None,
                   help="Checkpoint directory. Defaults to results/sim_checkpoints_gridN/")
    p.add_argument("--device",      type=int,   default=0,
                   help="CUDA device index (default: 0). Ignored if "
                        "CUDA_VISIBLE_DEVICES is already set.")
    return p.parse_args()


# =============================================================================
# Spectral mixture kernel
# =============================================================================

@jit
def spectral_mixture(x, y, weights, means, variances):
    """SM kernel matrix K(x, y).

    Args:
        x, y:      [N, D] and [M, D]
        weights:   [Q]   mixture weights
        means:     [Q, D] spectral frequencies
        variances: [Q, D] spectral bandwidths

    Returns: [N, M]
    """
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    diff = x[:, None, :] - y[None, :, :]   # [N, M, D]

    def component(w, mu, v):
        env  = jnp.exp(-2 * jnp.pi**2 * jnp.sum(v * diff**2, axis=-1))
        freq = jnp.cos(2 * jnp.pi * jnp.sum(mu * diff, axis=-1))
        return w * env * freq

    return jnp.sum(vmap(component)(weights, means, variances), axis=0)


# =============================================================================
# SM dataloader
# =============================================================================

def sm_dataloader(s, Q, batch_size):
    """Infinite generator of GP batches with randomly sampled SM hyperparameters."""
    N, D = s.shape
    jitter = 5e-4 * jnp.eye(N)

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
            conditionals = jnp.concatenate([weights, means.ravel(), vars_.ravel()])
            yield {"s": s, "z": z, "conditionals": conditionals, "f": f}

    return dataloader


# =============================================================================
# Synthetic data generation
# =============================================================================

def generate_observations(s_all, n_obs, noise_std, rng):
    """Sample from the true SM kernel and return train/test split.

    True hyperparameters are chosen to lie within the dataloader's sampling
    distribution (means ~ LogNormal(0,1), variances ~ InvGamma(2,1)).

    Returns:
        s_obs, y_obs, s_test, f_test, f_all, true_cond, obs_idx, tst_idx
    """
    N_all, D = s_all.shape
    n_test   = N_all - n_obs

    # True hyperparameters — in-distribution for the dataloader
    true_weights   = jnp.array([0.5, 0.3, 0.2])
    true_means     = jnp.array([[1.0, 0.8], [2.5, 1.5], [4.0, 3.0]])
    true_variances = jnp.array([[0.5, 0.5], [1.0, 0.8], [1.5, 1.2]])

    rng, rng_f, rng_noise, rng_idx = random.split(rng, 4)

    jitter = 5e-4 * jnp.eye(N_all)
    K      = spectral_mixture(s_all, s_all, true_weights, true_means,
                              true_variances) + jitter
    L      = jnp.linalg.cholesky(K)
    z_true = random.normal(rng_f, shape=(N_all,))
    f_all  = L @ z_true

    if jnp.any(jnp.isnan(f_all)):
        raise ValueError("f_all contains NaN — kernel matrix is near-singular.")

    idx     = random.permutation(rng_idx, N_all)
    obs_idx = idx[:n_obs]
    tst_idx = idx[n_obs:n_obs + n_test]

    s_obs  = s_all[obs_idx]
    f_obs  = f_all[obs_idx]
    s_test = s_all[tst_idx]
    f_test = f_all[tst_idx]
    y_obs  = f_obs + noise_std * random.normal(rng_noise, shape=(n_obs,))

    true_cond = jnp.concatenate([
        true_weights, true_means.ravel(), true_variances.ravel()
    ])
    return s_obs, y_obs, s_test, f_test, f_all, true_cond, obs_idx, tst_idx


# =============================================================================
# NumPyro MCMC inference in z-space
# =============================================================================

def nuts_inference(decoder, y_obs, y_mean, y_std, s_all, obs_idx, tst_idx,
                   conditionals, noise_std, rng,
                   num_warmup=500, num_samples=1000):
    """NUTS posterior inference over the latent z given observations.

    The model is:
        z ~ N(0, I)           (standard normal prior on latent code)
        f = decoder(z, cond, s_all)
        y_obs ~ N(f[obs_idx], noise_std²)

    We normalise observations to zero mean / unit std for numerical stability,
    matching the scale on which the decoder was trained.

    Args:
        decoder:      trained surrogate decoder
        y_obs:        raw observations [N_OBS]
        y_mean, y_std: normalisation constants computed from y_obs
        s_all:        full spatial grid [N_all, D]
        obs_idx:      indices of observed locations into s_all
        tst_idx:      indices of test locations into s_all
        conditionals: kernel hyperparameter vector
        noise_std:    observation noise standard deviation (in normalised units)
        rng:          JAX random key
        num_warmup:   NUTS warmup steps
        num_samples:  posterior samples to draw

    Returns:
        mean_pred:  posterior predictive mean [N_TEST] in original scale
        std_pred:   posterior predictive std  [N_TEST] in original scale
        z_samples:  posterior z samples [num_samples, N_all]
    """
    N_all = s_all.shape[0]
    y_norm = (y_obs - y_mean) / y_std
    noise_norm = noise_std / y_std   # noise in normalised units

    def numpyro_model():
        # Prior over latent code
        z = numpyro.sample("z", dist.Normal(jnp.zeros(N_all),
                                            jnp.ones(N_all)))
        # Decode — reshape to [1, N_all] as decoder expects batch dimension
        f_all = decoder(z[None], conditionals, s=s_all).squeeze()   # [N_all]
        # Likelihood at observed locations
        numpyro.sample("y", dist.Normal(f_all[obs_idx], noise_norm),
                       obs=y_norm)

    rng_key = jax.random.key(int(rng[0]))

    kernel = NUTS(numpyro_model, target_accept_prob=0.8)
    mcmc   = MCMC(kernel,
                  num_warmup=num_warmup,
                  num_samples=num_samples,
                  num_chains=1,
                  progress_bar=True)
    mcmc.run(rng_key)
    mcmc.print_summary()

    z_samples = mcmc.get_samples()["z"]   # [num_samples, N_all]

    # Posterior predictive at test locations
    def predict_one(z):
        f = decoder(z[None], conditionals, s=s_all).squeeze()
        return f[tst_idx]

    preds = vmap(predict_one)(z_samples)   # [num_samples, N_TEST]

    # Rescale back to original units
    preds     = preds * y_std + y_mean
    mean_pred = preds.mean(axis=0)
    std_pred  = preds.std(axis=0)

    return mean_pred, std_pred, z_samples


# =============================================================================
# Matérn-MLE baseline
# =============================================================================

def matern_baseline(s_obs, y_obs, s_test):
    """Fit a Matérn-3/2 GP by MLE and return predictive mean and std."""
    from dl4bi_sps.kernels import matern_3_2
    from dl4bi.mle import gp_mle_bfgs

    log.info("Fitting Matérn-3/2 MLE baseline …")
    theta = gp_mle_bfgs(s_obs, y_obs, matern_3_2)
    var, ls, noise = theta
    log.info(f"MLE result: var={var:.3f}  ls={ls:.3f}  noise={noise:.3f}")

    N_obs  = s_obs.shape[0]
    K_obs  = matern_3_2(s_obs, s_obs, var, ls) + noise * jnp.eye(N_obs)
    K_star = matern_3_2(s_test, s_obs, var, ls)
    K_ss   = matern_3_2(s_test, s_test, var, ls)

    L     = jnp.linalg.cholesky(K_obs)
    alpha = jnp.linalg.solve(K_obs, y_obs)

    mean_pred = K_star @ alpha
    v         = jnp.linalg.solve(L, K_star.T)
    var_pred  = jnp.diag(K_ss) - jnp.sum(v**2, axis=0)
    std_pred  = jnp.sqrt(jnp.clip(var_pred, 0.0))

    return mean_pred, std_pred, theta


# =============================================================================
# Evaluation metrics
# =============================================================================

def evaluate(mean_pred, std_pred, f_true):
    errors   = mean_pred - f_true
    rmse     = float(jnp.sqrt(jnp.mean(errors**2)))
    mae      = float(jnp.mean(jnp.abs(errors)))
    z_scores = jnp.abs(errors) / (std_pred + 1e-8)
    coverage = float(jnp.mean(z_scores < 1.96))
    mean_std = float(jnp.mean(std_pred))
    return {"RMSE": rmse, "MAE": mae, "Coverage_95": coverage,
            "Mean_std": mean_std}


# =============================================================================
# Plotting
# =============================================================================

def plot_results(s_all, f_all, s_obs, y_obs, s_test,
                 drv_mean, drv_std, mle_mean, mle_std,
                 out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def scatter(ax, s, vals, title, vmin=None, vmax=None, cmap="RdYlBu_r"):
        sc = ax.scatter(s[:, 0], s[:, 1], c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax, s=18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("lon / x"); ax.set_ylabel("lat / y")
        plt.colorbar(sc, ax=ax)

    vmin = float(f_all.min()); vmax = float(f_all.max())

    scatter(axes[0, 0], s_all, f_all, "True function", vmin, vmax)
    axes[0, 0].scatter(s_obs[:, 0], s_obs[:, 1],
                       c="k", s=20, marker="x", label="Observations", zorder=5)
    axes[0, 0].legend(fontsize=8)

    scatter(axes[0, 1], s_test, drv_mean,
            "SM-DeepRV predictive mean (NUTS)", vmin, vmax)

    scatter(axes[1, 0], s_test, mle_mean,
            "Matérn-MLE predictive mean", vmin, vmax)

    # Uncertainty comparison
    std_vmax = float(max(float(jnp.nanmax(drv_std)),
                         float(jnp.nanmax(mle_std))))
    axes[1, 1].scatter(s_test[:, 0], s_test[:, 1],
                       c=np.array(drv_std), cmap="Blues",
                       s=20, alpha=0.8, vmin=0, vmax=std_vmax,
                       label="SM-DeepRV")
    axes[1, 1].scatter(s_test[:, 0], s_test[:, 1],
                       c=np.array(mle_std), cmap="Oranges",
                       s=8, alpha=0.6, vmin=0, vmax=std_vmax,
                       label="Matérn-MLE")
    axes[1, 1].set_title("Predictive std (blue=SM-DeepRV, orange=Matérn-MLE)",
                          fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    axes[1, 1].legend(fontsize=8)

    plt.suptitle("SM-DeepRV (NUTS) vs Matérn-MLE — Simulation Study",
                 fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved → {path}")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def make_save_callback(ckpt_dir):
    def save_fn(step, rng, state, batch, extra):
        path = Path(ckpt_dir) / f"step_{step:07d}"
        path.mkdir(parents=True, exist_ok=True)
        ocp.PyTreeCheckpointer().save(
            str(path.absolute()),
            {"params": state.params, "kwargs": state.kwargs, "step": step},
            force=True,
        )
        log.info(f"Checkpoint saved at step {step}")
    return Callback(fn=save_fn, interval=None)   # interval set in main()


def load_latest_checkpoint(ckpt_dir, model, optimizer, loader):
    ckpt_dir = Path(ckpt_dir)
    checkpoints = sorted(
        ckpt_dir.glob("step_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
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
# Main
# =============================================================================

def main():
    args = parse_args()

    # Derived quantities
    N_all = args.grid_size ** 2
    n_obs = args.n_obs or int(0.6 * N_all)
    D     = 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"sim_{timestamp}_grid{args.grid_size}"
    out_dir   = Path(args.results_dir) / run_name
    ckpt_dir  = args.ckpt_dir or str(
        Path(args.results_dir) / f"sim_checkpoints_grid{args.grid_size}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Run: {run_name}")
    log.info(f"Grid: {args.grid_size}×{args.grid_size} = {N_all} locations")
    log.info(f"Q={args.q}  n_obs={n_obs}  n_test={N_all-n_obs}")

    rng = random.key(args.seed)
    rng, rng_data, rng_train, rng_infer = random.split(rng, 4)

    # ------------------------------------------------------------------
    # A. Synthetic data
    # ------------------------------------------------------------------
    log.info("=== A. Generating synthetic observations ===")
    s_all = build_grid(
        [{"start": 0.0, "stop": 1.0, "num": args.grid_size}] * 2
    ).reshape(-1, D)

    (s_obs, y_obs, s_test, f_test,
     f_all, true_cond, obs_idx, tst_idx) = generate_observations(
        s_all, n_obs, args.noise_std, rng_data
    )
    log.info(f"Observed={len(s_obs)}  Test={len(s_test)}")
    np.save(out_dir / "s_obs.npy",  np.array(s_obs))
    np.save(out_dir / "y_obs.npy",  np.array(y_obs))
    np.save(out_dir / "s_test.npy", np.array(s_test))
    np.save(out_dir / "f_test.npy", np.array(f_test))
    np.save(out_dir / "f_all.npy",  np.array(f_all))

    loader = sm_dataloader(s_all, args.q, args.batch_size)

    # ------------------------------------------------------------------
    # B. Train SM-DeepRV
    # ------------------------------------------------------------------
    log.info("=== B. Training SM-DeepRV ===")
    wandb.init(project="sm_deeprv", name=run_name,
               config=vars(args))

    nn_model  = gMLPDeepRV(num_blks=2)
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(args.train_steps, args.lr),
                    weight_decay=1e-2),
    )

    @jit
    def valid_step(rng, state, batch):
        output: VAEOutput = state.apply_fn(
            {"params": state.params, **state.kwargs},
            **batch, rngs={"extra": rng},
        )
        return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}

    cb = make_save_callback(ckpt_dir)
    cb.interval = args.ckpt_interval

    saved_state, start_step = load_latest_checkpoint(
        ckpt_dir, nn_model, optimizer, loader
    )
    remaining = args.train_steps - start_step

    if remaining > 0:
        state = train(
            rng_train, nn_model, optimizer, deep_rv_train_step,
            train_num_steps      = remaining,
            train_dataloader     = loader,
            valid_step           = valid_step,
            valid_interval       = 10_000,
            valid_num_steps      = 200,
            valid_dataloader     = loader,
            valid_monitor_metric = "norm MSE",
            callbacks            = [cb],
            state                = saved_state,
            return_state         = "best",
        )
    else:
        state = saved_state
        log.info("Training already complete.")

    decoder = generate_surrogate_decoder(state, nn_model)

    # ------------------------------------------------------------------
    # C. NUTS inference in z-space
    # ------------------------------------------------------------------
    log.info("=== C. SM-DeepRV inference (NUTS) ===")
    y_mean = float(jnp.mean(y_obs))
    y_std  = float(jnp.std(y_obs)) + 1e-6

    drv_mean, drv_std, z_samples = nuts_inference(
        decoder, y_obs, y_mean, y_std,
        s_all, obs_idx, tst_idx, true_cond,
        args.noise_std, rng_infer,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
    )
    np.save(out_dir / "drv_mean.npy",   np.array(drv_mean))
    np.save(out_dir / "drv_std.npy",    np.array(drv_std))
    np.save(out_dir / "z_samples.npy",  np.array(z_samples))

    # ------------------------------------------------------------------
    # D. Matérn-MLE baseline
    # ------------------------------------------------------------------
    log.info("=== D. Matérn-MLE baseline ===")
    mle_mean, mle_std, mle_theta = matern_baseline(s_obs, y_obs, s_test)
    np.save(out_dir / "mle_mean.npy",  np.array(mle_mean))
    np.save(out_dir / "mle_std.npy",   np.array(mle_std))
    np.save(out_dir / "mle_theta.npy", np.array(mle_theta))

    # ------------------------------------------------------------------
    # E. Evaluation
    # ------------------------------------------------------------------
    log.info("=== E. Evaluation ===")
    drv_metrics = evaluate(drv_mean, drv_std, f_test)
    mle_metrics = evaluate(mle_mean, mle_std, f_test)

    metrics_df = pd.DataFrame({
        "SM-DeepRV (NUTS)": drv_metrics,
        "Matérn-MLE":       mle_metrics,
    }).T
    metrics_df.index.name = "Method"
    print("\n" + metrics_df.to_string())
    metrics_df.to_csv(out_dir / "metrics.csv")

    wandb.log({
        "SM-DeepRV RMSE":    drv_metrics["RMSE"],
        "SM-DeepRV Coverage":drv_metrics["Coverage_95"],
        "MLE RMSE":          mle_metrics["RMSE"],
        "MLE Coverage":      mle_metrics["Coverage_95"],
        "grid_size":         args.grid_size,
    })

    # ------------------------------------------------------------------
    # F. Plot
    # ------------------------------------------------------------------
    log.info("=== F. Plotting ===")
    plot_results(
        s_all, f_all, s_obs, y_obs, s_test,
        drv_mean, drv_std, mle_mean, mle_std,
        out_dir,
    )

    wandb.finish()
    log.info(f"Done. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()