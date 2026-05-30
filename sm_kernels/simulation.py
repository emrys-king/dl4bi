"""
SM-DeepRV Simulation Experiment
================================
Validates the SM-DeepRV framework on synthetic data where the ground truth
kernel hyperparameters are known. The experiment:

  1. Defines a "true" SM kernel with fixed hyperparameters
  2. Generates synthetic GP observations from that kernel
  3. Trains gMLPDeepRV on the SM prior (randomised hyperparameters)
  4. Runs MAP inference in the latent z-space to fit the observations
  5. Evaluates posterior predictive accuracy on held-out locations
  6. Compares against a Matérn MLE baseline (dl4bi.mle.gp_mle_bfgs)
  7. Saves all results and plots

Run on nvidia7
--------------
    conda activate sm_deeprv
    python simulation.py

Results are written to results/sim_<timestamp>/
"""

# =============================================================================
# Configuration
# =============================================================================

SEED          = 42
Q             = 3        # SM mixture components
GRID_SIZE     = 16       # training grid: GRID_SIZE² locations
BATCH_SIZE    = 32
TRAIN_STEPS   = 100_000
VALID_INTERVAL= 10_000
VALID_STEPS   = 200
LR            = 1e-3
CKPT_INTERVAL = 10_000

# Synthetic observation settings
N_OBS         = 150      # number of observed locations (random subset of grid)
N_TEST        = 106      # remaining grid points used for held-out evaluation
NOISE_STD     = 0.05     # observation noise standard deviation

# MAP inference settings
MAP_STEPS     = 2_000    # optimisation steps for z
MAP_LR        = 1e-2
N_POSTERIOR   = 50       # number of posterior samples drawn after MAP

# True SM hyperparameters — chosen to produce non-degenerate spatial variation
# on a [0,1]² grid and to lie within the dataloader's sampling distribution.
# The dataloader samples means ~ LogNormal(0,1) (range ~0.3–7) and
# variances ~ InverseGamma(2,1) (range ~0.3–3), so these values are in-distribution.
TRUE_WEIGHTS   = [0.5,  0.3,  0.2]
TRUE_MEANS     = [[1.0, 0.8], [2.5, 1.5], [4.0, 3.0]]   # [Q, D] cycles per unit
TRUE_VARIANCES = [[0.5, 0.5], [1.0, 0.8], [1.5, 1.2]]   # [Q, D]

CKPT_DIR   = "results/sim_checkpoints"
RESULTS_DIR= "results"

# =============================================================================
# Imports
# =============================================================================

import os
import time
from pathlib import Path
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad

import wandb
import optax
import orbax.checkpoint as ocp
import flax.linen as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt

from dl4bi_sps.utils import build_grid
from dl4bi.core.model_output import VAEOutput
from dl4bi.train import Callback, TrainState, train, cosine_annealing_lr
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder

# =============================================================================
# 1.  Spectral mixture kernel
# =============================================================================

@jit
def spectral_mixture(x, y, weights, means, variances):
    """Evaluate the SM kernel matrix K(x, y).

    Args:
        x, y:      input locations [N, D] and [M, D]
        weights:   mixture weights [Q]
        means:     spectral frequencies [Q, D]
        variances: spectral bandwidths [Q, D]

    Returns:
        Kernel matrix [N, M]
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
# 2.  SM dataloader  (randomised hyperparameters for pre-training)
# =============================================================================

def sm_dataloader(s, Q=Q, batch_size=BATCH_SIZE):
    """Infinite generator of GP batches with randomly sampled SM hyperparameters.

    Each yielded dict contains:
        s:            spatial locations [N, D]
        z:            iid Gaussian latents [B, N]
        f:            GP samples via Cholesky [B, N]
        conditionals: flattened SM hyperparameters [Q + 2·Q·D]
    """
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

            conditionals = jnp.concatenate([
                weights, means.ravel(), vars_.ravel()
            ])
            yield {"s": s, "z": z, "conditionals": conditionals, "f": f}

    return dataloader


# =============================================================================
# 3.  Synthetic observation generation
# =============================================================================

def generate_observations(s_all, rng):
    """Sample a synthetic function from the true SM kernel and observe a subset.

    Args:
        s_all: full spatial grid [N_all, D]
        rng:   JAX random key

    Returns:
        s_obs:   observed locations   [N_OBS, D]
        y_obs:   noisy observations   [N_OBS]
        s_test:  held-out locations   [N_TEST, D]
        f_test:  true function values [N_TEST]   (for evaluation)
        f_all:   true function over full grid [N_all]
        cond:    true conditionals vector (for oracle inference)
    """
    N_all = s_all.shape[0]
    rng, rng_f, rng_noise, rng_idx = random.split(rng, 4)

    weights   = jnp.array(TRUE_WEIGHTS)
    means     = jnp.array(TRUE_MEANS)
    variances = jnp.array(TRUE_VARIANCES)

    jitter = 5e-4 * jnp.eye(N_all)   # match dataloader jitter for consistency
    K      = spectral_mixture(s_all, s_all, weights, means, variances) + jitter
    L      = jnp.linalg.cholesky(K)
    z_true = random.normal(rng_f, shape=(N_all,))
    f_all  = L @ z_true   # [N_all]

    if jnp.any(jnp.isnan(f_all)):
        raise ValueError(
            "f_all contains NaN — kernel matrix is near-singular. "
            "Check TRUE_MEANS and TRUE_VARIANCES are non-degenerate on the grid."
        )

    # Random train/test split
    idx     = random.permutation(rng_idx, N_all)
    obs_idx = idx[:N_OBS]
    tst_idx = idx[N_OBS:N_OBS + N_TEST]

    s_obs  = s_all[obs_idx]
    f_obs  = f_all[obs_idx]
    s_test = s_all[tst_idx]
    f_test = f_all[tst_idx]

    noise  = NOISE_STD * random.normal(rng_noise, shape=(N_OBS,))
    y_obs  = f_obs + noise

    # True conditionals vector
    cond = jnp.concatenate([weights, means.ravel(), variances.ravel()])

    return s_obs, y_obs, s_test, f_test, f_all, cond, obs_idx, tst_idx


# =============================================================================
# 4.  MAP inference in z-space
# =============================================================================

def map_inference(decoder, y_obs, s_all, obs_idx, tst_idx, conditionals, rng,
                  n_steps=MAP_STEPS, lr=MAP_LR, n_samples=N_POSTERIOR):
    """Fit z via MAP to the observations, then predict at test locations.

    The decoder requires the full training grid s_all — the gMLP spatial
    gating unit has a bias parameter fixed to the training sequence length,
    so we always pass s_all and index the output at obs_idx / tst_idx.

    The objective is:
        L(z) = ||decoder(z, cond, s_all)[obs_idx] - y_obs||² / (2·noise²)
               + ||z||² / 2

    Args:
        decoder:      trained surrogate decoder from generate_surrogate_decoder
        y_obs:        observations [N_OBS]
        s_all:        full spatial grid [N_all, D]
        obs_idx:      indices of observed locations into s_all [N_OBS]
        tst_idx:      indices of test locations into s_all [N_TEST]
        conditionals: kernel hyperparameter vector
        rng:          JAX random key
        n_steps:      gradient descent steps for MAP
        lr:           learning rate
        n_samples:    number of posterior samples drawn after MAP

    Returns:
        mean_pred:  posterior predictive mean [N_TEST]  (in original scale)
        std_pred:   posterior predictive std  [N_TEST]  (in original scale)
        z_map:      MAP estimate of z         [1, N_all]
    """
    N_all = s_all.shape[0]

    # Normalise observations to zero mean / unit std for numerical stability.
    # The decoder was trained on standardised GP samples so this is necessary.
    y_mean = jnp.mean(y_obs)
    y_std  = jnp.std(y_obs) + 1e-6
    y_norm = (y_obs - y_mean) / y_std

    z     = jnp.zeros((1, N_all))
    optimizer  = optax.adam(lr)
    opt_state  = optimizer.init(z)

    @jit
    def loss_fn(z):
        # Decode over full grid, index to observed locations for loss
        f_hat = decoder(z, conditionals, s=s_all).squeeze()   # [N_all]
        f_obs = f_hat[obs_idx]                                 # [N_OBS]
        recon = jnp.mean((f_obs - y_norm)**2) / (2 * (NOISE_STD / y_std)**2)
        prior = 0.5 * jnp.mean(z**2)
        return recon + prior

    val_and_grad_fn = jit(value_and_grad(loss_fn))

    for step in range(n_steps):
        loss, grad = val_and_grad_fn(z)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = optax.apply_updates(z, updates)
        if step % 500 == 0:
            print(f"    MAP step {step:5d}  loss={loss:.4f}")

    z_map = z

    # --- Predict at test locations, then rescale to original units ---
    rng, rng_samp = random.split(rng)
    z_noise = 0.1 * random.normal(rng_samp, shape=(n_samples, N_all))
    z_samps = z_map + z_noise                              # [n_samples, N_all]

    preds = decoder(z_samps, conditionals, s=s_all)        # [n_samples, N_all, 1]
    preds = preds.squeeze(-1)[:, tst_idx]                  # [n_samples, N_TEST]

    # Rescale back to original observation scale
    preds = preds * y_std + y_mean

    mean_pred = preds.mean(axis=0)
    std_pred  = preds.std(axis=0)

    return mean_pred, std_pred, z_map


# =============================================================================
# 5.  Matérn MLE baseline
# =============================================================================

def matern_baseline(s_obs, y_obs, s_test):
    """Fit a Matérn-3/2 GP by MLE and return test predictions.

    Uses dl4bi.mle.gp_mle_bfgs for hyperparameter estimation and the
    standard GP predictive equations for inference.

    Returns:
        mean_pred: predictive mean [N_TEST]
        std_pred:  predictive std  [N_TEST]
        theta:     estimated (var, ls, noise)
    """
    from dl4bi_sps.kernels import matern_3_2
    from dl4bi.mle import gp_mle_bfgs, gp_nll

    print("  Fitting Matérn-3/2 MLE baseline …")

    theta = gp_mle_bfgs(s_obs, y_obs, matern_3_2)
    var, ls, noise = theta

    print(f"  MLE result: var={var:.3f}  ls={ls:.3f}  noise={noise:.3f}")

    N_obs  = s_obs.shape[0]
    K_obs  = matern_3_2(s_obs, s_obs, var, ls) + noise * jnp.eye(N_obs)
    K_star = matern_3_2(s_test, s_obs, var, ls)
    K_ss   = matern_3_2(s_test, s_test, var, ls)

    L    = jnp.linalg.cholesky(K_obs)
    alpha = jnp.linalg.solve(K_obs, y_obs)

    mean_pred = K_star @ alpha

    v         = jnp.linalg.solve(L, K_star.T)
    var_pred  = jnp.diag(K_ss) - jnp.sum(v**2, axis=0)
    std_pred  = jnp.sqrt(jnp.clip(var_pred, 0.0))

    return mean_pred, std_pred, theta


# =============================================================================
# 6.  Evaluation metrics
# =============================================================================

def evaluate(mean_pred, std_pred, f_true):
    """Compute RMSE, MAE, and 95% interval coverage.

    Args:
        mean_pred: predictive mean [N]
        std_pred:  predictive std  [N]
        f_true:    ground truth    [N]

    Returns:
        dict of metric name → float
    """
    errors    = mean_pred - f_true
    rmse      = float(jnp.sqrt(jnp.mean(errors**2)))
    mae       = float(jnp.mean(jnp.abs(errors)))
    z_scores  = jnp.abs(errors) / (std_pred + 1e-8)
    coverage  = float(jnp.mean(z_scores < 1.96))   # nominal 95%
    mean_std  = float(jnp.mean(std_pred))
    return {"RMSE": rmse, "MAE": mae, "Coverage_95": coverage,
            "Mean_std": mean_std}


# =============================================================================
# 7.  Plotting
# =============================================================================

def plot_results(s_all, f_all, s_obs, y_obs, s_test,
                 drv_mean, drv_std, mle_mean, mle_std,
                 out_dir):
    """Save a 2×2 grid comparing DeepRV and Matérn-MLE predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    D = s_all.shape[1]

    def scatter(ax, s, vals, title, vmin=None, vmax=None, cmap="RdYlBu_r"):
        sc = ax.scatter(s[:, 0], s[:, 1], c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax, s=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("lon / x"); ax.set_ylabel("lat / y")
        plt.colorbar(sc, ax=ax)

    vmin = float(f_all.min()); vmax = float(f_all.max())

    scatter(axes[0, 0], s_all, f_all,  "True function",     vmin, vmax)
    axes[0, 0].scatter(s_obs[:, 0], s_obs[:, 1],
                       c="k", s=15, marker="x", label="Observations")
    axes[0, 0].legend(fontsize=8)

    scatter(axes[0, 1], s_test, drv_mean,
            "SM-DeepRV predictive mean", vmin, vmax)

    scatter(axes[1, 0], s_test, mle_mean,
            "Matérn-MLE predictive mean", vmin, vmax)

    # Uncertainty comparison — side-by-side scatter on same axes
    axes[1, 1].scatter(s_test[:, 0], s_test[:, 1],
                       c=np.array(drv_std), cmap="Blues",
                       s=20, alpha=0.8, vmin=0,
                       vmax=float(max(drv_std.max(), mle_std.max())),
                       label="SM-DeepRV")
    sc2 = axes[1, 1].scatter(s_test[:, 0], s_test[:, 1],
                       c=np.array(mle_std), cmap="Oranges",
                       s=8, alpha=0.5, vmin=0,
                       vmax=float(max(drv_std.max(), mle_std.max())),
                       label="Matérn-MLE")
    axes[1, 1].set_title("Predictive std (blue=SM-DeepRV, orange=Matérn-MLE)", fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    axes[1, 1].legend(fontsize=8)

    plt.suptitle("SM-DeepRV vs Matérn-MLE — Simulation Study", fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# =============================================================================
# 8.  Checkpoint helpers (reused from training notebook)
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
        print(f"  ✓ Checkpoint saved at step {step}")
    return Callback(fn=save_fn, interval=CKPT_INTERVAL)


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
    init_vars  = model.init({"params": dummy_rng, "extra": dummy_rng}, **dummy_batch)
    init_vars.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=ckpt["params"],
        kwargs=ckpt["kwargs"],
        tx=optimizer,
    )
    print(f"  Resumed from step {start_step} ({latest.name})")
    return state, start_step


# =============================================================================
# 9.  Main
# =============================================================================

def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = Path(RESULTS_DIR) / f"sim_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {out_dir}\n")

    rng = random.key(SEED)
    rng, rng_data, rng_train, rng_infer = random.split(rng, 4)

    # ------------------------------------------------------------------
    # A. Spatial grid and synthetic data
    # ------------------------------------------------------------------
    print("=== A. Generating synthetic observations ===")
    from dl4bi_sps.utils import build_grid
    s_all = build_grid(
        [{"start": 0.0, "stop": 1.0, "num": GRID_SIZE}] * 2
    ).reshape(-1, 2)
    N_all, D = s_all.shape

    s_obs, y_obs, s_test, f_test, f_all, true_cond, obs_idx, tst_idx = generate_observations(
        s_all, rng_data
    )
    print(f"  Grid:       {N_all} locations ({GRID_SIZE}×{GRID_SIZE})")
    print(f"  Observed:   {len(s_obs)}")
    print(f"  Test:       {len(s_test)}")
    print(f"  True cond:  {true_cond}")

    loader = sm_dataloader(s_all)

    # Save observation arrays for reproducibility
    np.save(out_dir / "s_obs.npy",  np.array(s_obs))
    np.save(out_dir / "y_obs.npy",  np.array(y_obs))
    np.save(out_dir / "s_test.npy", np.array(s_test))
    np.save(out_dir / "f_test.npy", np.array(f_test))
    np.save(out_dir / "f_all.npy",  np.array(f_all))

    # ------------------------------------------------------------------
    # B. Train SM-DeepRV
    # ------------------------------------------------------------------
    print("\n=== B. Training SM-DeepRV ===")
    wandb.init(project="sm_deeprv", name=f"sim_{timestamp}")
    nn_model  = gMLPDeepRV(num_blks=2)
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(TRAIN_STEPS, LR), weight_decay=1e-2),
    )

    @jit
    def valid_step(rng, state, batch):
        output: VAEOutput = state.apply_fn(
            {"params": state.params, **state.kwargs},
            **batch, rngs={"extra": rng},
        )
        return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}

    # Resume if a checkpoint exists
    saved_state, start_step = load_latest_checkpoint(
        CKPT_DIR, nn_model, optimizer, loader
    )
    remaining = TRAIN_STEPS - start_step

    if remaining > 0:
        state = train(
            rng_train,
            nn_model,
            optimizer,
            deep_rv_train_step,
            train_num_steps      = remaining,
            train_dataloader     = loader,
            valid_step           = valid_step,
            valid_interval       = VALID_INTERVAL,
            valid_num_steps      = VALID_STEPS,
            valid_dataloader     = loader,
            valid_monitor_metric = "norm MSE",
            callbacks            = [make_save_callback(CKPT_DIR)],
            state                = saved_state,
            return_state         = "best",
        )
    else:
        state = saved_state
        print("  Training already complete.")

    decoder = generate_surrogate_decoder(state, nn_model)

    # ------------------------------------------------------------------
    # C. SM-DeepRV inference
    # ------------------------------------------------------------------
    print("\n=== C. SM-DeepRV inference (MAP in z-space) ===")

    # Oracle inference: use the true hyperparameters as conditionals.
    # In a real application you would estimate these from data first
    # (e.g. via MLE on the SM NLL) — we use the true values here to
    # isolate the quality of the latent-space inference from
    # hyperparameter estimation error.
    drv_mean, drv_std, z_map = map_inference(
        decoder, y_obs, s_all, obs_idx, tst_idx, true_cond, rng_infer
    )
    np.save(out_dir / "z_map.npy",    np.array(z_map))
    np.save(out_dir / "drv_mean.npy", np.array(drv_mean))
    np.save(out_dir / "drv_std.npy",  np.array(drv_std))

    # ------------------------------------------------------------------
    # D. Matérn-MLE baseline
    # ------------------------------------------------------------------
    print("\n=== D. Matérn-MLE baseline ===")
    mle_mean, mle_std, mle_theta = matern_baseline(s_obs, y_obs, s_test)
    np.save(out_dir / "mle_mean.npy",  np.array(mle_mean))
    np.save(out_dir / "mle_std.npy",   np.array(mle_std))
    np.save(out_dir / "mle_theta.npy", np.array(mle_theta))

    # ------------------------------------------------------------------
    # E. Evaluation
    # ------------------------------------------------------------------
    print("\n=== E. Evaluation ===")
    drv_metrics = evaluate(drv_mean, drv_std, f_test)
    mle_metrics = evaluate(mle_mean, mle_std, f_test)

    metrics_df = pd.DataFrame({
        "SM-DeepRV": drv_metrics,
        "Matérn-MLE": mle_metrics,
    }).T
    metrics_df.index.name = "Method"

    print(metrics_df.to_string())
    metrics_df.to_csv(out_dir / "metrics.csv")

    # ------------------------------------------------------------------
    # F. Plot
    # ------------------------------------------------------------------
    print("\n=== F. Plotting ===")
    plot_results(
        s_all, f_all, s_obs, y_obs, s_test,
        drv_mean, drv_std, mle_mean, mle_std,
        out_dir,
    )

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()