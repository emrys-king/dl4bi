"""
SM-DeepRV Simulation Experiment  (multi-Q / multi-grid version)
================================================================
Extends q1_sim.py to support any number of SM components (--q) and
any grid size, with two new inference improvements for Q > 1:

  1. Label-switching fix (--q > 1 only)
     The x-direction frequency µ^(x)_q is ordered ascending across
     components by sampling a base frequency + Q-1 positive increments.
     This collapses the Q! permutation-equivalent posterior modes into
     one canonical ordering, consistent between training and inference.

  2. Sparse Dirichlet weights (--alpha_dirichlet, default 0.0)
     Setting alpha < 1.0 (e.g. 0.5) places mass at the corners of the
     weight simplex, driving unnecessary components toward zero and
     giving a soft approximation to learnable Q.  alpha = 0.0 uses the
     standard uniform Dirichlet (all components equally encouraged).

IMPORTANT — per-(grid_size, Q) training
     Each unique combination of grid_size and Q requires its own trained
     decoder because the conditional vector length is Q(1+2D).
     Use --fresh_run when switching grid_size OR Q.

     Rough training time on one RTX 6000 Ada (49 GB):
       grid16 Q=1: ~30 min  (500k steps)
       grid32 Q=1: ~2 hr    (500k steps, ~4× per-step cost)
       grid64 Q=1: ~3 hr    (700k steps)
       grid16 Q=3: ~45 min  (500k steps, larger cond vector)

Usage examples
--------------
    # Train Q=1 on grid16 (same as q1_sim.py)
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 16 --q 1 --fresh_run --train_only

    # Train Q=3 on grid16 with sparse Dirichlet
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 16 --q 3 \\
        --alpha_dirichlet 0.5 --fresh_run --train_only

    # Train Q=1 on grid32 (needs its own checkpoint)
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 32 --q 1 --fresh_run --train_only

    # Inference only, loading existing checkpoint
    CUDA_VISIBLE_DEVICES=0 python simulation.py --grid_size 16 --q 3 \\
        --alpha_dirichlet 0.5 --sm_config slow \\
        --load_ckpt results/sim_checkpoints_grid16_q3_blks2_emb64 \\
        --no_dense_mass --num_warmup 10000 --num_samples 12000
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import scipy.spatial.distance as spd
from scipy.optimize import root
from scipy.stats import invgamma

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, init_to_median
from numpyro.infer.autoguide import AutoMultivariateNormal

import optax
import orbax.checkpoint as ocp
import numpy as np
import pandas as pd
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dl4bi.mlp
import flax.linen as nn
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
JITTER        = 2e-3

# SM kernel configurations — parameters chosen to produce genuinely
# oscillatory behaviour (ell*mu > 0.5 for at least one component).
#
# smooth : Q=1, approximates RBF.  Useful as a baseline / sanity check.
# diverse: Q=3, one smooth + two oscillatory components.
#            Component 1: broad envelope, low frequency  (ell*mu ~0.21)
#            Component 2: medium frequency — deep negative lobes  (ell*mu ~1.29)
#            Component 3: fast oscillation — visible ripple  (ell*mu ~0.49)
#
# Note: both configs use Q as written here.  Pass --q matching the config
# (--q 1 for smooth, --q 3 for diverse) — the assertion in run_single_config
# will catch mismatches.
SM_CONFIGS = {
    "smooth": {
        "weights":   [1.0],
        "means":     [[0.005, 0.005]],
        "variances": [[3e-5, 3e-5]],
        "label":     "SM smooth (Q=1)",
    },
    "diverse": {
        "weights":   [0.50, 0.30, 0.20],
        "means":     [[0.005, 0.005], [0.070, 0.070], [0.170, 0.170]],
        "variances": [[3e-5, 3e-5],   [1.5e-4, 1.5e-4], [6e-3, 6e-3]],
        "label":     "SM diverse (Q=3)",
    },
}


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SM-DeepRV simulation vs exact GP baseline (multi-Q / multi-grid)"
    )
    # Grid / model
    p.add_argument("--grid_size",   type=int,   default=16)
    p.add_argument("--q",           type=int,   default=1,
                   help="Number of SM components Q (default: 1)")
    p.add_argument("--sm_config",   type=str,   default=None,
                   choices=list(SM_CONFIGS.keys()))
    p.add_argument("--embed_dim",   type=int,   default=64)
    p.add_argument("--num_blks",    type=int,   default=2)
    # Training
    p.add_argument("--train_steps", type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--lr_schedule", type=str,   default="cosine",
                   choices=["cosine", "flat_cosine", "warmup_cosine", "cosine_restarts"],
                   help="LR schedule: cosine (default), flat_cosine (60%% flat "
                        "then 40%% cosine decay), warmup_cosine (10k warmup + "
                        "cosine), cosine_restarts (SGDR, restarts every 500k steps)")
    p.add_argument("--valid_interval", type=int, default=None)
    p.add_argument("--valid_steps", type=int,   default=5000)
    # Inference
    p.add_argument("--num_warmup",  type=int,   default=4000)
    p.add_argument("--num_samples", type=int,   default=6000)
    p.add_argument("--num_chains",  type=int,   default=None,
                   help="Number of NUTS chains (default: 1)")
    p.add_argument("--noise_std",   type=float, default=0.05)
    p.add_argument("--likelihood",  type=str,   default="gaussian",
                   choices=["gaussian", "lognormal", "studentt"])
    p.add_argument("--studentt_df", type=float, default=7.0)
    p.add_argument("--alpha_dirichlet", type=float, default=0.0,
                   help="Dirichlet concentration for mixture weights.  "
                        "0.0 = uniform (standard); < 1.0 (e.g. 0.5) = sparse "
                        "(soft learnable-Q effect); > 1.0 = anti-sparse.  "
                        "Only relevant for Q > 1.")
    p.add_argument("--partial_dense_mass", action="store_true")
    p.add_argument("--no_dense_mass",      action="store_true")
    # ADVI
    p.add_argument("--advi",         action="store_true")
    p.add_argument("--advi_steps",   type=int,   default=50_000)
    p.add_argument("--advi_lr",      type=float, default=1e-4)
    p.add_argument("--advi_samples", type=int,   default=6000)
    # Baselines
    p.add_argument("--no_exact_gp",  action="store_true")
    # Masking
    p.add_argument("--mask_type",     type=str,   default="contiguous",
                   choices=["contiguous", "random", "blob"])
    p.add_argument("--mask_fraction", type=float, default=0.5)
    # Misc
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--results_dir",  type=str,   default="results")
    p.add_argument("--ckpt_dir",     type=str,   default=None)
    # Training mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fresh_run", action="store_true",
                      help="Delete existing checkpoints and train from scratch")
    mode.add_argument("--load_ckpt", type=str,  default=None, metavar="CKPT_DIR")
    p.add_argument("--train_only", action="store_true")
    return p.parse_args()


def paper_train_steps(grid_size, override=None):
    if override is not None:
        return override
    return 500_000 if grid_size <= 32 else 700_000


def paper_lr(grid_size, override=None):
    if override is not None:
        return override
    return 1e-3 if grid_size <= 32 else 2e-3


def paper_num_chains(override=None):
    """Default: 1 chain.  Pass --num_chains to override."""
    return override if override is not None else 1


def build_lr_schedule(schedule_name, train_steps, lr):
    """Build an LR schedule by name.

    cosine         : standard cosine annealing from lr to 0 (original default)
    flat_cosine    : flat at lr for 60% of steps, then cosine decay for 40%
                     — keeps full LR during fast-descent phase
    warmup_cosine  : linear warmup for 10k steps, then cosine decay
                     — matches original DeepRV paper setup
    cosine_restarts: SGDR (Loshchilov & Hutter 2017), restarts every 500k steps
                     — helps escape shallow local minima
    """
    if schedule_name == "cosine":
        return cosine_annealing_lr(train_steps, lr)
    elif schedule_name == "flat_cosine":
        flat_steps   = int(0.6 * train_steps)
        cosine_steps = train_steps - flat_steps
        flat         = optax.constant_schedule(lr)
        cosine       = optax.cosine_decay_schedule(lr, cosine_steps)
        return optax.join_schedules([flat, cosine], [flat_steps])
    elif schedule_name == "warmup_cosine":
        warmup_steps = min(10_000, train_steps // 10)
        warmup       = optax.linear_schedule(0.0, lr, warmup_steps)
        cosine       = optax.cosine_decay_schedule(lr, train_steps - warmup_steps)
        return optax.join_schedules([warmup, cosine], [warmup_steps])
    elif schedule_name == "cosine_restarts":
        restart_steps = 500_000
        return optax.cosine_decay_schedule(lr, restart_steps, alpha=0.1)
    else:
        raise ValueError(f"Unknown lr_schedule: {schedule_name}")


# =============================================================================
# Spectral mixture kernel
# =============================================================================

@jit
def spectral_mixture(x, y, weights, means, variances):
    """Wilson & Adams (2013) SM kernel — corrected multidimensional form.
    k(τ) = Σ_q w_q cos(2πτᵀµ_q) Π_d exp(-2π²τ_d²v_q^(d))
    """
    x    = x.reshape(-1, x.shape[-1])
    y    = y.reshape(-1, y.shape[-1])
    diff = x[:, None, :] - y[None, :, :]

    def component(w, mu, v):
        env  = jnp.exp(-2 * jnp.pi**2 * jnp.sum(v * diff**2, axis=-1))
        freq = jnp.cos(2 * jnp.pi * jnp.sum(mu * diff, axis=-1))
        return w * env * freq

    return jnp.sum(vmap(component)(weights, means, variances), axis=0)


# =============================================================================
# Training dataloader
# =============================================================================

def sm_dataloader(s, Q, batch_size=BATCH_SIZE, alpha_dirichlet=0.0):
    """JIT-compiled SM kernel dataloader.

    Hyperparameter sampling — must match _sm_hyperparameter_priors exactly:
      weights   : Dirichlet(alpha) — uniform if alpha=0 (treated as 1.0)
      means     : Q=1: Uniform(0.0005, 0.30) per dimension independently
                  Q>1: ordered by x-direction (label-switching fix) +
                       independent y-direction
      variances : derived from ell ~ Uniform(1, 35) (same prior as inference)

    Conditionals: [weights, log(means).ravel(), log(vars_).ravel()]
    """
    N, D   = s.shape
    jitter = JITTER * jnp.eye(N)
    alpha  = alpha_dirichlet if alpha_dirichlet > 0.0 else 1.0

    @jit
    def generate_batch(rng_w, rng_mu, rng_v, rng_z):
        # --- Weights ---
        if Q == 1:
            weights = jnp.array([1.0])
        else:
            raw     = random.gamma(rng_w, a=alpha, shape=(Q,))
            weights = raw / raw.sum()

        # --- Means (with ordering constraint for Q > 1) ---
        if Q == 1:
            means = random.uniform(rng_mu, shape=(1, D),
                                   minval=0.0005, maxval=0.30)
        else:
            # Order by x-direction frequency to break label-switching.
            # First component's x-frequency in [0.0005, 0.30/Q];
            # each subsequent component is strictly larger.
            rng_base, rng_delta, rng_y = random.split(rng_mu, 3)
            mu_x_base  = random.uniform(rng_base, shape=(),
                                        minval=0.0005, maxval=0.30 / Q)
            delta_mu_x = random.uniform(rng_delta, shape=(Q - 1,),
                                        minval=0.0,
                                        maxval=(0.30 - 0.0005) / Q)
            means_x = jnp.concatenate(
                [mu_x_base[None], mu_x_base + jnp.cumsum(delta_mu_x)]
            )                                                   # (Q,)
            means_y = random.uniform(rng_y, shape=(Q,),
                                     minval=0.0005, maxval=0.30)
            means = jnp.stack([means_x, means_y], axis=-1)    # (Q, D)

        # --- Variances via component-specific lengthscale ---
        # Bounds decrease linearly from 50 (lowest-µ component) to 10
        # (highest-µ component), preventing ill-conditioned high-µ + high-ell
        # combinations while covering the diverse config's true ell values.
        ell_maxes   = np.linspace(50.0, 10.0, Q)          # concrete Python floats
        rng_v_splits = random.split(rng_v, Q)
        ell = jnp.stack([
            random.uniform(rng_v_splits[q], shape=(D,),
                           minval=1.0, maxval=float(ell_maxes[q]))
            for q in range(Q)
        ], axis=0)                                         # (Q, D)
        vars_ = 1.0 / (2.0 * jnp.pi**2 * ell**2)

        K = spectral_mixture(s, s, weights, means, vars_) + jitter
        L = jnp.linalg.cholesky(K)
        z = random.normal(rng_z, shape=(batch_size, N))
        f = jnp.einsum("ij,bj->bi", L, z)

        conditionals = jnp.concatenate([
            weights,
            jnp.log(means).ravel(),
            jnp.log(vars_).ravel(),
        ])
        return z, f, conditionals, K

    def dataloader(rng):
        while True:
            rng, rng_w, rng_mu, rng_v, rng_z = random.split(rng, 5)
            z, f, conditionals, K = generate_batch(rng_w, rng_mu, rng_v, rng_z)
            yield {"s": s, "z": z, "conditionals": conditionals, "f": f, "K": K}

    return dataloader


# =============================================================================
# Masking
# =============================================================================

def contiguous_mask(grid_size, mask_fraction=0.5, rng_key=None):
    N        = grid_size ** 2
    n_masked = int(N * mask_fraction)
    rect_h   = min(int(np.sqrt(n_masked)), grid_size)
    rect_w   = min(int(np.ceil(n_masked / rect_h)), grid_size)
    seed     = int(jax.random.bits(rng_key)) if rng_key is not None else 0
    rng_np   = np.random.default_rng(seed)
    row0     = rng_np.integers(0, max(grid_size - rect_h, 0) + 1)
    col0     = rng_np.integers(0, max(grid_size - rect_w, 0) + 1)
    mask     = np.zeros((grid_size, grid_size), dtype=bool)
    mask[row0:row0 + rect_h, col0:col0 + rect_w] = True
    flat     = mask.ravel()
    tst_idx  = jnp.array(np.where( flat)[0])
    obs_idx  = jnp.array(np.where(~flat)[0])
    log.info(f"Mask: {rect_h}×{rect_w} at ({row0},{col0}) — "
             f"{len(tst_idx)}/{N} held out ({100*len(tst_idx)/N:.1f}%)")
    return obs_idx, tst_idx


def random_mask(grid_size, rng_key=None):
    N      = grid_size ** 2
    seed   = int(jax.random.bits(rng_key)) if rng_key is not None else 0
    perm   = np.random.default_rng(seed).permutation(N)
    tst_idx= jnp.array(np.sort(perm[:N // 2]))
    obs_idx= jnp.array(np.sort(perm[N // 2:]))
    log.info(f"Random mask: {len(tst_idx)}/{N} held out")
    return obs_idx, tst_idx


def blob_mask(grid_size, mask_fraction=0.5, rng_key=None):
    H = W       = grid_size
    obs_ratio   = 1.0 - mask_fraction
    total_points= H * W
    num_obs     = int(obs_ratio * total_points)
    seed        = int(jax.random.bits(rng_key)) if rng_key is not None else 0
    rng_np      = np.random.default_rng(seed)
    mask        = np.zeros((H, W), dtype=bool)
    yy, xx      = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    points_collected, n_blobs = 0, 0
    while points_collected < num_obs:
        center_x = rng_np.integers(0, H)
        center_y = rng_np.integers(0, W)
        radius_x = rng_np.integers(H // 8, H // 4 + 1)
        radius_y = rng_np.integers(W // 8, W // 4 + 1)
        ellipse  = (((xx - center_x) / radius_x) ** 2 +
                    ((yy - center_y) / radius_y) ** 2) <= 1.0
        new_mask = np.logical_or(mask, ellipse)
        points_collected += int(new_mask.sum() - mask.sum())
        mask = new_mask
        n_blobs += 1
    if points_collected > num_obs:
        flat_idxs  = np.flatnonzero(mask.ravel())
        selected   = rng_np.choice(flat_idxs, size=num_obs, replace=False)
        final_mask = np.zeros(total_points, dtype=bool)
        final_mask[selected] = True
    else:
        final_mask = mask.ravel()
    obs_idx = jnp.array(np.sort(np.where( final_mask)[0]))
    tst_idx = jnp.array(np.sort(np.where(~final_mask)[0]))
    log.info(f"Blob mask: {n_blobs} ellipse(s) — "
             f"{len(tst_idx)}/{total_points} held out "
             f"({100*len(tst_idx)/total_points:.1f}%)")
    return obs_idx, tst_idx


# =============================================================================
# Data generation
# =============================================================================

def generate_observations(s_all, grid_size, sm_config, noise_std, rng,
                           mask_type="contiguous", mask_fraction=0.5):
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
    if mask_type == "contiguous":
        obs_idx, tst_idx = contiguous_mask(grid_size, mask_fraction, rng_mask)
    elif mask_type == "blob":
        obs_idx, tst_idx = blob_mask(grid_size, mask_fraction, rng_mask)
    else:
        obs_idx, tst_idx = random_mask(grid_size, rng_mask)
    y_obs      = f_all[obs_idx] + noise_std * random.normal(rng_y, shape=(len(obs_idx),))
    y_mean     = float(jnp.mean(y_obs))
    y_std      = float(jnp.std(y_obs)) + 1e-6
    y_obs_norm = (y_obs - y_mean) / y_std
    true_cond  = jnp.concatenate([weights, means.ravel(), variances.ravel()])
    return y_obs_norm, y_mean, y_std, f_all, true_cond, obs_idx, tst_idx


# =============================================================================
# Observation likelihood helper
# =============================================================================

def _observe(name, f_obs, sigma, y_obs, likelihood, studentt_df=7.0):
    if likelihood == "gaussian":
        numpyro.sample(name, dist.Normal(f_obs, sigma), obs=y_obs)
    elif likelihood == "lognormal":
        numpyro.sample(name, dist.LogNormal(f_obs, sigma), obs=y_obs)
    elif likelihood == "studentt":
        numpyro.sample(name, dist.StudentT(studentt_df, f_obs, sigma), obs=y_obs)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")


# =============================================================================
# Marginal likelihood weight optimisation (Q > 1 only)
# =============================================================================

def optimise_weights(y_obs_norm, s_all, obs_idx, Q, noise_std, y_std,
                     n_steps=500, n_mc=8, lr=1e-2, rng=None):
    """Optimise SM mixture weights via expected log marginal likelihood."""
    N_all, D = s_all.shape
    sigma2   = (noise_std / y_std) ** 2
    y        = jnp.array(y_obs_norm)
    N_obs    = len(obs_idx)
    if rng is None:
        rng = random.key(0)
    logits   = jnp.zeros(Q)

    @jit
    def log_marg_lik(logits, rng_mc):
        weights = jax.nn.softmax(logits)
        total   = 0.0
        for b in range(n_mc):
            rng_mc, rng_mu, rng_v = random.split(rng_mc, 3)
            means = jnp.exp(random.normal(rng_mu, shape=(Q, D))) / 100.0
            g     = random.gamma(rng_v, a=2.0, shape=(Q, D))
            vars_ = 1.0 / g / 1_000.0
            K_all = spectral_mixture(s_all, s_all, weights, means, vars_)
            K_obs = K_all[jnp.ix_(obs_idx, obs_idx)] \
                    + (sigma2 + JITTER) * jnp.eye(N_obs)
            L     = jnp.linalg.cholesky(K_obs)
            alpha = jax.scipy.linalg.cho_solve((L, True), y)
            lml   = (-0.5 * jnp.dot(y, alpha)
                     - jnp.sum(jnp.log(jnp.diag(L)))
                     - 0.5 * N_obs * jnp.log(2 * jnp.pi))
            total = total + lml
        return total / n_mc

    grad_fn   = jax.value_and_grad(log_marg_lik)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(logits)
    log.info(f"Optimising SM weights ({n_steps} steps, {n_mc} MC samples/step)…")
    for step in range(n_steps):
        rng, rng_mc = random.split(rng)
        val, grads  = grad_fn(logits, rng_mc)
        updates, opt_state = optimizer.update(grads, opt_state)
        logits = optax.apply_updates(logits, updates)
        if step % 100 == 0:
            w = jax.nn.softmax(logits)
            log.info(f"  step {step:4d}  lml={float(val):.3f}  "
                     f"weights={[f'{float(wi):.3f}' for wi in w]}")
    opt_weights = jax.nn.softmax(logits)
    log.info(f"Optimised weights: {[f'{float(w):.4f}' for w in opt_weights]}")
    return opt_weights


# =============================================================================
# Shared NUTS model — hyperparameter priors
# =============================================================================

def _sm_hyperparameter_priors(Q, D, alpha_dirichlet=0.0, fixed_weights=None):
    """Sample SM hyperparameters for inference on the [0,100]² grid.

    Matches sm_dataloader exactly so the decoder is never called
    out-of-distribution.

    For Q > 1: label-switching fix via ascending x-direction ordering.
    alpha_dirichlet > 0: sparse Dirichlet on weights (soft learnable-Q).
    """
    alpha = alpha_dirichlet if alpha_dirichlet > 0.0 else 1.0

    # --- Weights ---
    if fixed_weights is not None:
        weights = fixed_weights
        numpyro.deterministic("weights", weights)
    elif Q == 1:
        weights = jnp.array([1.0])
        numpyro.deterministic("weights", weights)
    else:
        raw     = numpyro.sample("raw_w",
                                 dist.Gamma(alpha * jnp.ones(Q), jnp.ones(Q)))
        weights = raw / raw.sum()
        numpyro.deterministic("weights", weights)

    # --- Means ---
    if Q == 1:
        means = numpyro.sample(
            "means",
            dist.Uniform(0.0005 * jnp.ones((1, D)), 0.30 * jnp.ones((1, D)))
        )
    else:
        # Ascending x-direction ordering to collapse Q! label-switching modes.
        # Base x-frequency in lower part of range; positive increments thereafter.
        mu_x_base = numpyro.sample(
            "mu_x_base",
            dist.Uniform(0.0005, 0.30 / Q)
        )
        delta_mu_x = numpyro.sample(
            "delta_mu_x",
            dist.Uniform(jnp.zeros(Q - 1),
                         (0.30 - 0.0005) / Q * jnp.ones(Q - 1))
        )
        means_x = jnp.concatenate(
            [mu_x_base[None], mu_x_base + jnp.cumsum(delta_mu_x)]
        )                                                        # (Q,)
        means_y = numpyro.sample(
            "means_y",
            dist.Uniform(0.0005 * jnp.ones(Q), 0.30 * jnp.ones(Q))
        )
        means = jnp.stack([means_x, means_y], axis=-1)          # (Q, D)
        numpyro.deterministic("means", means)

    # --- Variances via component-specific lengthscale (matches dataloader) ---
    ell_maxes = np.linspace(50.0, 10.0, Q)                # concrete Python floats
    ell = jnp.stack([
        numpyro.sample(f"ell_{q}", dist.Uniform(
            1.0 * jnp.ones(D),
            float(ell_maxes[q]) * jnp.ones(D),
        ))
        for q in range(Q)
    ], axis=0)                                             # (Q, D)
    vars_ = 1.0 / (2.0 * jnp.pi**2 * ell**2)
    numpyro.deterministic("variances", vars_)
    numpyro.deterministic("vars_det", vars_)

    cond = jnp.concatenate([
        weights,
        jnp.log(means).ravel(),
        jnp.log(vars_).ravel(),
    ])
    return weights, means, vars_, cond


# =============================================================================
# SM-DeepRV NUTS inference
# =============================================================================

def deeprv_nuts_inference(decoder, y_obs_norm, y_mean, y_std,
                          s_all, obs_idx, tst_idx,
                          Q, noise_std,
                          num_warmup, num_samples, num_chains, rng,
                          partial_dense_mass=False,
                          no_dense_mass=False,
                          likelihood="gaussian", studentt_df=7.0,
                          fixed_weights=None, alpha_dirichlet=0.0):
    N_all, D    = s_all.shape
    sigma_fixed = noise_std / y_std

    def model():
        weights, means, vars_, cond = _sm_hyperparameter_priors(
            Q, D, alpha_dirichlet=alpha_dirichlet, fixed_weights=fixed_weights)
        z = numpyro.sample("z", dist.Normal(jnp.zeros(N_all), jnp.ones(N_all)))
        f = decoder(z[None], cond, s=s_all).squeeze()
        _observe("y", f[obs_idx], sigma_fixed, y_obs_norm, likelihood, studentt_df)
        numpyro.deterministic("f_all", f)

    dense_mass = (False if no_dense_mass
                  else ([["means", "variances"]] if partial_dense_mass else True))
    t0   = time.time()
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.8, dense_mass=dense_mass,
             init_strategy=init_to_median(num_samples=10)),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel" if num_chains > 1 else "sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.fold_in(rng, 0), extra_fields=("energy", "diverging",
                                                        "accept_prob", "num_steps"))
    elapsed = time.time() - t0
    log.info(f"SM-DeepRV NUTS: {elapsed:.1f}s")
    mcmc.print_summary()

    samples = mcmc.get_samples()
    extra   = mcmc.get_extra_fields()
    f_samps = samples["f_all"][:, tst_idx] * y_std + y_mean
    return f_samps.mean(0), f_samps.std(0), samples, extra, elapsed


# =============================================================================
# Exact SM-GP NUTS inference
# =============================================================================

def exact_gp_nuts_inference(y_obs_norm, y_mean, y_std,
                             s_all, obs_idx, tst_idx,
                             Q, noise_std,
                             num_warmup, num_samples, num_chains, rng,
                             partial_dense_mass=False,
                             no_dense_mass=False,
                             likelihood="gaussian", studentt_df=7.0,
                             fixed_weights=None, alpha_dirichlet=0.0):
    N_all, D    = s_all.shape
    sigma_fixed = noise_std / y_std

    def model():
        weights, means, vars_, _ = _sm_hyperparameter_priors(
            Q, D, alpha_dirichlet=alpha_dirichlet, fixed_weights=fixed_weights)
        K = spectral_mixture(s_all, s_all, weights, means, vars_) \
            + JITTER * jnp.eye(N_all)
        L = jnp.linalg.cholesky(K)
        z = numpyro.sample("z", dist.Normal(jnp.zeros(N_all), jnp.ones(N_all)))
        f = L @ z
        _observe("y", f[obs_idx], sigma_fixed, y_obs_norm, likelihood, studentt_df)
        numpyro.deterministic("f_all", f)

    dense_mass = (False if no_dense_mass
                  else ([["means", "variances"]] if partial_dense_mass else True))
    t0   = time.time()
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.8, dense_mass=dense_mass,
             init_strategy=init_to_median(num_samples=10)),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel" if num_chains > 1 else "sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.fold_in(rng, 1), extra_fields=("energy", "diverging",
                                                        "accept_prob", "num_steps"))
    elapsed = time.time() - t0
    log.info(f"Exact GP NUTS: {elapsed:.1f}s")
    mcmc.print_summary()

    samples = mcmc.get_samples()
    extra   = mcmc.get_extra_fields()
    f_samps = samples["f_all"][:, tst_idx] * y_std + y_mean
    return f_samps.mean(0), f_samps.std(0), samples, extra, elapsed


# =============================================================================
# ADVI inference
# =============================================================================

def advi_inference(decoder, y_obs_norm, y_mean, y_std,
                   s_all, obs_idx, tst_idx, Q, noise_std,
                   num_steps, lr, num_samples, rng, alpha_dirichlet=0.0):
    N_all, D    = s_all.shape
    sigma_fixed = noise_std / y_std

    def model():
        weights, means, vars_, cond = _sm_hyperparameter_priors(
            Q, D, alpha_dirichlet=alpha_dirichlet)
        z = numpyro.sample("z", dist.Normal(jnp.zeros(N_all), jnp.ones(N_all)))
        f = decoder(z[None], cond, s=s_all).squeeze()
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma_fixed), obs=y_obs_norm)
        numpyro.deterministic("f_all", f)

    guide      = AutoMultivariateNormal(model)
    optimizer  = numpyro.optim.Adam(step_size=lr)
    svi        = SVI(model, guide, optimizer, loss=Trace_ELBO())
    t0         = time.time()
    svi_result = svi.run(jax.random.fold_in(rng, 2), num_steps, progress_bar=True)
    elapsed    = time.time() - t0
    log.info(f"ADVI: {elapsed:.1f}s  final ELBO={float(-svi_result.losses[-1]):.2f}")

    samples    = Predictive(model, guide=guide, params=svi_result.params,
                            num_samples=num_samples)(jax.random.fold_in(rng, 3))
    f_samps    = samples["f_all"][:, tst_idx] * y_std + y_mean
    return f_samps.mean(0), f_samps.std(0), samples, elapsed, svi_result.losses


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
# Plots
# =============================================================================

def _scatter(ax, s, vals, title, cmap="RdYlBu_r", vmin=None, vmax=None):
    sc = ax.scatter(s[:, 0], s[:, 1], c=np.array(vals),
                    cmap=cmap, vmin=vmin, vmax=vmax, s=14)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def plot_spatial(s_all, f_all, y_obs,
                 obs_idx, tst_idx,
                 drv_mean, drv_std, egp_mean, egp_std,
                 config_label, grid_size, out_dir, tag):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    s_obs  = s_all[obs_idx]; s_test = s_all[tst_idx]; f_test = f_all[tst_idx]
    fv = (float(f_all.min()), float(f_all.max()))
    ev = float(max(np.abs(drv_mean - f_test).max(),
                   np.abs(egp_mean - f_test).max() if egp_mean is not None else 0))
    sv = float(max(drv_std.max(),
                   egp_std.max() if egp_std is not None else 0))
    _scatter(axes[0,0], s_all, f_all, f"True f ({config_label})", vmin=fv[0], vmax=fv[1])
    axes[0,0].scatter(s_obs[:,0], s_obs[:,1], c="k", s=5, alpha=0.3, label="Observed")
    axes[0,0].legend(fontsize=7)
    _scatter(axes[0,1], s_obs, np.array(y_obs), "Observed y", vmin=fv[0], vmax=fv[1])
    _scatter(axes[0,2], s_test, drv_mean, "SM-DeepRV  E[f|y]", vmin=fv[0], vmax=fv[1])
    _scatter(axes[0,3], s_test, drv_std,  "SM-DeepRV  Std[f|y]", cmap="Blues", vmin=0, vmax=sv)
    _scatter(axes[1,0], s_test, np.array(drv_mean)-np.array(f_test),
             "SM-DeepRV error", cmap="RdBu_r", vmin=-ev, vmax=ev)
    if egp_mean is not None:
        _scatter(axes[1,1], s_test, np.array(egp_mean)-np.array(f_test),
                 "Exact GP error", cmap="RdBu_r", vmin=-ev, vmax=ev)
        _scatter(axes[1,2], s_test, egp_mean, "Exact GP  E[f|y]", vmin=fv[0], vmax=fv[1])
        _scatter(axes[1,3], s_test, egp_std,  "Exact GP  Std[f|y]", cmap="Blues", vmin=0, vmax=sv)
    else:
        for ax in axes[1, 1:]:
            ax.set_visible(False)
    plt.suptitle(f"SM-DeepRV vs Exact GP — {config_label}  |  grid={grid_size}²", fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / f"spatial_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Spatial plot → {path}")


def plot_metrics_comparison(drv_metrics, egp_metrics, advi_metrics,
                             config_label, drv_time, egp_time, advi_time,
                             out_dir, tag):
    methods  = ["SM-DeepRV", "Exact GP", "ADVI"]
    metrics  = [drv_metrics, egp_metrics or {}, advi_metrics or {}]
    times    = [drv_time,
                egp_time  if egp_time  is not None else float("nan"),
                advi_time if advi_time is not None else float("nan")]
    colours  = ["#2196F3", "#FF5722", "#4CAF50"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, metric in enumerate(["RMSE", "MAE", "Coverage_95", "Mean_std"]):
        vals = [m.get(metric, float("nan")) for m in metrics]
        bars = axes[i].bar(methods, vals, color=colours, alpha=0.8,
                           edgecolor="k", linewidth=0.5)
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            axes[i].set_ylim(0, max(valid) * 1.3 + 1e-6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.01*axes[i].get_ylim()[1],
                             f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        if metric == "Coverage_95":
            axes[i].axhline(0.95, color="green", linestyle="--", linewidth=1,
                            label="Nominal 95%")
            axes[i].legend(fontsize=7)
        axes[i].set_title(metric, fontsize=11)
    valid_t = [t for t in times if not np.isnan(t)]
    bars = axes[4].bar(methods, times, color=colours, alpha=0.8,
                       edgecolor="k", linewidth=0.5)
    axes[4].set_title("Inference time (s)", fontsize=11)
    for bar, val in zip(bars, times):
        if not np.isnan(val):
            axes[4].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.01*(max(valid_t) or 1),
                         f"{val:.0f}s", ha="center", va="bottom", fontsize=8)
    plt.suptitle(f"Metrics — {config_label}", fontsize=12)
    plt.tight_layout()
    path = Path(out_dir) / f"metrics_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Metrics plot → {path}")


def plot_posterior_hyperparams(drv_samples, egp_samples, Q,
                                sm_config, out_dir, tag):
    fig, axes = plt.subplots(1, Q, figsize=(5*Q, 4))
    if Q == 1:
        axes = [axes]
    colours = ["#2196F3", "#FF5722"]
    for q in range(Q):
        ax = axes[q]
        ax.hist(np.array(drv_samples["weights"])[:, q], bins=40,
                alpha=0.6, color=colours[0], label="SM-DeepRV", density=True)
        if egp_samples is not None:
            ax.hist(np.array(egp_samples["weights"])[:, q], bins=40,
                    alpha=0.6, color=colours[1], label="Exact GP", density=True)
        ax.axvline(sm_config["weights"][q], color="k", linestyle="--",
                   linewidth=1.5, label=f"True = {sm_config['weights'][q]:.3f}")
        ax.set_title(f"Weight w_{q+1}", fontsize=11)
        ax.set_xlabel("Value"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Posterior SM mixture weights", fontsize=12)
    plt.tight_layout()
    path = Path(out_dir) / f"hyperparams_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Hyperparams plot → {path}")


def plot_scatter_pred_vs_true(drv_mean, egp_mean, advi_mean, f_test,
                               config_label, out_dir, tag):
    active = [(n, m, c) for n, m, c in [
        ("SM-DeepRV", drv_mean,  "#2196F3"),
        ("Exact GP",  egp_mean,  "#FF5722"),
        ("ADVI",      advi_mean, "#4CAF50"),
    ] if m is not None]
    fig, axes = plt.subplots(1, len(active), figsize=(6*len(active), 5))
    if len(active) == 1:
        axes = [axes]
    f_np = np.array(f_test)
    lims = (f_np.min()-0.1, f_np.max()+0.1)
    for ax, (title, mean, col) in zip(axes, active):
        m_np = np.array(mean)
        ax.scatter(f_np, m_np, alpha=0.5, s=15, color=col)
        ax.plot(lims, lims, "k--", linewidth=1, label="y=x")
        rmse = float(np.sqrt(np.mean((m_np-f_np)**2)))
        ax.set_title(f"{title}  (RMSE={rmse:.3f})", fontsize=11)
        ax.set_xlabel("True f"); ax.set_ylabel("Predicted f")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=8); ax.set_aspect("equal")
    plt.suptitle(f"Predicted vs True — {config_label}", fontsize=12)
    plt.tight_layout()
    path = Path(out_dir) / f"pred_vs_true_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Pred-vs-true plot → {path}")


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
    model.init({"params": dummy_rng, "extra": dummy_rng}, **dummy_batch)
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
    Q = args.q
    log.info(f"\n--- SM config: {config_name} ({sm_config['label']})  Q={Q} ---")
    assert Q == len(sm_config["weights"]), (
        f"--q {Q} does not match config '{config_name}' which has "
        f"{len(sm_config['weights'])} component(s). "
        f"Pass --q {len(sm_config['weights'])} or choose a different --sm_config."
    )
    rng, rng_data, rng_drv, rng_egp, rng_advi = random.split(rng, 5)

    y_obs_norm, y_mean, y_std, f_all, true_cond, obs_idx, tst_idx = \
        generate_observations(s_all, args.grid_size, sm_config,
                              args.noise_std, rng_data,
                              mask_type=args.mask_type,
                              mask_fraction=args.mask_fraction)
    f_test = f_all[tst_idx]
    log.info(f"Observed: {len(obs_idx)}  Held-out: {len(tst_idx)}  "
             f"y_mean={y_mean:.4f}  y_std={y_std:.4f}")

    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [("y_obs_norm", y_obs_norm), ("f_all", f_all),
                      ("true_cond", true_cond),
                      ("obs_idx", obs_idx), ("tst_idx", tst_idx)]:
        np.save(cfg_dir / f"{name}.npy", np.array(arr))
    np.save(cfg_dir / "y_scale.npy", np.array([y_mean, y_std]))

    # Weights: fixed to 1 for Q=1; inferred by sparse Dirichlet for Q>1
    rng, rng_wopt = random.split(rng)
    if Q == 1:
        fixed_weights = jnp.array([1.0])
        log.info("Q=1: weight fixed to 1.0.")
    else:
        fixed_weights = None
        log.info(f"Q={Q}: weights inferred (alpha_dirichlet={args.alpha_dirichlet:.2f}).")
    np.save(cfg_dir / "fixed_weights.npy",
            np.array(fixed_weights if fixed_weights is not None
                     else sm_config["weights"]))

    # --- SM-DeepRV NUTS ---
    log.info(f"Running SM-DeepRV NUTS ({num_chains} chain(s)) …")
    drv_mean, drv_std, drv_samples, drv_extra, drv_time = deeprv_nuts_inference(
        decoder, y_obs_norm, y_mean, y_std,
        s_all, obs_idx, tst_idx,
        Q, args.noise_std,
        args.num_warmup, args.num_samples, num_chains, rng_drv,
        partial_dense_mass=args.partial_dense_mass,
        no_dense_mass=args.no_dense_mass,
        likelihood=args.likelihood, studentt_df=args.studentt_df,
        fixed_weights=fixed_weights,
        alpha_dirichlet=args.alpha_dirichlet,
    )
    for name, arr in [("drv_mean", drv_mean), ("drv_std", drv_std)]:
        np.save(cfg_dir / f"{name}.npy", np.array(arr))
    for name, arr in drv_samples.items():       # save ALL sites (f_all, z included)
        np.save(cfg_dir / f"drv_samples_{name}.npy", np.array(arr))
    for key, fname in [("energy",       "drv_energy.npy"),
                       ("diverging",    "drv_diverging.npy"),
                       ("accept_prob",  "drv_accept_prob.npy"),
                       ("num_steps",    "drv_num_steps.npy")]:
        if key in drv_extra:
            np.save(cfg_dir / fname, np.array(drv_extra[key]))
    drv_metrics = evaluate(drv_mean, drv_std, f_test)
    log.info(f"SM-DeepRV — RMSE={drv_metrics['RMSE']:.4f}  "
             f"Coverage={drv_metrics['Coverage_95']:.3f}  Time={drv_time:.1f}s")

    # --- Exact GP NUTS ---
    egp_mean = egp_std = egp_samples = egp_time = None
    egp_metrics = {}
    if not args.no_exact_gp:
        log.info(f"Running Exact GP NUTS ({num_chains} chain(s)) …")
        egp_mean, egp_std, egp_samples, egp_extra, egp_time = exact_gp_nuts_inference(
            y_obs_norm, y_mean, y_std,
            s_all, obs_idx, tst_idx,
            Q, args.noise_std,
            args.num_warmup, args.num_samples, num_chains, rng_egp,
            partial_dense_mass=args.partial_dense_mass,
            no_dense_mass=args.no_dense_mass,
            likelihood=args.likelihood, studentt_df=args.studentt_df,
            fixed_weights=fixed_weights,
            alpha_dirichlet=args.alpha_dirichlet,
        )
        for name, arr in [("egp_mean", egp_mean), ("egp_std", egp_std)]:
            np.save(cfg_dir / f"{name}.npy", np.array(arr))
        for name, arr in egp_samples.items():
            np.save(cfg_dir / f"egp_samples_{name}.npy", np.array(arr))
        for key, fname in [("energy",       "egp_energy.npy"),
                           ("diverging",    "egp_diverging.npy"),
                           ("accept_prob",  "egp_accept_prob.npy"),
                           ("num_steps",    "egp_num_steps.npy")]:
            if key in egp_extra:
                np.save(cfg_dir / fname, np.array(egp_extra[key]))
        egp_metrics = evaluate(egp_mean, egp_std, f_test)
        log.info(f"Exact GP  — RMSE={egp_metrics['RMSE']:.4f}  "
                 f"Coverage={egp_metrics['Coverage_95']:.3f}  Time={egp_time:.1f}s")

    # --- ADVI ---
    advi_mean = advi_std = advi_time = None
    advi_metrics = {}
    if args.advi:
        log.info(f"Running ADVI ({args.advi_steps} steps, lr={args.advi_lr}) …")
        advi_mean, advi_std, advi_samples, advi_time, advi_losses = advi_inference(
            decoder, y_obs_norm, y_mean, y_std,
            s_all, obs_idx, tst_idx, Q, args.noise_std,
            args.advi_steps, args.advi_lr, args.advi_samples, rng_advi,
            alpha_dirichlet=args.alpha_dirichlet,
        )
        for name, arr in [("advi_mean", advi_mean), ("advi_std", advi_std),
                          ("advi_losses", advi_losses)]:
            np.save(cfg_dir / f"{name}.npy", np.array(arr))
        advi_metrics = evaluate(advi_mean, advi_std, f_test)
        log.info(f"ADVI      — RMSE={advi_metrics['RMSE']:.4f}  "
                 f"Coverage={advi_metrics['Coverage_95']:.3f}  Time={advi_time:.1f}s")

    # --- Plots ---
    y_obs_plot = y_obs_norm * y_std + y_mean
    plot_spatial(s_all, f_all, y_obs_plot,
                 obs_idx, tst_idx, drv_mean, drv_std, egp_mean, egp_std,
                 sm_config["label"], args.grid_size, cfg_dir, config_name)
    plot_metrics_comparison(drv_metrics, egp_metrics or None,
                            advi_metrics or None, sm_config["label"],
                            drv_time, egp_time, advi_time,
                            cfg_dir, config_name)
    plot_posterior_hyperparams(drv_samples,
                               egp_samples if egp_samples else None,
                               Q, sm_config, cfg_dir, config_name)
    plot_scatter_pred_vs_true(drv_mean, egp_mean, advi_mean, f_test,
                              sm_config["label"], cfg_dir, config_name)

    row = {"config": config_name, "grid_size": args.grid_size, "Q": Q,
           "drv_time_s": drv_time, "egp_time_s": egp_time, "advi_time_s": advi_time}
    for k, v in drv_metrics.items():  row[f"drv_{k}"] = v
    for k, v in egp_metrics.items():  row[f"egp_{k}"] = v
    for k, v in advi_metrics.items(): row[f"advi_{k}"] = v
    return row


# =============================================================================
# Custom training step with covariance matching loss
# =============================================================================

def sm_train_step(rng, state, batch):
    """MSE + covariance matching loss.

    Adds an explicit covariance penalty to the standard DeepRV MSE loss,
    directly targeting variance collapse: the decoder's output covariance
    across the batch is penalised for deviating from the true kernel matrix K.

    lambda_cov = 0.1 is a reasonable starting point — drop to 0.01 if the
    loss spikes or destabilises early in training.
    """
    lambda_cov = 0.1

    def loss_fn(params):
        f      = batch["f"]                                # (B, N)
        output: VAEOutput = state.apply_fn(
            {"params": params, **state.kwargs},
            **batch, rngs={"extra": rng},
        )
        mse_loss = output.mse(f)

        # Covariance matching — penalises variance collapse.
        # f_hat shape is (B, 1, N) due to atleast_3d in VAEOutput;
        # reshape to (B, N) anchored on the known batch shape.
        K        = batch["K"]                              # (N, N)
        f_hat    = output.f_hat.reshape(f.shape)           # (B, N)
        B        = f.shape[0]
        cov_pred = jnp.einsum("bi,bj->ij", f_hat, f_hat) / B
        cov_loss = jnp.mean((cov_pred - K) ** 2)

        return mse_loss + lambda_cov * cov_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


# =============================================================================
# Main
# =============================================================================

def main():
    args        = parse_args()
    numpyro.set_host_device_count(1)

    Q           = args.q
    train_steps = paper_train_steps(args.grid_size, args.train_steps)
    lr          = paper_lr(args.grid_size, args.lr)
    num_chains  = paper_num_chains(args.num_chains)
    configs     = ({args.sm_config: SM_CONFIGS[args.sm_config]}
                   if args.sm_config
                   else SM_CONFIGS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"sim_{timestamp}_grid{args.grid_size}_q{Q}"
    out_dir   = Path(args.results_dir) / run_name

    # Checkpoint dir is unique per (grid_size, Q, lr_schedule, embed_dim, num_blks)
    ckpt_dir  = (args.load_ckpt or args.ckpt_dir or str(
        Path(args.results_dir) /
        f"sim_checkpoints_grid{args.grid_size}_q{Q}"
        f"_blks{args.num_blks}_emb{args.embed_dim}_sched{args.lr_schedule}"
    ))

    if args.fresh_run and Path(ckpt_dir).exists():
        log.info(f"--fresh_run: deleting checkpoints at {ckpt_dir}")
        shutil.rmtree(ckpt_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== SM-DeepRV Simulation ===")
    log.info(f"Grid: {args.grid_size}²  Q={Q}  "
             f"Steps={train_steps}  LR={lr}")
    log.info(f"alpha_dirichlet={args.alpha_dirichlet}  "
             f"(sparse weights: {'yes' if 0 < args.alpha_dirichlet < 1 else 'no'})")
    log.info(f"NUTS: {args.num_warmup} warmup / {args.num_samples} samples / "
             f"{num_chains} chain(s)")
    log.info(f"Baselines: Exact GP={'OFF' if args.no_exact_gp else 'ON'}  "
             f"ADVI={'ON' if args.advi else 'OFF'}")
    log.info(f"Mode: {'load only' if args.load_ckpt else 'fresh' if args.fresh_run else 'resume/train'}")
    log.info(f"Checkpoint dir: {ckpt_dir}")
    log.info(f"Results → {out_dir}")

    rng = random.key(args.seed)
    rng, rng_train, rng_exp = random.split(rng, 3)

    s_all  = build_grid(
        [{"start": 0.0, "stop": 100.0, "num": args.grid_size}] * 2
    ).reshape(-1, 2)

    loader = sm_dataloader(s_all, Q, BATCH_SIZE, args.alpha_dirichlet)

    # --- Train ---
    log.info("\n=== Training SM-DeepRV ===")
    wandb.init(project="sm_deeprv",
               entity="emrys-king25-imperial-college-london",
               name=run_name,
               config={
                   "grid_size": args.grid_size, "train_steps": train_steps,
                   "lr": lr, "lr_schedule": args.lr_schedule,
                   "Q": Q, "num_blks": args.num_blks,
                   "embed_dim": args.embed_dim, "kernel": "spectral_mixture",
                   "likelihood": args.likelihood, "noise_std": args.noise_std,
                   "alpha_dirichlet": args.alpha_dirichlet,
               })
    wandb.define_metric("Train Loss",      step_metric="_step")
    wandb.define_metric("Valid norm MSE",  step_metric="_step")

    nn_model  = gMLPDeepRV(
        num_blks = args.num_blks,
        embed    = dl4bi.mlp.MLP([args.embed_dim, args.embed_dim], nn.gelu),
        proj_out = dl4bi.mlp.MLP([args.embed_dim, args.embed_dim], nn.gelu),
    )
    log.info(f"Architecture: gMLPDeepRV(num_blks={args.num_blks}, "
             f"embed_dim={args.embed_dim})")
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(build_lr_schedule(args.lr_schedule, train_steps, lr),
                    weight_decay=1e-2),
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
        if saved_state is None:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}.")
        state = saved_state
        log.info(f"Loaded from {ckpt_dir} (step {start_step}) — skipping training.")
    elif remaining > 0:
        valid_interval = args.valid_interval or max(train_steps // 4, 1)

        # Rebuild optimizer over remaining steps on resume (smoother LR continuation)
        if start_step > 0:
            log.info(f"Resuming from step {start_step}: rebuilding LR schedule "
                     f"({args.lr_schedule}) over remaining {remaining} steps "
                     f"(peak lr={lr}).")
            optimizer = optax.chain(
                optax.clip_by_global_norm(3.0),
                optax.adamw(build_lr_schedule(args.lr_schedule, remaining, lr),
                            weight_decay=1e-2),
            )

        state = train(
            rng_train, nn_model, optimizer, sm_train_step,
            train_num_steps      = remaining,
            train_dataloader     = loader,
            valid_step           = valid_step,
            valid_interval       = valid_interval,
            valid_num_steps      = args.valid_steps,
            valid_dataloader     = loader,
            valid_monitor_metric = "norm MSE",
            callbacks            = [make_save_callback(ckpt_dir, valid_interval)],
            state                = saved_state,
            return_state         = "best",
        )
    else:
        state = saved_state
        log.info("Training already complete.")

    wandb.finish()

    if args.train_only:
        log.info(f"--train_only: skipping inference. Checkpoint at {ckpt_dir}")
        return

    decoder = generate_surrogate_decoder(state, nn_model)

    # --- Inference ---
    log.info("\n=== Inference ===")
    all_rows = []
    for config_name, sm_config in configs.items():
        rng_exp, rng_cfg = random.split(rng_exp)
        row = run_single_config(
            args, s_all, decoder, config_name, sm_config,
            out_dir, num_chains, rng_cfg,
        )
        all_rows.append(row)

    log.info("\n=== Summary ===")
    summary = pd.DataFrame(all_rows).set_index("config")
    print("\n" + summary.to_string())
    summary.to_csv(out_dir / "summary.csv")
    log.info(f"\nDone. All outputs → {out_dir}")


if __name__ == "__main__":
    main()