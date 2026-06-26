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
from jax.scipy.linalg import solve_triangular
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
    p.add_argument("--model",       type=str,   default="gmlp",
                   choices=["gmlp", "qgated"],
                   help="Decoder architecture: gmlp (default, single gMLPDeepRV) or "
                        "qgated (Q separate gMLPDeepRV branches, one per SM component). "
                        "qgated gives each component its own spatial gating, conditioned "
                        "only on that component's parameters.")
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
    # Fixed-theta inference (conditioning on hyperparameters)
    p.add_argument("--fixed_theta", action="store_true",
                   help="Condition on fixed θ during inference rather than sampling "
                        "jointly. Reduces NUTS to z-space only — unimodal, log-concave "
                        "posterior that mixes efficiently. Matches Wilson & Adams (2013) "
                        "setup. Use --sm_config for true θ (simulation) or "
                        "--optimise_theta for ML estimates.")
    p.add_argument("--optimise_theta", action="store_true",
                   help="Estimate θ via marginal likelihood optimisation before "
                        "fixed-theta inference (used with --fixed_theta). "
                        "If --fixed_theta is set without this flag, uses true θ "
                        "from --sm_config.")
    p.add_argument("--opt_steps",    type=int,   default=2000,
                   help="Gradient steps for marginal likelihood optimisation.")
    p.add_argument("--opt_lr",       type=float, default=0.01)
    p.add_argument("--opt_restarts", type=int,   default=5,
                   help="Random restarts for marginal likelihood optimisation.")
    p.add_argument("--no_analytic",  action="store_true",
                   help="Skip analytic GP (Wilson & Adams) in fixed-theta mode.")
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

def sm_dataloader(s, Q, batch_size=BATCH_SIZE, alpha_dirichlet=0.0,
                  mu_ig_params=None):
    """JIT-compiled SM kernel dataloader.

    Hyperparameter sampling — must match _sm_hyperparameter_priors exactly:
      weights   : Dirichlet(alpha) — uniform if alpha=0 (treated as 1.0)
      means     : sampled via Betancourt InverseGamma on period T=1/µ if
                  mu_ig_params=(a,b) is provided; else Uniform(0.0005, mu_max)
                  Q>1: ordered by x-direction (label-switching fix)
      variances : derived from component-specific ell bounds (np.linspace)

    Conditionals: [weights, log(means).ravel(), log(vars_).ravel()]
    """
    N, D      = s.shape
    jitter    = JITTER * jnp.eye(N)
    alpha     = alpha_dirichlet if alpha_dirichlet > 0.0 else 1.0
    # Nyquist clamping — used as fallback if no Betancourt params provided
    spacing   = 100.0 / (int(round(N ** 0.5)) - 1)
    mu_max    = float(0.95 / (2.0 * spacing))

    # Betancourt period prior parameters (concrete Python floats, safe in @jit)
    if mu_ig_params is not None:
        a_mu, b_mu = float(mu_ig_params[0]), float(mu_ig_params[1])
        use_betancourt = True
    else:
        use_betancourt = False

    @jit
    def generate_batch(rng_w, rng_mu, rng_v, rng_z):
        # --- Weights ---
        if Q == 1:
            weights = jnp.array([1.0])
        else:
            raw     = random.gamma(rng_w, a=alpha, shape=(Q,))
            weights = raw / raw.sum()

        # --- Means via Betancourt period prior or Nyquist-clamped Uniform ---
        if use_betancourt:
            # T = 1/µ ~ InverseGamma(a_mu, b_mu)  =>  µ = 1/T
            # InverseGamma(a,b) sampled as b / Gamma(a)
            if Q == 1:
                T    = b_mu / random.gamma(rng_mu, a=a_mu, shape=(1, D))
                means = 1.0 / T                                    # (1, D)
            else:
                rng_base, rng_delta, rng_y = random.split(rng_mu, 3)
                # Sample Q periods in descending order (ascending frequency)
                T_base   = b_mu / random.gamma(rng_base, a=a_mu, shape=())
                # Positive increments in 1/T space to maintain ascending µ
                delta_T  = b_mu / random.gamma(
                    rng_delta, a=a_mu, shape=(Q - 1,)) * 0.5
                T_x = jnp.concatenate(
                    [T_base[None], T_base - jnp.cumsum(delta_T)]
                )                                                   # (Q,) descending T
                T_x = jnp.clip(T_x, 1.0 / mu_max, 1e6)            # clamp to valid range
                means_x = 1.0 / T_x                               # (Q,) ascending µ
                T_y  = b_mu / random.gamma(rng_y, a=a_mu, shape=(Q,))
                means_y = jnp.clip(1.0 / T_y, 0.0005, mu_max)
                means   = jnp.stack([means_x, means_y], axis=-1)  # (Q, D)
        else:
            # Fallback: Nyquist-clamped Uniform (original approach)
            if Q == 1:
                means = random.uniform(rng_mu, shape=(1, D),
                                       minval=0.0005, maxval=mu_max)
            else:
                rng_base, rng_delta, rng_y = random.split(rng_mu, 3)
                mu_x_base  = random.uniform(rng_base, shape=(),
                                            minval=0.0005, maxval=mu_max / Q)
                delta_mu_x = random.uniform(rng_delta, shape=(Q - 1,),
                                            minval=0.0,
                                            maxval=(mu_max - 0.0005) / Q)
                means_x = jnp.concatenate(
                    [mu_x_base[None], mu_x_base + jnp.cumsum(delta_mu_x)]
                )
                means_y = random.uniform(rng_y, shape=(Q,),
                                         minval=0.0005, maxval=mu_max)
                means   = jnp.stack([means_x, means_y], axis=-1)

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
        return z, f, conditionals

    def dataloader(rng):
        while True:
            rng, rng_w, rng_mu, rng_v, rng_z = random.split(rng, 5)
            z, f, conditionals = generate_batch(rng_w, rng_mu, rng_v, rng_z)
            yield {"s": s, "z": z, "conditionals": conditionals, "f": f}

    return dataloader


def calibrate_period_prior(grid_size, domain=100.0, lower_q=0.05, upper_q=0.05):
    """Betancourt-calibrated InverseGamma prior on the spectral period T = 1/µ.

    Analogous to calibrated_ig_params() for lengthscales, but applied to the
    oscillation period.  The period should lie between:
      T_min = 2Δ  (Nyquist period — shortest representable oscillation)
      T_max = domain  (full domain — no more than one cycle visible)

    Finds InverseGamma(a, b) such that:
      P(T < T_min) = lower_q   (soft Nyquist upper bound on µ)
      P(T > T_max) = upper_q   (soft lower bound on µ)

    This is a principled alternative to hard Nyquist clamping: rather than
    truncating the prior at µ_nyq, it places diminishing but nonzero probability
    on above-Nyquist frequencies, with the calibration ensuring that most mass
    lies in the representable regime.  Frequencies are then µ = 1/T.

    Returns: (a, b) — InverseGamma parameters for the period distribution
    """
    spacing = domain / (grid_size - 1)
    T_min   = 2.0 * spacing          # Nyquist period
    T_max   = domain                  # full domain period

    def equations(log_ab):
        a = np.exp(log_ab[0])
        b = np.exp(log_ab[1])
        return [
            invgamma.cdf(T_min, a=a, scale=b) - lower_q,
            invgamma.cdf(T_max, a=a, scale=b) - (1.0 - upper_q),
        ]

    best_result, best_resid = None, np.inf
    for log_a0 in np.log([1.5, 3.0, 7.0]):
        for log_b0 in np.log([T_min, np.sqrt(T_min * T_max), T_max / 2]):
            try:
                res   = root(equations, [log_a0, log_b0], method="hybr")
                resid = np.max(np.abs(res.fun))
                if res.success and resid < best_resid:
                    best_resid  = resid
                    best_result = res
            except Exception:
                continue

    if best_result is not None and best_resid < 1e-6:
        a = float(np.exp(best_result.x[0]))
        b = float(np.exp(best_result.x[1]))
        mu_nyq   = 1.0 / T_min
        mu_lower = 1.0 / T_max
        log.info(f"Betancourt period prior: IG({a:.3f}, {b:.3f})  "
                 f"[T_min={T_min:.3f}, T_max={T_max:.1f}]  "
                 f"=> µ in [{mu_lower:.5f}, {mu_nyq:.4f}]  "
                 f"residual={best_resid:.2e}")
    else:
        # Fallback: use a weakly informative prior centred in the valid range
        a = 3.0
        b = float(np.sqrt(T_min * T_max))
        log.warning(f"Betancourt period prior solver failed "
                    f"(best residual={best_resid:.2e}). "
                    f"Falling back to IG({a:.3f}, {b:.3f}).")
    return a, b


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
# Fixed-theta inference (conditioning on hyperparameters)
# =============================================================================

def optimise_sm_hyperparameters(y_obs_norm, s_all, obs_idx, Q, D,
                                  noise_std, y_std, mu_max,
                                  n_steps=2000, lr=0.01, n_restarts=5, rng=None):
    """Full marginal likelihood optimisation over all SM hyperparameters.

    Optimises weights, means, and variances jointly — matching the Wilson &
    Adams (2013) approach.  O(N_obs³) per gradient step, run once before
    inference.
    """
    if rng is None:
        rng = random.key(0)
    N_obs  = len(obs_idx)
    sigma2 = (noise_std / y_std) ** 2
    y      = jnp.array(y_obs_norm)
    ell_maxes = np.linspace(50.0, 10.0, Q)

    @jit
    def neg_lml(params):
        log_w_raw = params[:Q]
        log_means = params[Q:Q*(1+D)].reshape(Q, D)
        log_vars  = params[Q*(1+D):].reshape(Q, D)
        weights   = jax.nn.softmax(log_w_raw)
        means     = jnp.exp(jnp.clip(log_means, -10.0, jnp.log(mu_max)))
        vars_     = jnp.exp(jnp.clip(log_vars,  -20.0, -2.0))
        K_all     = spectral_mixture(s_all, s_all, weights, means, vars_)
        K_obs     = K_all[jnp.ix_(obs_idx, obs_idx)] \
                    + (sigma2 + JITTER) * jnp.eye(N_obs)
        L         = jnp.linalg.cholesky(K_obs)
        alpha     = jax.scipy.linalg.cho_solve((L, True), y)
        lml       = (-0.5 * jnp.dot(y, alpha)
                     - jnp.sum(jnp.log(jnp.diag(L)))
                     - 0.5 * N_obs * jnp.log(2.0 * jnp.pi))
        return -lml

    grad_fn = jax.value_and_grad(neg_lml)
    best_val, best_params = np.inf, None

    for restart in range(n_restarts):
        rng, rng_init = random.split(rng)
        log_w_init  = random.normal(rng_init, shape=(Q,)) * 0.3
        log_mu_init = jnp.log(random.uniform(
            rng_init, shape=(Q, D), minval=0.001, maxval=mu_max * 0.8))
        log_v_init  = jnp.stack([
            jnp.log(1.0 / (2.0 * jnp.pi**2 *
                    random.uniform(rng_init, shape=(D,),
                                   minval=1.0, maxval=float(ell_maxes[q]))**2))
            for q in range(Q)
        ])
        params     = jnp.concatenate([log_w_init,
                                       log_mu_init.ravel(),
                                       log_v_init.ravel()])
        optimizer  = optax.adam(lr)
        opt_state  = optimizer.init(params)
        for step in range(n_steps):
            val, grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if step % 500 == 0:
                log.info(f"  Restart {restart+1}/{n_restarts}  "
                         f"step {step:4d}  -LML={float(val):.3f}")
        if float(val) < best_val:
            best_val    = float(val)
            best_params = params

    weights_opt = jnp.array(jax.nn.softmax(best_params[:Q]))
    means_opt   = jnp.exp(jnp.clip(
        best_params[Q:Q*(1+D)].reshape(Q, D), -10.0, jnp.log(mu_max)))
    vars_opt    = jnp.exp(jnp.clip(best_params[Q*(1+D):].reshape(Q, D), -20.0, -2.0))
    log.info(f"Optimised theta (best -LML={best_val:.3f}):")
    for q in range(Q):
        ell = 1.0 / np.sqrt(2.0 * np.pi**2 * np.array(vars_opt[q]))
        log.info(f"  q{q+1}: w={float(weights_opt[q]):.3f}  "
                 f"µ={np.array(means_opt[q])}  ell={ell}")
    return weights_opt, means_opt, vars_opt


def analytic_gp_predict(y_obs_norm, y_mean, y_std,
                         s_all, obs_idx, tst_idx,
                         weights, means, vars_, noise_std):
    """Closed-form GP posterior predictive at fixed theta (Wilson & Adams 2013).

    Returns posterior mean and std at test locations — no NUTS, no samples.
    Provides a point-prediction baseline for comparison with full posterior methods.
    """
    sigma2 = (noise_std / y_std) ** 2
    y      = jnp.array(y_obs_norm)
    N_obs  = len(obs_idx)

    t0          = time.time()
    K_all       = spectral_mixture(s_all, s_all, weights, means, vars_)
    K_obs       = K_all[jnp.ix_(obs_idx, obs_idx)] \
                  + (sigma2 + JITTER) * jnp.eye(N_obs)
    K_star_obs  = K_all[jnp.ix_(tst_idx, obs_idx)]
    K_star_star = jnp.diag(K_all[jnp.ix_(tst_idx, tst_idx)])

    L            = jnp.linalg.cholesky(K_obs)
    alpha        = jax.scipy.linalg.cho_solve((L, True), y)
    mu_star_norm = K_star_obs @ alpha
    v            = solve_triangular(L, K_star_obs.T, lower=True)
    var_star     = K_star_star - jnp.sum(v**2, axis=0)
    std_star     = jnp.sqrt(jnp.clip(var_star, 1e-10, None))
    elapsed      = time.time() - t0
    log.info(f"Analytic GP (W&A): {elapsed:.1f}s")

    return (np.array(mu_star_norm * y_std + y_mean),
            np.array(std_star * y_std),
            elapsed)


def compute_z_init(y_obs_norm, s_all, obs_idx,
                    weights, means, vars_, noise_std, y_std, L_full):
    """Compute analytic posterior mean in z-space as warm start for NUTS.

    With fixed theta, the GP posterior mean is:
        f_post = K_{all,obs} (K_{obs,obs} + σ²I)^{-1} y_obs

    Converting to z-space: z_init = L^{-1} f_post

    This places NUTS at the posterior mode rather than z=0 (the prior mean),
    avoiding the enormous gradient magnitude at z=0 that causes NUTS to adopt
    a near-zero step size and fail to explore.
    """
    sigma2    = (noise_std / y_std) ** 2
    N_obs     = len(obs_idx)
    y         = jnp.array(y_obs_norm)
    K_all     = spectral_mixture(s_all, s_all, weights, means, vars_)
    K_obs     = K_all[jnp.ix_(obs_idx, obs_idx)] \
                + (sigma2 + JITTER) * jnp.eye(N_obs)
    K_all_obs = K_all[:, obs_idx]
    L_obs     = jnp.linalg.cholesky(K_obs)
    alpha     = jax.scipy.linalg.cho_solve((L_obs, True), y)
    f_post    = K_all_obs @ alpha                              # (N,) posterior mean
    z_init    = solve_triangular(L_full, f_post, lower=True)  # (N,) in z-space
    return z_init


def deeprv_nuts_fixed_theta(decoder, y_obs_norm, y_mean, y_std,
                             s_all, obs_idx, tst_idx,
                             weights, means, vars_, noise_std,
                             num_warmup, num_samples, rng):
    """SM-DeepRV NUTS with fixed theta — samples only z ~ N(0,I).

    The z-posterior is log-concave and unimodal, so NUTS mixes efficiently
    regardless of the SM kernel's hyperparameter identifiability issues.

    Uses the analytic GP posterior mean as a warm start for z, placing NUTS
    at the posterior mode rather than z=0.  This avoids the enormous gradient
    magnitude at z=0 (caused by the 1/σ² amplification) that otherwise forces
    NUTS to adopt a near-zero step size and fail to explore.
    """
    from numpyro.infer import init_to_value
    N_all, D    = s_all.shape
    sigma_fixed = noise_std / y_std
    cond = jnp.concatenate([weights,
                            jnp.log(means).ravel(),
                            jnp.log(vars_).ravel()])

    # Compute Cholesky and analytic warm start
    K      = spectral_mixture(s_all, s_all, weights, means, vars_) \
             + JITTER * jnp.eye(N_all)
    L_full = jnp.linalg.cholesky(K)
    z_init = compute_z_init(y_obs_norm, s_all, obs_idx,
                            weights, means, vars_, noise_std, y_std, L_full)
    log.info(f"z_init: mean={float(jnp.mean(z_init)):.4f}  "
             f"std={float(jnp.std(z_init)):.4f}  "
             f"max|z|={float(jnp.max(jnp.abs(z_init))):.4f}")

    def model():
        z = numpyro.sample("z", dist.Normal(jnp.zeros(N_all), jnp.ones(N_all)))
        f = decoder(z[None], cond, s=s_all).squeeze()
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma_fixed), obs=y_obs_norm)
        numpyro.deterministic("f_all", f)

    t0   = time.time()
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.8, dense_mass=False,
             init_strategy=init_to_value(values={"z": z_init})),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=1, progress_bar=True,
    )
    mcmc.run(rng, extra_fields=("energy", "diverging", "accept_prob", "num_steps"))
    elapsed = time.time() - t0
    log.info(f"SM-DeepRV (fixed θ) NUTS: {elapsed:.1f}s")
    mcmc.print_summary()

    samples = mcmc.get_samples()
    extra   = mcmc.get_extra_fields()
    f_samps = samples["f_all"][:, tst_idx] * y_std + y_mean
    return f_samps.mean(0), f_samps.std(0), samples, extra, elapsed


def exact_gp_nuts_fixed_theta(y_obs_norm, y_mean, y_std,
                               s_all, obs_idx, tst_idx,
                               weights, means, vars_, noise_std,
                               num_warmup, num_samples, rng):
    """Exact GP NUTS with fixed theta — samples only z ~ N(0,I).

    Direct comparison with DeepRV: same fixed theta, same NUTS setup,
    same analytic warm start — isolates decoder approximation error.
    """
    from numpyro.infer import init_to_value
    N_all, D    = s_all.shape
    sigma_fixed = noise_std / y_std
    K = spectral_mixture(s_all, s_all, weights, means, vars_) \
        + JITTER * jnp.eye(N_all)
    L = jnp.linalg.cholesky(K)

    # Same analytic warm start as DeepRV for a fair comparison
    z_init = compute_z_init(y_obs_norm, s_all, obs_idx,
                            weights, means, vars_, noise_std, y_std, L)

    def model():
        z = numpyro.sample("z", dist.Normal(jnp.zeros(N_all), jnp.ones(N_all)))
        f = L @ z
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma_fixed), obs=y_obs_norm)
        numpyro.deterministic("f_all", f)

    t0   = time.time()
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.8, dense_mass=False,
             init_strategy=init_to_value(values={"z": z_init})),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=1, progress_bar=True,
    )
    mcmc.run(rng, extra_fields=("energy", "diverging", "accept_prob", "num_steps"))
    elapsed = time.time() - t0
    log.info(f"Exact GP (fixed θ) NUTS: {elapsed:.1f}s")
    mcmc.print_summary()

    samples = mcmc.get_samples()
    extra   = mcmc.get_extra_fields()
    f_samps = samples["f_all"][:, tst_idx] * y_std + y_mean
    return f_samps.mean(0), f_samps.std(0), samples, extra, elapsed


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

    # --- Fixed-theta inference (conditioning on hyperparameters) ---
    if args.fixed_theta:
        spacing = 100.0 / (args.grid_size - 1)
        mu_max  = float(0.95 / (2.0 * spacing))

        if args.optimise_theta:
            log.info("Optimising theta via marginal likelihood …")
            rng, rng_opt = random.split(rng)
            weights_th, means_th, vars_th = optimise_sm_hyperparameters(
                y_obs_norm, s_all, obs_idx, Q, s_all.shape[1],
                args.noise_std, y_std, mu_max,
                n_steps=args.opt_steps, lr=args.opt_lr,
                n_restarts=args.opt_restarts, rng=rng_opt,
            )
        else:
            log.info("Using true theta from SM config.")
            weights_th = jnp.array(sm_config["weights"])
            means_th   = jnp.array(sm_config["means"])
            vars_th    = jnp.array(sm_config["variances"])

        log.info("Fixed theta:")
        for q in range(Q):
            ell = 1.0 / np.sqrt(2.0 * np.pi**2 * np.array(vars_th[q]))
            log.info(f"  q{q+1}: w={float(weights_th[q]):.3f}  "
                     f"µ={np.array(means_th[q])}  ell={ell}")
        np.save(cfg_dir / "theta_weights.npy", np.array(weights_th))
        np.save(cfg_dir / "theta_means.npy",   np.array(means_th))
        np.save(cfg_dir / "theta_vars.npy",    np.array(vars_th))

        # SM-DeepRV fixed theta
        log.info(f"Running SM-DeepRV NUTS fixed-θ …")
        drv_mean, drv_std, drv_samples, drv_extra, drv_time = \
            deeprv_nuts_fixed_theta(
                decoder, y_obs_norm, y_mean, y_std,
                s_all, obs_idx, tst_idx,
                weights_th, means_th, vars_th, args.noise_std,
                args.num_warmup, args.num_samples,
                jax.random.fold_in(rng_drv, 0),
            )
        for name, arr in [("drv_mean", drv_mean), ("drv_std", drv_std)]:
            np.save(cfg_dir / f"{name}.npy", np.array(arr))
        for name, arr in drv_samples.items():
            np.save(cfg_dir / f"drv_samples_{name}.npy", np.array(arr))
        drv_metrics = evaluate(drv_mean, drv_std, f_test)
        log.info(f"SM-DeepRV (fixed θ) — RMSE={drv_metrics['RMSE']:.4f}  "
                 f"Coverage={drv_metrics['Coverage_95']:.3f}  "
                 f"Time={drv_time:.1f}s")

        # Exact GP fixed theta
        egp_mean = egp_std = egp_samples = egp_time = None
        egp_metrics = {}
        if not args.no_exact_gp:
            log.info("Running Exact GP NUTS fixed-θ …")
            egp_mean, egp_std, egp_samples, egp_extra, egp_time = \
                exact_gp_nuts_fixed_theta(
                    y_obs_norm, y_mean, y_std,
                    s_all, obs_idx, tst_idx,
                    weights_th, means_th, vars_th, args.noise_std,
                    args.num_warmup, args.num_samples,
                    jax.random.fold_in(rng_egp, 0),
                )
            for name, arr in [("egp_mean", egp_mean), ("egp_std", egp_std)]:
                np.save(cfg_dir / f"{name}.npy", np.array(arr))
            for name, arr in egp_samples.items():
                np.save(cfg_dir / f"egp_samples_{name}.npy", np.array(arr))
            egp_metrics = evaluate(egp_mean, egp_std, f_test)
            log.info(f"Exact GP (fixed θ)  — RMSE={egp_metrics['RMSE']:.4f}  "
                     f"Coverage={egp_metrics['Coverage_95']:.3f}  "
                     f"Time={egp_time:.1f}s")

        # Analytic GP (Wilson & Adams)
        ana_mean = ana_std = ana_time = None
        ana_metrics = {}
        if not args.no_analytic:
            log.info("Running analytic GP (Wilson & Adams) …")
            ana_mean, ana_std, ana_time = analytic_gp_predict(
                y_obs_norm, y_mean, y_std,
                s_all, obs_idx, tst_idx,
                weights_th, means_th, vars_th, args.noise_std,
            )
            np.save(cfg_dir / "ana_mean.npy", ana_mean)
            np.save(cfg_dir / "ana_std.npy",  ana_std)
            ana_metrics = evaluate(ana_mean, ana_std, f_test)
            log.info(f"Analytic GP (W&A)  — RMSE={ana_metrics['RMSE']:.4f}  "
                     f"Coverage={ana_metrics['Coverage_95']:.3f}  "
                     f"Time={ana_time:.1f}s")

        # Plots
        y_obs_plot = y_obs_norm * y_std + y_mean
        plot_spatial(s_all, f_all, y_obs_plot,
                     obs_idx, tst_idx, drv_mean, drv_std, egp_mean, egp_std,
                     sm_config["label"], args.grid_size, cfg_dir, config_name)
        plot_metrics_comparison(drv_metrics, egp_metrics or None,
                                ana_metrics or None, sm_config["label"],
                                drv_time, egp_time, ana_time,
                                cfg_dir, config_name)
        plot_scatter_pred_vs_true(drv_mean, egp_mean, ana_mean, f_test,
                                  sm_config["label"], cfg_dir, config_name)

        row = {"config": config_name, "grid_size": args.grid_size, "Q": Q,
               "inference": "fixed_theta",
               "drv_time_s": drv_time,
               "egp_time_s": egp_time,
               "ana_time_s": ana_time}
        for k, v in drv_metrics.items():  row[f"drv_{k}"] = v
        for k, v in egp_metrics.items():  row[f"egp_{k}"] = v
        for k, v in ana_metrics.items():  row[f"ana_{k}"] = v
        return row

    # --- Joint (z, theta) inference (original behaviour) ---
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
           "inference": "joint",
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
# Q-Gated DeepRV architecture
# =============================================================================

class QGatedDeepRV(nn.Module):
    """Q-component SM-DeepRV with per-component spatial gating.

    Instead of one gMLP conditioned on the full (Q*15-dim) conditional
    vector, runs Q separate gMLPDeepRV branches each conditioned only on
    their own component's parameters:
        cond_q = [log(w_q), log(mu_qx), log(mu_qy), log(v_qx), log(v_qy)]  (5-dim for D=2)

    Outputs are combined with mixture weights:
        f_hat = sum_q  w_q * gMLP_q(z, cond_q)

    Each branch learns the spatial correlation structure appropriate to its
    own kernel — component 1 (smooth, long-range) learns broad spatial gates;
    component 3 (fast oscillation) learns local gates.  The shared z vector
    preserves correlations across components as in the true GP sample.
    """
    Q:         int
    D:         int
    num_blks:  int
    embed_dim: int

    def setup(self):
        self.decoders = [
            gMLPDeepRV(
                num_blks = self.num_blks,
                embed    = dl4bi.mlp.MLP([self.embed_dim, self.embed_dim], nn.gelu),
                proj_out = dl4bi.mlp.MLP([self.embed_dim, self.embed_dim], nn.gelu),
            )
            for _ in range(self.Q)
        ]

    def __call__(self, z, conditionals, s, **kwargs):
        Q, D = self.Q, self.D
        # Parse: [weights(Q), log_means(Q*D), log_vars(Q*D)]
        weights   = conditionals[:Q]                               # (Q,)
        log_means = conditionals[Q:Q*(1 + D)].reshape(Q, D)       # (Q, D)
        log_vars  = conditionals[Q*(1 + D):].reshape(Q, D)        # (Q, D)

        f_hat = None
        for q in range(Q):
            # Per-component conditional — same dimensionality as a Q=1 decoder,
            # so each branch is architecturally identical to the baseline gMLP.
            cond_q = jnp.concatenate([
                jnp.log(jnp.clip(weights[q:q+1], 1e-8, 1.0)),    # log w_q   (1,)
                log_means[q],                                       # log mu_q  (D,)
                log_vars[q],                                        # log v_q   (D,)
            ])                                                      # (1+2D,) = (5,) for D=2

            output_q = self.decoders[q](z, cond_q, s=s)
            f_q      = weights[q] * output_q.f_hat
            f_hat    = f_q if f_hat is None else f_hat + f_q

        return VAEOutput(f_hat=f_hat)


def generate_q_gated_surrogate_decoder(state, model):
    """Surrogate decoder for QGatedDeepRV — returns f_hat array directly,
    matching the interface expected by deeprv_nuts_inference."""
    @jit
    def decoder(z, conditionals, s):
        output = state.apply_fn(
            {"params": state.params, **state.kwargs},
            z, conditionals, s=s,
            rngs={"extra": random.key(0)},
        )
        return output.f_hat
    return decoder


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

    # Checkpoint dir is unique per (grid_size, Q, lr_schedule, embed_dim, num_blks, model)
    ckpt_dir  = (args.load_ckpt or args.ckpt_dir or str(
        Path(args.results_dir) /
        f"sim_checkpoints_grid{args.grid_size}_q{Q}"
        f"_blks{args.num_blks}_emb{args.embed_dim}"
        f"_sched{args.lr_schedule}_model{args.model}"
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
                   "embed_dim": args.embed_dim, "model": args.model,
                   "kernel": "spectral_mixture",
                   "likelihood": args.likelihood, "noise_std": args.noise_std,
                   "alpha_dirichlet": args.alpha_dirichlet,
               })
    wandb.define_metric("Train Loss",      step_metric="_step")
    wandb.define_metric("Valid norm MSE",  step_metric="_step")

    if args.model == "qgated":
        nn_model = QGatedDeepRV(
            Q        = Q,
            D        = 2,
            num_blks = args.num_blks,
            embed_dim= args.embed_dim,
        )
        log.info(f"Architecture: QGatedDeepRV(Q={Q}, num_blks={args.num_blks}, "
                 f"embed_dim={args.embed_dim}) — {Q} separate gMLP branches")
    else:
        nn_model = gMLPDeepRV(
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
            rng_train, nn_model, optimizer, deep_rv_train_step,
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

    if args.model == "qgated":
        decoder = generate_q_gated_surrogate_decoder(state, nn_model)
    else:
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