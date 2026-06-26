"""
test_weight_sensitivity.py
==========================
Diagnostic: does the trained decoder's output vary meaningfully with
mixture weights, or has it learned to ignore them?

Usage:
    python test_weight_sensitivity.py \
        --ckpt_dir results/sim_checkpoints_grid32_q3_blks2_emb64_schedflat_cosine_modelgmlp \
        --grid_size 32 --q 3 --embed_dim 64 --num_blks 2

Output: a table showing how f_hat statistics change as w_1 varies from
0.1 to 0.9, with means and variances held fixed at the diverse config
true values.  If std/mean barely changes across rows, the decoder is
ignoring weights and NUTS cannot infer them.
"""

import argparse
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path

import dl4bi.mlp
import flax.linen as nn
from dl4bi_sps.utils import build_grid
from dl4bi.train import TrainState
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import generate_surrogate_decoder
import optax


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",  type=str, required=True)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--q",         type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--num_blks",  type=int, default=2)
    p.add_argument("--model",     type=str, default="gmlp",
                   choices=["gmlp", "qgated"])
    return p.parse_args()


def load_decoder(ckpt_dir, nn_model, s_all):
    """Load latest checkpoint and return surrogate decoder."""
    ckpt_dir    = Path(ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("step_*"),
                         key=lambda p: int(p.name.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest     = checkpoints[-1]
    print(f"Loading checkpoint: {latest.name}")
    ckpt       = ocp.PyTreeCheckpointer().restore(str(latest.absolute()))
    optimizer  = optax.adamw(1e-3)
    dummy_rng  = random.key(0)
    # Build a dummy batch to initialise the model
    N = s_all.shape[0]
    dummy_batch = {
        "s": s_all,
        "z": jnp.zeros((1, N)),
        "conditionals": jnp.zeros(15),   # Q*(1+2D) = 3*(1+4) = 15 for Q=3, D=2
        "f": jnp.zeros((1, N)),
    }
    nn_model.init({"params": dummy_rng, "extra": dummy_rng}, **dummy_batch)
    state = TrainState.create(
        apply_fn=nn_model.apply,
        params=ckpt["params"],
        kwargs=ckpt["kwargs"],
        tx=optimizer,
    )
    return generate_surrogate_decoder(state, nn_model)


def main():
    args  = parse_args()
    Q, D  = args.q, 2
    s_all = build_grid(
        [{"start": 0.0, "stop": 100.0, "num": args.grid_size}] * 2
    ).reshape(-1, 2)
    N = s_all.shape[0]

    nn_model = gMLPDeepRV(
        num_blks = args.num_blks,
        embed    = dl4bi.mlp.MLP([args.embed_dim, args.embed_dim], nn.gelu),
        proj_out = dl4bi.mlp.MLP([args.embed_dim, args.embed_dim], nn.gelu),
    )

    decoder = load_decoder(args.ckpt_dir, nn_model, s_all)

    # Base conditionals from diverse config true values
    true_means = np.array([[0.005,0.005],[0.070,0.070],[0.170,0.170]])
    true_vars  = np.array([[3e-5,3e-5],[1.5e-4,1.5e-4],[6e-3,6e-3]])
    lm_base    = jnp.log(jnp.array(true_means)).ravel()
    lv_base    = jnp.log(jnp.array(true_vars)).ravel()

    # Test with zero z (prior mean) and a random z
    rng      = random.key(42)
    z_zero   = jnp.zeros(N)
    z_random = random.normal(rng, shape=(N,))

    print(f"\nDecoder weight sensitivity test")
    print(f"Grid: {args.grid_size}² (N={N})  Q={Q}  embed_dim={args.embed_dim}")
    print(f"Means and variances fixed to true diverse config values.")
    print(f"Varying w_1 from 0.1 to 0.9, w_2=(1-w_1)*0.6, w_3=(1-w_1)*0.4\n")

    print(f"{'w_1':>6} {'w_2':>6} {'w_3':>6} | "
          f"{'z=0 std':>10} {'z=0 mean':>10} | "
          f"{'z~N std':>10} {'z~N mean':>10}")
    print("-" * 70)

    for w1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        w2   = (1 - w1) * 0.6
        w3   = (1 - w1) * 0.4
        w    = jnp.array([w1, w2, w3])
        cond = jnp.concatenate([w, lm_base, lv_base])

        f0 = decoder(z_zero[None],   cond, s=s_all).squeeze()
        fr = decoder(z_random[None], cond, s=s_all).squeeze()

        print(f"{w1:>6.2f} {w2:>6.2f} {w3:>6.2f} | "
              f"{float(jnp.std(f0)):>10.4f} {float(jnp.mean(f0)):>10.4f} | "
              f"{float(jnp.std(fr)):>10.4f} {float(jnp.mean(fr)):>10.4f}")

    # Also test: fix weights, vary means frequency (should definitely change output)
    print(f"\nSanity check — varying mu_2 with fixed weights:")
    print(f"{'mu_2':>8} | {'z=0 std':>10} {'z=0 mean':>10}")
    print("-" * 35)
    w_fixed = jnp.array([0.5, 0.3, 0.2])
    for mu2 in [0.01, 0.03, 0.05, 0.07, 0.09]:
        means_varied    = true_means.copy()
        means_varied[1] = [mu2, mu2]
        lm_varied       = jnp.log(jnp.array(means_varied)).ravel()
        cond            = jnp.concatenate([w_fixed, lm_varied, lv_base])
        f0              = decoder(z_zero[None], cond, s=s_all).squeeze()
        print(f"{mu2:>8.3f} | {float(jnp.std(f0)):>10.4f} {float(jnp.mean(f0)):>10.4f}")

    print("\nInterpretation:")
    print("  Weight section varies widely → decoder is weight-sensitive → NUTS can infer weights")
    print("  Weight section barely varies → decoder ignores weights → NUTS falls back to prior")
    print("  Sanity check should always vary — if it doesn't, checkpoint may not have loaded correctly")

    # ==========================================================================
    # Z-sensitivity test
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"Z-sensitivity test")
    print(f"Fixed theta at true diverse config values.")
    print(f"Varying z scale — if f std barely changes, decoder ignores z\n"
          f"and NUTS will collapse regardless of inference mode.\n")

    w_true    = jnp.array([0.5, 0.3, 0.2])
    lm_true   = jnp.log(jnp.array(true_means)).ravel()
    lv_true   = jnp.log(jnp.array(true_vars)).ravel()
    cond_true = jnp.concatenate([w_true, lm_true, lv_true])

    print(f"{'z_scale':>10} {'f std':>10} {'f mean':>10} {'f min':>10} {'f max':>10}")
    print("-" * 55)
    for scale in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        z = random.normal(random.key(0), shape=(N,)) * scale
        f = decoder(z[None], cond_true, s=s_all).squeeze()
        print(f"{scale:>10.2f} {float(jnp.std(f)):>10.4f} "
              f"{float(jnp.mean(f)):>10.4f} "
              f"{float(jnp.min(f)):>10.4f} "
              f"{float(jnp.max(f)):>10.4f}")

    # Also test multiple random z at scale=1 to check variance across samples
    print(f"\nVariance across 10 independent z ~ N(0,I) samples (scale=1):")
    print(f"  (should show meaningful spread if decoder is z-sensitive)\n")
    f_samples = []
    for i in range(10):
        z = random.normal(random.key(i), shape=(N,))
        f = decoder(z[None], cond_true, s=s_all).squeeze()
        f_samples.append(np.array(f))
    f_samples = np.stack(f_samples)           # (10, N)
    sample_std_per_location = f_samples.std(axis=0)
    print(f"  Mean std across locations: {sample_std_per_location.mean():.4f}")
    print(f"  Max  std across locations: {sample_std_per_location.max():.4f}")
    print(f"  Min  std across locations: {sample_std_per_location.min():.4f}")
    print(f"  (Exact GP posterior std at unobserved locations is ~0.5-1.0 "
          f"for this kernel)")

    print(f"\nInterpretation:")
    print(f"  f std grows proportionally with z_scale → decoder z-sensitive → "
          f"NUTS can explore posterior")
    print(f"  f std flat across z_scale           → decoder ignores z    → "
          f"NUTS collapses regardless of theta fixing")
    print(f"  z_scale=0.0 shows decoder bias (should be near-zero mean and std)")
    print(f"  Sample std ~0.5-1.0                 → decoder well-calibrated")
    print(f"  Sample std <<0.1                    → variance collapse in decoder")


if __name__ == "__main__":
    main()