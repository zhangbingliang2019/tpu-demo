# train_ddpm_cifar10.py
# All code and comments are in English as requested.

import os
import time
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state

import optax

import tensorflow as tf
import tensorflow_datasets as tfds


# ----------------------------
# Distributed utilities
# ----------------------------

def maybe_init_distributed():
    """
    Initialize JAX distributed runtime for multi-host (TPU VM) setups.

    - If running single-host, this is a no-op.
    - If already initialized, it will raise; we ignore safely.
    """
    try:
        if "JAX_PROCESS_COUNT" in os.environ and int(os.environ["JAX_PROCESS_COUNT"]) > 1:
            jax.distributed.initialize()
    except Exception:
        pass


def is_main_process() -> bool:
    """Only the main process should write logs/samples to avoid duplicates."""
    return jax.process_index() == 0


# ----------------------------
# Data: TFDS CIFAR-10
# ----------------------------

def preprocess(example: Dict[str, Any], training: bool) -> Dict[str, Any]:
    """
    Preprocess CIFAR-10 image:
    - Convert to float32
    - Scale to [-1, 1] (common for diffusion)
    - Optional random flip (training only)
    """
    x = tf.cast(example["image"], tf.float32) / 255.0
    if training:
        x = tf.image.random_flip_left_right(x)
    x = x * 2.0 - 1.0
    return {"image": x}


def make_dataset(split: str, batch_size_global: int, training: bool, seed: int) -> tf.data.Dataset:
    """
    Build tf.data pipeline and shard by JAX process for multi-host.
    """
    ds = tfds.load("cifar10", split=split, shuffle_files=training)
    ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())

    if training:
        ds = ds.shuffle(50_000, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda ex: preprocess(ex, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size_global, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def tfds_iter(ds: tf.data.Dataset):
    """Yield numpy batches from TFDS."""
    for b in tfds.as_numpy(ds):
        yield b


def shard_for_pmap(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reshape batch from [global_batch, ...] to [local_devices, per_device_batch, ...].
    """
    n_local = jax.local_device_count()
    x = batch["image"]
    assert x.shape[0] % n_local == 0, "batch_size_global must be divisible by local_device_count"
    per = x.shape[0] // n_local
    x = x.reshape((n_local, per) + x.shape[1:])
    return {"image": x}


# ----------------------------
# Diffusion schedule (DDPM)
# ----------------------------

@dataclass
class DDPMSchedule:
    """
    Precomputed arrays for the DDPM forward and reverse process.

    We use a simple linear beta schedule.
    """
    betas: jnp.ndarray                 # [T]
    alphas: jnp.ndarray                # [T]
    alpha_bars: jnp.ndarray            # [T]
    sqrt_alpha_bars: jnp.ndarray       # [T]
    sqrt_one_minus_alpha_bars: jnp.ndarray  # [T]


def make_linear_schedule(T: int, beta_start: float, beta_end: float) -> DDPMSchedule:
    """
    Create a linear beta schedule from beta_start to beta_end over T steps.
    """
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    sqrt_alpha_bars = jnp.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - alpha_bars)
    return DDPMSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )


def extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Tuple[int, ...]) -> jnp.ndarray:
    """
    Extract values from a[t] and reshape to broadcast over x.

    a: [T]
    t: [B] integer timesteps
    returns: [B, 1, 1, 1] (or broadcastable to x)
    """
    out = a[t]  # [B]
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


def q_sample(rng: jax.Array, schedule: DDPMSchedule, x0: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward diffusion: sample x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps

    Returns (x_t, eps).
    """
    eps = random.normal(rng, x0.shape, dtype=jnp.float32)
    s1 = extract(schedule.sqrt_alpha_bars, t, x0.shape)
    s2 = extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)
    xt = s1 * x0 + s2 * eps
    return xt, eps


# ----------------------------
# Time embedding + U-Net (minimal)
# ----------------------------

def sinusoidal_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Classic sinusoidal time embedding.

    t: [B] integer timesteps
    returns: [B, dim]
    """
    half = dim // 2
    freqs = jnp.exp(-math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    # Convert t to float for embedding
    args = t.astype(jnp.float32)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class ResBlock(nn.Module):
    """
    A minimal residual block with time conditioning.
    """
    out_ch: int
    time_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        h = nn.GroupNorm(num_groups=8)(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)

        # Add time embedding (project to channels and broadcast)
        te = nn.swish(t_emb)
        te = nn.Dense(self.out_ch)(te)  # [B, out_ch]
        h = h + te[:, None, None, :]

        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)

        # Residual connection (match channels if needed)
        if x.shape[-1] != self.out_ch:
            x = nn.Conv(self.out_ch, (1, 1), padding="SAME")(x)
        return x + h


class Downsample(nn.Module):
    """Downsample by 2 using strided conv."""
    out_ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(self.out_ch, (3, 3), strides=(2, 2), padding="SAME")(x)


class Upsample(nn.Module):
    """Upsample by 2 using nearest neighbor + conv (simple and stable)."""
    out_ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
        return x


class MiniUNet(nn.Module):
    """
    A very small U-Net suitable for CIFAR-10 diffusion baseline.

    Input:
      x_t: [B, 32, 32, 3]
      t:   [B] timesteps
    Output:
      eps_pred: [B, 32, 32, 3]
    """
    base_ch: int = 64
    time_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        # Time embedding MLP
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = nn.Dense(self.time_dim)(t_emb)
        t_emb = nn.swish(t_emb)
        t_emb = nn.Dense(self.time_dim)(t_emb)

        # Stem
        x0 = nn.Conv(self.base_ch, (3, 3), padding="SAME")(x)

        # Down 1: 32 -> 16
        d1 = ResBlock(self.base_ch, self.time_dim)(x0, t_emb, train)
        d1 = ResBlock(self.base_ch, self.time_dim)(d1, t_emb, train)
        x1 = Downsample(self.base_ch * 2)(d1)

        # Down 2: 16 -> 8
        d2 = ResBlock(self.base_ch * 2, self.time_dim)(x1, t_emb, train)
        d2 = ResBlock(self.base_ch * 2, self.time_dim)(d2, t_emb, train)
        x2 = Downsample(self.base_ch * 4)(d2)

        # Bottleneck: 8x8
        mid = ResBlock(self.base_ch * 4, self.time_dim)(x2, t_emb, train)
        mid = ResBlock(self.base_ch * 4, self.time_dim)(mid, t_emb, train)

        # Up 2: 8 -> 16
        u2 = Upsample(self.base_ch * 2)(mid)
        u2 = jnp.concatenate([u2, d2], axis=-1)
        u2 = ResBlock(self.base_ch * 2, self.time_dim)(u2, t_emb, train)
        u2 = ResBlock(self.base_ch * 2, self.time_dim)(u2, t_emb, train)

        # Up 1: 16 -> 32
        u1 = Upsample(self.base_ch)(u2)
        u1 = jnp.concatenate([u1, d1], axis=-1)
        u1 = ResBlock(self.base_ch, self.time_dim)(u1, t_emb, train)
        u1 = ResBlock(self.base_ch, self.time_dim)(u1, t_emb, train)

        # Output head: predict epsilon
        h = nn.GroupNorm(num_groups=8)(u1)
        h = nn.swish(h)
        eps = nn.Conv(3, (3, 3), padding="SAME")(h)
        return eps


# ----------------------------
# Training state and step
# ----------------------------

class TrainState(train_state.TrainState):
    pass


def create_state(rng: jax.Array, model: nn.Module, lr: float, wd: float) -> TrainState:
    """
    Initialize parameters and optimizer state.
    """
    dummy_x = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init({"params": rng}, dummy_x, dummy_t, train=True)["params"]

    tx = optax.adamw(learning_rate=lr, weight_decay=wd)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@dataclass
class Metrics:
    loss: jnp.ndarray


def ddpm_loss_fn(params, apply_fn, schedule: DDPMSchedule, rng: jax.Array, x0: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    DDPM training objective:
      - sample x_t from q(x_t | x0)
      - predict eps with model
      - minimize MSE(eps_pred, eps)
    """
    xt, eps = q_sample(rng, schedule, x0, t)
    eps_pred = apply_fn({"params": params}, xt, t, train=True)
    loss = jnp.mean((eps_pred - eps) ** 2)
    return loss


def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array, schedule: DDPMSchedule, T: int):
    """
    One pmap'ed training step.

    Notes:
    - rng is per-device.
    - batch["image"] is per-device batch: [B, 32, 32, 3]
    - t is sampled uniformly from [0, T-1]
    """
    x0 = batch["image"]
    bsz = x0.shape[0]

    # Sample timesteps uniformly
    t = random.randint(rng, (bsz,), minval=0, maxval=T, dtype=jnp.int32)

    def loss_fn(params):
        return ddpm_loss_fn(params, state.apply_fn, schedule, rng, x0, t)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")

    new_state = state.apply_gradients(grads=grads)
    return new_state, Metrics(loss=loss)


# ----------------------------
# Sampling (ancestral DDPM)
# ----------------------------

def p_sample(rng: jax.Array, model_apply, params, schedule: DDPMSchedule, xt: jnp.ndarray, t: int) -> jnp.ndarray:
    """
    One reverse sampling step: x_{t-1} ~ p(x_{t-1} | x_t)

    We use the standard DDPM mean formula derived from epsilon prediction.

    This implementation is minimal and not optimized.
    """
    # Predict epsilon
    tt = jnp.full((xt.shape[0],), t, dtype=jnp.int32)
    eps_pred = model_apply({"params": params}, xt, tt, train=False)

    beta_t = schedule.betas[t]
    alpha_t = schedule.alphas[t]
    alpha_bar_t = schedule.alpha_bars[t]

    # Mean for p(x_{t-1}|x_t)
    # mu = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1-alpha_bar_t))*eps_pred)
    coef1 = 1.0 / jnp.sqrt(alpha_t)
    coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
    mean = coef1 * (xt - coef2 * eps_pred)

    if t == 0:
        return mean

    # Add noise with variance beta_t (a common simple choice for minimal code)
    noise = random.normal(rng, xt.shape, dtype=jnp.float32)
    return mean + jnp.sqrt(beta_t) * noise


def sample_images(rng: jax.Array, state: TrainState, model: nn.Module, schedule: DDPMSchedule, T: int, n: int = 64) -> jnp.ndarray:
    """
    Generate images by starting from Gaussian noise and iteratively denoising.

    Returns images in [-1, 1].
    """
    x = random.normal(rng, (n, 32, 32, 3), dtype=jnp.float32)
    for t in reversed(range(T)):
        rng, step_rng = random.split(rng)
        x = p_sample(step_rng, model.apply, state.params, schedule, x, t)
    return x


def save_grid(images: np.ndarray, path: str, grid: int = 8):
    """
    Save a grid of images.

    images: [-1,1] float32
    """
    images = (images + 1.0) * 0.5  # to [0,1]
    images = (images * 255.0).clip(0, 255).astype(np.uint8)

    n, h, w, c = images.shape
    assert n == grid * grid

    rows = []
    for r in range(grid):
        row = np.concatenate(images[r * grid:(r + 1) * grid], axis=1)
        rows.append(row)
    grid_img = np.concatenate(rows, axis=0)
    Image.fromarray(grid_img).save(path)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workdir", type=str, default="./runs/ddpm_cifar10")
    parser.add_argument("--seed", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size_global", type=int, default=1024)  # per-host global batch
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=1e-4)

    # Diffusion
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    # Model
    parser.add_argument("--base_ch", type=int, default=64)

    # Logging/sampling
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=64)

    args = parser.parse_args()

    maybe_init_distributed()

    if is_main_process():
        os.makedirs(args.workdir, exist_ok=True)
        os.makedirs(os.path.join(args.workdir, "samples"), exist_ok=True)

    tf.random.set_seed(args.seed + jax.process_index())

    if is_main_process():
        print(f"process_count={jax.process_count()} process_index={jax.process_index()}")
        print(f"device_count global={jax.device_count()} local={jax.local_device_count()}")
        print("first local device:", jax.local_devices()[0])

    # Datasets (per-host shards)
    train_ds = make_dataset("train", args.batch_size_global, training=True, seed=args.seed)
    train_it = tfds_iter(train_ds)

    # CIFAR-10 has 50,000 training examples
    train_examples_per_host = 50_000 // jax.process_count()
    steps_per_epoch = train_examples_per_host // args.batch_size_global

    if is_main_process():
        print(f"steps_per_epoch(per-host)={steps_per_epoch}")

    # Model + schedule
    model = MiniUNet(base_ch=args.base_ch, time_dim=256)
    schedule = make_linear_schedule(args.T, args.beta_start, args.beta_end)

    # Initialize state
    rng = random.PRNGKey(args.seed)
    rng, init_rng = random.split(rng)
    state = create_state(init_rng, model, lr=args.lr, wd=args.wd)

    # Replicate for pmap
    local_devices = jax.local_devices()
    state = flax.jax_utils.replicate(state, devices=local_devices)

    # Put schedule on device (it is small, replicate it)
    schedule = jax.device_put(schedule)

    # pmap train step
    p_train = jax.pmap(
        lambda s, b, r: train_step(s, b, r, schedule, args.T),
        axis_name="data",
        devices=local_devices,
    )

    global_step = 0
    start = time.time()

    for epoch in range(args.epochs):
        for _ in range(steps_per_epoch):
            global_step += 1
            batch = next(train_it)
            batch = shard_for_pmap(batch)

            rng, step_rng = random.split(rng)
            # Create per-device rngs
            step_rngs = random.split(step_rng, jax.local_device_count())
            step_rngs = jax.device_put_sharded(list(step_rngs), local_devices)

            state, metrics = p_train(state, batch, step_rngs)

            if is_main_process() and (global_step % args.log_every == 0):
                m = flax.jax_utils.unreplicate(metrics)
                elapsed = time.time() - start
                print(f"[train] step={global_step} epoch={epoch+1}/{args.epochs} loss={float(m.loss):.6f} elapsed={elapsed:.1f}s")

            if is_main_process() and (global_step % args.sample_every == 0):
                # Sample on host (unreplicate state)
                s = flax.jax_utils.unreplicate(state)
                rng, samp_rng = random.split(rng)
                images = sample_images(samp_rng, s, model, schedule, args.T, n=args.num_samples)
                images = np.array(images)
                out = os.path.join(args.workdir, "samples", f"samples_step_{global_step}.png")
                save_grid(images, out, grid=int(math.sqrt(args.num_samples)))
                print(f"[sample] saved {out}")

        if is_main_process():
            print(f"finished epoch {epoch+1}/{args.epochs}")

    if is_main_process():
        print("done.")


if __name__ == "__main__":
    # Recommended: prevent TF from grabbing GPU memory on GPU hosts
    tf.config.set_visible_devices([], "GPU")
    main()
