#!/usr/bin/env python3
"""
================================================================================
DDPM (Denoising Diffusion Probabilistic Model) Training on CIFAR-10
================================================================================

This script trains a simple diffusion model on CIFAR-10 using JAX/Flax.
Designed for multi-host TPU pods (e.g., 256 TPU chips across 32 workers).

ARCHITECTURE OVERVIEW:
----------------------
- Model: Mini U-Net with residual blocks and timestep conditioning
- Diffusion: DDPM with linear beta schedule (1000 timesteps)
- Optimizer: AdamW with configurable learning rate and weight decay
- Data: CIFAR-10 (50k training images, 32x32x3)

MULTI-HOST TPU SETUP (256 TPUs / 32 Workers):
---------------------------------------------
- Each worker (host) manages 8 local TPU chips
- Total: 32 workers x 8 chips = 256 TPU chips
- Data parallelism: Each chip processes a shard of the batch
- Gradient sync: All-reduce (pmean) across all 256 chips via TPU ICI

ENVIRONMENT VARIABLES (set automatically by TPU VM launcher):
-------------------------------------------------------------
- JAX_PROCESS_COUNT: Total number of workers (e.g., 32)
- JAX_PROCESS_INDEX: This worker's index (0-31)
- JAX_COORDINATOR_ADDRESS: Coordinator IP for distributed init
- JAX_COORDINATOR_PORT: Coordinator port (typically 1234)

USAGE:
------
    # Single host (testing)
    python train_ddpm_cifar10.py --batch_size_global 256

    # Multi-host TPU (via gcloud ssh --worker=all)
    python train_ddpm_cifar10.py --batch_size_global 8192 --epochs 100

================================================================================
"""

import os
import sys
import time
import math
import argparse
import datetime
from typing import Any, Dict, Tuple, Optional

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

# Rich library for beautiful terminal output (optional but recommended)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Weights & Biases for experiment tracking (optional but recommended)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==============================================================================
# LOGGING UTILITIES
# ==============================================================================
# We implement comprehensive logging that works in multi-host settings.
# Only the main process (process_index=0) prints to avoid duplicate output.

class Logger:
    """
    Centralized logger for multi-host TPU training.

    Features:
    - Only main process (index 0) prints to stdout
    - Rich formatting when available (tables, panels, colors)
    - Fallback to plain text if Rich is not installed
    - Timestamps for all log entries
    - Structured logging for metrics

    LOG LEVELS:
    - INFO: General progress information
    - CONFIG: Configuration and hyperparameter logging
    - METRIC: Training metrics (loss, throughput, etc.)
    - SAMPLE: Sample generation events
    - DEBUG: Detailed debugging information
    """

    def __init__(self, is_main: bool, workdir: str):
        """
        Initialize the logger.

        Args:
            is_main: True if this is the main process (index 0)
            workdir: Directory to save log files
        """
        self.is_main = is_main
        self.workdir = workdir
        self.start_time = time.time()

        # Rich console for formatted output
        if RICH_AVAILABLE and is_main:
            self.console = Console()
        else:
            self.console = None

        # Log file for persistent logging
        if is_main:
            os.makedirs(workdir, exist_ok=True)
            self.log_file = open(os.path.join(workdir, "train.log"), "a")
        else:
            self.log_file = None

    def _timestamp(self) -> str:
        """Generate ISO format timestamp."""
        return datetime.datetime.now().isoformat(timespec='seconds')

    def _elapsed(self) -> str:
        """Get elapsed time since training started."""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_file(self, message: str):
        """Write message to log file."""
        if self.log_file:
            self.log_file.write(f"[{self._timestamp()}] {message}\n")
            self.log_file.flush()

    def info(self, message: str):
        """Log general information."""
        if not self.is_main:
            return

        formatted = f"[{self._elapsed()}] {message}"
        self._write_file(f"INFO: {message}")

        if self.console:
            self.console.print(f"[blue][INFO][/blue] {formatted}")
        else:
            print(f"[INFO] {formatted}")

    def config(self, title: str, config_dict: Dict[str, Any]):
        """Log configuration as a formatted table."""
        if not self.is_main:
            return

        self._write_file(f"CONFIG: {title} = {config_dict}")

        if self.console:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            for key, value in config_dict.items():
                table.add_row(str(key), str(value))

            self.console.print(table)
        else:
            print(f"\n{'='*60}")
            print(f" {title}")
            print(f"{'='*60}")
            for key, value in config_dict.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")

    def metric(self, step: int, epoch: int, metrics: Dict[str, float]):
        """
        Log training metrics.

        Args:
            step: Global training step
            epoch: Current epoch
            metrics: Dictionary of metric names to values
        """
        if not self.is_main:
            return

        # Format metrics string
        metrics_str = " | ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items()])

        self._write_file(f"METRIC: step={step} epoch={epoch} {metrics_str}")

        formatted = f"[{self._elapsed()}] step={step:>6d} epoch={epoch:>3d} | {metrics_str}"

        if self.console:
            self.console.print(f"[green][TRAIN][/green] {formatted}")
        else:
            print(f"[TRAIN] {formatted}")

    def sample(self, step: int, path: str, num_samples: int):
        """Log sample generation event."""
        if not self.is_main:
            return

        message = f"Generated {num_samples} samples -> {path}"
        self._write_file(f"SAMPLE: step={step} {message}")

        if self.console:
            self.console.print(f"[yellow][SAMPLE][/yellow] [{self._elapsed()}] step={step:>6d} | {message}")
        else:
            print(f"[SAMPLE] [{self._elapsed()}] step={step:>6d} | {message}")

    def separator(self, title: str = ""):
        """Print a visual separator."""
        if not self.is_main:
            return

        if self.console:
            if title:
                self.console.print(Panel(title, style="bold blue"))
            else:
                self.console.print("─" * 60)
        else:
            if title:
                print(f"\n{'='*60}")
                print(f" {title}")
                print(f"{'='*60}")
            else:
                print("-" * 60)

    def close(self):
        """Close log file."""
        if self.log_file:
            self.log_file.close()


# ==============================================================================
# DISTRIBUTED TRAINING UTILITIES
# ==============================================================================
# Functions for initializing and managing multi-host TPU training.

def maybe_init_distributed() -> Dict[str, Any]:
    """
    Initialize JAX distributed runtime for multi-host TPU VM setups.

    HOW MULTI-HOST TPU WORKS:
    -------------------------
    1. Each TPU VM host runs a separate Python process
    2. The first host (worker 0) acts as the coordinator
    3. jax.distributed.initialize() connects all hosts via gRPC
    4. After init, jax.device_count() returns GLOBAL device count (e.g., 256)
    5. pmap operations automatically sync gradients across all hosts

    ENVIRONMENT VARIABLES (set by TPU launcher):
    - JAX_PROCESS_COUNT: Total workers (e.g., 32)
    - JAX_PROCESS_INDEX: This worker's index (0-31)
    - JAX_COORDINATOR_ADDRESS: IP of coordinator (worker 0)
    - JAX_COORDINATOR_PORT: gRPC port (typically 1234)

    Returns:
        Dictionary with distributed training info for logging
    """
    dist_info = {
        "initialized": False,
        "process_count": 1,
        "process_index": 0,
        "coordinator": "N/A",
    }

    try:
        # Check if we're in a multi-host environment
        process_count = int(os.environ.get("JAX_PROCESS_COUNT", 1))

        if process_count > 1:
            # Multi-host: Initialize distributed runtime
            # This blocks until all workers have connected
            jax.distributed.initialize()
            dist_info["initialized"] = True
            dist_info["coordinator"] = os.environ.get("JAX_COORDINATOR_ADDRESS", "unknown")

        dist_info["process_count"] = jax.process_count()
        dist_info["process_index"] = jax.process_index()

    except Exception as e:
        # Log error but continue (might be single-host)
        print(f"[WARNING] Distributed init failed: {e}")

    return dist_info


def is_main_process() -> bool:
    """
    Check if this is the main process (index 0).

    Only the main process should:
    - Print logs to stdout
    - Save checkpoints
    - Generate and save samples
    - Write metrics to disk

    This prevents duplicate output and file corruption in multi-host settings.
    """
    return jax.process_index() == 0


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for logging.

    Returns:
        Dictionary with device counts, types, and memory info
    """
    local_devices = jax.local_devices()

    return {
        "global_device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "device_type": str(local_devices[0].device_kind) if local_devices else "unknown",
        "local_devices": [str(d) for d in local_devices],
    }


# ==============================================================================
# DATA PIPELINE (CIFAR-10)
# ==============================================================================
# Efficient tf.data pipeline with proper sharding for multi-host training.

def preprocess_image(example: Dict[str, Any], training: bool) -> Dict[str, Any]:
    """
    Preprocess a single CIFAR-10 example.

    PREPROCESSING STEPS:
    1. Cast uint8 [0, 255] to float32 [0, 1]
    2. Random horizontal flip (training only) for data augmentation
    3. Scale from [0, 1] to [-1, 1] (standard for diffusion models)

    WHY [-1, 1] RANGE:
    - Matches the range of Gaussian noise (unbounded, centered at 0)
    - Makes the forward diffusion process symmetric
    - Standard convention for generative models

    Args:
        example: Dictionary with 'image' key (uint8 tensor)
        training: Whether to apply data augmentation

    Returns:
        Dictionary with preprocessed 'image' (float32 in [-1, 1])
    """
    # Step 1: Convert to float and normalize to [0, 1]
    image = tf.cast(example["image"], tf.float32) / 255.0

    # Step 2: Random horizontal flip for training (data augmentation)
    # CIFAR-10 images are horizontally symmetric (cars, planes, etc.)
    if training:
        image = tf.image.random_flip_left_right(image)

    # Step 3: Scale to [-1, 1] range
    image = image * 2.0 - 1.0

    return {"image": image}


def make_dataset(
    split: str,
    batch_size_per_host: int,
    training: bool,
    seed: int
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline for CIFAR-10 with multi-host sharding.

    MULTI-HOST DATA SHARDING:
    -------------------------
    Each host processes a different shard of the dataset:
    - Host 0 sees examples 0, 32, 64, ... (every 32nd starting at 0)
    - Host 1 sees examples 1, 33, 65, ... (every 32nd starting at 1)
    - etc.

    This ensures:
    - No duplicate examples across hosts
    - Equal workload distribution
    - Deterministic data order (with seed)

    Args:
        split: TFDS split name ("train" or "test")
        batch_size_per_host: Batch size for THIS host (will be sharded across local devices)
        training: Whether to shuffle and augment
        seed: Random seed for shuffling

    Returns:
        tf.data.Dataset yielding batches of shape [batch_size_per_host, 32, 32, 3]
    """
    # Load CIFAR-10 from TensorFlow Datasets
    # shuffle_files=True ensures different file order per epoch
    ds = tfds.load("cifar10", split=split, shuffle_files=training)

    # CRITICAL: Shard dataset across hosts for multi-host training
    # Each host gets every Nth example, where N = process_count
    num_hosts = jax.process_count()
    host_index = jax.process_index()
    ds = ds.shard(num_shards=num_hosts, index=host_index)

    # Shuffle with a large buffer for good randomization
    # Buffer of 50k / num_hosts ensures seeing most examples before repeat
    if training:
        ds = ds.shuffle(
            buffer_size=50_000 // num_hosts,
            seed=seed + host_index,  # Different seed per host for variety
            reshuffle_each_iteration=True
        )

    # Apply preprocessing (normalization, augmentation)
    ds = ds.map(
        lambda ex: preprocess_image(ex, training),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and drop remainder (ensures consistent batch sizes)
    ds = ds.batch(batch_size_per_host, drop_remainder=True)

    # Prefetch for pipelining (overlap data loading with training)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Repeat infinitely for training (we control epochs in the loop)
    if training:
        ds = ds.repeat()

    return ds


def shard_batch_for_pmap(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reshape batch for pmap across local TPU devices.

    PMAP DATA LAYOUT:
    -----------------
    pmap expects the first axis to be the device axis:
    - Input: [batch_size_per_host, H, W, C]
    - Output: [local_device_count, batch_per_device, H, W, C]

    EXAMPLE (8 local TPUs, batch 256 per host):
    - Input: [256, 32, 32, 3]
    - Output: [8, 32, 32, 32, 3]
    - Each TPU processes 32 images

    Args:
        batch: Dictionary with 'image' of shape [batch_size, 32, 32, 3]

    Returns:
        Dictionary with 'image' of shape [local_devices, batch_per_device, 32, 32, 3]
    """
    num_local_devices = jax.local_device_count()
    images = batch["image"]

    # Validate batch size is divisible by local device count
    if images.shape[0] % num_local_devices != 0:
        raise ValueError(
            f"Batch size per host ({images.shape[0]}) must be divisible by "
            f"local device count ({num_local_devices})."
        )

    batch_per_device = images.shape[0] // num_local_devices

    # Reshape: [B, H, W, C] -> [D, B//D, H, W, C]
    images = images.reshape((num_local_devices, batch_per_device) + images.shape[1:])

    return {"image": images}


# ==============================================================================
# DIFFUSION SCHEDULE (DDPM)
# ==============================================================================
# Linear beta schedule and forward diffusion utilities.

def make_linear_schedule(
    num_timesteps: int,
    beta_start: float,
    beta_end: float
) -> Dict[str, jnp.ndarray]:
    """
    Create a linear beta schedule and precompute diffusion coefficients.

    DDPM FORWARD PROCESS:
    ---------------------
    The forward diffusion adds Gaussian noise over T timesteps:

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Where:
        - beta_t: Noise variance at timestep t
        - alpha_t = 1 - beta_t: Signal retention at timestep t
        - alpha_bar_t = prod(alpha_1, ..., alpha_t): Cumulative signal retention

    LINEAR SCHEDULE:
    - beta increases linearly from beta_start to beta_end
    - beta_start=1e-4, beta_end=0.02 are standard DDPM values
    - T=1000 timesteps is standard

    Args:
        num_timesteps: Total diffusion timesteps T
        beta_start: Initial noise variance (small, e.g., 1e-4)
        beta_end: Final noise variance (larger, e.g., 0.02)

    Returns:
        Dictionary with precomputed schedule arrays:
        - betas: [T] noise variances
        - alphas: [T] signal retention (1 - beta)
        - alpha_bars: [T] cumulative product of alphas
        - sqrt_alpha_bars: [T] for forward process mean
        - sqrt_one_minus_alpha_bars: [T] for forward process std
    """
    # Linear interpolation from beta_start to beta_end
    betas = jnp.linspace(beta_start, beta_end, num_timesteps, dtype=jnp.float32)

    # Alpha = 1 - beta (signal retention per step)
    alphas = 1.0 - betas

    # Alpha_bar = cumulative product (total signal retention at each step)
    alpha_bars = jnp.cumprod(alphas, axis=0)

    # Precompute square roots for efficiency
    sqrt_alpha_bars = jnp.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - alpha_bars)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
    }


def extract_schedule_value(
    schedule_array: jnp.ndarray,
    timesteps: jnp.ndarray,
    broadcast_shape: Tuple[int, ...]
) -> jnp.ndarray:
    """
    Extract schedule values at given timesteps and reshape for broadcasting.

    Args:
        schedule_array: [T] array of schedule values
        timesteps: [B] array of timestep indices
        broadcast_shape: Shape of tensor to broadcast to (e.g., [B, H, W, C])

    Returns:
        [B, 1, 1, 1] array broadcastable to [B, H, W, C]
    """
    # Gather values at specified timesteps
    values = schedule_array[timesteps]  # [B]

    # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
    num_dims = len(broadcast_shape)
    return values.reshape((timesteps.shape[0],) + (1,) * (num_dims - 1))


def forward_diffusion(
    rng: jax.Array,
    schedule: Dict[str, jnp.ndarray],
    x_0: jnp.ndarray,
    timesteps: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from the forward diffusion process q(x_t | x_0).

    FORWARD DIFFUSION EQUATION:
    ---------------------------
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Where epsilon ~ N(0, I) is standard Gaussian noise.

    This closed-form sampling allows jumping directly to any timestep t
    without iterating through intermediate steps (key DDPM insight).

    Args:
        rng: JAX random key
        schedule: Precomputed diffusion schedule
        x_0: Clean images [B, H, W, C]
        timesteps: Timestep indices [B]

    Returns:
        Tuple of:
        - x_t: Noisy images at timestep t [B, H, W, C]
        - epsilon: The noise that was added [B, H, W, C]
    """
    # Sample noise
    epsilon = random.normal(rng, x_0.shape, dtype=jnp.float32)

    # Get schedule coefficients
    sqrt_alpha_bar = extract_schedule_value(
        schedule["sqrt_alpha_bars"], timesteps, x_0.shape
    )
    sqrt_one_minus_alpha_bar = extract_schedule_value(
        schedule["sqrt_one_minus_alpha_bars"], timesteps, x_0.shape
    )

    # Apply forward diffusion
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon

    return x_t, epsilon


# ==============================================================================
# MODEL: MINI U-NET FOR DIFFUSION
# ==============================================================================
# A minimal U-Net architecture for CIFAR-10 diffusion.

def sinusoidal_timestep_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Create sinusoidal positional embeddings for timesteps.

    POSITIONAL ENCODING:
    --------------------
    Similar to Transformer positional encodings, but for scalar timesteps.
    Uses different frequencies to encode the timestep value:

        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    This allows the model to distinguish between different timesteps
    and learn timestep-dependent denoising behavior.

    Args:
        timesteps: [B] integer timestep values
        dim: Embedding dimension (must be even)

    Returns:
        [B, dim] timestep embeddings
    """
    half_dim = dim // 2

    # Frequency scaling factor
    # log(10000) / (half_dim - 1) gives geometric series of frequencies
    freq_scale = -math.log(10_000.0) / (half_dim - 1)
    frequencies = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * freq_scale)

    # Compute angles: [B, half_dim]
    timesteps_float = timesteps.astype(jnp.float32)[:, None]
    angles = timesteps_float * frequencies[None, :]

    # Concatenate sin and cos embeddings
    embeddings = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    # Pad if dim is odd
    if dim % 2 == 1:
        embeddings = jnp.pad(embeddings, ((0, 0), (0, 1)))

    return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    ARCHITECTURE:
    -------------
    1. GroupNorm -> Swish -> Conv3x3
    2. Add timestep embedding (projected to channel dim)
    3. GroupNorm -> Swish -> Conv3x3
    4. Residual connection (with 1x1 conv if channels change)

    The timestep embedding is added after the first convolution,
    allowing the model to modulate its behavior based on noise level.
    """
    out_channels: int
    time_embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, time_embed: jnp.ndarray, train: bool) -> jnp.ndarray:
        """
        Forward pass through residual block.

        Args:
            x: Input features [B, H, W, C_in]
            time_embed: Timestep embeddings [B, time_embed_dim]
            train: Training mode flag (unused here, but standard Flax convention)

        Returns:
            Output features [B, H, W, out_channels]
        """
        # First half: norm -> activation -> conv
        h = nn.GroupNorm(num_groups=8)(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME")(h)

        # Inject timestep information
        # Project time embedding to channel dimension and add spatially
        time_proj = nn.swish(time_embed)
        time_proj = nn.Dense(self.out_channels)(time_proj)
        h = h + time_proj[:, None, None, :]  # Broadcast over spatial dims

        # Second half: norm -> activation -> conv
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME")(h)

        # Residual connection
        # Use 1x1 conv to match channels if needed
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, (1, 1), padding="SAME")(x)

        return x + h


class DownsampleBlock(nn.Module):
    """
    Spatial downsampling by factor of 2 using strided convolution.

    Strided convolution is preferred over pooling because:
    - Learnable downsampling
    - Better gradient flow
    - Preserves more information
    """
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 3x3 conv with stride 2: [B, H, W, C] -> [B, H/2, W/2, out_channels]
        return nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME"
        )(x)


class UpsampleBlock(nn.Module):
    """
    Spatial upsampling by factor of 2 using resize + convolution.

    Two-step upsampling:
    1. Nearest neighbor resize (fast, no learnable params)
    2. 3x3 convolution (learnable refinement)

    This avoids checkerboard artifacts from transposed convolution.
    """
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape

        # Step 1: Nearest neighbor upsampling
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method="nearest")

        # Step 2: Convolution for refinement
        x = nn.Conv(self.out_channels, (3, 3), padding="SAME")(x)

        return x


class MiniUNet(nn.Module):
    """
    Minimal U-Net for CIFAR-10 diffusion.

    ARCHITECTURE OVERVIEW:
    ----------------------
    Encoder (downsampling path):
        32x32 -> 16x16 -> 8x8

    Bottleneck:
        8x8 processing

    Decoder (upsampling path):
        8x8 -> 16x16 -> 32x32

    Skip connections: Concatenate encoder features to decoder

    CHANNEL PROGRESSION:
    - Level 0 (32x32): base_channels (e.g., 64)
    - Level 1 (16x16): base_channels * 2 (e.g., 128)
    - Level 2 (8x8): base_channels * 4 (e.g., 256)

    INPUT/OUTPUT:
    - Input: (x_t, t) where x_t is noisy image [B, 32, 32, 3], t is timestep [B]
    - Output: epsilon_pred [B, 32, 32, 3] (predicted noise)
    """
    base_channels: int = 64
    time_embed_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool) -> jnp.ndarray:
        """
        Forward pass: predict noise from noisy image and timestep.

        Args:
            x: Noisy images [B, 32, 32, 3]
            timesteps: Diffusion timesteps [B]
            train: Training mode flag

        Returns:
            Predicted noise [B, 32, 32, 3]
        """
        # =====================================================================
        # TIMESTEP EMBEDDING
        # =====================================================================
        # Convert scalar timestep to high-dimensional embedding
        # This allows the model to condition on the noise level

        time_embed = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        time_embed = nn.Dense(self.time_embed_dim)(time_embed)
        time_embed = nn.swish(time_embed)
        time_embed = nn.Dense(self.time_embed_dim)(time_embed)

        # =====================================================================
        # ENCODER (Downsampling Path)
        # =====================================================================

        # Initial projection: 3 -> base_channels
        # 32x32x3 -> 32x32x64
        h0 = nn.Conv(self.base_channels, (3, 3), padding="SAME")(x)

        # Level 0: 32x32, base_channels
        d0 = ResidualBlock(self.base_channels, self.time_embed_dim)(h0, time_embed, train)
        d0 = ResidualBlock(self.base_channels, self.time_embed_dim)(d0, time_embed, train)

        # Downsample: 32x32 -> 16x16
        h1 = DownsampleBlock(self.base_channels * 2)(d0)

        # Level 1: 16x16, base_channels * 2
        d1 = ResidualBlock(self.base_channels * 2, self.time_embed_dim)(h1, time_embed, train)
        d1 = ResidualBlock(self.base_channels * 2, self.time_embed_dim)(d1, time_embed, train)

        # Downsample: 16x16 -> 8x8
        h2 = DownsampleBlock(self.base_channels * 4)(d1)

        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        # 8x8, base_channels * 4 (256 channels with base=64)

        mid = ResidualBlock(self.base_channels * 4, self.time_embed_dim)(h2, time_embed, train)
        mid = ResidualBlock(self.base_channels * 4, self.time_embed_dim)(mid, time_embed, train)

        # =====================================================================
        # DECODER (Upsampling Path)
        # =====================================================================

        # Upsample: 8x8 -> 16x16
        u1 = UpsampleBlock(self.base_channels * 2)(mid)

        # Skip connection: concatenate encoder features
        u1 = jnp.concatenate([u1, d1], axis=-1)

        # Level 1: 16x16
        u1 = ResidualBlock(self.base_channels * 2, self.time_embed_dim)(u1, time_embed, train)
        u1 = ResidualBlock(self.base_channels * 2, self.time_embed_dim)(u1, time_embed, train)

        # Upsample: 16x16 -> 32x32
        u0 = UpsampleBlock(self.base_channels)(u1)

        # Skip connection: concatenate encoder features
        u0 = jnp.concatenate([u0, d0], axis=-1)

        # Level 0: 32x32
        u0 = ResidualBlock(self.base_channels, self.time_embed_dim)(u0, time_embed, train)
        u0 = ResidualBlock(self.base_channels, self.time_embed_dim)(u0, time_embed, train)

        # =====================================================================
        # OUTPUT PROJECTION
        # =====================================================================
        # Project back to image channels (3 for RGB)

        h = nn.GroupNorm(num_groups=8)(u0)
        h = nn.swish(h)
        epsilon_pred = nn.Conv(3, (3, 3), padding="SAME")(h)

        return epsilon_pred


# ==============================================================================
# TRAINING STATE AND OPTIMIZATION
# ==============================================================================

class DDPMTrainState(train_state.TrainState):
    """
    Extended Flax TrainState for DDPM training.

    Inherits from train_state.TrainState which provides:
    - step: Global training step counter
    - apply_fn: Model forward function
    - params: Model parameters
    - tx: Optimizer (optax)
    - opt_state: Optimizer state

    We use the base class as-is for this simple demo.
    For advanced use cases, you might add:
    - EMA parameters for sampling
    - Learning rate schedule state
    - Gradient clipping statistics
    """
    pass


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float
) -> DDPMTrainState:
    """
    Initialize model parameters and optimizer state.

    PARAMETER INITIALIZATION:
    -------------------------
    - Uses Flax default initializers (Lecun normal for Dense, He for Conv)
    - Initialized with dummy input to infer shapes
    - Random seed ensures reproducibility

    OPTIMIZER (AdamW):
    ------------------
    - Adam with decoupled weight decay
    - Good default for diffusion models
    - Weight decay helps regularization

    Args:
        rng: JAX random key for initialization
        model: MiniUNet instance
        learning_rate: Adam learning rate (e.g., 2e-4)
        weight_decay: L2 regularization strength (e.g., 1e-4)

    Returns:
        Initialized DDPMTrainState
    """
    # Create dummy inputs for parameter initialization
    dummy_images = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    dummy_timesteps = jnp.zeros((1,), dtype=jnp.int32)

    # Initialize parameters
    params = model.init(
        {"params": rng},
        dummy_images,
        dummy_timesteps,
        train=True
    )["params"]

    # Create AdamW optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    return DDPMTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )


def compute_ddpm_loss(
    params,
    apply_fn,
    schedule: Dict[str, jnp.ndarray],
    rng: jax.Array,
    x_0: jnp.ndarray,
    timesteps: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute DDPM training loss (simple MSE on noise prediction).

    DDPM TRAINING OBJECTIVE:
    ------------------------
    The model learns to predict the noise that was added to x_0:

        L = E[||epsilon - epsilon_theta(x_t, t)||^2]

    Where:
        - epsilon: The actual noise added (sampled from N(0, I))
        - epsilon_theta: Model's noise prediction
        - x_t: Noisy image at timestep t
        - t: Timestep (uniformly sampled from [0, T))

    This is equivalent to learning the score function (gradient of log density).

    Args:
        params: Model parameters
        apply_fn: Model forward function
        schedule: Precomputed diffusion schedule
        rng: Random key for noise sampling
        x_0: Clean images [B, H, W, C]
        timesteps: Sampled timesteps [B]

    Returns:
        Scalar MSE loss
    """
    # Forward diffusion: add noise to clean images
    x_t, epsilon = forward_diffusion(rng, schedule, x_0, timesteps)

    # Model predicts the noise
    epsilon_pred = apply_fn({"params": params}, x_t, timesteps, train=True)

    # MSE loss between predicted and actual noise
    loss = jnp.mean((epsilon_pred - epsilon) ** 2)

    return loss


def train_step(
    state: DDPMTrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    schedule: Dict[str, jnp.ndarray],
    num_timesteps: int
) -> Tuple[DDPMTrainState, Dict[str, jnp.ndarray]]:
    """
    Execute one training step (designed to be pmapped).

    TRAINING STEP OVERVIEW:
    -----------------------
    1. Sample random timesteps for each image in batch
    2. Compute loss (forward diffusion + noise prediction)
    3. Compute gradients via backprop
    4. All-reduce gradients across all devices (pmean)
    5. Update parameters with optimizer

    GRADIENT SYNCHRONIZATION:
    -------------------------
    jax.lax.pmean averages gradients across all devices:
    - axis_name="data" identifies the pmap axis
    - Works across all 256 TPU chips via high-speed interconnect
    - Ensures all devices have identical parameters after update

    Args:
        state: Current training state (params, optimizer state, step)
        batch: Dictionary with "image" [B, H, W, C]
        rng: Per-device random key
        schedule: Diffusion schedule
        num_timesteps: Total diffusion steps T

    Returns:
        Tuple of (new_state, metrics_dict)
    """
    x_0 = batch["image"]
    batch_size = x_0.shape[0]

    # Sample random timesteps uniformly from [0, T)
    rng, timestep_rng, noise_rng = random.split(rng, 3)
    timesteps = random.randint(
        timestep_rng,
        (batch_size,),
        minval=0,
        maxval=num_timesteps,
        dtype=jnp.int32
    )

    # Compute loss and gradients
    def loss_fn(params):
        return compute_ddpm_loss(
            params, state.apply_fn, schedule, noise_rng, x_0, timesteps
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # CRITICAL: Synchronize gradients across all devices
    # This is what makes distributed training work!
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")

    # Apply gradients (optimizer step)
    new_state = state.apply_gradients(grads=grads)

    # Collect metrics for logging
    metrics = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }

    return new_state, metrics


# ==============================================================================
# SAMPLING (Reverse Diffusion)
# ==============================================================================

def reverse_diffusion_step(
    rng: jax.Array,
    apply_fn,
    params,
    schedule: Dict[str, jnp.ndarray],
    x_t: jnp.ndarray,
    t: int
) -> jnp.ndarray:
    """
    One step of reverse diffusion: sample x_{t-1} from p(x_{t-1} | x_t).

    REVERSE PROCESS:
    ----------------
    The model approximates the reverse diffusion:

        p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 I)

    Where the mean is computed from the noise prediction:

        mu_theta = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta)

    And variance is simply beta_t (could also use learned variance).

    Args:
        rng: Random key for sampling
        apply_fn: Model forward function
        params: Model parameters
        schedule: Diffusion schedule
        x_t: Noisy images at timestep t [B, H, W, C]
        t: Current timestep (integer)

    Returns:
        x_{t-1}: Slightly denoised images [B, H, W, C]
    """
    # Broadcast timestep to batch
    timesteps = jnp.full((x_t.shape[0],), t, dtype=jnp.int32)

    # Predict noise
    epsilon_pred = apply_fn({"params": params}, x_t, timesteps, train=False)

    # Get schedule values for this timestep
    beta_t = schedule["betas"][t]
    alpha_t = schedule["alphas"][t]
    alpha_bar_t = schedule["alpha_bars"][t]

    # Compute mean of p(x_{t-1} | x_t)
    # mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon)
    coef = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
    mean = (1.0 / jnp.sqrt(alpha_t)) * (x_t - coef * epsilon_pred)

    # At t=0, we're done (return the mean directly)
    if t == 0:
        return mean

    # Otherwise, add noise scaled by sqrt(beta_t)
    noise = random.normal(rng, x_t.shape, dtype=jnp.float32)
    return mean + jnp.sqrt(beta_t) * noise


def generate_samples(
    rng: jax.Array,
    state: DDPMTrainState,
    model: nn.Module,
    schedule: Dict[str, jnp.ndarray],
    num_timesteps: int,
    num_samples: int
) -> jnp.ndarray:
    """
    Generate images by iterating the reverse diffusion process.

    SAMPLING PROCEDURE:
    -------------------
    1. Start from pure Gaussian noise x_T ~ N(0, I)
    2. Iterate: x_{t-1} = reverse_step(x_t, t) for t = T-1, ..., 0
    3. Final x_0 is the generated image

    This is slow (T forward passes) but generates high-quality samples.
    For faster sampling, consider DDIM or other accelerated methods.

    Args:
        rng: Random key
        state: Training state with model params
        model: MiniUNet instance
        schedule: Diffusion schedule
        num_timesteps: T (e.g., 1000)
        num_samples: Number of images to generate

    Returns:
        Generated images [num_samples, 32, 32, 3] in range [-1, 1]
    """
    # Start from pure noise
    rng, init_rng = random.split(rng)
    x = random.normal(init_rng, (num_samples, 32, 32, 3), dtype=jnp.float32)

    # Iterate reverse diffusion
    for t in reversed(range(num_timesteps)):
        rng, step_rng = random.split(rng)
        x = reverse_diffusion_step(
            step_rng,
            model.apply,
            state.params,
            schedule,
            x,
            t
        )

    return x


def save_image_grid(
    images: np.ndarray,
    path: str,
    grid_size: int
) -> None:
    """
    Save a grid of images to disk.

    Args:
        images: float32 array in [-1, 1] range [N, H, W, C]
        path: Output file path
        grid_size: Number of images per row/column
    """
    # Convert from [-1, 1] to [0, 255] uint8
    images = (images + 1.0) * 0.5  # [-1, 1] -> [0, 1]
    images = (images * 255.0).clip(0, 255).astype(np.uint8)

    n, h, w, c = images.shape

    if n != grid_size * grid_size:
        raise ValueError(
            f"Number of images ({n}) must equal grid_size^2 ({grid_size**2})"
        )

    # Arrange into grid
    rows = []
    for r in range(grid_size):
        row_images = images[r * grid_size : (r + 1) * grid_size]
        row = np.concatenate(row_images, axis=1)  # Horizontal concatenation
        rows.append(row)

    grid = np.concatenate(rows, axis=0)  # Vertical concatenation

    # Save as PNG
    Image.fromarray(grid).save(path)


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    """
    Main training function.

    EXECUTION FLOW:
    ---------------
    1. Parse command line arguments
    2. Initialize distributed runtime (for multi-host TPU)
    3. Set up logging
    4. Load dataset with proper sharding
    5. Create model and initialize parameters
    6. Replicate state across local devices
    7. Training loop with logging and periodic sampling
    """

    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================

    parser = argparse.ArgumentParser(
        description="Train DDPM on CIFAR-10 with JAX/TPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Output and reproducibility
    parser.add_argument("--workdir", type=str, default="./runs/ddpm_cifar10",
                        help="Directory for checkpoints, samples, and logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size_global", type=int, default=2048,
                        help="Total batch size across ALL devices (will be sharded)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for AdamW")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay for AdamW")

    # Diffusion hyperparameters
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps T")
    parser.add_argument("--beta_start", type=float, default=1e-4,
                        help="Starting beta (noise variance)")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="Ending beta (noise variance)")

    # Model hyperparameters
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channel count for U-Net")

    # Logging and sampling
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log training metrics every N steps")
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate samples every N steps")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples to generate (must be perfect square)")

    # Weights & Biases configuration
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="ddpm-cifar10",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team name)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")

    args = parser.parse_args()

    # =========================================================================
    # DISTRIBUTED INITIALIZATION
    # =========================================================================
    # CRITICAL: Must be called before any JAX operations in multi-host setting

    dist_info = maybe_init_distributed()

    # =========================================================================
    # LOGGING SETUP
    # =========================================================================

    logger = Logger(is_main_process(), args.workdir)

    # Log startup banner
    logger.separator("DDPM Training on CIFAR-10")

    # Log distributed setup info
    device_info = get_device_info()
    logger.config("Distributed Setup", {
        "process_count (workers)": device_info["process_count"],
        "process_index (this worker)": device_info["process_index"],
        "global_device_count (total TPUs)": device_info["global_device_count"],
        "local_device_count (TPUs per worker)": device_info["local_device_count"],
        "device_type": device_info["device_type"],
        "distributed_initialized": dist_info["initialized"],
    })

    # =========================================================================
    # WEIGHTS & BIASES INITIALIZATION
    # =========================================================================
    # Initialize wandb only on main process to avoid duplicate logging.
    # wandb tracks experiments, logs metrics, and stores generated samples.

    use_wandb = args.use_wandb and WANDB_AVAILABLE and is_main_process()

    if args.use_wandb and not WANDB_AVAILABLE:
        logger.info("WARNING: --use_wandb specified but wandb not installed. Skipping.")

    if use_wandb:
        logger.info("Initializing Weights & Biases...")

        # Prepare wandb config with all hyperparameters
        wandb_config = {
            # Training
            "epochs": args.epochs,
            "batch_size_global": args.batch_size_global,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "seed": args.seed,
            # Diffusion
            "num_timesteps": args.num_timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            # Model
            "architecture": "MiniUNet",
            "base_channels": args.base_channels,
            "time_embed_dim": 256,
            # Distributed
            "num_hosts": device_info["process_count"],
            "total_devices": device_info["global_device_count"],
            "device_type": device_info["device_type"],
        }

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config,
            dir=args.workdir,
        )

        logger.info(f"W&B run initialized: {wandb.run.name}")
        logger.info(f"W&B dashboard: {wandb.run.get_url()}")

    # =========================================================================
    # BATCH SIZE CALCULATION
    # =========================================================================
    # Global batch is split across all devices

    num_devices_global = jax.device_count()
    num_devices_local = jax.local_device_count()
    num_hosts = jax.process_count()

    # Validate batch size
    if args.batch_size_global % num_devices_global != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size_global}) must be divisible by "
            f"total device count ({num_devices_global})"
        )

    batch_size_per_device = args.batch_size_global // num_devices_global
    batch_size_per_host = batch_size_per_device * num_devices_local

    logger.config("Batch Size Configuration", {
        "batch_size_global": args.batch_size_global,
        "batch_size_per_host": batch_size_per_host,
        "batch_size_per_device": batch_size_per_device,
        "num_hosts": num_hosts,
        "devices_per_host": num_devices_local,
    })

    # =========================================================================
    # HYPERPARAMETERS LOGGING
    # =========================================================================

    logger.config("Training Hyperparameters", {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "seed": args.seed,
    })

    logger.config("Diffusion Hyperparameters", {
        "num_timesteps": args.num_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
    })

    logger.config("Model Hyperparameters", {
        "architecture": "MiniUNet",
        "base_channels": args.base_channels,
        "time_embed_dim": 256,
    })

    # =========================================================================
    # DATASET SETUP
    # =========================================================================

    logger.info("Loading CIFAR-10 dataset...")

    # Prevent TensorFlow from using GPU memory (we only use it for data loading)
    tf.config.set_visible_devices([], "GPU")

    # Set TF random seed per host for different shuffling
    tf.random.set_seed(args.seed + jax.process_index())

    # Create training dataset with proper sharding
    train_ds = make_dataset(
        split="train",
        batch_size_per_host=batch_size_per_host,
        training=True,
        seed=args.seed
    )

    # Create iterator
    train_iter = iter(tfds.as_numpy(train_ds))

    # Calculate steps per epoch
    # CIFAR-10 has 50,000 training images, sharded across hosts
    train_examples_total = 50_000
    train_examples_per_host = train_examples_total // num_hosts
    steps_per_epoch = train_examples_per_host // batch_size_per_host

    # Handle edge case where batch is larger than shard
    if steps_per_epoch == 0:
        steps_per_epoch = 1
        logger.info("WARNING: Batch size larger than data shard, using 1 step per epoch")

    total_steps = args.epochs * steps_per_epoch

    logger.config("Dataset Configuration", {
        "dataset": "CIFAR-10",
        "train_examples_total": train_examples_total,
        "train_examples_per_host": train_examples_per_host,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    })

    # =========================================================================
    # MODEL AND SCHEDULE CREATION
    # =========================================================================

    logger.info("Creating model and diffusion schedule...")

    # Create diffusion schedule
    schedule = make_linear_schedule(
        args.num_timesteps,
        args.beta_start,
        args.beta_end
    )

    # Create model
    model = MiniUNet(
        base_channels=args.base_channels,
        time_embed_dim=256
    )

    # Initialize training state
    rng = random.PRNGKey(args.seed)
    rng, init_rng = random.split(rng)

    state = create_train_state(
        init_rng,
        model,
        learning_rate=args.lr,
        weight_decay=args.wd
    )

    # Count parameters
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    logger.info(f"Model initialized with {num_params:,} parameters")

    # =========================================================================
    # REPLICATE STATE FOR PMAP
    # =========================================================================
    # Each local device gets a copy of the parameters

    logger.info("Replicating model across local devices...")

    local_devices = jax.local_devices()
    state = flax.jax_utils.replicate(state, devices=local_devices)

    # =========================================================================
    # COMPILE PMAP'D TRAINING STEP
    # =========================================================================

    logger.info("Compiling training step (this may take a minute)...")

    # pmap'd training step with gradient sync across all devices
    p_train_step = jax.pmap(
        lambda s, b, r: train_step(s, b, r, schedule, args.num_timesteps),
        axis_name="data",  # Name of the data-parallel axis
        devices=local_devices,
    )

    # =========================================================================
    # CREATE OUTPUT DIRECTORIES
    # =========================================================================

    if is_main_process():
        os.makedirs(args.workdir, exist_ok=True)
        os.makedirs(os.path.join(args.workdir, "samples"), exist_ok=True)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    logger.separator("Starting Training")

    global_step = 0
    epoch_start_time = time.time()
    step_times = []

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step_in_epoch in range(steps_per_epoch):
            step_start_time = time.time()
            global_step += 1

            # -----------------------------------------------------------------
            # GET BATCH
            # -----------------------------------------------------------------
            try:
                batch = next(train_iter)
            except StopIteration:
                # Shouldn't happen with repeat(), but just in case
                train_iter = iter(tfds.as_numpy(train_ds))
                batch = next(train_iter)

            # Reshape batch for pmap: [B, H, W, C] -> [D, B/D, H, W, C]
            batch = shard_batch_for_pmap(batch)

            # -----------------------------------------------------------------
            # PREPARE RNG
            # -----------------------------------------------------------------
            # Each device needs its own RNG key
            rng, step_rng = random.split(rng)
            step_rngs = random.split(step_rng, num_devices_local)
            step_rngs = jax.device_put_sharded(list(step_rngs), local_devices)

            # -----------------------------------------------------------------
            # TRAINING STEP
            # -----------------------------------------------------------------
            state, metrics = p_train_step(state, batch, step_rngs)

            # Track timing
            step_time = time.time() - step_start_time
            step_times.append(step_time)

            # Accumulate epoch loss
            loss_value = float(flax.jax_utils.unreplicate(metrics["loss"]))
            epoch_loss_sum += loss_value
            epoch_steps += 1

            # -----------------------------------------------------------------
            # LOGGING
            # -----------------------------------------------------------------
            if global_step % args.log_every == 0:
                # Calculate throughput
                recent_times = step_times[-args.log_every:]
                avg_step_time = sum(recent_times) / len(recent_times)
                images_per_sec = args.batch_size_global / avg_step_time

                # Get metrics from device
                grad_norm = float(flax.jax_utils.unreplicate(metrics["grad_norm"]))

                logger.metric(global_step, epoch + 1, {
                    "loss": loss_value,
                    "grad_norm": grad_norm,
                    "imgs/sec": int(images_per_sec),
                    "step_time": f"{avg_step_time:.3f}s",
                })

                # Log to Weights & Biases
                if use_wandb:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/grad_norm": grad_norm,
                        "train/images_per_sec": images_per_sec,
                        "train/step_time": avg_step_time,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

            # -----------------------------------------------------------------
            # SAMPLE GENERATION
            # -----------------------------------------------------------------
            if global_step % args.sample_every == 0 and is_main_process():
                logger.info("Generating samples...")

                # Unreplicate state for sampling (run on single device)
                state_unrep = flax.jax_utils.unreplicate(state)

                rng, sample_rng = random.split(rng)
                samples = generate_samples(
                    sample_rng,
                    state_unrep,
                    model,
                    schedule,
                    args.num_timesteps,
                    args.num_samples
                )

                # Save grid locally
                samples_np = np.array(samples)
                grid_size = int(math.sqrt(args.num_samples))
                sample_path = os.path.join(
                    args.workdir, "samples", f"samples_step_{global_step:06d}.png"
                )
                save_image_grid(samples_np, sample_path, grid_size)

                logger.sample(global_step, sample_path, args.num_samples)

                # Log samples to Weights & Biases
                if use_wandb:
                    # Convert samples from [-1, 1] to [0, 255] for wandb
                    samples_uint8 = ((samples_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                    # Log individual samples as a grid of images
                    wandb.log({
                        "samples/grid": wandb.Image(
                            sample_path,
                            caption=f"Generated samples at step {global_step}"
                        ),
                        # Also log a few individual samples for closer inspection
                        "samples/individual": [
                            wandb.Image(samples_uint8[i], caption=f"Sample {i}")
                            for i in range(min(8, len(samples_uint8)))
                        ],
                    }, step=global_step)

        # ---------------------------------------------------------------------
        # EPOCH SUMMARY
        # ---------------------------------------------------------------------
        epoch_time = time.time() - epoch_start_time
        epoch_loss_avg = epoch_loss_sum / max(epoch_steps, 1)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} completed | "
            f"avg_loss={epoch_loss_avg:.6f} | "
            f"time={epoch_time:.1f}s"
        )

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================

    logger.separator("Training Complete")

    total_time = time.time() - logger.start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Samples saved to: {os.path.join(args.workdir, 'samples')}")

    # Finalize Weights & Biases
    if use_wandb:
        # Log final summary metrics
        wandb.summary["total_steps"] = global_step
        wandb.summary["total_time_seconds"] = total_time
        wandb.summary["final_loss"] = epoch_loss_avg
        wandb.finish()
        logger.info("W&B run finished and synced.")

    # =========================================================================
    # CLEAN SHUTDOWN FOR MULTI-HOST TPU
    # =========================================================================
    # Synchronize all hosts before exit to avoid "GetSliceInfo" errors.
    # This ensures all workers finish at the same time.

    if jax.process_count() > 1:
        logger.info("Synchronizing all workers before exit...")

        # Create a simple barrier by doing a collective operation
        # This forces all hosts to sync before any can exit
        x = jnp.ones((1,))
        x = jax.pmap(lambda x: jax.lax.psum(x, axis_name='i'), axis_name='i')(
            x[None].repeat(jax.local_device_count(), axis=0)
        )
        x.block_until_ready()

        # Small delay to ensure all logging/IO completes
        time.sleep(2)

        logger.info("All workers synchronized. Exiting cleanly.")

    logger.close()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
