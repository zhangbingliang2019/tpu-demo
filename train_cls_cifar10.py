#!/usr/bin/env python3
"""
================================================================================
CIFAR-10 Image Classification with JAX/Flax on TPU (Multi-Host Ready)
================================================================================

This script trains a simple CNN classifier on CIFAR-10 using JAX/Flax.
It closely follows the structure, logging style, and multi-host TPU setup
of `train_ddpm_cifar10.py`, but with a standard supervised classification
objective instead of diffusion.

Key Features:
------------
- JAX/Flax model (simple CNN) with ~few million parameters
- Cross-entropy loss with accuracy metrics
- tf.data input pipeline with host-level sharding for multi-host TPU pods
- pmap-based data parallel training with gradient all-reduce
- Rich-based console logging (if available)
- Detailed logging to Weights & Biases (wandb), mirroring DDPM script style

Usage (single host test):
-------------------------
    python train_cifar10_classification.py --batch_size_global 256 --epochs 2

Usage (multi-host TPU via gcloud ssh --worker=all):
---------------------------------------------------
    python train_cifar10_classification.py \\
        --batch_size_global 8192 \\
        --epochs 100 \\
        --use_wandb --wandb_project cifar10-classification

================================================================================
"""

import os
import time
import math
import argparse
import datetime
from typing import Any, Dict, Tuple

import numpy as np

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn
from flax.training import train_state

import optax

import tensorflow as tf
import tensorflow_datasets as tfds

# Rich library for pretty logging
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Weights & Biases
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==============================================================================
# LOGGING UTILITIES (copied & slightly adapted from DDPM script)
# ==============================================================================


class Logger:
    """
    Centralized logger for multi-host TPU training.

    Same behavior as in `train_ddpm_cifar10.py`:
    - Only main process prints to stdout
    - Optional Rich formatting
    - Logs to a persistent log file in workdir
    """

    def __init__(self, is_main: bool, workdir: str):
        self.is_main = is_main
        self.workdir = workdir
        self.start_time = time.time()

        if RICH_AVAILABLE and is_main:
            self.console = Console()
        else:
            self.console = None

        if is_main:
            os.makedirs(workdir, exist_ok=True)
            self.log_file = open(os.path.join(workdir, "train.log"), "a")
        else:
            self.log_file = None

    def _timestamp(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_file(self, message: str):
        if self.log_file:
            self.log_file.write(f"[{self._timestamp()}] {message}\n")
            self.log_file.flush()

    def info(self, message: str):
        if not self.is_main:
            return
        formatted = f"[{self._elapsed()}] {message}"
        self._write_file(f"INFO: {message}")
        if self.console:
            self.console.print(f"[blue][INFO][/blue] {formatted}")
        else:
            print(f"[INFO] {formatted}")

    def config(self, title: str, config_dict: Dict[str, Any]):
        if not self.is_main:
            return

        self._write_file(f"CONFIG: {title} = {config_dict}")

        if self.console:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            for k, v in config_dict.items():
                table.add_row(str(k), str(v))
            self.console.print(table)
        else:
            print(f"\n{'=' * 60}")
            print(f" {title}")
            print(f"{'=' * 60}")
            for k, v in config_dict.items():
                print(f"  {k}: {v}")
            print(f"{'=' * 60}\n")

    def metric(self, step: int, epoch: int, metrics: Dict[str, float], prefix: str = "TRAIN"):
        if not self.is_main:
            return

        metrics_str = " | ".join(
            [
                f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ]
        )

        self._write_file(f"METRIC: step={step} epoch={epoch} {metrics_str}")
        formatted = f"[{self._elapsed()}] step={step:>6d} epoch={epoch:>3d} | {metrics_str}"

        if self.console:
            color = "green" if prefix.upper() == "TRAIN" else "magenta"
            self.console.print(f"[{color}][{prefix.upper()}][/{color}] {formatted}")
        else:
            print(f"[{prefix.upper()}] {formatted}")

    def separator(self, title: str = ""):
        if not self.is_main:
            return
        if self.console:
            if title:
                self.console.print(Panel(title, style="bold blue"))
            else:
                self.console.print("─" * 60)
        else:
            if title:
                print(f"\n{'=' * 60}")
                print(f" {title}")
                print(f"{'=' * 60}")
            else:
                print("-" * 60)

    def close(self):
        if self.log_file:
            self.log_file.close()


# ==============================================================================
# DISTRIBUTED TRAINING UTILITIES (same as DDPM script)
# ==============================================================================


def maybe_init_distributed() -> Dict[str, Any]:
    dist_info = {
        "initialized": False,
        "process_count": 1,
        "process_index": 0,
        "coordinator": "N/A",
    }
    try:
        process_count = int(os.environ.get("JAX_PROCESS_COUNT", 1))
        if process_count > 1:
            jax.distributed.initialize()
            dist_info["initialized"] = True
            dist_info["coordinator"] = os.environ.get("JAX_COORDINATOR_ADDRESS", "unknown")

        dist_info["process_count"] = jax.process_count()
        dist_info["process_index"] = jax.process_index()
    except Exception as e:
        print(f"[WARNING] Distributed init failed: {e}")
    return dist_info


def is_main_process() -> bool:
    return jax.process_index() == 0


def get_device_info() -> Dict[str, Any]:
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
# DATA PIPELINE (CIFAR-10) FOR CLASSIFICATION
# ==============================================================================


def preprocess_example(example: Dict[str, Any], training: bool) -> Dict[str, Any]:
    """
    Preprocess CIFAR-10 example for classification.

    Steps:
    - Convert uint8 image [0, 255] -> float32 [0, 1]
    - Random horizontal flip for training
    - Scale to [-1, 1] (consistent with DDPM script, but fine for classification)
    """
    image = tf.cast(example["image"], tf.float32) / 255.0

    if training:
        # Standard CIFAR-10 augmentation: padding + random crop + horizontal flip
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)

    # Keep the same [-1, 1] normalization convention as the DDPM script
    image = image * 2.0 - 1.0

    label = tf.cast(example["label"], tf.int32)
    return {"image": image, "label": label}


def make_dataset(
    split: str,
    batch_size_per_host: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Build tf.data pipeline with host-level sharding, same logic as DDPM.
    Returns batches:
        {"image": [B, 32, 32, 3], "label": [B]}
    """
    ds = tfds.load("cifar10", split=split, shuffle_files=training)

    num_hosts = jax.process_count()
    host_index = jax.process_index()
    ds = ds.shard(num_shards=num_hosts, index=host_index)

    if training:
        ds = ds.shuffle(
            buffer_size=50_000 // num_hosts,
            seed=seed + host_index,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda ex: preprocess_example(ex, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size_per_host, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if training:
        ds = ds.repeat()

    return ds


def shard_batch_for_pmap(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reshape batch for pmap across local devices.
    Input:
        "image": [B, H, W, C]
        "label": [B]
    Output:
        "image": [D, B/D, H, W, C]
        "label": [D, B/D]
    """
    num_local_devices = jax.local_device_count()
    images = batch["image"]
    labels = batch["label"]

    if images.shape[0] % num_local_devices != 0:
        raise ValueError(
            f"Batch size per host ({images.shape[0]}) must be divisible by "
            f"local device count ({num_local_devices})."
        )

    batch_per_device = images.shape[0] // num_local_devices

    images = images.reshape((num_local_devices, batch_per_device) + images.shape[1:])
    labels = labels.reshape((num_local_devices, batch_per_device))

    return {"image": images, "label": labels}


# ==============================================================================
# MODEL: SIMPLE CNN CLASSIFIER
# ==============================================================================


class CIFAR10CNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.

    Architecture:
        - Conv -> ReLU -> Conv -> ReLU -> MaxPool
        - Conv -> ReLU -> Conv -> ReLU -> MaxPool
        - Global average pool
        - Dense -> logits (10 classes)
    """

    num_classes: int = 10
    base_channels: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """A slightly deeper CNN with normalization, good enough for >80% on CIFAR-10."""

        def conv_block(x, channels: int) -> jnp.ndarray:
            # Use GroupNorm (no running stats state) to keep things simple with TrainState.
            x = nn.Conv(channels, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.relu(x)
            x = nn.Conv(channels, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.relu(x)
            return x

        # 32x32 -> 16x16
        x = conv_block(x, self.base_channels)          # 32x32, C
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")  # 16x16

        # 16x16 -> 8x8
        x = conv_block(x, self.base_channels * 2)      # 16x16, 2C
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")  # 8x8

        # 8x8, higher channels
        x = conv_block(x, self.base_channels * 4)      # 8x8, 4C

        # Global average pooling
        x = x.mean(axis=(1, 2))

        x = nn.Dense(self.base_channels * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x  # logits


class ClassificationTrainState(train_state.TrainState):
    """Basic TrainState wrapper (no extras needed here)."""

    pass


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> ClassificationTrainState:
    dummy_images = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    params = model.init({"params": rng}, dummy_images, train=True)["params"]

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return ClassificationTrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss_and_metrics(
    params,
    apply_fn,
    batch: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute softmax cross-entropy loss and accuracy."""
    images = batch["image"]
    labels = batch["label"]

    logits = apply_fn({"params": params}, images, train=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return loss, metrics


def train_step(
    state: ClassificationTrainState,
    batch: Dict[str, jnp.ndarray],
) -> Tuple[ClassificationTrainState, Dict[str, jnp.ndarray]]:
    """
    Single training step (to be pmapped).
    """

    def loss_fn(params):
        loss, metrics = compute_loss_and_metrics(params, state.apply_fn, batch)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # All-reduce across devices
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")
    # jax.tree_map was removed in newer JAX; use tree_util.tree_map for compatibility.
    metrics = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="data"), metrics
    )

    new_state = state.apply_gradients(grads=grads)

    metrics_out = {
        "loss": loss,
        "accuracy": metrics["accuracy"],
        "grad_norm": optax.global_norm(grads),
    }
    return new_state, metrics_out


def eval_step(
    state: ClassificationTrainState,
    batch: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """
    Evaluation step (to be pmapped).
    """
    images = batch["image"]
    labels = batch["label"]

    logits = state.apply_fn({"params": state.params}, images, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)

    loss = jax.lax.pmean(loss, axis_name="data")
    accuracy = jax.lax.pmean(accuracy, axis_name="data")

    return {
        "loss": loss,
        "accuracy": accuracy,
    }


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 classifier with JAX/TPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output and reproducibility
    parser.add_argument(
        "--workdir",
        type=str,
        default="./runs/cifar10_classification",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch_size_global",
        type=int,
        default=2048,
        help="Global batch size across all devices",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")

    # Model hyperparameters
    parser.add_argument(
        "--base_channels", type=int, default=64, help="Base channels for CNN"
    )

    # Logging
    parser.add_argument(
        "--log_every", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="Run evaluation every N epochs",
    )

    # Weights & Biases
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cifar10-classification",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not specified)",
    )

    args = parser.parse_args()

    # Distributed init
    dist_info = maybe_init_distributed()

    # Logger
    logger = Logger(is_main_process(), args.workdir)
    logger.separator("CIFAR-10 Classification Training")

    # Device info
    device_info = get_device_info()
    logger.config(
        "Distributed Setup",
        {
            "process_count (workers)": device_info["process_count"],
            "process_index (this worker)": device_info["process_index"],
            "global_device_count (total TPUs)": device_info["global_device_count"],
            "local_device_count (TPUs per worker)": device_info["local_device_count"],
            "device_type": device_info["device_type"],
            "distributed_initialized": dist_info["initialized"],
        },
    )

    # W&B init
    use_wandb = args.use_wandb and WANDB_AVAILABLE and is_main_process()
    if args.use_wandb and not WANDB_AVAILABLE:
        logger.info("WARNING: --use_wandb specified but wandb not installed. Skipping.")

    if use_wandb:
        logger.info("Initializing Weights & Biases...")
        wandb_config = {
            "epochs": args.epochs,
            "batch_size_global": args.batch_size_global,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "seed": args.seed,
            "architecture": "CIFAR10CNN",
            "base_channels": args.base_channels,
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

    # Batch size config
    num_devices_global = jax.device_count()
    num_devices_local = jax.local_device_count()
    num_hosts = jax.process_count()

    if args.batch_size_global % num_devices_global != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size_global}) must be divisible by "
            f"total device count ({num_devices_global})"
        )

    batch_size_per_device = args.batch_size_global // num_devices_global
    batch_size_per_host = batch_size_per_device * num_devices_local

    logger.config(
        "Batch Size Configuration",
        {
            "batch_size_global": args.batch_size_global,
            "batch_size_per_host": batch_size_per_host,
            "batch_size_per_device": batch_size_per_device,
            "num_hosts": num_hosts,
            "devices_per_host": num_devices_local,
        },
    )

    # Log hyperparameters
    logger.config(
        "Training Hyperparameters",
        {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "seed": args.seed,
        },
    )
    logger.config(
        "Model Hyperparameters",
        {
            "architecture": "CIFAR10CNN",
            "base_channels": args.base_channels,
            "num_classes": 10,
        },
    )

    # Dataset setup
    logger.info("Loading CIFAR-10 dataset...")

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(args.seed + jax.process_index())

    train_ds = make_dataset(
        split="train",
        batch_size_per_host=batch_size_per_host,
        training=True,
        seed=args.seed,
    )
    train_iter = iter(tfds.as_numpy(train_ds))

    eval_ds = make_dataset(
        split="test",
        batch_size_per_host=batch_size_per_host,
        training=False,
        seed=args.seed,
    )

    train_examples_total = 50_000
    eval_examples_total = 10_000
    train_examples_per_host = train_examples_total // num_hosts
    eval_examples_per_host = eval_examples_total // num_hosts

    steps_per_epoch = train_examples_per_host // batch_size_per_host
    eval_steps_per_epoch = max(eval_examples_per_host // batch_size_per_host, 1)

    if steps_per_epoch == 0:
        steps_per_epoch = 1
        logger.info(
            "WARNING: Batch size larger than data shard, using 1 step per epoch"
        )

    total_steps = args.epochs * steps_per_epoch

    logger.config(
        "Dataset Configuration",
        {
            "dataset": "CIFAR-10",
            "train_examples_total": train_examples_total,
            "train_examples_per_host": train_examples_per_host,
            "eval_examples_total": eval_examples_total,
            "eval_examples_per_host": eval_examples_per_host,
            "steps_per_epoch": steps_per_epoch,
            "eval_steps_per_epoch": eval_steps_per_epoch,
            "total_steps": total_steps,
        },
    )

    # Model & optimizer
    logger.info("Creating model...")
    model = CIFAR10CNN(num_classes=10, base_channels=args.base_channels)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        init_rng,
        model,
        learning_rate=args.lr,
        weight_decay=args.wd,
    )

    num_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    logger.info(f"Model initialized with {num_params:,} parameters")

    # Replicate
    logger.info("Replicating model across local devices...")
    local_devices = jax.local_devices()
    state = flax.jax_utils.replicate(state, devices=local_devices)

    # pmapped steps
    logger.info("Compiling training and evaluation steps...")
    p_train_step = jax.pmap(
        lambda s, b: train_step(s, b),
        axis_name="data",
        devices=local_devices,
    )
    p_eval_step = jax.pmap(
        lambda s, b: eval_step(s, b),
        axis_name="data",
        devices=local_devices,
    )

    if is_main_process():
        os.makedirs(args.workdir, exist_ok=True)

    # Training loop
    logger.separator("Starting Training")
    global_step = 0
    step_times = []

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        epoch_steps = 0

        for step_in_epoch in range(steps_per_epoch):
            step_start_time = time.time()
            global_step += 1

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(tfds.as_numpy(train_ds))
                batch = next(train_iter)

            batch = shard_batch_for_pmap(batch)

            state, metrics = p_train_step(state, batch)

            step_time = time.time() - step_start_time
            step_times.append(step_time)

            loss_value = float(flax.jax_utils.unreplicate(metrics["loss"]))
            acc_value = float(flax.jax_utils.unreplicate(metrics["accuracy"]))
            grad_norm = float(flax.jax_utils.unreplicate(metrics["grad_norm"]))

            epoch_loss_sum += loss_value
            epoch_acc_sum += acc_value
            epoch_steps += 1

            if global_step % args.log_every == 0:
                recent_times = step_times[-args.log_every :]
                avg_step_time = sum(recent_times) / len(recent_times)
                images_per_sec = args.batch_size_global / avg_step_time

                logger.metric(
                    global_step,
                    epoch + 1,
                    {
                        "loss": loss_value,
                        "accuracy": acc_value,
                        "grad_norm": grad_norm,
                        "imgs/sec": int(images_per_sec),
                        "step_time": avg_step_time,
                    },
                    prefix="TRAIN",
                )

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss_value,
                            "train/accuracy": acc_value,
                            "train/grad_norm": grad_norm,
                            "train/images_per_sec": images_per_sec,
                            "train/step_time": avg_step_time,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_loss_avg = epoch_loss_sum / max(epoch_steps, 1)
        epoch_acc_avg = epoch_acc_sum / max(epoch_steps, 1)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} completed | "
            f"train_loss={epoch_loss_avg:.6f} | "
            f"train_acc={epoch_acc_avg:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if use_wandb:
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss_avg,
                    "train/epoch_accuracy": epoch_acc_avg,
                    "train/epoch_time": epoch_time,
                },
                step=global_step,
            )

        # Evaluation
        if (epoch + 1) % args.eval_every_epochs == 0:
            logger.info("Running evaluation...")

            eval_iter = iter(tfds.as_numpy(eval_ds))

            eval_loss_sum = 0.0
            eval_acc_sum = 0.0
            eval_steps_done = 0

            for _ in range(eval_steps_per_epoch):
                try:
                    eval_batch = next(eval_iter)
                except StopIteration:
                    break

                eval_batch = shard_batch_for_pmap(eval_batch)
                eval_metrics = p_eval_step(state, eval_batch)

                eval_loss = float(flax.jax_utils.unreplicate(eval_metrics["loss"]))
                eval_acc = float(flax.jax_utils.unreplicate(eval_metrics["accuracy"]))

                eval_loss_sum += eval_loss
                eval_acc_sum += eval_acc
                eval_steps_done += 1

            if eval_steps_done > 0:
                eval_loss_avg = eval_loss_sum / eval_steps_done
                eval_acc_avg = eval_acc_sum / eval_steps_done
            else:
                eval_loss_avg = float("nan")
                eval_acc_avg = float("nan")

            logger.metric(
                global_step,
                epoch + 1,
                {
                    "loss": eval_loss_avg,
                    "accuracy": eval_acc_avg,
                },
                prefix="EVAL",
            )

            if use_wandb:
                wandb.log(
                    {
                        "eval/loss": eval_loss_avg,
                        "eval/accuracy": eval_acc_avg,
                        "eval/epoch": epoch + 1,
                    },
                    step=global_step,
                )

    # Training complete
    logger.separator("Training Complete")

    total_time = time.time() - logger.start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Total steps: {global_step}")

    if use_wandb:
        wandb.summary["total_steps"] = global_step
        wandb.summary["total_time_seconds"] = total_time
        wandb.finish()
        logger.info("W&B run finished and synced.")

    # Barrier for multi-host
    if jax.process_count() > 1:
        logger.info("Synchronizing all workers before exit...")
        x = jnp.ones((1,))
        x = jax.pmap(lambda y: jax.lax.psum(y, axis_name="i"), axis_name="i")(
            x[None].repeat(jax.local_device_count(), axis=0)
        )
        x.block_until_ready()
        time.sleep(2)
        logger.info("All workers synchronized. Exiting cleanly.")

    logger.close()


if __name__ == "__main__":
    main()

