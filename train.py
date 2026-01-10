import jax
import jax.numpy as jnp
from jax import grad, pmap
from functools import partial

def run():
    # 1. Initialize the distributed environment (Critical for Pods)
    jax.distributed.initialize()

    # Verify we see all chips
    num_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_idx = jax.process_index()

    if process_idx == 0:
        print(f"Global Device Count: {num_devices}")
        print(f"Local Device Count:  {local_devices}")

    # 2. Hyperparameters
    learning_rate = 0.1
    num_steps = 10000
    features = 128
    batch_size_per_chip = 64

    # 3. Generate Dummy Data (Target function: y = 2x)
    # We create a batch for EVERY device on this host
    # Shape: (local_devices, batch_size, features)
    key = jax.random.PRNGKey(42 + process_idx) # Seed differently per host
    k1, k2, k3 = jax.random.split(key, 3)

    # Sharded input data for this specific host
    inputs = jax.random.normal(k1, (local_devices, batch_size_per_chip, features))
    targets = inputs * 2.0  # The model should learn to multiply by 2

    # 4. Initialize Weights
    # Replicate weights across the 'devices' dimension so each chip has a copy
    # Shape: (local_devices, features, features) (simplification for pmap)
    w_key = jax.random.PRNGKey(0) # Same seed everywhere so all chips start same
    w_init = jax.random.normal(w_key, (features, features)) * 0.1
    # Replicate to (local_devices, features, features)
    params = jnp.stack([w_init] * local_devices)

    # 5. Define Loss and Update Step
    def loss_fn(w, x, y):
        pred = jnp.dot(x, w)
        return jnp.mean((pred - y) ** 2)

    @partial(pmap, axis_name='batch') # <--- The Magic: Parallel Map across chips
    def update(w, x, y):
        # Calculate gradients on this chip
        grads = grad(loss_fn)(w, x, y)

        # SYNC: Average gradients across ALL 128 chips using optical interconnect
        # This proves the pod is networked correctly.
        grads = jax.lax.pmean(grads, axis_name='batch')

        # Update weights
        return w - learning_rate * grads, jax.lax.pmean(loss_fn(w, x, y), axis_name='batch')

    # 6. Training Loop
    if process_idx == 0:
        print("\nStarting Distributed Training...")

    for i in range(num_steps):
        # pmap automatically distributes the 'params', 'inputs', 'targets'
        # to the local chips corresponding to the first dimension of the arrays.
        params, loss_val = update(params, inputs, targets)

        # We only print from the first chip on the first host to avoid spam
        if process_idx == 0 and i % 10 == 0:
            # loss_val is a sharded array, grab the first one
            print(f"Step {i:03d} | Loss: {loss_val[0]:.5f}")

    # 7. Final Check
    if process_idx == 0:
        print(f"Final Loss: {loss_val[0]:.5f}")
        if loss_val[0] < 0.01:
            print("✅ SUCCESS: Distributed training works and model converged.")
        else:
            print("❌ FAILURE: Model did not converge.")

if __name__ == "__main__":
    run()