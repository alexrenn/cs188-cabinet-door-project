"""
Step 6: Train a Diffusion Policy
==================================
This script provides a self-contained training loop for a simple
behavior-cloning policy on the OpenCabinet task, suitable for
understanding the training pipeline.

For production-quality training, use the official Diffusion Policy repo:
    git clone https://github.com/robocasa-benchmark/diffusion_policy
    cd diffusion_policy && pip install -e .
    python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet

This simplified version trains a small CNN+MLP policy to demonstrate
the data loading -> training -> checkpoint pipeline.

Usage:
    python 06_train_policy.py [--epochs 50] [--batch_size 32] [--lr 1e-4]
    python 06_train_policy.py --use_diffusion_policy   # Use official repo

    # NEW CHANGES - Action chunking: predict K future actions per step (smoother behavior)
    python 06_train_policy.py --chunk_size 8
    python 06_train_policy.py --sweep_chunk_sizes 4 8 16


"""

import argparse
import os
import sys
import yaml

import numpy as np


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def train_simple_policy(config, chunk_size=1):
    """
    Train a simple behavior-cloning policy.

    This is a simplified training loop to illustrate the pipeline.
    For real results, use the official Diffusion Policy codebase.

    Args:
        config: Training configuration dict.
        chunk_size: Number of future actions to predict per step (action chunking).
            K=1 is standard single-step BC; K>1 enables temporal action chunking.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    print_section("Simple Behavior Cloning Policy")
    if chunk_size > 1:
        print(f"Action chunking enabled: K={chunk_size}")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    # ----------------------------------------------------------------
    # 1. Build a simple dataset from the LeRobot format
    # ----------------------------------------------------------------
    print("\nLoading dataset...")

    class CabinetDemoDataset(Dataset):
        """
        Loads state-action pairs from the LeRobot-format dataset.

        For simplicity, this uses only the low-dimensional state observations
        (gripper qpos, base pose, eef pose) rather than images.
        Full visuomotor training with images requires the Diffusion Policy repo.
        """

        def __init__(self, dataset_path, max_episodes=None, chunk_size=1):
            import pyarrow.parquet as pq

            self.chunk_size = chunk_size
            self.states = []
            self.actions = []

            # The dataset path from get_ds_path may point to the lerobot dir directly
            # or to the parent. Try both layouts.
            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(dataset_path, "lerobot", "data")
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"Data directory not found under: {dataset_path}\n"
                    "Make sure you downloaded the dataset with 04_download_dataset.py"
                )

            # Load parquet files
            chunk_dir = os.path.join(data_dir, "chunk-000")
            if not os.path.exists(chunk_dir):
                raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

            parquet_files = sorted(
                f for f in os.listdir(chunk_dir) if f.endswith(".parquet")
            )
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {chunk_dir}")

            # First pass: collect per-episode state-action sequences
            episode_states = []  # list of lists
            episode_actions = []  # list of lists

            episodes_loaded = 0
            for pf in parquet_files:
                table = pq.read_table(os.path.join(chunk_dir, pf))
                df = table.to_pandas()

                # Extract state and action columns
                state_cols = [
                    c for c in df.columns if c.startswith("observation.state")
                ]
                action_cols = [
                    c for c in df.columns
                    if c == "action" or c.startswith("action.")
                ]

                if not state_cols or not action_cols:
                    # Try alternative column naming
                    state_cols = [
                        c
                        for c in df.columns
                        if "gripper" in c or "base" in c or "eef" in c
                    ]
                    action_cols = [c for c in df.columns if "action" in c]

                if state_cols and action_cols:
                    # Detect episode boundaries via episode_index column
                    ep_col = None
                    for candidate in ("episode_index", "episode", "episode_id"):
                        if candidate in df.columns:
                            ep_col = candidate
                            break

                    if ep_col is not None:
                        groups = df.groupby(ep_col)
                    else:
                        # Treat the entire file as one episode
                        groups = [(0, df)]

                    for _ep_id, ep_df in groups:
                        ep_s, ep_a = [], []
                        for _, row in ep_df.iterrows():
                            state_parts = []
                            for c in state_cols:
                                val = row[c]
                                if isinstance(val, np.ndarray):
                                    state_parts.extend(val.flatten().tolist())
                                elif isinstance(val, (int, float, np.floating)):
                                    state_parts.append(float(val))
                            action_parts = []
                            for c in action_cols:
                                val = row[c]
                                if isinstance(val, np.ndarray):
                                    action_parts.extend(val.flatten().tolist())
                                elif isinstance(val, (int, float, np.floating)):
                                    action_parts.append(float(val))

                            if state_parts and action_parts:
                                ep_s.append(np.array(state_parts, dtype=np.float32))
                                ep_a.append(np.array(action_parts, dtype=np.float32))

                        if ep_s:
                            episode_states.append(ep_s)
                            episode_actions.append(ep_a)
                            episodes_loaded += 1

                if max_episodes and episodes_loaded >= max_episodes:
                    break

            if not episode_states:
                print("WARNING: Could not extract state-action pairs from parquet files.")
                print("The dataset may use a different format.")
                print("Generating synthetic demo data for illustration...")
                self._generate_synthetic_data()
            else:
                # Build (state, action_chunk) pairs respecting episode boundaries
                K = self.chunk_size
                for ep_s, ep_a in zip(episode_states, episode_actions):
                    T = len(ep_s)
                    for t in range(T):
                        remaining = T - t
                        if remaining >= K:
                            chunk = np.stack(ep_a[t : t + K], axis=0)  # (K, action_dim)
                        else:
                            # Pad by repeating the last action
                            chunk = np.stack(
                                ep_a[t:] + [ep_a[-1]] * (K - remaining), axis=0
                            )
                        self.states.append(ep_s[t])
                        self.actions.append(chunk.flatten())  # (K * action_dim,)

            self.states = np.array(self.states, dtype=np.float32)
            self.actions = np.array(self.actions, dtype=np.float32)

            # Infer action_dim per single step
            if len(episode_actions) > 0 and len(episode_actions[0]) > 0:
                self.single_action_dim = len(episode_actions[0][0])
            else:
                self.single_action_dim = self.actions.shape[-1] // max(K, 1)

            print(f"Loaded {len(self.states)} state-action_chunk pairs")
            print(f"State dim:       {self.states.shape[-1]}")
            print(f"Action dim:      {self.single_action_dim}")
            print(f"Chunk size (K):  {K}")
            print(f"Output dim:      {self.actions.shape[-1]}")

        def _generate_synthetic_data(self):
            """Generate synthetic data for demonstration purposes."""
            K = self.chunk_size
            rng = np.random.default_rng(42)
            self.single_action_dim = 12
            # Generate synthetic episodes of length 50
            for _ep in range(20):
                ep_actions = [
                    rng.standard_normal(12).astype(np.float32) * 0.1
                    for _ in range(50)
                ]
                for t in range(50):
                    state = rng.standard_normal(16).astype(np.float32)
                    remaining = 50 - t
                    if remaining >= K:
                        chunk = np.stack(ep_actions[t : t + K], axis=0)
                    else:
                        chunk = np.stack(
                            ep_actions[t:] + [ep_actions[-1]] * (K - remaining),
                            axis=0,
                        )
                    self.states.append(state)
                    self.actions.append(chunk.flatten())

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.actions[idx]),
            )

    dataset = CabinetDemoDataset(dataset_path, max_episodes=50, chunk_size=chunk_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # 2. Define a simple MLP policy
    # ----------------------------------------------------------------
    state_dim = dataset.states.shape[-1]
    action_dim = dataset.single_action_dim  # per-step action dim
    output_dim = chunk_size * action_dim     # total output width

    class SimplePolicy(nn.Module):
        def __init__(self, state_dim, output_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SimplePolicy(state_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    print_section("Training")
    print(f"Epochs:     {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"LR:         {config['learning_rate']}")

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            pred_actions = model(states_batch)  # (B, K * action_dim)
            loss = nn.functional.mse_loss(pred_actions, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{config['epochs']}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "chunk_size": chunk_size,
                },
                ckpt_path,
            )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
        },
        final_path,
    )

    print(f"\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "This simple MLP policy is for educational purposes only.\n"
        "For a policy that can actually solve the task, use the\n"
        "official Diffusion Policy codebase:\n"
        "\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "Alternatively, try pi-0 or GR00T N1.5:\n"
        "  https://github.com/robocasa-benchmark/openpi\n"
        "  https://github.com/robocasa-benchmark/Isaac-GR00T"
    )


def print_diffusion_policy_instructions():
    """Print instructions for using the official Diffusion Policy repo."""
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  # Train\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "  # Evaluate\n"
        "  python eval_robocasa.py \\\n"
        "    --checkpoint <path-to-checkpoint> \\\n"
        "    --task_set atomic \\\n"
        "    --split target\n"
        "\n"
        "Option B: pi-0 via OpenPi (for foundation model fine-tuning)\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "  cd openpi && pip install -e . && pip install -e packages/openpi-client/\n"
        "\n"
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \\\n"
        "    robocasa_OpenCabinet --exp-name=cabinet_door\n"
        "\n"
        "Option C: GR00T N1.5 (NVIDIA foundation model)\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
        "  cd groot && pip install -e .\n"
        "\n"
        "  python scripts/gr00t_finetune.py \\\n"
        "    --output-dir experiments/cabinet_door \\\n"
        "    --dataset_soup robocasa_OpenCabinet \\\n"
        "    --max_steps 50000\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a policy for OpenCabinet")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1,
        help="Action chunk size K: predict K future actions per step (1=single-step BC)",
    )
    parser.add_argument(
        "--sweep_chunk_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Sweep over multiple chunk sizes, e.g. --sweep_chunk_sizes 4 8 16",
    )
    parser.add_argument(
        "--use_diffusion_policy",
        action="store_true",
        help="Print instructions for using the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    # Build config from args or YAML file
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
        }

    # CLI --chunk_size / --sweep_chunk_sizes override config file value
    if args.sweep_chunk_sizes:
        chunk_sizes = args.sweep_chunk_sizes
    elif args.chunk_size != 1:
        chunk_sizes = [args.chunk_size]
    else:
        chunk_sizes = [config.get("chunk_size", args.chunk_size)]
    for K in chunk_sizes:
        if len(chunk_sizes) > 1:
            print_section(f"Sweep: chunk_size K={K}")
            # Use a subdirectory per chunk size
            sweep_config = dict(config)
            sweep_config["checkpoint_dir"] = os.path.join(
                config["checkpoint_dir"], f"chunk_K{K}"
            )
        else:
            sweep_config = config
        train_simple_policy(sweep_config, chunk_size=K)


if __name__ == "__main__":
    main()
