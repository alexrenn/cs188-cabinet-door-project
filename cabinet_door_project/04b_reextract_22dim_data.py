"""
Step 4b: Re-extract 22-dim state data from existing demonstrations
====================================================================
The downloaded OpenCabinet dataset stores a 16-dim observation.state
(gripper_qpos, base_pos/quat, eef_pos/quat).  This script replays
every episode through the simulator so we can read the cabinet handle
position at each timestep and produce a new dataset with a 22-dim
state vector that also includes ``handle_pos`` (3) and
``handle_to_eef_pos`` (3).

The output parquet files are saved to ``--output`` and can be passed
to the training script via ``--dataset_path``:

    python 04b_reextract_22dim_data.py [--output /tmp/cabinet_22dim_dataset]
    python 06_train_policy.py --dataset_path /tmp/cabinet_22dim_dataset \
                              --config configs/diffusion_policy.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Headless rendering setup (Linux only; macOS uses native CGL) ────────────
import platform
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.utils.dataset_registry_utils import get_ds_path
import robosuite

from policy_utils import augment_obs_with_handle

# Canonical key order — must match extract_state() in 07/08 scripts.
ROBOSUITE_STATE_KEYS = [
    "robot0_gripper_qpos",     # 2
    "robot0_base_pos",         # 3
    "robot0_base_quat",        # 4
    "robot0_base_to_eef_pos",  # 3
    "robot0_base_to_eef_quat", # 4
    "handle_pos",              # 3
    "handle_to_eef_pos",       # 3
]


# ── Sim reset helper (mirrors playback_dataset.py) ─────────────────────────
def reset_to(env, state):
    """Reset the environment to a specific simulator state."""
    if "model" in state:
        ep_meta = json.loads(state.get("ep_meta", "{}"))
        if hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        elif hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def build_state_vector(obs):
    """Build a 22-dim state vector from the observation dict in
    ROBOSUITE_STATE_KEYS order."""
    parts = []
    for key in ROBOSUITE_STATE_KEYS:
        if key in obs:
            parts.append(np.asarray(obs[key]).flatten())
    return np.concatenate(parts).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract 22-dim states from existing OpenCabinet demos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/cabinet_22dim_dataset",
        help="Directory to write the new parquet files",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit the number of episodes to process (default: all)",
    )
    args = parser.parse_args()

    # ── Locate original dataset ─────────────────────────────────────────
    dataset_path = Path(get_ds_path("OpenCabinet", source="human"))
    episodes = LU.get_episodes(dataset_path)
    num_episodes = len(episodes)
    if args.max_episodes:
        num_episodes = min(num_episodes, args.max_episodes)
    print(f"Dataset:   {dataset_path}")
    print(f"Episodes:  {num_episodes}")

    # ── Create environment (headless, no cameras) ───────────────────────
    env_meta = LU.get_env_metadata(dataset_path)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)

    # ── Output directory ────────────────────────────────────────────────
    out_dir = Path(args.output) / "data" / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_steps = 0
    for ep_idx in range(num_episodes):
        # Load episode data
        states = LU.get_episode_states(dataset_path, ep_idx)
        model_xml = LU.get_episode_model_xml(dataset_path, ep_idx)
        ep_meta = LU.get_episode_meta(dataset_path, ep_idx)

        # Read original actions from parquet (keep native format)
        src_files = sorted(dataset_path.glob(f"data/*/episode_{ep_idx:06d}.parquet"))
        if not src_files:
            print(f"  [skip] no parquet for episode {ep_idx}")
            continue
        src_df = pd.read_parquet(src_files[0])
        original_actions = src_df["action"].to_list()

        T = min(len(states), len(original_actions))

        # Reset to initial state (loads XML + first sim state)
        reset_to(env, {
            "states": states[0],
            "model": model_xml,
            "ep_meta": json.dumps(ep_meta),
        })

        new_states = []
        for t in range(T):
            if t > 0:
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                if hasattr(env, "update_state"):
                    env.update_state()

            # Get standard robot observations
            obs = env._get_observations(force_update=True)

            # Inject handle_pos and handle_to_eef_pos
            augment_obs_with_handle(obs, env)

            # Build 22-dim vector
            state_vec = build_state_vector(obs)
            new_states.append(state_vec.tolist())

        # Build output table preserving episode_index column
        out_data = {
            "observation.state": new_states,
            "action": original_actions[:T],
        }
        if "episode_index" in src_df.columns:
            out_data["episode_index"] = src_df["episode_index"].to_list()[:T]
        else:
            out_data["episode_index"] = [ep_idx] * T

        table = pa.table(out_data)
        pq.write_table(table, out_dir / f"episode_{ep_idx:06d}.parquet")

        total_steps += T
        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(
                f"  Episode {ep_idx + 1:3d}/{num_episodes}  "
                f"({T} steps, state_dim={len(new_states[0])})"
            )

    env.close()

    print(f"\nDone — wrote {num_episodes} episodes ({total_steps} total steps)")
    print(f"State dim: {len(new_states[0]) if new_states else '?'}")
    print(f"Output:    {args.output}")
    print(
        f"\nTo train:  python 06_train_policy.py "
        f"--dataset_path {args.output} --config configs/diffusion_policy.yaml"
    )


if __name__ == "__main__":
    main()
