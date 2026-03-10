"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint from 06_train_policy.py and runs it
live in the OpenCabinet environment so you can watch the robot.

This is your primary debugging tool: watch exactly where and why the policy
fails — does it reach for the handle? Does it grasp? Does it pull correctly?

Two rendering modes:
  On-screen  (default)  — interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) — renders to a video file, works without a display

Usage:
    # Watch live in a window (WSL/Linux) + save video
    python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Save to video only (no display needed — works headless / in notebooks)
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen

    # Run 3 episodes, slow down playback so you can follow along
    python 08_visualize_policy_rollout.py --checkpoint ... --num_episodes 3 --max_steps 200

    # Mac users must use mjpython for the on-screen window
    mjpython 08_visualize_policy_rollout.py --checkpoint ...
"""

import os
import sys

# ── Rendering mode detection ────────────────────────────────────────────────
# We peek at sys.argv *before* argparse so we can configure the GL backend
# before any library is imported.  Wrong GL backend = gladLoadGL error.
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    # Off-screen mode: use Mesa's software osmesa renderer.
    # EGL is the default on headless Linux but fails on WSL2 (no /dev/dri).
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    # On-screen mode: re-exec with correct display vars baked into the OS
    # environment so Mesa (GLFW) sees them before any C library initializes.
    # On WSLg the .bashrc often sets a stale VcXsrv-style DISPLAY that
    # breaks GLFW; os.execve() restarts the process cleanly.
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import time
from collections import deque

import numpy as np
import robocasa  # noqa: F401 — registers OpenCabinet environment
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


# ── Policy loading (identical to 07_evaluate_policy.py) ─────────────────────

def load_policy(checkpoint_path, device):
    """Load the SimplePolicy trained by 06_train_policy.py."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    chunk_size = ckpt.get("chunk_size", 1)
    output_dim = chunk_size * action_dim

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

    model = SimplePolicy(state_dim, output_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, state_dim, action_dim, chunk_size, ckpt


def extract_state(obs, state_dim):
    """Flatten non-image observations into a state vector of length state_dim."""
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray) and not key.endswith("_image"):
            parts.append(val.flatten())
    if not parts:
        return np.zeros(state_dim, dtype=np.float32)
    state = np.concatenate(parts).astype(np.float32)
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]
    return state


# ── On-screen rollout ────────────────────────────────────────────────────────

def run_onscreen(model, state_dim, action_dim, chunk_size, args):
    """
    Run the policy with an interactive MuJoCo viewer window.

    The viewer opens automatically; you can pan/zoom/rotate the camera
    with the mouse while the robot executes the policy.
    """
    import torch

    device = next(model.parameters()).device

    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")
        print(f"  Running for up to {args.max_steps} steps...")
        print(f"  (Watch the viewer window — use mouse to orbit the camera)")
        if chunk_size > 1:
            print(f"  Action chunking: K={chunk_size} (re-plan every {chunk_size} steps)")
        print()

        success = False
        hold_count = 0
        action_buffer = deque()  # FIFO buffer for action chunking

        for step in range(args.max_steps):
            # If the action buffer is empty, run the policy to get K actions
            if len(action_buffer) == 0:
                state = extract_state(obs, state_dim)
                with torch.no_grad():
                    raw_output = model(
                        torch.from_numpy(state).unsqueeze(0).to(device)
                    ).cpu().numpy().squeeze(0)

                action_chunk = raw_output.reshape(chunk_size, action_dim)
                for a in action_chunk:
                    action_buffer.append(a)

            action = action_buffer.popleft()

            # Pad / trim to environment's expected action dimension
            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            # Print a brief status every 20 steps
            if step % 20 == 0:
                checking = env._check_success()
                status = "cabinet OPEN" if checking else "in progress"
                act_mag = float(np.abs(action).mean())
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  "
                    f"action_mag={act_mag:.3f}  [{status}]"
                )

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            # Pace the rollout so it is easy to watch
            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"\n  Result: {result}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout with video ────────────────────────────────────────────

def run_offscreen(model, state_dim, action_dim, chunk_size, args):
    """
    Run the policy headlessly and save a side-by-side annotated video.

    Each frame shows the robot from the front-view camera; per-step
    diagnostics (step count, reward, success flag) are printed to the
    terminal.
    """
    import torch
    import imageio
    from robocasa.utils.env_utils import create_env

    device = next(model.parameters()).device

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768

    successes = 0
    all_frames = []  # collect frames across episodes

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")

        success = False
        hold_count = 0
        ep_frames = []
        action_buffer = deque()  # FIFO buffer for action chunking

        for step in range(args.max_steps):
            # If the action buffer is empty, run the policy to get K actions
            if len(action_buffer) == 0:
                state = extract_state(obs, state_dim)
                with torch.no_grad():
                    raw_output = model(
                        torch.from_numpy(state).unsqueeze(0).to(device)
                    ).cpu().numpy().squeeze(0)

                action_chunk = raw_output.reshape(chunk_size, action_dim)
                for a in action_chunk:
                    action_buffer.append(a)

            action = action_buffer.popleft()

            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            # Render from the agent view camera
            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]  # MuJoCo renders upside-down
            ep_frames.append(frame)

            if step % 20 == 0:
                checking = env._check_success()
                status = "cabinet OPEN" if checking else "in progress"
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  [{status}]"
                )

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
        env.close()

    # Write video
    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")

    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/cabinet_policy_checkpoints/best_policy.pt",
        help="Path to policy checkpoint (.pt) saved by 06_train_policy.py",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render to video file instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/policy_rollout.mp4",
        help="Output video path (used with --offscreen)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the saved video",
    )
    parser.add_argument(
        "--max_fr",
        type=int,
        default=20,
        help="On-screen playback rate cap (frames/second); lower = slower",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment layout/style selection",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)
    print()

    # Load policy
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 06_train_policy.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state_dim, action_dim, chunk_size, ckpt = load_policy(args.checkpoint, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
    print(f"  State dim: {state_dim},  Action dim: {action_dim},  Chunk size: {chunk_size}")
    print(f"  Device: {device}")
    print()

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:     {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps/ep: {args.max_steps}")
    if args.offscreen:
        print(f"Output:   {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(model, state_dim, action_dim, chunk_size, args)
    else:
        print("Opening viewer window...")
        print("  Tip: orbit the camera with the mouse to see the gripper.\n")
        run_onscreen(model, state_dim, action_dim, chunk_size, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
