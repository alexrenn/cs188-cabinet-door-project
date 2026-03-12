"""
Utility functions for policy handling in the cabinet door project.

state extraction, normalization, model definitions, data loading, etc. for the diffusion policy implementation.
used by training, evaluation, and visualization scripts.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definitions (shared by 06, 07, 08)
# ---------------------------------------------------------------------------

class SimplePolicy(nn.Module):
    """3-layer MLP behavior cloning policy."""

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


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionPolicy(nn.Module):
    """DDPM-based diffusion policy for action generation.

    During training, call ``predict_noise(noisy_action, state, timestep)``
    to get the predicted noise.  During inference, call ``forward(state)``
    (or equivalently ``sample(state)``) to generate actions via iterative
    DDPM denoising.
    """

    def __init__(self, state_dim, output_dim, hidden_dim=256,
                 n_diffusion_steps=100):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.n_diffusion_steps = n_diffusion_steps

        # Timestep embedding
        time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Noise prediction network (MLP with residual connections)
        input_dim = output_dim + state_dim + time_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(4)
        ])
        self.out_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

        # DDPM noise schedule (linear beta)
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def predict_noise(self, noisy_action, state, timestep):
        """Predict the noise that was added to *noisy_action*."""
        t_emb = self.time_mlp(timestep)
        h = self.input_proj(torch.cat([noisy_action, state, t_emb], dim=-1))
        for block in self.blocks:
            h = h + block(h)  # residual
        return self.out_proj(h)

    # -- Inference ---------------------------------------------------------

    def forward(self, state):
        """Generate actions via DDPM sampling (used at eval time)."""
        return self.sample(state)

    @torch.no_grad()
    def sample(self, state):
        """DDPM reverse process: iteratively denoise from Gaussian noise."""
        batch_size = state.shape[0]
        device = state.device

        x = torch.randn(batch_size, self.output_dim, device=device)

        for t in reversed(range(self.n_diffusion_steps)):
            t_batch = torch.full(
                (batch_size,), t, device=device, dtype=torch.long
            )
            predicted_noise = self.predict_noise(x, state, t_batch)

            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_cumprod)) * predicted_noise
            )

            if t > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        return x.clamp(-1, 1)


def load_policy_from_checkpoint(checkpoint_path, device):
    """Load a trained policy (SimplePolicy or DiffusionPolicy) from a checkpoint.

    Returns (model, state_dim, action_dim, chunk_size).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    chunk_size = ckpt.get("chunk_size", 1)
    output_dim = chunk_size * action_dim
    policy_type = ckpt.get("policy_type", "simple")

    if policy_type == "diffusion":
        n_steps = ckpt.get("n_diffusion_steps", 100)
        model = DiffusionPolicy(
            state_dim, output_dim, n_diffusion_steps=n_steps
        ).to(device)
    else:
        model = SimplePolicy(state_dim, output_dim).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"Loaded {policy_type} policy from: {checkpoint_path}")
    print(f"  Trained for {epoch} epochs, loss={loss:.6f}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}, Chunk size: {chunk_size}")

    return model, state_dim, action_dim, chunk_size


def _handle_name_to_body(handle_name):
    """Convert a fixture handle name to the corresponding MuJoCo body name.

    Fixture properties like ``fxtr.handle_name`` return names ending in
    ``_handle`` (e.g. ``cab_3_main_group_door_handle_handle``), but the
    MuJoCo body for that handle ends in ``_main`` instead
    (e.g. ``cab_3_main_group_door_handle_main``).
    """
    return handle_name.rsplit("_handle", 1)[0] + "_main"


def get_handle_pos(env):
    """Get the 3D world position of the cabinet door handle.

    For HingeCabinet (two doors), returns the handle closest to the
    end-effector so the policy tracks whichever door the robot is
    currently manipulating.
    """
    from robocasa.models.fixtures.cabinets import HingeCabinet

    fxtr = env.fxtr
    if isinstance(fxtr, HingeCabinet):
        left_body = _handle_name_to_body(fxtr.left_handle_name)
        right_body = _handle_name_to_body(fxtr.right_handle_name)
        left_id = env.sim.model.body_name2id(left_body)
        right_id = env.sim.model.body_name2id(right_body)
        left_pos = env.sim.data.body_xpos[left_id].copy()
        right_pos = env.sim.data.body_xpos[right_id].copy()
        eef_pos = _get_eef_world_pos(env)
        if np.linalg.norm(left_pos - eef_pos) <= np.linalg.norm(right_pos - eef_pos):
            return left_pos
        return right_pos
    else:
        body_name = _handle_name_to_body(fxtr.handle_name)
        body_id = env.sim.model.body_name2id(body_name)
        return env.sim.data.body_xpos[body_id].copy()


def _get_eef_world_pos(env):
    """Get the world-frame end-effector position from the MuJoCo sim."""
    robot = env.robots[0]
    sid = robot.eef_site_id
    if isinstance(sid, dict):
        sid = list(sid.values())[0]
    return env.sim.data.site_xpos[sid].copy()


def augment_obs_with_handle(obs, env):
    """Inject ``handle_pos`` and ``handle_to_eef_pos`` into *obs*.

    Call this right after ``env.reset()`` or ``env.step()`` so that
    ``extract_state`` can find these keys in the observation dict.
    """
    handle_pos = get_handle_pos(env)
    eef_pos = _get_eef_world_pos(env)
    obs["handle_pos"] = handle_pos
    obs["handle_to_eef_pos"] = handle_pos - eef_pos
    return obs

