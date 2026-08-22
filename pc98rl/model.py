"""CPU-friendly entity-set actor critic for TH05.

The compact observation has a stable layout produced by the Rust reader:
37 global floats, 16 x 7 special-projectile floats, 16 x 7 bullet floats,
and 12 drop floats.  Shared token encoders avoid spending most of the CPU and
memory bandwidth on mostly redundant dense spatial maps.
"""

import torch
from torch import nn

from .contracts import KinematicSpec


GLOBAL_DIM = 37
ENTITY_COUNT = 16
ENTITY_DIM = 7
DROP_DIM = 12
FEATURE_DIM = GLOBAL_DIM + 2 * ENTITY_COUNT * ENTITY_DIM + DROP_DIM
KINEMATIC_DIM = 5


def add_kinematic_features(
    tokens: torch.Tensor,
    player_velocity: torch.Tensor,
    spec: KinematicSpec,
) -> torch.Tensor:
    """Append bounded constant-velocity collision geometry to entity tokens.

    The native schema stores position in playfield units and both player and
    entity velocity in units of 12 pixels per game frame.  Converting them back
    to the same physical units is important: using normalized x/y directly
    would distort trajectories because the playfield is not square.
    """
    if tokens.shape[-1] != ENTITY_DIM:
        raise ValueError(f"expected {ENTITY_DIM}-float entity tokens")
    if player_velocity.shape[-1] != 2:
        raise ValueError("player_velocity must end in two components")

    dx = tokens[..., 0] * spec.position_scale[0]
    dy = tokens[..., 1] * spec.position_scale[1]
    relative_vx = (
        tokens[..., 2] - player_velocity[..., None, 0]
    ) * spec.velocity_scale[0]
    relative_vy = (
        tokens[..., 3] - player_velocity[..., None, 1]
    ) * spec.velocity_scale[1]

    distance = torch.sqrt(dx.square() + dy.square()).clamp_min(1e-6)
    speed_squared = relative_vx.square() + relative_vy.square()
    radial_dot = dx * relative_vx + dy * relative_vy
    approaching = (radial_dot < 0.0) & (speed_squared > 1e-8)
    approaching_time = (-radial_dot / speed_squared.clamp_min(1e-8)).clamp(
        0.0, spec.horizon_steps
    )
    time_to_closest = torch.where(
        approaching,
        approaching_time,
        torch.full_like(approaching_time, spec.horizon_steps),
    )
    projection_time = torch.where(
        approaching, approaching_time, torch.zeros_like(approaching_time)
    )
    closest_x = dx + relative_vx * projection_time
    closest_y = dy + relative_vy * projection_time
    miss_distance = torch.sqrt(closest_x.square() + closest_y.square())
    relative_velocity_scale = sum(spec.velocity_scale)
    closing_speed = (-radial_dot / distance / relative_velocity_scale).clamp(
        -1.0, 1.0
    )

    derived = torch.stack(
        (
            (relative_vx / (2.0 * spec.velocity_scale[0])).clamp(-1.0, 1.0),
            (relative_vy / (2.0 * spec.velocity_scale[1])).clamp(-1.0, 1.0),
            closing_speed,
            time_to_closest / spec.horizon_steps,
            (miss_distance / spec.distance_scale).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    return torch.cat((tokens, derived), dim=-1)


class EntitySetEncoder(nn.Module):
    """Encode a distance-sentinel-padded set with attention and max pooling."""

    def __init__(self, token_dim: int = ENTITY_DIM, hidden_dim: int = 32):
        super().__init__()
        self.token_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Rust pads absent entities as [0, 0, 0, 0, 0, 0, 1], where the final
        # value is normalized distance.  Looking at all seven values would mark
        # every padding token as present and dilute the useful nearest entities.
        mask = (tokens[..., :6].abs().sum(dim=-1) > 1e-7) | (
            tokens[..., 6] < 1.0 - 1e-7
        )
        encoded = self.token_net(tokens)

        scores = self.attention(encoded).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = (encoded * weights.unsqueeze(-1)).sum(dim=-2)

        max_pooled = encoded.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=-2)
        any_present = mask.any(dim=-1, keepdim=True)
        max_pooled = torch.where(any_present, max_pooled, torch.zeros_like(max_pooled))
        return self.output(torch.cat((attended, max_pooled), dim=-1))


class CompactFeatureEncoder(nn.Module):
    output_dim = 128

    def __init__(
        self,
        analytic_geometry: bool = False,
        kinematic_spec: KinematicSpec | None = None,
    ):
        super().__init__()
        self.analytic_geometry = analytic_geometry
        if analytic_geometry and kinematic_spec is None:
            raise ValueError("analytic geometry requires an adapter KinematicSpec")
        self.kinematic_spec = kinematic_spec
        self.global_net = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        token_dim = ENTITY_DIM + (KINEMATIC_DIM if analytic_geometry else 0)
        self.projectile_net = EntitySetEncoder(token_dim=token_dim)
        self.bullet_net = EntitySetEncoder(token_dim=token_dim)
        self.drop_net = nn.Sequential(nn.Linear(DROP_DIM, 16), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 16, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"expected {FEATURE_DIM} features, got {features.shape[-1]}")

        projectile_start = GLOBAL_DIM
        bullet_start = projectile_start + ENTITY_COUNT * ENTITY_DIM
        drop_start = bullet_start + ENTITY_COUNT * ENTITY_DIM

        global_features = features[..., :GLOBAL_DIM]
        projectiles = features[..., projectile_start:bullet_start].reshape(
            *features.shape[:-1], ENTITY_COUNT, ENTITY_DIM
        )
        bullets = features[..., bullet_start:drop_start].reshape(
            *features.shape[:-1], ENTITY_COUNT, ENTITY_DIM
        )
        drops = features[..., drop_start:]
        if self.analytic_geometry:
            player_velocity = global_features[..., 2:4]
            projectiles = add_kinematic_features(
                projectiles, player_velocity, self.kinematic_spec
            )
            bullets = add_kinematic_features(
                bullets, player_velocity, self.kinematic_spec
            )

        return self.fusion(
            torch.cat(
                (
                    self.global_net(global_features),
                    self.projectile_net(projectiles),
                    self.bullet_net(bullets),
                    self.drop_net(drops),
                ),
                dim=-1,
            )
        )


class EntityActorCritic(nn.Module):
    """Recurrent actor critic with a shared three-objective value trunk."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        action_dim: int = 19,
        hidden_size: int = 128,
        num_objectives: int = 3,
        analytic_geometry: bool = False,
        kinematic_spec: KinematicSpec | None = None,
    ):
        super().__init__()
        if feature_dim != FEATURE_DIM:
            raise ValueError(f"entity model requires feature_dim={FEATURE_DIM}")
        self.hidden_size = hidden_size
        self.gru_hidden_size = hidden_size
        self.num_objectives = num_objectives
        self.analytic_geometry = analytic_geometry
        self.encoder = CompactFeatureEncoder(
            analytic_geometry=analytic_geometry,
            kinematic_spec=kinematic_spec,
        )
        self.gru = nn.GRU(self.encoder.output_dim, hidden_size, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, num_objectives)
        )
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward_step(self, features: torch.Tensor, hidden=None):
        encoded = self.encoder(features).unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(
                1,
                features.shape[0],
                self.hidden_size,
                device=features.device,
                dtype=features.dtype,
            )
        recurrent, new_hidden = self.gru(encoded, hidden)
        recurrent = recurrent[:, 0]
        return self.actor(recurrent), self.critic(recurrent), new_hidden

    def forward_sequence(
        self,
        features_seq: torch.Tensor,
        initial_hidden: torch.Tensor,
        dones_seq: torch.Tensor,
    ):
        batch_size, seq_len = features_seq.shape[:2]
        encoded = self.encoder(features_seq)

        # Most 16-step slices contain no boundary.  Use the fused GRU path for
        # those batches and retain exact hidden resets for the uncommon case.
        if seq_len == 1 or not torch.any(dones_seq[:, :-1] > 0):
            recurrent, _ = self.gru(encoded, initial_hidden)
        else:
            hidden = initial_hidden
            outputs = []
            for step in range(seq_len):
                if step:
                    keep = (1.0 - dones_seq[:, step - 1]).view(1, batch_size, 1)
                    hidden = hidden * keep
                output, hidden = self.gru(encoded[:, step : step + 1], hidden)
                outputs.append(output)
            recurrent = torch.cat(outputs, dim=1)

        flat = recurrent.reshape(batch_size * seq_len, self.hidden_size)
        return self.actor(flat), self.critic(flat)
