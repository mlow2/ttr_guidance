"""Scenario configuration, vehicle sampling, and rejection constraints.

The paper's 100 scenarios (scenarios/random_8v_cdc.json) store fixed initial states drawn by
rejection sampling: 8 vehicles with x ~ U(-5, -0.5), y ~ U(-3, 3), heading ~ U(-pi, pi),
speed 0.07, accepted when every pair starts at least 1.0 apart and every group of 3 vehicles
spans at least 10 s of TTR. Seeds 0-104 were drawn; seeds with a pairwise CBF value below the
safe threshold at t = 0 (27, 45, 78) and seeds with vehicles left unfinished by the TTR grid
limits (48, 65) are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ParamDist:
    """Base class for a parameter that is either fixed or sampled."""

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError


class Fixed(ParamDist):
    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, rng: np.random.Generator) -> float:
        return self.value

    def __repr__(self):
        return f"Fixed({self.value})"


def _wrap_param(v) -> ParamDist:
    """Convert a plain float/int to Fixed; pass through ParamDist instances."""
    if isinstance(v, ParamDist):
        return v
    return Fixed(float(v))


class VehicleSampler:
    """Samples one vehicle's initial state: [x, y, theta], plus speed when it is given."""

    def __init__(self, x=0.0, y=0.0, heading=0.0, speed=None):
        self.x = _wrap_param(x)
        self.y = _wrap_param(y)
        self.heading = _wrap_param(heading)
        self.speed = _wrap_param(speed) if speed is not None else None

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        state = [self.x.sample(rng), self.y.sample(rng), self.heading.sample(rng)]
        if self.speed is not None:
            state.append(self.speed.sample(rng))
        return np.array(state)

    def __repr__(self):
        parts = f"VehicleSampler(x={self.x}, y={self.y}, heading={self.heading}"
        if self.speed is not None:
            parts += f", speed={self.speed}"
        return parts + ")"


class RejectionConstraint:
    """Base class.  `check` returns True if the sample is *acceptable*."""

    def check(self, states: list[np.ndarray], **context: Any) -> bool:
        raise NotImplementedError


@dataclass
class ScenarioConfig:
    """One scenario: per-vehicle samplers, rejection constraints, and a base seed."""

    name: str
    vehicle_samplers: list[VehicleSampler]
    constraints: list[RejectionConstraint] = field(default_factory=list)
    max_rejection_attempts: int = 10_000
    seed: int | None = None

    def generate(self, seed: int | None = None,
                 raise_on_failure: bool = True,
                 **constraint_context: Any) -> list[np.ndarray] | None:
        """Rejection-sample initial states; after max_rejection_attempts raise, or return None."""
        effective_seed = seed
        if self.seed is not None:
            effective_seed = self.seed if seed is None else self.seed + seed
        rng = np.random.default_rng(effective_seed)

        for _ in range(self.max_rejection_attempts):
            states = [s.sample(rng) for s in self.vehicle_samplers]
            if all(c.check(states, **constraint_context) for c in self.constraints):
                return states

        if raise_on_failure:
            raise RuntimeError(
                f"Scenario '{self.name}': failed to satisfy constraints "
                f"after {self.max_rejection_attempts} attempts"
            )
        return None


def scenario_from_states(name: str, states: list[np.ndarray]) -> ScenarioConfig:
    """Wrap a list of fixed vehicle states as a deterministic ScenarioConfig."""
    samplers = []
    for s in states:
        kwargs = dict(x=float(s[0]), y=float(s[1]), heading=float(s[2]))
        if len(s) > 3:
            kwargs["speed"] = float(s[3])
        samplers.append(VehicleSampler(**kwargs))
    return ScenarioConfig(name=name, vehicle_samplers=samplers)
