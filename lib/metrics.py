"""Performance metrics for multi-vehicle simulation evaluation.

Metrics
-------
- **Kendall's tau-b**: agreement between priority ordering and arrival ordering.
- **True TTR**: actual time each vehicle takes to reach the corridor.
- **Safety violations (pair)**: count of pairwise separation breaches per timestep.
- **Safety violations (vehicle)**: count of distinct vehicles in any violation
  per timestep, summed across timesteps.
- **Safety violations (vehicle, normalized)**: fraction of total vehicle-flight
  time spent in safety violation, in free space, inside the corridor, and
  globally.  Bounded [0, 1].
- **Minimum separation distance**: closest pairwise pass over all active
  vehicles, in free space, inside the corridor, and anywhere.
- **TTR delay avg**: mean TTR delay across vehicles (lower = tighter tracking).
  TTR delay uses raw initial TTR and clamps early arrivals to zero.
- **Normalized TTR delay std**: standard deviation of per-vehicle TTR delay
  normalized by each vehicle's initial raw TTR.
- **Wall-clock solve time**: mean per-vehicle per-timestep wall-clock time
  (ms) to resolve a vehicle's safety-filter QP(s), computed from
  ``cbf_log.csv``.  Only timesteps with a vehicle outside the corridor count.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Kendall's tau-b
# ---------------------------------------------------------------------------

def kendall_tau_b(priority_ranks, arrival_ranks) -> float | None:
    """Compute Kendall's tau-b between two rankings.

    Parameters
    ----------
    priority_ranks : sequence of numeric values
        Lower value means higher priority / earlier expected arrival.
    arrival_ranks : sequence of numeric values
        Lower value means earlier actual arrival.

    Returns
    -------
    float or None
        tau-b in [-1, 1], or None if undefined (< 2 items or zero denominator).
    """
    n = len(priority_ranks)
    if n != len(arrival_ranks):
        raise ValueError("priority_ranks and arrival_ranks must have the same length")
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n - 1):
        xi = priority_ranks[i]
        yi = arrival_ranks[i]
        for j in range(i + 1, n):
            dx = xi - priority_ranks[j]
            dy = yi - arrival_ranks[j]

            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue

            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1

    n0 = n * (n - 1) // 2
    denom_x = n0 - ties_x
    denom_y = n0 - ties_y
    denom = (denom_x * denom_y) ** 0.5

    if denom == 0:
        return None

    return (concordant - discordant) / denom


# ---------------------------------------------------------------------------
# TTR helpers
# ---------------------------------------------------------------------------

def compute_actual_ttr(trajectory_records: list[dict],
                       num_vehicles: int) -> list[float | None]:
    """Return per-vehicle actual TTR from trajectory log records.

    Actual TTR for vehicle *i* is the first `sim_time` where
    `in_corridor == 1`.  Returns `None` for vehicles that never reach
    the corridor.
    """
    first_arrival: dict[int, float] = {}
    for row in trajectory_records:
        vid = int(row["vehicle_id"])
        if vid in first_arrival:
            continue
        if int(row["in_corridor"]) == 1:
            first_arrival[vid] = float(row["sim_time"])

    return [first_arrival.get(i) for i in range(num_vehicles)]


def _correct_early_termination_ttr(
    records: list[dict],
    actual_ttr: list[float | None],
    num_vehicles: int,
    dt: float,
    corridor_entrance: np.ndarray,
    corridor_half_width: float,
    corridor_heading: float = 0.0,
    position_tol: float = 0.05,
) -> list[float | None]:
    """Assign ``floor(final_time / dt) * dt`` to vehicles that triggered the exit event but
    log ``in_corridor=0`` at the off-grid event time (within *position_tol* of the corridor)."""
    if not records or dt <= 0:
        return actual_ttr

    final_time = max(float(r["sim_time"]) for r in records)

    remainder = final_time % dt
    if remainder < 1e-6 or (dt - remainder) < 1e-6:
        return actual_ttr

    if all(a is not None for a in actual_ttr):
        return actual_ttr

    corrected_time = (final_time // dt) * dt

    final_rows: dict[int, dict] = {}
    for row in records:
        if abs(float(row["sim_time"]) - final_time) < 1e-9:
            final_rows[int(row["vehicle_id"])] = row

    heading_vec = np.array([np.cos(corridor_heading), np.sin(corridor_heading)])
    entrance = np.asarray(corridor_entrance, dtype=np.float64).flatten()[:2]

    result = list(actual_ttr)
    for vid in range(num_vehicles):
        if result[vid] is not None:
            continue
        if vid not in final_rows:
            continue
        row = final_rows[vid]
        pos = np.array([float(row["x"]), float(row["y"])])
        rel = pos - entrance
        progress = heading_vec @ rel
        lateral_offset = np.cross(heading_vec, rel)
        if progress >= -position_tol and abs(lateral_offset) <= corridor_half_width + position_tol:
            result[vid] = corrected_time

    return result


def compute_arrival_order(actual_ttr: list[float | None]) -> list[int]:
    """Convert actual TTR list to arrival-rank list (0-indexed, lower = earlier).

    Vehicles that never arrive get rank = num_vehicles.
    """
    n = len(actual_ttr)
    inf = float("inf")
    ttr_with_idx = [(actual_ttr[i] if actual_ttr[i] is not None else inf, i)
                    for i in range(n)]
    ttr_with_idx.sort(key=lambda x: x[0])
    ranks = [0] * n
    for rank, (_, vid) in enumerate(ttr_with_idx):
        ranks[vid] = rank
    return ranks


# ---------------------------------------------------------------------------
# Safety violations
# ---------------------------------------------------------------------------

@dataclass
class SafetyViolationResult:
    total_violations_pair: int
    total_violations_vehicle: int


def completed_onset(trajectory_records: list[dict]) -> dict[int, float]:
    """Per-vehicle first time with ``completed == 1``; a completed vehicle is frozen and no longer active traffic."""
    onset: dict[int, float] = {}
    for r in trajectory_records:
        if int(r["completed"]) != 1:
            continue
        vid = int(r["vehicle_id"])
        t = float(r["sim_time"])
        if vid not in onset or t < onset[vid]:
            onset[vid] = t
    return onset


def compute_safety_violations(
    trajectory_records: list[dict],
    separation_distance: float,
    corridor_filter: str = "outside",
    completed_onset_by_vid: dict[int, float] | None = None,
) -> SafetyViolationResult:
    """Count separation breaches across all timesteps.

    Returns both pairwise counts (one per violating pair per timestep) and
    vehicle-level counts (number of distinct vehicles involved in any
    violation per timestep).

    Parameters
    ----------
    corridor_filter : ``"outside"`` | ``"inside"`` | ``"all"``
        Which region to count violations in:

        - ``"outside"`` -- skip pairs where either vehicle has
          ``in_corridor == 1``.
        - ``"inside"`` -- only count pairs where **both** vehicles have
          ``within_corridor == 1`` (hysteresis-based corridor status).
        - ``"all"`` -- no corridor filtering; count violations everywhere.
    """
    if corridor_filter not in ("outside", "inside", "all"):
        raise ValueError(f"Unknown corridor_filter: {corridor_filter!r}")

    by_time: dict[str, list[dict]] = {}
    for row in trajectory_records:
        by_time.setdefault(row["sim_time"], []).append(row)

    total_pair = 0
    total_vehicle = 0

    onset = completed_onset_by_vid or {}

    for t_key, rows in by_time.items():
        t_val = float(t_key)
        rows_sorted = sorted(rows, key=lambda r: int(r["vehicle_id"]))
        positions = {}
        skip_vids: set[int] = set()
        include_vids: set[int] = set()

        for r in rows_sorted:
            vid = int(r["vehicle_id"])
            if vid in onset and t_val >= onset[vid]:
                skip_vids.add(vid)
            positions[vid] = np.array([float(r["x"]), float(r["y"])])

            if corridor_filter == "outside":
                if int(r.get("in_corridor", 0)) == 1:
                    skip_vids.add(vid)
            elif corridor_filter == "inside":
                if int(r["within_corridor"]) == 1:
                    include_vids.add(vid)

        violating_vehicles: set[int] = set()
        for i, j in itertools.combinations(sorted(positions.keys()), 2):
            if i in skip_vids or j in skip_vids:
                continue
            if corridor_filter == "inside" and (
                i not in include_vids or j not in include_vids
            ):
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < separation_distance:
                total_pair += 1
                violating_vehicles.add(i)
                violating_vehicles.add(j)

        total_vehicle += len(violating_vehicles)

    return SafetyViolationResult(
        total_violations_pair=total_pair,
        total_violations_vehicle=total_vehicle,
    )


# ---------------------------------------------------------------------------
# Proximity: most-severe incident
# ---------------------------------------------------------------------------

def compute_min_separation(
    positions_by_time: dict[str, dict[int, np.ndarray]],
) -> float | None:
    """Smallest pairwise distance over all timesteps of masked positions.

    ``positions_by_time`` maps a timestep key to ``{vehicle_id: [x, y]}`` and
    must contain **only the vehicles that should count** at that timestep
    (the caller masks out frozen / corridor-interior vehicles as appropriate).
    Returns ``None`` when no timestep has >= 2 vehicles.
    """
    min_sep: float | None = None
    for posmap in positions_by_time.values():
        for i, j in itertools.combinations(sorted(posmap.keys()), 2):
            dist = float(np.linalg.norm(posmap[i] - posmap[j]))
            if min_sep is None or dist < min_sep:
                min_sep = dist
    return min_sep


def positions_by_time_from_records(
    trajectory_records: list[dict],
    include_in_corridor: bool = False,
    completed_onset_by_vid: dict[int, float] | None = None,
    in_corridor_only: bool = False,
) -> dict[str, dict[int, np.ndarray]]:
    """Build a ``positions_by_time`` map from trajectory CSV records.

    Region selection mirrors :func:`compute_safety_violations`: *outside*
    (default) drops ``in_corridor == 1``; *all* (``include_in_corridor=True``)
    keeps everyone; *inside* (``in_corridor_only=True``) keeps only
    ``in_corridor == 1``.  Completed (landed / frozen) vehicles are dropped
    when *completed_onset_by_vid* is given.
    """
    onset = completed_onset_by_vid or {}
    by_time: dict[str, dict[int, np.ndarray]] = {}
    for row in trajectory_records:
        ic = int(row.get("in_corridor", 0) or 0)
        if in_corridor_only:
            if ic != 1:
                continue
        elif not include_in_corridor and ic == 1:
            continue
        vid = int(row["vehicle_id"])
        if vid in onset and float(row["sim_time"]) >= onset[vid]:
            continue
        by_time.setdefault(row["sim_time"], {})[vid] = np.array(
            [float(row["x"]), float(row["y"])])
    return by_time


# ---------------------------------------------------------------------------
# Normalized TTR delay spread
# ---------------------------------------------------------------------------

def compute_norm_ttr_delay_std(
    ttr_delay: list[float | None],
    initial_ttr: list[float],
) -> float | None:
    """Standard deviation of per-vehicle TTR delay normalized by initial TTR.

    Returns the population standard deviation, or None if fewer than 2 valid
    values exist.
    """
    norm_delays = [delay / init for delay, init in zip(ttr_delay, initial_ttr)
                   if delay is not None and init > 0]
    if len(norm_delays) < 2:
        return None
    return float(np.std(norm_delays, ddof=0))


# ---------------------------------------------------------------------------
# Wall-clock solve time
# ---------------------------------------------------------------------------

def compute_wall_clock_solve_time(
    cbf_log_records: list[dict],
    trajectory_records: list[dict],
) -> float | None:
    """Mean per-vehicle per-timestep wall-clock QP solve time (ms).

    Each ego vehicle's solves are summed within a timestep. The centralized
    cluster QP is charged to every vehicle. Only timesteps with a
    vehicle outside the corridor count. Returns None when there are none.
    """
    if not cbf_log_records:
        return None

    # Trajectory records may hold floats while cbf records hold CSV strings.
    def _tkey(v):
        return round(float(v), 6)

    outside_times: set[float] = set()
    for row in trajectory_records:
        if int(float(row.get("in_corridor", 0) or 0)) == 0:
            outside_times.add(_tkey(row["sim_time"]))

    rows_by_time: dict[float, list[dict]] = {}
    for row in cbf_log_records:
        t_key = _tkey(row["sim_time"])
        if t_key not in outside_times:
            continue
        rows_by_time.setdefault(t_key, []).append(row)

    per_vehicle_values: list[float] = []
    for rows in rows_by_time.values():
        prev = 0.0
        ego_time: dict[str, float] = {}
        for row in rows:
            val = row.get("wall_clock_ms", "")
            if val == "":
                continue
            ms = float(val)
            delta = ms - prev  # wall_clock_ms is cumulative within a step
            prev = ms
            if row.get("solver_type") == "multi_stage":  # summary row, not a solve
                continue
            vehicles = [v.strip() for v in str(row.get("vehicles", "")).split(",")]
            vehicles = [v for v in vehicles if v != ""]
            if not vehicles:
                continue
            egos = vehicles if row.get("solver_type") == "cluster" else vehicles[:1]
            for ego in egos:
                ego_time[ego] = ego_time.get(ego, 0.0) + delta
        per_vehicle_values.extend(ego_time.values())

    if not per_vehicle_values:
        return None
    return sum(per_vehicle_values) / len(per_vehicle_values)


# ---------------------------------------------------------------------------
# Control override
# ---------------------------------------------------------------------------

def normalized_control_override(control, control_ref, control_ranges) -> float:
    """``(1/u_dim) * sum_d |u_d - u_ref_d| / (u_max_d - u_min_d)`` for one vehicle."""
    u_dim = len(control_ranges)
    mag = 0.0
    for d in range(u_dim):
        mag += abs(float(control[d]) - float(control_ref[d])) / control_ranges[d]
    return mag / u_dim


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class ScenarioMetrics:
    kendall_tau: float | None
    actual_ttr: list[float | None]
    ttr_delay: list[float | None]
    safety_violations_pair: int
    safety_violations_vehicle: int
    safety_violations_vehicle_normalized: float | None
    safety_violations_corridor_normalized: float | None
    safety_violations_global_normalized: float | None
    min_separation_distance: float | None
    min_separation_distance_corridor: float | None
    min_separation_distance_free: float | None
    ttr_delay_avg: float | None
    norm_ttr_delay_std: float | None
    total_sim_time: float
    has_unfinished_vehicles: bool
    wall_clock_solve_time: float | None
    initial_raw_ttr: list[float]
    initial_target_ttr: list[float]
    priority_order: list[int]

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for a summary CSV row.

        Per-vehicle columns are indexed by priority rank (``_p0`` is the
        highest-priority vehicle).
        """
        d: dict[str, Any] = {
            "kendall_tau": self.kendall_tau,
            "safety_violations_pair": self.safety_violations_pair,
            "safety_violations_vehicle": self.safety_violations_vehicle,
            "safety_violations_vehicle_normalized": self.safety_violations_vehicle_normalized,
            "safety_violations_corridor_normalized": self.safety_violations_corridor_normalized,
            "safety_violations_global_normalized": self.safety_violations_global_normalized,
            "min_separation_distance": self.min_separation_distance,
            "min_separation_distance_corridor": self.min_separation_distance_corridor,
            "min_separation_distance_free": self.min_separation_distance_free,
            "ttr_delay_avg": self.ttr_delay_avg,
            "norm_ttr_delay_std": self.norm_ttr_delay_std,
            "total_sim_time": self.total_sim_time,
            "has_unfinished_vehicles": self.has_unfinished_vehicles,
            "wall_clock_solve_time": self.wall_clock_solve_time,
        }
        for rank, vid in enumerate(self.priority_order):
            d[f"init_raw_ttr_p{rank}"] = self.initial_raw_ttr[vid]
            d[f"init_target_ttr_p{rank}"] = self.initial_target_ttr[vid]
            d[f"actual_ttr_p{rank}"] = self.actual_ttr[vid]
            d[f"ttr_delay_p{rank}"] = self.ttr_delay[vid]
        return d


def compute_metrics(
    trajectory_csv_path: str,
    initial_ttr: list[float],
    initial_priority: list[int] | None,
    num_vehicles: int,
    separation_distance: float,
    initial_target_ttr: list[float],
    dt: float,
    corridor_entrance: tuple[float, float],
    corridor_half_width: float,
    corridor_heading: float = 0.0,
    cbf_log_csv_path: str | None = None,
) -> ScenarioMetrics:
    """Compute all metrics from a trajectory CSV and initial conditions.

    Parameters
    ----------
    trajectory_csv_path : str
        Path to the per-timestep trajectory CSV written by
        :class:`SimulationLogger`.
    initial_ttr : list[float]
        Raw initial TTR estimate per vehicle (from HJ value function).
    initial_priority : list[int] | None
        Vehicle indices in priority order (index 0 = highest priority).
        When *None*, priority is ascending initial TTR.
    num_vehicles : int
        Number of vehicles.
    separation_distance : float
        Safety threshold.
    initial_target_ttr : list[float]
        Per-vehicle target TTR used by the controller.
    dt : float
        Simulation timestep.  Converts violation counts to vehicle-time and
        detects unfinished vehicles caused by early termination.
    corridor_entrance : tuple[float, float]
        (x, y) of the corridor entrance.
    corridor_half_width : float
        Half-width of the corridor.
    corridor_heading : float
        Heading angle (radians) of the corridor axis.  Default ``0.0``.
    cbf_log_csv_path : str | None
        Path to ``cbf_log.csv``.  When provided, ``wall_clock_solve_time``
        is computed from this file.
    """
    from lib.simulation_logger import SimulationLogger
    records = SimulationLogger.load_csv(trajectory_csv_path)

    total_sim_time = 0.0
    if records:
        total_sim_time = max(float(r["sim_time"]) for r in records)

    actual = compute_actual_ttr(records, num_vehicles)
    actual = _correct_early_termination_ttr(
        records, actual, num_vehicles, dt,
        np.asarray(corridor_entrance, dtype=np.float64),
        corridor_half_width, corridor_heading,
    )

    arrival_order = compute_arrival_order(actual)

    if initial_priority is not None:
        priority_order = list(initial_priority)
    else:
        priority_order = sorted(range(num_vehicles),
                                key=lambda i: initial_ttr[i])

    priority_ranks = [0] * num_vehicles
    for rank, vid in enumerate(priority_order):
        priority_ranks[vid] = rank
    tau = kendall_tau_b(priority_ranks, arrival_order)

    ttr_delay: list[float | None] = []
    for raw, act in zip(initial_ttr, actual):
        ttr_delay.append(max(0.0, act - raw) if act is not None else None)

    valid_delays = [d for d in ttr_delay if d is not None]
    ttr_delay_avg = sum(valid_delays) / len(valid_delays) if valid_delays else None

    norm_delay_std = compute_norm_ttr_delay_std(ttr_delay, list(initial_ttr))

    has_unfinished = any(a is None for a in actual)

    # Completed vehicles are frozen, so they count neither as violations nor as flight time.
    onset = completed_onset(records)

    def _completed_at(r) -> bool:
        vid = int(r["vehicle_id"])
        return vid in onset and float(r["sim_time"]) >= onset[vid]

    safety = compute_safety_violations(
        records, separation_distance,
        corridor_filter="outside", completed_onset_by_vid=onset)
    safety_corridor = compute_safety_violations(
        records, separation_distance,
        corridor_filter="inside", completed_onset_by_vid=onset)
    safety_global = compute_safety_violations(
        records, separation_distance,
        corridor_filter="all", completed_onset_by_vid=onset)

    min_sep = compute_min_separation(positions_by_time_from_records(
        records, include_in_corridor=True, completed_onset_by_vid=onset))
    min_sep_corridor = compute_min_separation(positions_by_time_from_records(
        records, in_corridor_only=True, completed_onset_by_vid=onset))
    min_sep_free = compute_min_separation(positions_by_time_from_records(
        records, include_in_corridor=False, completed_onset_by_vid=onset))

    # Unfinished vehicles contribute the full run time to the denominator.
    ttr_sum = sum(t if t is not None else total_sim_time for t in actual)
    safety_vehicle_norm: float | None = None
    if ttr_sum > 0:
        safety_vehicle_norm = safety.total_violations_vehicle * dt / ttr_sum

    # Completed vehicles keep within_corridor == 1, so they are excluded from the denominator explicitly.
    safety_corridor_norm: float | None = None
    corridor_vehicle_time = dt * sum(
        1 for r in records
        if int(r["within_corridor"]) == 1 and not _completed_at(r))
    if corridor_vehicle_time > 0:
        safety_corridor_norm = (
            safety_corridor.total_violations_vehicle * dt / corridor_vehicle_time)

    safety_global_norm: float | None = None
    global_vehicle_time = dt * sum(1 for r in records if not _completed_at(r))
    if global_vehicle_time > 0:
        safety_global_norm = (
            safety_global.total_violations_vehicle * dt / global_vehicle_time)

    wc_solve_time: float | None = None
    if cbf_log_csv_path is not None:
        from lib.cbf_logger import CBFLogger
        cbf_records = CBFLogger.load_csv(cbf_log_csv_path)
        wc_solve_time = compute_wall_clock_solve_time(cbf_records, records)

    return ScenarioMetrics(
        kendall_tau=tau,
        actual_ttr=actual,
        ttr_delay=ttr_delay,
        safety_violations_pair=safety.total_violations_pair,
        safety_violations_vehicle=safety.total_violations_vehicle,
        safety_violations_vehicle_normalized=safety_vehicle_norm,
        safety_violations_corridor_normalized=safety_corridor_norm,
        safety_violations_global_normalized=safety_global_norm,
        min_separation_distance=min_sep,
        min_separation_distance_corridor=min_sep_corridor,
        min_separation_distance_free=min_sep_free,
        ttr_delay_avg=ttr_delay_avg,
        norm_ttr_delay_std=norm_delay_std,
        total_sim_time=total_sim_time,
        has_unfinished_vehicles=has_unfinished,
        wall_clock_solve_time=wc_solve_time,
        initial_raw_ttr=list(initial_ttr),
        initial_target_ttr=list(initial_target_ttr),
        priority_order=priority_order,
    )
