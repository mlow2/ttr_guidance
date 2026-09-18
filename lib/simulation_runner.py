"""Loads HJ reachability data once and runs (scenario, safety_mode, nominal_mode) simulations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hj_reachability_torch.grid import get_hj_numpy_grid_from_meta_data
from hj_reachability_torch.utils import to_numpy
from hj_reachability_torch.eval import rollout_controller

from lib.dubins_car import (
    DubinsCar, KinematicVehicle, MultipleVehicle,
    TwoDubinsCarRelative, TwoKinematicVehicleRelative,
)
from lib.safety_filter import (
    MultiVehicleSafetyFilterHandle,
    ClusteredMultiVehicleSafetyFilterHandle,
    MultiStageCBFFilter,
    build_stages_from_names,
    DEFAULT_HMS_STAGE_NAMES,
    SAFETY_MODES,
    DEFAULT_SAFETY_MODE,
    CLUSTER_HANDLE_MODES,
    MODES_NEEDING_PRIORITY,
    compute_most_critical_safety_order,
    compute_critical_neighbors,
)
from lib.metrics import normalized_control_override
from lib.cbf_logger import CBFLogger
from lib.sim_and_ctrl import (
    CorridorHandle,
    CorridorStatusTracker,
    MissionAwareDynamics,
    MissionCompleteTracker,
    switching_control_multi_vehicle_static,
    switching_control_multi_vehicle_static_with_ttr_separation,
    convert_state_stacked_to_list,
    convert_action_stacked_to_list,
    convert_action_list_to_stacked,
    evaluate_ttr_priority,
    get_ttr_per_vehicle,
    event_function_all_vehicle_reach_target_corridor_based,
)
from lib.scenario import ScenarioConfig

VALID_SAFETY_MODES = SAFETY_MODES
CLUSTERED_MODES = CLUSTER_HANDLE_MODES
NEEDS_PRIORITY = MODES_NEEDING_PRIORITY

VALID_NOMINAL_MODES = {
    "ttr_min",
    "ttr_track_bang",
}

_NOMINAL_NEEDS_PRIORITY = {
    "ttr_track_bang",
}


@dataclass
class SimulationResult:
    """Container for all outputs of a single simulation run."""
    ts: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    extras: dict[str, Any]
    initial_priority: list[int] | None
    initial_ttr: list[float]
    initial_target_ttr: list[float]
    safety_mode: str
    nominal_mode: str
    cbf_logger: CBFLogger | None = None
    num_vehicles: int = 0
    vehicle_x_dim: int = 0
    separation_distance: float = 0.0
    corridor_handle: CorridorHandle | None = None


_DEFAULT_DATA_PATHS = {
    "dubins_car": {
        "hj_data": "data/dubins_ttr_grid.pkl",
        "safety_data": "data/dubins_relative_dynamics_cbf.pkl",
    },
    "kinematic_vehicle": {
        "hj_data": "data/ttr_grid.pkl",
        "safety_data": "data/relative_dynamics_cbf.pkl",
    },
}

VALID_DYNAMICS_TYPES = set(_DEFAULT_DATA_PATHS.keys())

TTR_TARGET_CAP_RATIO = 0.98
TTR_TARGET_CAP_QUANTILE = 0.99

DEFAULT_OVERRIDE_TOL = 0.02


def compute_ttr_grid_cap(values_ttr,
                         ratio: float = TTR_TARGET_CAP_RATIO,
                         quantile: float = TTR_TARGET_CAP_QUANTILE) -> float:
    """ratio * quantile of the finite TTR table, sampled on a stride to avoid a full-size finite mask."""
    strides = tuple(slice(None, None, 4 if axis < 2 else 2)
                    for axis in range(np.ndim(values_ttr)))
    sample = np.asarray(values_ttr[strides]).ravel()
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise ValueError("TTR table has no finite entries")
    return float(ratio * np.quantile(finite, quantile))


class SimulationRunner:
    """Loads HJ data once, then runs arbitrary (scenario, safety_mode) pairs."""

    def __init__(self,
                 dynamics_type: str,
                 hj_data_path: str | None = None,
                 safety_data_path: str | None = None):
        if dynamics_type not in VALID_DYNAMICS_TYPES:
            raise ValueError(f"Unknown dynamics_type: {dynamics_type}")
        self.dynamics_type = dynamics_type
        defaults = _DEFAULT_DATA_PATHS[dynamics_type]
        hj_data_path = hj_data_path or defaults["hj_data"]
        safety_data_path = safety_data_path or defaults["safety_data"]
        self._setup_torch()
        self._load_data(hj_data_path, safety_data_path)
        self._mission_tracker = None

    @staticmethod
    def _setup_torch():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

    def _load_data(self, hj_data_path: str, safety_data_path: str):
        with open(hj_data_path, "rb") as f:
            hj_data = pickle.load(f)
        self.grid = get_hj_numpy_grid_from_meta_data(hj_data.grid_meta_data)
        self.values_ttr = to_numpy(hj_data.values)
        self.target_function = to_numpy(hj_data.target_function)
        self.corridor_width = hj_data.info["corridor_width"]
        self.angular_rate_max = hj_data.info["angular_rate_max"]
        self.ttr_max = hj_data.ttr_max

        if self.dynamics_type == "dubins_car":
            self.speed = hj_data.info["speed"]
        elif self.dynamics_type == "kinematic_vehicle":
            self.speed_min = hj_data.info["speed_min"]
            self.speed_max = hj_data.info["speed_max"]
            self.speed = hj_data.info["speed_target"]
            self.accel_min = hj_data.info["accel_min"]
            self.accel_max = hj_data.info["accel_max"]
        else:
            raise NotImplementedError(f"Unknown dynamics type: {self.dynamics_type}")

        with open(safety_data_path, "rb") as f:
            hj_data_safety = pickle.load(f)
        self.values_safety = to_numpy(hj_data_safety.values)
        self.grid_safety = get_hj_numpy_grid_from_meta_data(hj_data_safety.grid_meta_data)
        self.separation_distance = hj_data_safety.info["separation_distance"]

        self.corridor_handle = CorridorHandle(
            half_width=0.5 * self.corridor_width,
            entrance_position=np.array([0.0, 0.0]),
            heading=0,
        )
        self._ttr_grid_cap = None

    @property
    def ttr_grid_cap(self) -> float:
        """Trackable-TTR cap: a high quantile of the finite table, since the table max sits at the solver horizon."""
        if self._ttr_grid_cap is None:
            self._ttr_grid_cap = compute_ttr_grid_cap(self.values_ttr)
        return self._ttr_grid_cap

    def control_ranges(self) -> list[float]:
        """Per-dimension control span, from the loaded parameters."""
        span = [2.0 * self.angular_rate_max]
        if self.dynamics_type == "kinematic_vehicle":
            span.append(self.accel_max - self.accel_min)
        return span

    def run(self,
            scenario: ScenarioConfig,
            safety_mode: str = DEFAULT_SAFETY_MODE,
            nominal_mode: str = "ttr_min",
            ttr_separation: float = 10.0,
            cbf_rate: float = 1.0,
            cbf_h_margin: float = 0.01,
            t_sim: float = 150.0,
            dt: float = 0.1,
            enable_early_termination: bool = True,
            enable_cbf_logging: bool = True,
            seed: int | None = None,
            verbose: bool = False,
            hybrid_stage_names: list[str] | None = None,
            hybrid_order: str = "most_critical_safety_pair",
            enable_mission_complete: bool = True,
            min_corridor_traverse_dist: float = 3.5,
            corridor_final_threshold: float = 4.0,
            corridor_enter_delay: int = 3,
            corridor_exit_delay: int = 15,
            override_tol: float = DEFAULT_OVERRIDE_TOL,
            ) -> SimulationResult:
        """Run a single simulation and return a SimulationResult."""
        if safety_mode not in VALID_SAFETY_MODES:
            raise ValueError(f"Unknown safety_mode: {safety_mode}")
        if nominal_mode not in VALID_NOMINAL_MODES:
            raise ValueError(f"Unknown nominal_mode: {nominal_mode}")

        effective_seed = seed
        if scenario.seed is not None:
            effective_seed = scenario.seed if seed is None else scenario.seed + seed
        if effective_seed is not None:
            s = int(effective_seed)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        constraint_context = dict(grid=self.grid, ttr_values=self.values_ttr)
        x0_list = scenario.generate(seed=seed, **constraint_context)
        num_vehicles = len(x0_list)
        x0_serial = np.concatenate(x0_list).astype(np.float32)

        if self.dynamics_type == "dubins_car":
            dynsys = DubinsCar(speed=self.speed, angular_rate_max=self.angular_rate_max)
            safety_filter_dynamics = TwoDubinsCarRelative(
                speed=self.speed, angular_rate_max=self.angular_rate_max)
        elif self.dynamics_type == "kinematic_vehicle":
            dynsys = KinematicVehicle(
                v_min=self.speed_min, v_max=self.speed_max,
                accel_min=self.accel_min, accel_max=self.accel_max,
                angular_rate_max=self.angular_rate_max,
                speed_target=self.speed)
            safety_filter_dynamics = TwoKinematicVehicleRelative(
                v_min=self.speed_min, v_max=self.speed_max,
                accel_min=self.accel_min, accel_max=self.accel_max,
                angular_rate_max=self.angular_rate_max)
        else:
            raise NotImplementedError(f"Unknown dynamics type: {self.dynamics_type}")

        dynsys_multi = MultipleVehicle(
            num_vehicles=num_vehicles, vehicle_dynamics=dynsys)

        cbf_logger = CBFLogger() if enable_cbf_logging else None

        safety_filter_handle = MultiVehicleSafetyFilterHandle(
            self.grid_safety, self.values_safety, safety_filter_dynamics,
            cbf_rate=cbf_rate, h_margin=cbf_h_margin, logger=cbf_logger)

        clustered_handle = None
        if safety_mode in CLUSTERED_MODES:
            clustered_handle = ClusteredMultiVehicleSafetyFilterHandle(
                self.grid_safety, self.values_safety, safety_filter_dynamics,
                cbf_rate=cbf_rate, h_margin=cbf_h_margin,
                logger=cbf_logger,
            )

        needs_priority = (
            safety_mode in NEEDS_PRIORITY
            or nominal_mode in _NOMINAL_NEEDS_PRIORITY
        )
        if needs_priority:
            initial_priority = evaluate_ttr_priority(
                x0_list, self.corridor_handle, self.grid,
                self.values_ttr,
                target_function_corridor=self.target_function)
        else:
            initial_priority = None

        if cbf_logger is not None:
            cbf_logger.set_priority_map(initial_priority)

        initial_ttr = get_ttr_per_vehicle(x0_list, self.grid, self.values_ttr)

        # Always built: grid escape must freeze the vehicle even when mission completion is off.
        mission_tracker = MissionCompleteTracker(
            corridor_handle=self.corridor_handle,
            vehicle_x_dim=dynsys.x_dim,
            num_vehicles=num_vehicles,
            min_corridor_traverse_dist=min_corridor_traverse_dist,
            corridor_final_threshold=corridor_final_threshold,
            grid=self.grid,
            track_completion=enable_mission_complete,
        )
        rollout_dynamics = MissionAwareDynamics(dynsys_multi, mission_tracker)
        self._mission_tracker = mission_tracker

        corridor_tracker = CorridorStatusTracker(
            corridor_handle=self.corridor_handle,
            vehicle_x_dim=dynsys.x_dim,
            num_vehicles=num_vehicles,
            enter_delay=corridor_enter_delay,
            exit_delay=corridor_exit_delay,
        )

        nominal_ctrl = self._build_nominal_controller(
            nominal_mode=nominal_mode,
            dynsys_multi=dynsys_multi,
            static_priority_list=initial_priority,
            ttr_separation=ttr_separation,
        )
        vehicle_clusters = [list(range(num_vehicles))]
        raw_controller = self._build_controller(
            safety_mode=safety_mode,
            dynsys=dynsys,
            safety_filter_handle=safety_filter_handle,
            clustered_handle=clustered_handle,
            cbf_logger=cbf_logger,
            static_priority_list=initial_priority,
            vehicle_clusters=vehicle_clusters,
            nominal_ctrl=nominal_ctrl,
            hybrid_stage_names=hybrid_stage_names,
            hybrid_order=hybrid_order,
        )

        xd = dynsys.x_dim
        ud = dynsys.u_dim
        _ind_handle = safety_filter_handle.individual_handle
        _u_ranges = self.control_ranges()
        _ttr_cap = self.ttr_grid_cap

        def _safety_diagnostics(u, extras, x_mod, mt):
            """Per-vehicle critical neighbour, override flag and TTR-target cap flag."""
            state_list = convert_state_stacked_to_list(x_mod, xd)
            active = [i for i in range(num_vehicles) if not mt.completed[i]]
            neighbours = [None] * num_vehicles
            values = [None] * num_vehicles
            if len(active) > 1:
                a_nb, a_val = compute_critical_neighbors(
                    [state_list[i] for i in active], _ind_handle, active)
                for local_idx, i in enumerate(active):
                    neighbours[i] = a_nb[local_idx]
                    values[i] = a_val[local_idx]

            override = [False] * num_vehicles
            u_ref = extras.get("u_ref")
            if u_ref is not None:
                ref = np.asarray(u_ref).ravel()
                for i in active:
                    mag = normalized_control_override(
                        u[ud * i: ud * (i + 1)], ref[ud * i: ud * (i + 1)],
                        _u_ranges)
                    override[i] = bool(mag > override_tol)

            capped = [False] * num_vehicles
            targets = extras.get("ttr_target")
            ttr = extras.get("ttr")
            for i in active:
                # Targets are clamped at the source, so sitting at the cap means the target was clamped.
                at_cap = (targets is not None and i < len(targets)
                          and float(targets[i]) >= _ttr_cap - 1e-6)
                lost_signal = (ttr is not None and i < len(ttr)
                               and not np.isfinite(float(ttr[i])))
                capped[i] = bool(at_cap or lost_signal)

            extras["safety_critical"] = neighbours
            extras["safety_critical_h"] = values
            extras["safety_override"] = override
            extras["ttr_target_capped"] = capped

        def controller(t, x,
                       _mt=mission_tracker,
                       _ct=corridor_tracker,
                       _inner=raw_controller):
            _mt.update(x)
            _ct.update(x)

            x_mod = x.copy()
            for i, frozen in enumerate(_mt.frozen_mask):
                if frozen:
                    x_mod[xd * i: xd * (i + 1)] = _mt.frozen_states[i]

            u, extras = _inner(t, x_mod)

            for i, frozen in enumerate(_mt.frozen_mask):
                if frozen:
                    u[ud * i: ud * (i + 1)] = 0
            extras["mission_complete"] = list(_mt.completed)
            extras["escaped"] = list(_mt.escaped)

            extras["within_corridor"] = list(_ct.within_corridor)
            _safety_diagnostics(u, extras, x_mod, _mt)
            return u, extras

        target_reach_event = None
        if enable_early_termination:
            target_reach_event = (
                lambda t, state: event_function_all_vehicle_reach_target_corridor_based(
                    t, state, self.corridor_handle, vehicle_x_dim=dynsys.x_dim))
            target_reach_event.terminal = True
            target_reach_event.direction = -1

        ts, xs, us, _, extras = rollout_controller(
            x0_serial, rollout_dynamics, controller, t_sim,
            dt=dt, exit_event_func=target_reach_event, verbose=verbose)
        self._mission_tracker = None

        ttr_target_series = extras.get("ttr_target")
        if ttr_target_series is not None and len(ttr_target_series) > 0:
            initial_target_ttr = [float(v) for v in ttr_target_series[0]]
        else:
            initial_target_ttr = list(initial_ttr)

        return SimulationResult(
            ts=ts, xs=xs, us=us, extras=extras,
            initial_priority=initial_priority,
            initial_ttr=initial_ttr,
            initial_target_ttr=initial_target_ttr,
            safety_mode=safety_mode,
            nominal_mode=nominal_mode,
            cbf_logger=cbf_logger,
            num_vehicles=num_vehicles,
            vehicle_x_dim=dynsys.x_dim,
            separation_distance=self.separation_distance,
            corridor_handle=self.corridor_handle,
        )

    def _build_nominal_controller(self, *, nominal_mode, dynsys_multi,
                                  static_priority_list, ttr_separation):
        """Return a callable `(t, x) -> (u_opt, extras)` for the nominal policy."""
        corridor_handle = self.corridor_handle
        grid = self.grid
        values_ttr = self.values_ttr
        target_function = self.target_function

        if nominal_mode == "ttr_min":
            def nominal_ctrl(t, x):
                return switching_control_multi_vehicle_static(
                    t, x, dynamics=dynsys_multi, corridor_handle=corridor_handle,
                    grid=grid, values=values_ttr,
                    target_function_corridor=target_function, control_mode="min")
        elif nominal_mode == "ttr_track_bang":
            def nominal_ctrl(t, x):
                return switching_control_multi_vehicle_static_with_ttr_separation(
                    t, x, ttr_separation=ttr_separation,
                    dynamics=dynsys_multi, corridor_handle=corridor_handle,
                    grid=grid, values_ttr=values_ttr,
                    priority_list=static_priority_list)
        else:
            raise ValueError(f"Unhandled nominal_mode: {nominal_mode}")
        return nominal_ctrl

    def _build_controller(self, *, safety_mode, dynsys,
                          safety_filter_handle, clustered_handle,
                          cbf_logger, static_priority_list, vehicle_clusters,
                          nominal_ctrl, hybrid_stage_names=None,
                          hybrid_order="most_critical_safety_pair"):
        ind_handle = safety_filter_handle.individual_handle
        ho = hybrid_order

        if safety_mode == "none":
            def controller(t, x):
                u_opt, extras = nominal_ctrl(t, x)
                state_list = convert_state_stacked_to_list(x, dynsys.x_dim)
                extras["ttr"] = get_ttr_per_vehicle(
                    state_list, self.grid, self.values_ttr)
                extras["priority"] = static_priority_list
                return u_opt, extras
        elif safety_mode in CLUSTERED_MODES:
            def controller(t, x):
                return self._ctrl_clustered(
                    t, x, dynsys, clustered_handle, cbf_logger,
                    vehicle_clusters, static_priority_list, nominal_ctrl)
        elif safety_mode == "hybrid_multi_stage":
            names = hybrid_stage_names or DEFAULT_HMS_STAGE_NAMES
            stages = build_stages_from_names(
                names, safety_filter_handle.individual_handle)
            def controller(t, x):
                return self._ctrl_multi_stage(
                    t, x, dynsys, stages, cbf_logger,
                    static_priority_list, nominal_ctrl,
                    individual_handle=ind_handle, hybrid_order=ho)
        elif safety_mode == "decentralized_nearest_joint_hard_then_slack":
            def controller(t, x):
                return self._ctrl_flattened_no_priority_fallback_relaxed(
                    t, x, dynsys, safety_filter_handle,
                    cbf_logger, nominal_ctrl, ind_handle, ho)
        else:
            raise ValueError(f"Unhandled safety_mode: {safety_mode}")
        return controller

    def _get_active_for_safety(self, state_list, action_list,
                               priority_list=None, clusters=None):
        """Drop mission-complete vehicles; returns (states, actions, priority, clusters, active_indices or None)."""
        mt = self._mission_tracker

        excluded: set[int] = set()
        if mt is not None:
            for i, done in enumerate(mt.completed):
                if done:
                    excluded.add(i)

        if not excluded:
            return state_list, action_list, priority_list, clusters, None

        active = [i for i in range(len(state_list)) if i not in excluded]
        if len(active) == len(state_list):
            return state_list, action_list, priority_list, clusters, None
        if not active:
            return [], [], None, None, active

        old_to_new = {old: new for new, old in enumerate(active)}
        a_states = [state_list[i] for i in active]
        a_actions = [action_list[i] for i in active]

        a_priority = None
        if priority_list is not None:
            a_priority = [old_to_new[i] for i in priority_list
                          if i in old_to_new]

        a_clusters = None
        if clusters is not None:
            a_clusters = []
            for cluster in clusters:
                remapped = [old_to_new[i] for i in cluster if i in old_to_new]
                if remapped:
                    a_clusters.append(remapped)

        return a_states, a_actions, a_priority, a_clusters, active

    def _expand_safety_result(self, filtered_u, nominal_action_list,
                              active_indices):
        """Map a filtered safety-filter result back to the full vehicle list."""
        if active_indices is None:
            return filtered_u
        result = list(nominal_action_list)
        for new_idx, old_idx in enumerate(active_indices):
            result[old_idx] = filtered_u[new_idx]
        return result


    def _ctrl_clustered(self, t, x, dynsys, clustered_handle,
                        cbf_logger, vehicle_clusters, priority_list,
                        nominal_ctrl):
        if cbf_logger is not None:
            cbf_logger.set_time(t)
        u_opt, extras = nominal_ctrl(t, x)
        state_list = convert_state_stacked_to_list(x, dynsys.x_dim)
        action_list = convert_action_stacked_to_list(u_opt, dynsys.u_dim)
        a_s, a_a, _, a_c, active = self._get_active_for_safety(
            state_list, action_list, clusters=vehicle_clusters)
        if active is not None and len(active) <= 1:
            u_safe_list = action_list
        else:
            filtered = clustered_handle.apply_safety_filter_with_clusters(
                a_s, a_a, clusters=a_c)
            u_safe_list = self._expand_safety_result(
                filtered, action_list, active)
        extras["priority"] = priority_list
        extras["u_ref"] = u_opt
        extras["ttr"] = get_ttr_per_vehicle(state_list, self.grid, self.values_ttr)
        return convert_action_list_to_stacked(u_safe_list), extras

    def _ctrl_multi_stage(self, t, x, dynsys, stages,
                          cbf_logger, priority_list, nominal_ctrl,
                          individual_handle=None, hybrid_order=None):
        if cbf_logger is not None:
            cbf_logger.set_time(t)
        u_opt, extras = nominal_ctrl(t, x)
        state_list = convert_state_stacked_to_list(x, dynsys.x_dim)
        action_list = convert_action_stacked_to_list(u_opt, dynsys.u_dim)
        a_s, a_a, a_p, _, active = self._get_active_for_safety(
            state_list, action_list, priority_list)
        if active is not None and len(active) <= 1:
            u_sol_list = action_list
        else:
            vehicle_order = self._compute_vehicle_order(
                a_s, individual_handle, hybrid_order)
            ms_filter = MultiStageCBFFilter(stages, logger=cbf_logger)
            filtered = ms_filter.apply(a_s, a_a, priority_list=a_p,
                                       vehicle_order=vehicle_order)
            u_sol_list = self._expand_safety_result(
                filtered, action_list, active)
        extras["priority"] = priority_list
        extras["u_ref"] = u_opt
        extras["ttr"] = get_ttr_per_vehicle(state_list, self.grid, self.values_ttr)
        return convert_action_list_to_stacked(u_sol_list), extras

    @staticmethod
    def _compute_vehicle_order(state_list, individual_handle, hybrid_order):
        if hybrid_order == "most_critical_safety_pair" and individual_handle is not None:
            return compute_most_critical_safety_order(state_list, individual_handle)
        return None

    def _ctrl_flattened_no_priority_fallback_relaxed(
            self, t, x, dynsys, sfh, cbf_logger,
            nominal_ctrl, individual_handle, hybrid_order):
        if cbf_logger is not None:
            cbf_logger.set_time(t)
        u_opt, extras = nominal_ctrl(t, x)
        state_list = convert_state_stacked_to_list(x, dynsys.x_dim)
        action_list = convert_action_stacked_to_list(u_opt, dynsys.u_dim)
        extras["priority"] = None
        extras["ttr"] = get_ttr_per_vehicle(state_list, self.grid, self.values_ttr)
        a_s, a_a, _, _, active = self._get_active_for_safety(
            state_list, action_list)
        if active is not None and len(active) <= 1:
            u_safe_list = action_list
        else:
            vehicle_order = self._compute_vehicle_order(
                a_s, individual_handle, hybrid_order)
            filtered = sfh.apply_safety_filter_no_priority_fallback_relaxed(
                a_s, a_a, vehicle_order=vehicle_order)
            u_safe_list = self._expand_safety_result(
                filtered, action_list, active)
        extras["u_ref"] = u_opt
        return convert_action_list_to_stacked(u_safe_list), extras
