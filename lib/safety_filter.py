from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from hj_reachability_torch.grid import HjNumpyGrid
from hj_reachability_torch.dynamics import AffineDynamicsWithBoxInputBounds
from lib.dubins_car import TwoDubinsCarRelative, TwoKinematicVehicleRelative


VALID_HYBRID_ORDERS = {"most_critical_safety_pair", "priority_list"}

# CLARABEL's user_limit on these badly scaled but feasible QPs still yields a usable iterate.
_ACCEPTED_QP_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE,
                                   'user_limit', 'user_limit_inaccurate'})


def _solve_qp(problem, solver=cp.CLARABEL):
    """Solve with CLARABEL; on hard QPs it raises SolverError for genuine infeasibility, mapped to a status."""
    try:
        problem.solve(solver=solver)
        return problem.status or 'unknown'
    except SolverError:
        return cp.INFEASIBLE


class CBFQPSolveFailure(RuntimeError):
    """A CBF-QP that is feasible by construction failed to solve."""


def _logger_time(logger):
    return getattr(logger, "_current_time", None)


def _raise_qp_failure(context, vehicles, status, sim_time, reason):
    raise CBFQPSolveFailure(
        f"{context}: solver status {status!r} at t={sim_time} for vehicles "
        f"{list(vehicles)}. {reason} Passing the unfiltered nominal through would "
        "void the safety certificate, so the run is aborted instead."
    )


_ALWAYS_FEASIBLE_REASON = (
    "This QP is feasible by construction: the slack variables are nonnegative and "
    "unbounded above, the control box is nonempty, and the objective is convex and "
    "coercive, so every CBF row can be satisfied. A non-optimal status is therefore "
    "a solver or numerical failure, not genuine infeasibility."
)


def _wrap_angle(a):
    """Relative heading into [-pi, pi); the CBVF grid's heading axis spans exactly that."""
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def compute_most_critical_safety_order(
    state_list: List,
    individual_handle: 'IndividualVehicleSafetyFilterHandle',
) -> List[int]:
    """Vehicle indices sorted by their minimum pairwise CBF value, most critical first."""
    n = len(state_list)
    min_values = [float('inf')] * n
    for i in range(n):
        for j in range(i + 1, n):
            state_i = state_list[i]
            state_j = state_list[j]
            rel_dist = individual_handle.get_relative_distance(state_i, state_j)
            if rel_dist > individual_handle.safety_filter_range:
                continue
            state_rel = individual_handle.construct_state_from_two_vehicle_state(
                state_i, state_j)
            value = individual_handle.grid.eval_value_and_deriv_from_table(
                state_rel, individual_handle.values, return_deriv=False)
            min_values[i] = min(min_values[i], float(value))
            min_values[j] = min(min_values[j], float(value))
    return sorted(range(n), key=lambda v: min_values[v])


def compute_critical_neighbors(
    state_list: List,
    individual_handle: 'IndividualVehicleSafetyFilterHandle',
    active_indices: Optional[List[int]] = None,
):
    """Per-ego (neighbour_ids, min_values); ego-centric like the decentralized filter, not the symmetric ordering scan."""
    n = len(state_list)
    ids = list(active_indices) if active_indices is not None else list(range(n))
    neighbours: List[Optional[int]] = [None] * n
    min_values: List[Optional[float]] = [None] * n
    for i in range(n):
        others = [state_list[j] for j in range(n) if j != i]
        other_ids = [ids[j] for j in range(n) if j != i]
        res = individual_handle.evaluate_target_from_other_vehicle_state_list(
            state_list[i], others)
        if res is None:
            continue
        local_idx, _, value = res
        if local_idx is None:
            continue
        neighbours[i] = other_ids[local_idx]
        min_values[i] = float(value)
    return neighbours, min_values


@dataclass(frozen=True)
class ClusterPairwiseCBFData:
    local_i: int
    local_j: int
    vehicle_i: int
    vehicle_j: int
    state_relative: np.ndarray
    value: float
    grad_value: np.ndarray


@dataclass
class ClusteredQPResult:
    status: str
    control_solution: Optional[np.ndarray]
    slack_by_pair: Optional[Dict[tuple, float]] = None


@dataclass
class StageResult:
    """Result from a single stage of the multi-stage CBF pipeline."""
    vehicle_controls: Dict[int, np.ndarray]
    feasible_vehicles: Set[int] = field(default_factory=set)
    infeasible_vehicles: Set[int] = field(default_factory=set)


class CBFSolveStage:
    """A solve stage in the multi-stage CBF pipeline; subclasses implement `solve`."""

    def solve(self, state_list: List, action_list: List, vehicle_indices: Set[int],
              priority_list: Optional[List[int]] = None,
              fixed_controls: Optional[Dict[int, np.ndarray]] = None,
              vehicle_order: Optional[List[int]] = None) -> StageResult:
        """Solve CBF-QPs for vehicle_indices; fixed_controls from earlier stages replace the references in the constraints."""
        raise NotImplementedError


class IndividualVehicleSafetyFilterHandle:
    def __init__(self, grid: HjNumpyGrid,
                 values: np.ndarray,
                 safety_filter_dynamics: AffineDynamicsWithBoxInputBounds,
                 u_ego_index: list = [0],
                 safety_filter_range: float = np.inf,
                 cbf_rate: float = 5.0,
                 h_margin: float = 0.0,
                 logger=None):
        self.grid = grid
        # Assumes grid axes [0] and [1] are relative x and y.
        if np.isinf(safety_filter_range):
            self.safety_filter_range = min(grid.domain_hi[0], grid.domain_hi[1])
        else:
            self.safety_filter_range = safety_filter_range
        # used as CBF, positive inside the safe set.
        self.values = values
        self.safety_filter_dynamics = safety_filter_dynamics
        assert self.grid.ndim == self.safety_filter_dynamics.x_dim, "Grid dimension and dynamics state dimension should match."
        assert self.grid.ndim == self.values.ndim, "Grid dimension and values shape should match."
        self.u_dim = self.safety_filter_dynamics.u_dim
        self.u_min = self.safety_filter_dynamics.u_min
        self.u_max = self.safety_filter_dynamics.u_max
        self.cbf_rate = cbf_rate
        # Enforce h >= h_margin: the table is not control-invariant in the thin h ~ 0 layer.
        self.h_margin = h_margin
        self.u_ego_index = u_ego_index
        self.logger = logger
        self._current_ego_idx = None
        self._current_other_idx = None
        self._last_qp_feasible = True

    def set_current_pair(self, ego_idx, other_idx):
        """Set vehicle indices for the current pairwise QP."""
        self._current_ego_idx = ego_idx
        self._current_other_idx = other_idx

    def _sim_time(self, t=None):
        return t if t is not None else _logger_time(self.logger)

    def combine_actions(self, action_ego, action_target):
        """Place ego and target controls at the interleaved indices the pair dynamics expects, not per-vehicle blocks."""
        combined = np.empty(self.u_dim, dtype=np.float32)
        u_ego_idx = list(self.u_ego_index)
        u_other_idx = [i for i in range(self.u_dim) if i not in u_ego_idx]
        for k, idx in enumerate(u_ego_idx):
            combined[idx] = action_ego[k]
        for k, idx in enumerate(u_other_idx):
            combined[idx] = action_target[k]
        return combined

    def evaluate_target_from_other_vehicle_state_list(self, state_ego, state_list_other):
        """Evaluate the CBF value for each ego-other pair; return the index of the other vehicle with the lowest value."""
        if len(state_list_other) == 0:
            return None
        min_value = float('inf')
        min_index = None
        state_for_safety_target = None
        for j, state_other in enumerate(state_list_other):
            relative_distance = self.get_relative_distance(state_ego, state_other)
            if relative_distance > self.safety_filter_range:
                continue
            state_for_safety = self.construct_state_from_two_vehicle_state(state_ego, state_other)
            value = self.grid.eval_value_and_deriv_from_table(state_for_safety, self.values, return_deriv=False)
            if value < min_value:
                min_value = value
                min_index = j
                state_for_safety_target = state_for_safety
        return min_index, state_for_safety_target, min_value


    def construct_state_from_two_vehicle_state(self, state_ego, state_target):
        raise NotImplementedError

    def get_relative_distance(self, state_ego, state_target):
        raise NotImplementedError

    def apply_safety_filter(self, state_ego: np.ndarray, action_ego: np.ndarray,
                            state_list_other: List, action_list_other: List,
                            ego_vehicle_idx=None, other_vehicle_indices=None):
        """Apply the safety filter to the ego vehicle against the other vehicles."""
        target_index, state_for_safety_target, _ = self.evaluate_target_from_other_vehicle_state_list(state_ego, state_list_other)
        if target_index is None:
            return action_ego

        if ego_vehicle_idx is not None and other_vehicle_indices is not None:
            self.set_current_pair(ego_vehicle_idx, other_vehicle_indices[target_index])

        action_target = action_list_other[target_index]
        action_combined = self.combine_actions(action_ego, action_target)
        value, grad_value = self.grid.eval_value_and_deriv_from_table(state_for_safety_target, self.values, return_deriv=True)
        u_sol = self.cbf_qp(None, state_for_safety_target, action_combined, value, grad_value)
        return u_sol[np.asarray(self.u_ego_index, dtype=np.intp)]

    def apply_safety_filter_relaxed(self, state_ego: np.ndarray, action_ego: np.ndarray,
                                    state_list_other: List, action_list_other: List,
                                    ego_vehicle_idx=None, other_vehicle_indices=None,
                                    lambda_slack=10.0, eps_slack=1e-3):
        """Like apply_safety_filter but uses slack-relaxed CBF-QPs."""
        target_index, state_for_safety_target, _ = self.evaluate_target_from_other_vehicle_state_list(state_ego, state_list_other)
        if target_index is None:
            return action_ego

        if ego_vehicle_idx is not None and other_vehicle_indices is not None:
            self.set_current_pair(ego_vehicle_idx, other_vehicle_indices[target_index])

        action_target = action_list_other[target_index]
        action_combined = self.combine_actions(action_ego, action_target)
        value, grad_value = self.grid.eval_value_and_deriv_from_table(state_for_safety_target, self.values, return_deriv=True)
        u_sol = self.cbf_qp_relaxed(None, state_for_safety_target, action_combined, value, grad_value,
                                    lambda_slack=lambda_slack, eps_slack=eps_slack)
        return u_sol[np.asarray(self.u_ego_index, dtype=np.intp)]


    def apply_safety_filter_priority_stage(self, state_ego: np.ndarray, action_ego: np.ndarray,
                            state_list_other: List, action_list_other: List,
                            priority_than_ego_list_other: List,
                            ego_vehicle_idx=None, other_vehicle_indices=None):
        """HMS Stage 1: neighbour frozen at its reference; a higher-priority ego is only feasibility-checked, a lower-priority ego yields."""
        target_index, state_for_safety_target, _ = self.evaluate_target_from_other_vehicle_state_list(state_ego, state_list_other)
        if target_index is None:
            self._last_qp_feasible = True
            return action_ego

        if ego_vehicle_idx is not None and other_vehicle_indices is not None:
            self.set_current_pair(ego_vehicle_idx, other_vehicle_indices[target_index])

        target_priority_than_ego = priority_than_ego_list_other[target_index]
        action_target = action_list_other[target_index]
        action_combined = self.combine_actions(action_ego, action_target)
        value, grad_value = self.grid.eval_value_and_deriv_from_table(state_for_safety_target, self.values, return_deriv=True)
        priority = 'other' if target_priority_than_ego else 'both'
        u_sol = self.cbf_qp(None, state_for_safety_target, action_combined, value, grad_value, priority=priority)
        return u_sol[np.asarray(self.u_ego_index, dtype=np.intp)]

    def cbf_qp(self, t, x, u_ref, value, grad, priority=None):
        assert priority in [None, 'other', 'both'], "priority should be None, 'other', or 'both'"
        assert u_ref.shape == (self.u_dim,), f"u_ref should have shape ({self.u_dim},)"
        u = cp.Variable(self.u_dim)
        objective = cp.Minimize(cp.sum_squares(u-u_ref))

        constraints = [
            grad @ self.safety_filter_dynamics.dynamics_for_filter(t, x, u)
            + self.cbf_rate * (value - self.h_margin) >= 0,
            u >= self.u_min,
            u <= self.u_max,
        ]
        u_ego_idx = list(self.u_ego_index)
        u_other_idx = [i for i in range(self.u_dim) if i not in u_ego_idx]
        if priority == 'other':
            for idx in u_other_idx:
                constraints.append(u[idx] == u_ref[idx])
        elif priority == 'both':
            for idx in range(self.u_dim):
                constraints.append(u[idx] == u_ref[idx])
        problem = cp.Problem(objective, constraints)
        status = _solve_qp(problem)
        self._last_qp_feasible = (status == "optimal")

        if self.logger is not None and self._current_ego_idx is not None:
            ego, other = self._current_ego_idx, self._current_other_idx
            self.logger.log('pairwise', f'V{self.logger.vid_label(ego)}',
                            [ego, other], status)

        if not self._last_qp_feasible:
            # No slack here, so infeasibility is genuine; the caller cascades via _last_qp_feasible.
            return u_ref

        u_sol = np.asarray(u.value, dtype=np.float32)
        u_sol = np.clip(u_sol, self.u_min, self.u_max)
        return u_sol

    def cbf_qp_relaxed(self, t, x, u_ref, value, grad,
                       lambda_slack=10.0, eps_slack=1e-3):
        assert u_ref.shape == (self.u_dim,), f"u_ref should have shape ({self.u_dim},)"
        u = cp.Variable(self.u_dim)
        # Unbounded-above nonneg slack over a nonempty control box: always feasible.
        slack = cp.Variable(nonneg=True)
        objective = cp.Minimize(
            cp.sum_squares(u - u_ref)
            + lambda_slack * slack
            + 0.5 * eps_slack * cp.square(slack)
        )

        constraints = [
            grad @ self.safety_filter_dynamics.dynamics_for_filter(t, x, u)
            + self.cbf_rate * (value - self.h_margin) + slack >= 0,
            u >= self.u_min,
            u <= self.u_max,
        ]
        problem = cp.Problem(objective, constraints)
        status = _solve_qp(problem)
        self._last_qp_feasible = (status in _ACCEPTED_QP_STATUSES
                                  and u.value is not None)

        if self.logger is not None and self._current_ego_idx is not None:
            ego, other = self._current_ego_idx, self._current_other_idx
            slack_val = float(slack.value) if slack.value is not None else 0.0
            self.logger.log('pairwise_relaxed', f'V{self.logger.vid_label(ego)}',
                            [ego, other], status,
                            slack=f"{slack_val:.6f}")

        if not self._last_qp_feasible:
            _raise_qp_failure(
                "pairwise relaxed CBF-QP",
                [self._current_ego_idx, self._current_other_idx],
                status, self._sim_time(t), _ALWAYS_FEASIBLE_REASON)

        u_sol = np.asarray(u.value, dtype=np.float32)
        u_sol = np.clip(u_sol, self.u_min, self.u_max)
        return u_sol

class TwoDubinsCarRelativeSafetyFilterIndividualHandle(IndividualVehicleSafetyFilterHandle):
    def __init__(self, grid: HjNumpyGrid,
                 values: np.ndarray,
                 safety_filter_dynamics: TwoDubinsCarRelative,
                 u_ego_index: list = [0],
                 safety_filter_range: float = np.inf,
                 cbf_rate: float = 5.0,
                 h_margin: float = 0.0,
                 logger=None):
        super().__init__(grid, values, safety_filter_dynamics, u_ego_index, safety_filter_range, cbf_rate,
                         h_margin=h_margin, logger=logger)

    def construct_state_from_two_vehicle_state(self, state_ego, state_target):
        relative_distance = np.linalg.norm(state_target[:2] - state_ego[:2])
        relative_heading = _wrap_angle(state_target[2] - state_ego[2])
        relative_angle = np.arctan2(state_target[1] - state_ego[1], state_target[0] - state_ego[0])
        x_r = relative_distance * np.cos(relative_angle - state_ego[2])
        y_r = relative_distance * np.sin(relative_angle - state_ego[2])
        return np.array([x_r, y_r, relative_heading])

    def get_relative_distance(self, state_ego, state_target):
        return np.linalg.norm(state_target[:2] - state_ego[:2])

class TwoKinematicVehicleRelativeSafetyFilterIndividualHandle(IndividualVehicleSafetyFilterHandle):
    def __init__(self, grid: HjNumpyGrid,
                 values: np.ndarray,
                 safety_filter_dynamics: TwoKinematicVehicleRelative,
                 u_ego_index: list = [0, 2],
                 safety_filter_range: float = np.inf,
                 cbf_rate: float = 5.0,
                 h_margin: float = 0.0,
                 logger=None):
        super().__init__(grid, values, safety_filter_dynamics, u_ego_index, safety_filter_range, cbf_rate,
                         h_margin=h_margin, logger=logger)

    def construct_state_from_two_vehicle_state(self, state_ego, state_target):
        relative_distance = np.linalg.norm(state_target[:2] - state_ego[:2])
        relative_heading = _wrap_angle(state_target[2] - state_ego[2])
        relative_angle = np.arctan2(state_target[1] - state_ego[1], state_target[0] - state_ego[0])
        x_r = relative_distance * np.cos(relative_angle - state_ego[2])
        y_r = relative_distance * np.sin(relative_angle - state_ego[2])
        return np.array([x_r, y_r, relative_heading, state_ego[3], state_target[3]])

    def get_relative_distance(self, state_ego, state_target):
        return np.linalg.norm(state_target[:2] - state_ego[:2])

class MultiVehicleSafetyFilterHandle:
    def __init__(self, grid: HjNumpyGrid,
                 values: np.ndarray,
                 safety_filter_dynamics: AffineDynamicsWithBoxInputBounds,
                 safety_filter_range: float = np.inf,
                 cbf_rate: float = 5.0,
                 h_margin: float = 0.0,
                 logger=None):
        if isinstance(safety_filter_dynamics, TwoDubinsCarRelative):
            self.individual_handle = TwoDubinsCarRelativeSafetyFilterIndividualHandle(
                grid, values, safety_filter_dynamics,
                safety_filter_range=safety_filter_range, cbf_rate=cbf_rate,
                h_margin=h_margin, logger=logger)
        elif isinstance(safety_filter_dynamics, TwoKinematicVehicleRelative):
            self.individual_handle = TwoKinematicVehicleRelativeSafetyFilterIndividualHandle(grid, values, safety_filter_dynamics,
                                                                                              safety_filter_range=safety_filter_range, cbf_rate=cbf_rate,
                                                                                              h_margin=h_margin, logger=logger)
        else:
            raise ValueError(f"Safety filter dynamics type {type(safety_filter_dynamics)} not supported.")

    def apply_safety_filter_no_priority_fallback_relaxed(
        self, state_list: List, action_list: List,
        vehicle_order: Optional[List[int]] = None,
        lambda_slack: float = 10.0, eps_slack: float = 1e-3,
    ) -> List[np.ndarray]:
        """Hard symmetric CBF-QP per ego with relaxed fallback; resolved controls are not propagated between egos."""
        n = len(state_list)
        if vehicle_order is None:
            vehicle_order = list(range(n))

        u_sol_list: List[Optional[np.ndarray]] = [None] * n
        for i in vehicle_order:
            state_ego = state_list[i]
            action_ego = action_list[i]
            other_indices = [j for j in range(n) if j != i]
            state_list_other = [state_list[j] for j in other_indices]
            action_list_other = [action_list[j] for j in other_indices]

            self.individual_handle._last_qp_feasible = True
            u_sol = self.individual_handle.apply_safety_filter(
                state_ego, action_ego, state_list_other, action_list_other,
                ego_vehicle_idx=i, other_vehicle_indices=other_indices)

            if self.individual_handle._last_qp_feasible:
                u_sol_list[i] = u_sol
            else:
                u_sol = self.individual_handle.apply_safety_filter_relaxed(
                    state_ego, action_ego,
                    state_list_other, action_list_other,
                    ego_vehicle_idx=i, other_vehicle_indices=other_indices,
                    lambda_slack=lambda_slack, eps_slack=eps_slack)
                u_sol_list[i] = u_sol

        return u_sol_list


def validate_vehicle_clusters(clusters: List[List[int]], num_vehicles: int):
    """Validate that `clusters` form a partition of vehicle indices."""
    assert isinstance(clusters, list), "clusters should be a list of clusters."
    seen_indices = []
    for cluster_id, cluster in enumerate(clusters):
        assert isinstance(cluster, list), f"Cluster {cluster_id} should be a list of vehicle indices."
        assert len(cluster) > 0, f"Cluster {cluster_id} should be non-empty."
        for vehicle_index in cluster:
            assert isinstance(vehicle_index, (int, np.integer)), "Vehicle indices should be integers."
            assert 0 <= int(vehicle_index) < num_vehicles, (
                f"Vehicle index {vehicle_index} in cluster {cluster_id} is out of range for {num_vehicles} vehicles."
            )
            seen_indices.append(int(vehicle_index))
    assert len(seen_indices) == num_vehicles, "Clusters should contain every vehicle exactly once."
    assert len(set(seen_indices)) == num_vehicles, "Each vehicle should belong to exactly one cluster."
    assert set(seen_indices) == set(range(num_vehicles)), "Clusters should form a partition of all vehicles."


class ClusteredMultiVehicleSafetyFilterHandle:
    """Solve one centralized CBF-QP per cluster in a user-supplied vehicle partition."""

    def __init__(self, grid: HjNumpyGrid,
                 values: np.ndarray,
                 safety_filter_dynamics: AffineDynamicsWithBoxInputBounds,
                 safety_filter_range: float = np.inf,
                 cbf_rate: float = 5.0,
                 h_margin: float = 0.0,
                 lambda_slack: float = 10.0,
                 eps_slack: float = 1e-3,
                 logger=None):
        if isinstance(safety_filter_dynamics, TwoDubinsCarRelative):
            self.individual_handle = TwoDubinsCarRelativeSafetyFilterIndividualHandle(
                grid,
                values,
                safety_filter_dynamics,
                safety_filter_range=safety_filter_range,
                cbf_rate=cbf_rate,
                h_margin=h_margin,
            )
        elif isinstance(safety_filter_dynamics, TwoKinematicVehicleRelative):
            self.individual_handle = TwoKinematicVehicleRelativeSafetyFilterIndividualHandle(
                grid,
                values,
                safety_filter_dynamics,
                safety_filter_range=safety_filter_range,
                cbf_rate=cbf_rate,
                h_margin=h_margin,
                logger=logger,
            )
        else:
            raise ValueError(f"Safety filter dynamics type {type(safety_filter_dynamics)} not supported.")
        self.grid = grid
        self.values = values
        self.safety_filter_dynamics = safety_filter_dynamics
        self.cbf_rate = cbf_rate
        self.h_margin = float(h_margin)
        self.safety_filter_range = self.individual_handle.safety_filter_range
        self.lambda_slack = float(lambda_slack)
        self.eps_slack = float(eps_slack)
        self.logger = logger
        self._last_cluster_feasible = True

    def _collect_active_cluster_pairs(self, cluster_states, cluster_vehicle_indices):
        pairwise_data = []
        for local_i in range(len(cluster_vehicle_indices)):
            for local_j in range(local_i + 1, len(cluster_vehicle_indices)):
                state_i = cluster_states[local_i]
                state_j = cluster_states[local_j]
                relative_distance = self.individual_handle.get_relative_distance(state_i, state_j)
                if relative_distance > self.safety_filter_range:
                    continue

                state_relative = self.individual_handle.construct_state_from_two_vehicle_state(state_i, state_j)
                value, grad_value = self.grid.eval_value_and_deriv_from_table(
                    state_relative,
                    self.values,
                    return_deriv=True,
                )

                pairwise_data.append(ClusterPairwiseCBFData(
                    local_i=local_i,
                    local_j=local_j,
                    vehicle_i=cluster_vehicle_indices[local_i],
                    vehicle_j=cluster_vehicle_indices[local_j],
                    state_relative=np.asarray(state_relative, dtype=np.float32),
                    value=float(value),
                    grad_value=np.asarray(grad_value, dtype=np.float32).reshape(-1),
                ))
        return pairwise_data

    def _build_pairwise_cbf_lhs(self, cluster_control, pair_data: ClusterPairwiseCBFData, constraints):
        pair_u_dim = self.safety_filter_dynamics.u_dim
        ego_u_dim = len(self.individual_handle.u_ego_index)
        u_ego_idx = list(self.individual_handle.u_ego_index)
        u_other_idx = [k for k in range(pair_u_dim) if k not in u_ego_idx]

        pair_control = cp.Variable(pair_u_dim)
        for k, eidx in enumerate(u_ego_idx):
            constraints.append(pair_control[eidx] == cluster_control[pair_data.local_i * ego_u_dim + k])
        for k, oidx in enumerate(u_other_idx):
            constraints.append(pair_control[oidx] == cluster_control[pair_data.local_j * ego_u_dim + k])

        return (
            pair_data.grad_value
            @ self.safety_filter_dynamics.dynamics_for_filter(None, pair_data.state_relative, pair_control)
            + self.cbf_rate * (pair_data.value - self.h_margin)
        )

    def _build_cluster_control_bounds(self, cluster_control):
        ego_u_dim = len(self.individual_handle.u_ego_index)
        u_ego_idx = list(self.individual_handle.u_ego_index)
        u_min_ego = np.array([self.safety_filter_dynamics.u_min[k] for k in u_ego_idx], dtype=np.float32)
        u_max_ego = np.array([self.safety_filter_dynamics.u_max[k] for k in u_ego_idx], dtype=np.float32)
        num_vehicles = cluster_control.shape[0] // ego_u_dim
        return [
            cluster_control >= np.tile(u_min_ego, num_vehicles),
            cluster_control <= np.tile(u_max_ego, num_vehicles),
        ]


    def _solve_relaxed_cluster_cbf_qp(self, u_ref_cluster, pairwise_data):
        """Per-pair slack relaxation, symmetric (no priority)."""
        num_pairs = len(pairwise_data)
        cluster_control = cp.Variable(u_ref_cluster.shape[0])
        # Unbounded-above nonneg slack over a nonempty control box: always feasible.
        pair_slack = cp.Variable(num_pairs, nonneg=True) if num_pairs > 0 else None

        objective_expr = cp.sum_squares(cluster_control - u_ref_cluster)
        if num_pairs > 0:
            objective_expr += (
                self.lambda_slack * cp.sum(pair_slack)
                + 0.5 * self.eps_slack * cp.sum_squares(pair_slack)
            )

        constraints = self._build_cluster_control_bounds(cluster_control)
        for pair_idx, pair_data in enumerate(pairwise_data):
            cbf_lhs = self._build_pairwise_cbf_lhs(cluster_control, pair_data, constraints)
            constraints.append(cbf_lhs + pair_slack[pair_idx] >= 0)

        objective = cp.Minimize(objective_expr)

        problem = cp.Problem(objective, constraints)
        status = _solve_qp(problem)

        control_solution = None
        if cluster_control.value is not None:
            control_solution = np.asarray(cluster_control.value, dtype=np.float32).reshape(-1)

        slack_by_pair = None
        if pair_slack is not None and pair_slack.value is not None:
            slack_values = np.asarray(pair_slack.value, dtype=np.float32).reshape(-1)
            slack_by_pair = {
                (pair_data.vehicle_i, pair_data.vehicle_j): float(slack_values[pair_idx])
                for pair_idx, pair_data in enumerate(pairwise_data)
            }

        return ClusteredQPResult(
            status=status,
            control_solution=control_solution,
            slack_by_pair=slack_by_pair,
        )

    def _build_cluster_log_fields(self, slack_by_pair=None):
        log_fields = {}
        _label = self.logger.vid_label if self.logger else lambda v: v
        if slack_by_pair is not None:
            for (vi, vj), value in slack_by_pair.items():
                log_fields[f"slack_p{_label(vi)}_{_label(vj)}"] = f"{float(value):.6f}"
        return log_fields

    def _solve_cluster_cbf_qp(self, state_list: List, action_list: List, cluster_vehicle_indices: List[int],
                               cluster_idx: int = 0):
        if len(cluster_vehicle_indices) == 1:
            vehicle_index = cluster_vehicle_indices[0]
            self._last_cluster_feasible = True
            return [np.asarray(action_list[vehicle_index], dtype=np.float32).reshape(-1)]

        ego_u_dim = len(self.individual_handle.u_ego_index)
        cluster_states = [np.asarray(state_list[i], dtype=np.float32).reshape(-1) for i in cluster_vehicle_indices]
        cluster_actions = [np.asarray(action_list[i], dtype=np.float32).reshape(-1) for i in cluster_vehicle_indices]
        for action in cluster_actions:
            assert action.shape == (ego_u_dim,), (
                f"All vehicle actions should have shape ({ego_u_dim},), got {action.shape}"
            )

        u_ref_cluster = np.concatenate(cluster_actions, axis=0)
        pairwise_data = self._collect_active_cluster_pairs(
            cluster_states,
            cluster_vehicle_indices,
        )

        result = self._solve_relaxed_cluster_cbf_qp(u_ref_cluster, pairwise_data)

        if self.logger is not None:
            self.logger.log('cluster', f'C{cluster_idx}', cluster_vehicle_indices,
                            result.status, **self._build_cluster_log_fields(
                                slack_by_pair=result.slack_by_pair))

        self._last_cluster_feasible = (
            result.status in _ACCEPTED_QP_STATUSES
            and result.control_solution is not None
        )

        if not self._last_cluster_feasible:
            _raise_qp_failure(
                "centralized cluster CBF-QP (slack_pair)",
                cluster_vehicle_indices, result.status,
                _logger_time(self.logger), _ALWAYS_FEASIBLE_REASON)

        cluster_control_solution = result.control_solution
        u_ego_idx = list(self.individual_handle.u_ego_index)
        clip_min = np.tile(
            np.array([self.safety_filter_dynamics.u_min[k] for k in u_ego_idx], dtype=np.float32),
            len(cluster_vehicle_indices),
        )
        clip_max = np.tile(
            np.array([self.safety_filter_dynamics.u_max[k] for k in u_ego_idx], dtype=np.float32),
            len(cluster_vehicle_indices),
        )
        cluster_control_solution = np.clip(cluster_control_solution, clip_min, clip_max)
        return [
            cluster_control_solution[i * ego_u_dim:(i + 1) * ego_u_dim]
            for i in range(len(cluster_vehicle_indices))
        ]

    def apply_safety_filter_with_clusters(self,
                                          state_list: List,
                                          action_list: List,
                                          clusters: List[List[int]]):
        validate_vehicle_clusters(clusters, len(state_list))
        u_sol_list = [None] * len(state_list)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_solution = self._solve_cluster_cbf_qp(
                state_list, action_list, cluster, cluster_idx=cluster_idx)
            for local_index, vehicle_index in enumerate(cluster):
                u_sol_list[vehicle_index] = cluster_solution[local_index]
        return u_sol_list


class PairwisePriorityCBFStage(CBFSolveStage):
    """Sequential pairwise-priority CBF-QPs (decentralised), tracking per-vehicle feasibility."""

    def __init__(self, individual_handle: IndividualVehicleSafetyFilterHandle,
                 stage_name: str = "pairwise_priority"):
        self.individual_handle = individual_handle
        self.stage_name = stage_name

    def solve(self, state_list, action_list, vehicle_indices,
              priority_list=None, fixed_controls=None,
              vehicle_order=None):
        assert priority_list is not None, (
            "PairwisePriorityCBFStage requires a priority_list."
        )
        vehicle_indices = set(vehicle_indices)

        action_list_updated = list(action_list)
        if fixed_controls is not None:
            for v, ctrl in fixed_controls.items():
                action_list_updated[v] = ctrl

        if vehicle_order is not None:
            ordered = [v for v in vehicle_order if v in vehicle_indices]
        else:
            ordered = [v for v in priority_list if v in vehicle_indices]

        controls: Dict[int, np.ndarray] = {}
        feasible: Set[int] = set()
        infeasible: Set[int] = set()

        for i in ordered:
            state_ego = state_list[i]
            action_ego = action_list_updated[i]
            other_indices = [j for j in priority_list if j != i]
            state_list_other = [state_list[j] for j in other_indices]
            action_list_other = [action_list_updated[j] for j in other_indices]
            priority_than_ego_list_other = [
                (priority_list.index(j) < priority_list.index(i))
                for j in other_indices
            ]
            u_sol = self.individual_handle.apply_safety_filter_priority_stage(
                state_ego, action_ego,
                state_list_other, action_list_other,
                priority_than_ego_list_other,
                ego_vehicle_idx=i, other_vehicle_indices=other_indices,
            )
            controls[i] = np.asarray(u_sol, dtype=np.float32)

            if self.individual_handle._last_qp_feasible:
                feasible.add(i)
            else:
                infeasible.add(i)

            action_list_updated[i] = controls[i]

        return StageResult(
            vehicle_controls=controls,
            feasible_vehicles=feasible,
            infeasible_vehicles=infeasible,
        )


class PairwiseCBFStage(CBFSolveStage):
    """Pairwise CBF-QPs without priority; resolved controls propagate to later vehicles in the iteration order."""

    def __init__(self, individual_handle: IndividualVehicleSafetyFilterHandle,
                 stage_name: str = "pairwise"):
        self.individual_handle = individual_handle
        self.stage_name = stage_name

    def solve(self, state_list, action_list, vehicle_indices,
              priority_list=None, fixed_controls=None,
              vehicle_order=None):
        vehicle_indices = set(vehicle_indices)

        action_list_updated = list(action_list)
        if fixed_controls is not None:
            for v, ctrl in fixed_controls.items():
                action_list_updated[v] = ctrl

        if vehicle_order is not None:
            ordered = [v for v in vehicle_order if v in vehicle_indices]
        elif priority_list is not None:
            ordered = [v for v in priority_list if v in vehicle_indices]
        else:
            ordered = sorted(vehicle_indices)

        controls: Dict[int, np.ndarray] = {}
        feasible: Set[int] = set()
        infeasible: Set[int] = set()

        for i in ordered:
            state_ego = state_list[i]
            action_ego = action_list_updated[i]
            other_indices = [j for j in range(len(state_list)) if j != i]
            state_list_other = [state_list[j] for j in other_indices]
            action_list_other = [action_list_updated[j] for j in other_indices]

            self.individual_handle._last_qp_feasible = True
            u_sol = self.individual_handle.apply_safety_filter(
                state_ego, action_ego,
                state_list_other, action_list_other,
                ego_vehicle_idx=i, other_vehicle_indices=other_indices,
            )
            controls[i] = np.asarray(u_sol, dtype=np.float32)
            action_list_updated[i] = controls[i]

            if self.individual_handle._last_qp_feasible:
                feasible.add(i)
            else:
                infeasible.add(i)

        return StageResult(
            vehicle_controls=controls,
            feasible_vehicles=feasible,
            infeasible_vehicles=infeasible,
        )


class PairwiseRelaxedCBFStage(CBFSolveStage):
    """PairwiseCBFStage with slack-relaxed QPs, so every solve is feasible."""

    def __init__(self, individual_handle: IndividualVehicleSafetyFilterHandle,
                 lambda_slack: float = 10.0, eps_slack: float = 1e-3,
                 stage_name: str = "pairwise_relaxed"):
        self.individual_handle = individual_handle
        self.lambda_slack = lambda_slack
        self.eps_slack = eps_slack
        self.stage_name = stage_name

    def solve(self, state_list, action_list, vehicle_indices,
              priority_list=None, fixed_controls=None,
              vehicle_order=None):
        vehicle_indices = set(vehicle_indices)

        action_list_updated = list(action_list)
        if fixed_controls is not None:
            for v, ctrl in fixed_controls.items():
                action_list_updated[v] = ctrl

        if vehicle_order is not None:
            ordered = [v for v in vehicle_order if v in vehicle_indices]
        elif priority_list is not None:
            ordered = [v for v in priority_list if v in vehicle_indices]
        else:
            ordered = sorted(vehicle_indices)

        controls: Dict[int, np.ndarray] = {}
        feasible: Set[int] = set()
        infeasible: Set[int] = set()

        for i in ordered:
            state_ego = state_list[i]
            action_ego = action_list_updated[i]
            other_indices = [j for j in range(len(state_list)) if j != i]
            state_list_other = [state_list[j] for j in other_indices]
            action_list_other = [action_list_updated[j] for j in other_indices]

            self.individual_handle._last_qp_feasible = True
            u_sol = self.individual_handle.apply_safety_filter_relaxed(
                state_ego, action_ego,
                state_list_other, action_list_other,
                ego_vehicle_idx=i, other_vehicle_indices=other_indices,
                lambda_slack=self.lambda_slack, eps_slack=self.eps_slack,
            )
            controls[i] = np.asarray(u_sol, dtype=np.float32)
            action_list_updated[i] = controls[i]

            if self.individual_handle._last_qp_feasible:
                feasible.add(i)
            else:
                infeasible.add(i)

        return StageResult(
            vehicle_controls=controls,
            feasible_vehicles=feasible,
            infeasible_vehicles=infeasible,
        )


class MultiStageCBFFilter:
    """Ordered pipeline of CBF stages; resolved controls are passed forward as fixed_controls."""

    def __init__(self, stages: List[CBFSolveStage], logger=None):
        self.stages = list(stages)
        self.logger = logger

    def apply(self, state_list: List, action_list: List,
              priority_list: Optional[List[int]] = None,
              vehicle_order: Optional[List[int]] = None) -> List[np.ndarray]:
        """Run the stages and return one control per vehicle in index order; vehicle_order overrides within-stage ordering."""
        num_vehicles = len(state_list)
        resolved: Dict[int, np.ndarray] = {}
        remaining: Set[int] = set(range(num_vehicles))

        if self.logger is not None and vehicle_order is not None:
            order_str = ','.join(
                str(self.logger.vid_label(v)) for v in vehicle_order)
            self.logger.log(
                'multi_stage_order', 'VO',
                vehicle_order, 'info',
                vehicle_order_raw=order_str,
            )

        for stage_idx, stage in enumerate(self.stages):
            if not remaining:
                break

            fixed_controls = dict(resolved)

            result = stage.solve(
                state_list, action_list, remaining,
                priority_list=priority_list,
                fixed_controls=fixed_controls,
                vehicle_order=vehicle_order,
            )

            for v in result.feasible_vehicles:
                resolved[v] = result.vehicle_controls[v]
            remaining -= result.feasible_vehicles

            if self.logger is not None:
                stage_name = getattr(stage, 'stage_name', f'stage_{stage_idx}')
                self.logger.log(
                    'multi_stage', f'S{stage_idx}',
                    sorted(result.feasible_vehicles | result.infeasible_vehicles),
                    'partial' if result.infeasible_vehicles else 'ok',
                    stage_index=stage_idx,
                    stage_name=stage_name,
                    num_feasible=len(result.feasible_vehicles),
                    num_infeasible=len(result.infeasible_vehicles),
                )

        for v in remaining:
            resolved[v] = np.asarray(action_list[v], dtype=np.float32)

        return [resolved[i] for i in range(num_vehicles)]


STAGE_REGISTRY: Dict[str, type] = {
    "decentralized_nearest_joint_priority_hard": PairwisePriorityCBFStage,
    "decentralized_nearest_joint_hard": PairwiseCBFStage,
    "decentralized_nearest_joint_slack": PairwiseRelaxedCBFStage,
}

DEFAULT_HMS_STAGE_NAMES: List[str] = [
    "decentralized_nearest_joint_priority_hard",
    "decentralized_nearest_joint_hard",
    "decentralized_nearest_joint_slack",
]


def build_stages_from_names(
    stage_names: List[str],
    individual_handle: IndividualVehicleSafetyFilterHandle,
) -> List[CBFSolveStage]:
    """Instantiate CBFSolveStage objects from registry key names."""
    stages = []
    for name in stage_names:
        cls = STAGE_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown stage type '{name}'. "
                f"Valid types: {sorted(STAGE_REGISTRY)}"
            )
        stages.append(cls(individual_handle=individual_handle, stage_name=name))
    return stages


# Mode names read <decomposition>_<coverage>_<peer treatment>[_priority]_<relaxation>.
SAFETY_MODES = {
    "none",
    "decentralized_nearest_joint_hard_then_slack",
    "centralized_all_slack_pair",
    "hybrid_multi_stage",
}

DEFAULT_SAFETY_MODE = "decentralized_nearest_joint_hard_then_slack"

MODES_NEEDING_PRIORITY = {
    "hybrid_multi_stage",
}

CENTRALIZED_MODES = {
    "centralized_all_slack_pair",
}

CLUSTER_HANDLE_MODES = set(CENTRALIZED_MODES)
