import numpy as np
from functools import lru_cache
from control import lqr
import matplotlib.pyplot as plt
from hj_reachability_torch.grid import HjNumpyGrid
from lib.dubins_car import MultipleVehicle, DubinsCar, KinematicVehicle

# The Riccati solve dominates each tracking-LQR call, so gains are memoized on the exact speed.
@lru_cache(maxsize=8192)
def _lqr_gain_dubins(speed: float) -> np.ndarray:
    A = np.array([[0.0, speed], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = 10 * np.eye(1)
    K, _, _ = lqr(A, B, Q, R)
    K.setflags(write=False)
    return K


@lru_cache(maxsize=8192)
def _lqr_gain_kinematic(speed: float) -> np.ndarray:
    A = np.array([[0.0, speed, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = np.diag([1.0, 1.0, 0.01]) * 10
    R = 1e-2 * np.eye(2)
    K, _, _ = lqr(A, B, Q, R)
    K.setflags(write=False)
    return K

class CorridorHandle():
    def __init__(self,
        half_width: float = 0.5,
        entrance_position = np.array([0.0, 0.0]),
        heading: float = 0.0):
        self.half_width = half_width
        self.entrance_position = entrance_position
        self.heading = heading

    def get_progress(self, car_state):
        return np.array([np.cos(self.heading), np.sin(self.heading)]) @ (car_state[:2] - self.entrance_position)

    def get_lateral_offset(self, car_state):
        return np.cross(np.array([np.cos(self.heading), np.sin(self.heading)]), car_state[:2] - self.entrance_position)

    def check_in_corridor(self, car_state, tolerance: float = 0.0):
        progress = self.get_progress(car_state)
        lateral_offset = self.get_lateral_offset(car_state)
        return (np.abs(lateral_offset) <= self.half_width + tolerance
                and progress >= -tolerance)

    def visualize_corridor(self, ax):
        """Draw the entrance edge and the two lane boundaries out to the axis limits."""
        if ax is None:
            ax = plt.gca()

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        heading = self.heading
        center = np.asarray(self.entrance_position)
        dir_perp = np.array([-np.sin(heading), np.cos(heading)])

        entrance_left_right = [center + self.half_width * dir_perp,
                                center - self.half_width * dir_perp]
        lines_to_plot = []
        lines_to_plot.append(entrance_left_right)

        for entrance_point in entrance_left_right:
            intersections = []
            box_edges = [
                ([x_min, y_min], [x_min, y_max]),
                ([x_min, y_max], [x_max, y_max]),
                ([x_max, y_max], [x_max, y_min]),
                ([x_max, y_min], [x_min, y_min])
            ]

            for (p0, p1) in box_edges:
                edge_vec = np.array(p1) - np.array(p0)
                entrance_pt = entrance_point
                a = -np.sin(heading) * edge_vec[0] + np.cos(heading) * edge_vec[1]
                b = -np.sin(heading) * (p0[0] - entrance_pt[0]) + np.cos(heading) * (p0[1] - entrance_pt[1])
                if np.isclose(a, 0):
                    continue
                t = -b / a
                if 0 <= t <= 1:
                    intersection = np.array(p0) + t * edge_vec
                    if (x_min <= intersection[0] <= x_max and
                        y_min <= intersection[1] <= y_max) and self.get_progress(intersection) >= 0:
                        intersections.append(intersection)
            for intersection in intersections:
                lines_to_plot.append([entrance_point, intersection])
        for line in lines_to_plot:
            ax.plot(
                [line[0][0], line[1][0]],
                [line[0][1], line[1][1]],
                color='k',
                linestyle='-',
                linewidth=1,
                alpha=0.7
            )
        return ax

def get_state_list_outside_corridor(state_stacked, corridor_handle, x_dim: int):
    num_vehicles = state_stacked.shape[0] // x_dim
    state_list = []
    index_list = []
    for i in range(num_vehicles):
        car_state = state_stacked[x_dim*i:x_dim*(i+1), ...]
        if not corridor_handle.check_in_corridor(car_state):
            state_list.append(car_state)
            index_list.append(i)
    return state_list, index_list

def convert_state_stacked_to_list(state_stacked: np.ndarray, x_dim: int) -> list:
    """[x1, y1, theta1, x2, y2, theta2] -> [[x1, y1, theta1], [x2, y2, theta2]]"""
    num_vehicles = state_stacked.shape[0] // x_dim
    state_list = []
    for i in range(num_vehicles):
        car_state = state_stacked[x_dim*i:x_dim*(i+1), ...]
        state_list.append(car_state)
    return state_list

def convert_action_stacked_to_list(action_stacked: np.ndarray, u_dim: int) -> list:
    """[u1, u2] -> [[u1], [u2]]"""
    num_vehicles = action_stacked.shape[0] // u_dim
    action_list = []
    for i in range(num_vehicles):
        car_action = action_stacked[u_dim*i:u_dim*(i+1), ...]
        action_list.append(car_action)
    return action_list

def convert_action_list_to_stacked(action_list: list) -> np.ndarray:
    """[[u1], [u2]] -> [u1, u2]"""
    action_stacked = np.concatenate(action_list, axis=0, dtype=np.float32)
    return action_stacked

def event_function_all_vehicle_reach_target_corridor_based(t, state, corridor_handle, vehicle_x_dim):
    """Rollout termination event: negative once every vehicle is inside the corridor."""
    target_state_max = -np.inf
    num_vehicles = state.shape[0] // vehicle_x_dim
    for i in range(num_vehicles):
        car_state = state[vehicle_x_dim*i:vehicle_x_dim*(i+1), ...]
        target_state_i = -1.0 if corridor_handle.check_in_corridor(car_state) else 1.0
        if target_state_i > target_state_max:
            target_state_max = target_state_i
    return target_state_max

def linear_tracking_controller_dubins(t, relative_state, speed, target_offset=0.0):
    """relative_state = [lateral offset, heading error]"""
    K = _lqr_gain_dubins(float(speed))
    control = -K @ (relative_state.reshape(-1, 1) - np.array([[target_offset], [0]]))
    return control.astype(np.float32)

def linear_tracking_controller_kinematic(t, relative_state, target_speed, target_offset=0.0):
    """relative_state = [lateral offset, heading error, speed]"""
    K = _lqr_gain_kinematic(float(target_speed))
    control = -K @ (relative_state.reshape(-1, 1) - np.array([[target_offset], [0], [target_speed]]))
    return control.astype(np.float32).squeeze(-1)

def linear_tracking_controller(t, car_state, single_vehicle_dynamics, corridor_handle,
                               target_offset=0.0, target_speed=None):
    lateral_offset = corridor_handle.get_lateral_offset(car_state)
    heading_error = (car_state[2] - corridor_handle.heading + np.pi) % (2 * np.pi) - np.pi
    if isinstance(single_vehicle_dynamics, DubinsCar):
        relative_state = np.array([lateral_offset, heading_error])
        speed = single_vehicle_dynamics.speed
        u_opt_unbounded = linear_tracking_controller_dubins(t, relative_state, speed, target_offset)
    elif isinstance(single_vehicle_dynamics, KinematicVehicle):
        relative_state = np.array([lateral_offset, heading_error, car_state[3]])
        if target_speed is None:
            target_speed = single_vehicle_dynamics.speed_target
        u_opt_unbounded = linear_tracking_controller_kinematic(t, relative_state, target_speed, target_offset)
    else:
        raise ValueError(f"Invalid dynamics type: {type(single_vehicle_dynamics)}")
    u_opt = np.clip(u_opt_unbounded, single_vehicle_dynamics.u_min, single_vehicle_dynamics.u_max)
    return u_opt.astype(np.float32).flatten()

def switching_control_multi_vehicle_static(time, state,
                              dynamics: MultipleVehicle,
                              corridor_handle: CorridorHandle,
                              grid: HjNumpyGrid,
                              values: np.ndarray,
                              target_function_corridor: np.ndarray,
                              control_mode: str):
    assert control_mode in ['min', 'max'], "control_mode should be either 'min' or 'max'"

    u_opt_list = []
    vehicle_x_dim = dynamics.vehicle_x_dim
    for i in range(dynamics.num_vehicles):
        car_state = state[vehicle_x_dim*i:vehicle_x_dim*(i+1), ...]
        car_in_corridor = corridor_handle.check_in_corridor(car_state)
        if car_in_corridor:
            if dynamics.target_offset[i] is None:
                dynamics.target_offset[i] = 0.0
            car_state[2] = (car_state[2] + np.pi) % (2 * np.pi) - np.pi
            u_opt_i = linear_tracking_controller(time, car_state, dynamics.single_vehicle, corridor_handle, dynamics.target_offset[i])
            u_opt_list.append(u_opt_i)
        else:
            value, grad_value = grid.eval_value_and_deriv_from_table(car_state, values, return_deriv=True)
            u_opt_i = dynamics.single_vehicle.optimal_control(time, car_state, grad_value, mode=control_mode).squeeze(-1)
            u_opt_list.append(u_opt_i)

    u_opt = np.asarray(u_opt_list).flatten().astype(np.float32)
    extras = {}
    extras['priority'] = None
    return u_opt, extras

# Inset so the state recorded at the escape step is still on the table.
GRID_ESCAPE_MARGIN = 0.05


def grid_escape_bounds(grid, margin: float = GRID_ESCAPE_MARGIN):
    lo = np.asarray(grid.domain_lo).reshape(-1)[:2].astype(float) + margin
    hi = np.asarray(grid.domain_hi).reshape(-1)[:2].astype(float) - margin
    return lo, hi


def is_outside_grid(position, lo, hi) -> bool:
    p = np.asarray(position, dtype=float).reshape(-1)[:2]
    return bool(np.any(p < lo) or np.any(p > hi))


_TTR_GRID_CAP_CACHE = []


def ttr_grid_cap_for_table(values: np.ndarray) -> float:
    """Memoized `compute_ttr_grid_cap`, keyed on table identity with a strong reference."""
    for table, cap in _TTR_GRID_CAP_CACHE:
        if table is values:
            return cap
    from lib.simulation_runner import compute_ttr_grid_cap
    cap = compute_ttr_grid_cap(values)
    _TTR_GRID_CAP_CACHE.append((values, cap))
    return cap


def ttr_cap(ttr_min: float, rank: int, ttr_separation: float,
            values: np.ndarray) -> float:
    """``min(ttr_min + rank * sep, grid cap)``; the cap also absorbs a non-finite ``ttr_min``."""
    return min(ttr_min + rank * ttr_separation, ttr_grid_cap_for_table(values))


def evaluate_target_separated_ttr_sequence_static(
                                ttr_separation: float,
                                state_stacked,
                                priority_list,
                                dynamics: MultipleVehicle,
                              corridor_handle: CorridorHandle,
                              grid: HjNumpyGrid,
                              values: np.ndarray,
                              ttr_list=None):
    """Per-vehicle TTR targets; ``ttr_list`` reuses a caller's table query (out-of-corridor entries only)."""
    x_dim = dynamics.vehicle_x_dim
    num_vehicles = dynamics.num_vehicles
    state_outside_list, index_outside_list = get_state_list_outside_corridor(state_stacked, corridor_handle, x_dim)
    if len(state_outside_list) == 0:
        return [0.0] * num_vehicles
    if ttr_list is None:
        state_in_list = convert_state_stacked_to_list(state_stacked, x_dim)
        ttr_list = get_ttr_per_vehicle(state_in_list, grid, values)
    priority_outside_list = []
    for i in range(num_vehicles):
        if priority_list[i] in index_outside_list:
            priority_outside_list.append(priority_list[i])
    ttr_min = ttr_list[priority_outside_list[0]]
    ttr_sequence = [ttr_min] * num_vehicles
    for i in range(num_vehicles):
        if i in index_outside_list:
            index_i = priority_outside_list.index(i)
            ttr_sequence[i] = ttr_cap(
                ttr_min, index_i, ttr_separation, values)
        else:
            ttr_sequence[i] = 0.0
    return ttr_sequence

def bang_bang_ttr_action(time, car_state, single_vehicle, value, grad_value,
                         ttr_target: float):
    """Race (`min`) while TTR is above the target, delay (`max`) once below it."""
    mode = 'max' if value < ttr_target else 'min'
    return single_vehicle.optimal_control(
        time, car_state, grad_value, mode=mode).squeeze(-1)


def switching_control_multi_vehicle_static_with_ttr_separation(time, state,
                              ttr_separation: float,
                              dynamics: MultipleVehicle,
                              corridor_handle: CorridorHandle,
                              grid: HjNumpyGrid,
                              values_ttr: np.ndarray,
                              priority_list=None):
    u_opt_list = []
    vehicle_x_dim = dynamics.vehicle_x_dim
    # One TTR query per free vehicle serves both the target ladder and the bang-bang switch.
    ttr_query = [None] * dynamics.num_vehicles
    for i in range(dynamics.num_vehicles):
        car_state = state[vehicle_x_dim*i:vehicle_x_dim*(i+1), ...]
        if not corridor_handle.check_in_corridor(car_state):
            ttr_query[i] = grid.eval_value_and_deriv_from_table(
                car_state, values_ttr, return_deriv=True)
    ttr_target_sequence = evaluate_target_separated_ttr_sequence_static(
        ttr_separation,
        state,
        priority_list,
        dynamics,
        corridor_handle,
        grid, values_ttr,
        ttr_list=[np.inf if q is None else q[0] for q in ttr_query])
    for i in range(dynamics.num_vehicles):
        car_state = state[vehicle_x_dim*i:vehicle_x_dim*(i+1), ...]
        if ttr_query[i] is None:
            if dynamics.target_offset[i] is None:
                dynamics.target_offset[i] = 0.0
            car_state[2] = (car_state[2] + np.pi) % (2 * np.pi) - np.pi
            u_opt_i = linear_tracking_controller(time, car_state, dynamics.single_vehicle, corridor_handle, dynamics.target_offset[i])
            u_opt_list.append(u_opt_i)
        else:
            value, grad_value = ttr_query[i]
            u_opt_i = bang_bang_ttr_action(
                time, car_state, dynamics.single_vehicle, value, grad_value,
                ttr_target_sequence[i])
            u_opt_list.append(u_opt_i)
    u_opt = np.asarray(u_opt_list).flatten().astype(np.float32)
    extras = {}
    extras['priority'] = None
    extras['ttr_target'] = ttr_target_sequence
    return u_opt, extras

def evaluate_ttr_priority(state_in_list, corridor_handle, grid, ttr_values, target_function_corridor):
    """Priority order (index 0 = highest): in-corridor by progress first, then approaching by TTR."""
    in_corridor = []
    approaching = []
    for i, car_state in enumerate(state_in_list):
        car_state = np.asarray(car_state).flatten()
        if corridor_handle.check_in_corridor(car_state):
            dist = np.sqrt(car_state[0] ** 2 + car_state[1] ** 2)
            in_corridor.append((i, float(dist)))
        else:
            ttr = grid.eval_value_and_deriv_from_table(car_state, ttr_values, return_deriv=False)
            ttr = float(ttr) if np.isscalar(ttr) else float(np.asarray(ttr).flat[0])
            approaching.append((i, ttr))
    in_corridor.sort(key=lambda x: -x[1])
    approaching.sort(key=lambda x: x[1])
    priority_list = [idx for idx, _ in in_corridor] + [idx for idx, _ in approaching]
    return priority_list

def get_ttr_per_vehicle(state_in_list, grid, ttr_values):
    ttrs = []
    for car_state in state_in_list:
        car_state = np.asarray(car_state).flatten()
        ttr = grid.eval_value_and_deriv_from_table(car_state, ttr_values, return_deriv=False)
        ttr = float(ttr) if np.isscalar(ttr) else float(np.asarray(ttr).flat[0])
        ttrs.append(ttr)
    return ttrs


class MissionCompleteTracker:
    """Complete = traversed ``min_corridor_traverse_dist`` in the corridor and progress >= ``corridor_final_threshold``; escaped = left the TTR grid in free space (frozen, still an obstacle)."""

    def __init__(self, corridor_handle: CorridorHandle, vehicle_x_dim: int,
                 num_vehicles: int, grid,
                 min_corridor_traverse_dist: float = 3.5,
                 corridor_final_threshold: float = 4.0,
                 track_completion: bool = True):
        self.corridor_handle = corridor_handle
        self.vehicle_x_dim = vehicle_x_dim
        self.num_vehicles = num_vehicles
        self.min_traverse_dist = min_corridor_traverse_dist
        self.final_threshold = corridor_final_threshold
        self.track_completion = track_completion
        self.grid_xy_lo, self.grid_xy_hi = grid_escape_bounds(grid)

        self.reset()

    def reset(self):
        n = self.num_vehicles
        self.traverse_dist = [0.0] * n
        self.completed = [False] * n
        self.escaped = [False] * n
        self.frozen_states: list[np.ndarray | None] = [None] * n
        self._prev_positions: list[np.ndarray | None] = [None] * n

    @property
    def frozen_mask(self):
        return [self.completed[i] or self.escaped[i]
                for i in range(self.num_vehicles)]

    def update(self, x_stacked: np.ndarray):
        for i in range(self.num_vehicles):
            if self.completed[i] or self.escaped[i]:
                continue
            state = x_stacked[self.vehicle_x_dim * i:
                              self.vehicle_x_dim * (i + 1)]
            pos = state[:2].copy()
            in_corridor = self.corridor_handle.check_in_corridor(state)

            if not in_corridor and is_outside_grid(
                    pos, self.grid_xy_lo, self.grid_xy_hi):
                self.escaped[i] = True
                self.frozen_states[i] = state.copy()
                continue

            if in_corridor and self._prev_positions[i] is not None:
                self.traverse_dist[i] += float(
                    np.linalg.norm(pos - self._prev_positions[i]))

            self._prev_positions[i] = pos

            if not self.track_completion:
                continue
            progress = self.corridor_handle.get_progress(state)
            if (progress >= self.final_threshold
                    and self.traverse_dist[i] >= self.min_traverse_dist):
                self.completed[i] = True
                self.frozen_states[i] = state.copy()


class CorridorStatusTracker:
    """Hysteretic ``within_corridor``: set after ``enter_delay`` consecutive inside steps, held for ``exit_delay`` steps."""

    def __init__(self, corridor_handle: CorridorHandle, vehicle_x_dim: int,
                 num_vehicles: int, enter_delay: int = 3,
                 exit_delay: int = 15):
        self.corridor_handle = corridor_handle
        self.vehicle_x_dim = vehicle_x_dim
        self.num_vehicles = num_vehicles
        self.enter_delay = enter_delay
        self.exit_delay = exit_delay

        self.within_corridor: list[bool] = [False] * num_vehicles
        self._consecutive_inside: list[int] = [0] * num_vehicles
        self._remaining_lock: list[int] = [0] * num_vehicles

    def update(self, x_stacked: np.ndarray):
        for i in range(self.num_vehicles):
            state = x_stacked[self.vehicle_x_dim * i:
                              self.vehicle_x_dim * (i + 1)]
            inside = self.corridor_handle.check_in_corridor(state)

            if self.within_corridor[i]:
                if self._remaining_lock[i] > 0:
                    self._remaining_lock[i] -= 1
                if not inside and self._remaining_lock[i] <= 0:
                    self.within_corridor[i] = False
                    self._consecutive_inside[i] = 0
            else:
                if inside:
                    self._consecutive_inside[i] += 1
                    if self._consecutive_inside[i] >= self.enter_delay:
                        self.within_corridor[i] = True
                        self._remaining_lock[i] = self.exit_delay
                else:
                    self._consecutive_inside[i] = 0


class MissionAwareDynamics:
    """MultipleVehicle wrapper that zeros drift and control Jacobian for vehicles in the tracker's ``frozen_mask``."""

    def __init__(self, base_dynamics: MultipleVehicle,
                 tracker: MissionCompleteTracker):
        self._base = base_dynamics
        self._tracker = tracker

    def __getattr__(self, name):
        return getattr(self._base, name)

    # Dunder lookup bypasses __getattr__, so __call__ is forwarded explicitly.
    def __call__(self, time, state, control, disturbance=None):
        result = self._base(time, state, control, disturbance)
        xd = self._base.vehicle_x_dim
        for i, frozen in enumerate(self._tracker.frozen_mask):
            if frozen:
                result[xd * i: xd * (i + 1)] = 0
        return result

    def drift(self, time, state):
        dx = self._base.drift(time, state)
        xd = self._base.vehicle_x_dim
        for i, frozen in enumerate(self._tracker.frozen_mask):
            if frozen:
                dx[xd * i: xd * (i + 1)] = 0
        return dx

    def control_jacobian(self, time, state):
        g = self._base.control_jacobian(time, state)
        xd = self._base.vehicle_x_dim
        ud = self._base.vehicle_u_dim
        for i, frozen in enumerate(self._tracker.frozen_mask):
            if frozen:
                g[xd * i: xd * (i + 1), ud * i: ud * (i + 1)] = 0
        return g
