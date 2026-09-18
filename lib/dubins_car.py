import numpy as np
import torch
import cvxpy as cp

from hj_reachability_torch.dynamics import Dynamics, AffineDynamicsWithBoxInputBounds
from hj_reachability_torch.utils import to_numpy

# Must match the cruise setpoint the corridor TTR table was solved against.
DEFAULT_SPEED_TARGET = 0.07


class DubinsCar(AffineDynamicsWithBoxInputBounds):
    def __init__(self, speed: float = 1.0, angular_rate_max: float = 1.0):
        self.speed = speed
        self.angular_rate_max = angular_rate_max
        super().__init__(x_dim=3, u_min=[-self.angular_rate_max], u_max=[self.angular_rate_max])

    def drift(self, time, state):
        dx = torch.zeros_like(state)
        dx[0, ...] = self.speed * torch.cos(state[2, ...])
        dx[1, ...] = self.speed * torch.sin(state[2, ...])
        dx[2, ...] = 0.0
        return dx

    def control_jacobian(self, time, state):
        g = torch.zeros(3, 1, dtype=torch.float32)
        g[2] = 1.0
        return g



class TwoDubinsCarRelative(AffineDynamicsWithBoxInputBounds):
    def __init__(self, speed: float = 1.0, angular_rate_max: float = 1.0):
        self.speed = speed
        self.angular_rate_max = angular_rate_max
        super().__init__(x_dim=3, u_min=[-self.angular_rate_max, -self.angular_rate_max], u_max=[self.angular_rate_max, self.angular_rate_max])

    def drift(self, time, state):
        # state: [relative x, relative y, relative theta]
        fx = torch.zeros_like(state)
        fx[0, ...] = - self.speed + self.speed* torch.cos(state[2, ...])
        fx[1, ...] = self.speed * torch.sin(state[2, ...])
        fx[2, ...] = 0.0
        return fx

    def control_jacobian(self, time, state):
        gx = torch.zeros(3, 2, state.shape[1], dtype=torch.float32)
        gx[0, 0, ...] = state[1, ...]
        gx[1, 0, ...] = -state[0, ...]
        gx[2, 0, ...] = -1.0
        gx[2, 1, ...] = 1.0
        return gx

    def dynamics_for_filter(self, time, state, action):
        if isinstance(action, cp.Variable):
            return cp.hstack([
                -self.speed + self.speed * np.cos(state[2]) + state[1] * action[0],
                self.speed * np.sin(state[2]) - state[0] * action[0],
                -action[0] + action[1]
            ])
        dx = np.zeros_like(state)
        action_flat = np.asarray(action).flatten()
        dx[0] = - self.speed + self.speed * np.cos(state[2]) + state[1] * action_flat[0]
        dx[1] = self.speed * np.sin(state[2]) - state[0] * action_flat[0]
        dx[2] = -action_flat[0] + action_flat[1]
        return dx

class TwoDubinsCarStacked(AffineDynamicsWithBoxInputBounds):
    """Two Dubins cars stacked into one 6-dimensional state."""
    def __init__(self, speed: float = 1.0, angular_rate_max: float = 1.0):
        self.speed = speed
        self.angular_rate_max = angular_rate_max
        super().__init__(x_dim=6, u_min=[-self.angular_rate_max, -self.angular_rate_max], u_max=[self.angular_rate_max, self.angular_rate_max])

    def drift(self, time, state):
        fx = torch.zeros_like(state)
        fx[0, ...] = self.speed * torch.cos(state[2, ...])
        fx[1, ...] = self.speed * torch.sin(state[2, ...])
        fx[2, ...] = 0.0
        fx[3, ...] = self.speed * torch.cos(state[5, ...])
        fx[4, ...] = self.speed * torch.sin(state[5, ...])
        fx[5, ...] = 0.0
        return fx

    def control_jacobian(self, time, state):
        g = torch.zeros(6, 2, dtype=torch.float32)
        g[2, 0] = 1.0
        g[5, 1] = 1.0
        return g

class TwoKinematicVehicleRelative(AffineDynamicsWithBoxInputBounds):
    def __init__(self, v_min = 0.5, v_max = 1.5, accel_min=-1, accel_max=1, angular_rate_max=1.0):
        self.speed_min  = v_min
        self.speed_max  = v_max
        self.accel_min  = accel_min
        self.accel_max  = accel_max
        self.angular_rate_max = angular_rate_max

        self.u_min_va_min = [-self.angular_rate_max, -self.angular_rate_max, 0, self.accel_min]
        self.u_max_va_min = [self.angular_rate_max, self.angular_rate_max, self.accel_max, self.accel_max]

        self.u_min_va_max = [-self.angular_rate_max, -self.angular_rate_max, self.accel_min, self.accel_min]
        self.u_max_va_max = [self.angular_rate_max, self.angular_rate_max, 0, self.accel_max]

        self.u_min_vb_min = [-self.angular_rate_max, -self.angular_rate_max, self.accel_min, 0]
        self.u_max_vb_min = [self.angular_rate_max, self.angular_rate_max, self.accel_max, self.accel_max]

        self.u_min_vb_max = [-self.angular_rate_max, -self.angular_rate_max, self.accel_min, self.accel_min]
        self.u_max_vb_max = [self.angular_rate_max, self.angular_rate_max, self.accel_max, 0]

        # control: angular_rate_a, angular_rate_b, acceleration_a, acceleration_b
        super().__init__(x_dim=5,
                         u_min=[-self.angular_rate_max, -self.angular_rate_max, self.accel_min, self.accel_min],
                         u_max=[self.angular_rate_max, self.angular_rate_max, self.accel_max, self.accel_max])

    def drift(self, time, state):
        # state: [relative x, relative y, relative theta, speed a, speed b]
        theta = state[2, ...]
        va = state[3, ...]
        vb = state[4, ...]
        fx = torch.zeros_like(state)
        fx[0, ...] = - va + vb * torch.cos(theta)
        fx[1, ...] = vb * torch.sin(theta)
        return fx

    def control_jacobian(self, time, state):
        gx = torch.zeros(5, 4, state.shape[1], dtype=torch.float32)
        gx[0, 0, ...] = state[1, ...]
        gx[1, 0, ...] = -state[0, ...]
        gx[2, 0, ...] = -1.0
        gx[2, 1, ...] = 1.0
        gx[3, 2, ...] = 1.0
        gx[4, 3, ...] = 1.0
        return gx

    def _eval_u_opt_with_specific_bound(self, LgV, u_min, u_max, mode):
        assert mode in ["min", "max"]
        u_opt = torch.zeros(self.u_dim, LgV.shape[1])
        for i in range(self.u_dim):
            if mode == "max":
                u_opt[i, LgV[i, ...] >= 0] = u_max[i]
                u_opt[i, LgV[i, ...] < 0] = u_min[i]
            else:
                u_opt[i, LgV[i, ...] >= 0] = u_min[i]
                u_opt[i, LgV[i, ...] < 0] = u_max[i]
        return u_opt

    def optimal_control(self, time, state, gradient, mode):
        assert mode in ["min", "max"]
        convert_to_np = False
        if type(state) == np.ndarray:
            state = torch.tensor(state)
            gradient = torch.tensor(gradient)
            convert_to_np = True
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        assert state.ndim == 2
        assert state.shape[0] == self.x_dim
        if gradient.ndim == 1 and gradient.shape[0] == self.x_dim:
            gradient = gradient.unsqueeze(-1)
        assert gradient.ndim == 2
        assert gradient.shape[0] == self.x_dim
        assert gradient.shape[1] == state.shape[1]

        gx = self.control_jacobian(time, state)
        if gx.ndim == 2:
            if gx.shape[0] != self.x_dim or gx.shape[1] != self.u_dim:
                raise ValueError(f"Invalid control Jacobian shape: {gx.shape}")
            gx = gx.unsqueeze(2).repeat(1, 1, state.shape[1])
        assert gx.ndim == 3
        assert gx.shape[0] == self.x_dim
        assert gx.shape[1] == self.u_dim
        assert gx.shape[2] == state.shape[1]

        LgV = torch.sum(gx * gradient.unsqueeze(1), dim=0)
        assert LgV.ndim == 2
        u_opt_interior = self._eval_u_opt_with_specific_bound(LgV, self.u_min, self.u_max, mode)
        u_opt_va_min = self._eval_u_opt_with_specific_bound(LgV, self.u_min_va_min, self.u_max_va_min, mode)
        u_opt_va_max = self._eval_u_opt_with_specific_bound(LgV, self.u_min_va_max, self.u_max_va_max, mode)
        u_opt_vb_min = self._eval_u_opt_with_specific_bound(LgV, self.u_min_vb_min, self.u_max_vb_min, mode)
        u_opt_vb_max = self._eval_u_opt_with_specific_bound(LgV, self.u_min_vb_max, self.u_max_vb_max, mode)
        u_opt = torch.where((state[3, ...] <= self.speed_min), u_opt_va_min, u_opt_interior)
        u_opt = torch.where((state[3, ...] >= self.speed_max), u_opt_va_max, u_opt)
        u_opt = torch.where((state[4, ...] <= self.speed_min), u_opt_vb_min, u_opt)
        u_opt = torch.where((state[4, ...] >= self.speed_max), u_opt_vb_max, u_opt)

        if convert_to_np:
            u_opt = to_numpy(u_opt)
        return u_opt

    def dynamics_for_filter(self, time, state, action):
        if isinstance(action, cp.Variable):
            return cp.hstack([
                -state[3] + state[4] * np.cos(state[2]) + state[1] * action[0],
                state[4] * np.sin(state[2]) - state[0] * action[0],
                -action[0] + action[1],
                action[2],
                action[3]
            ])
        dx = np.zeros_like(state)
        action_flat = np.asarray(action).flatten()
        dx[0] = - state[3] + state[4] * np.cos(state[2]) + state[1] * action_flat[0]
        dx[1] = state[4] * np.sin(state[2]) - state[0] * action_flat[0]
        dx[2] = -action_flat[0] + action_flat[1]
        dx[3] = action_flat[2]
        dx[4] = action_flat[3]
        return dx

class KinematicVehicle(AffineDynamicsWithBoxInputBounds):
    def __init__(self, v_min = 0.5, v_max = 1.5, accel_min=-1, accel_max=1, angular_rate_max=1.0,
                 speed_target=DEFAULT_SPEED_TARGET):
        self.speed_min  = v_min
        self.speed_max  = v_max
        self.speed_target = speed_target
        self.accel_min  = accel_min
        self.accel_max  = accel_max
        self.angular_rate_max = angular_rate_max

        self.u_min_v_min = [-self.angular_rate_max, 0]
        self.u_max_v_min = [self.angular_rate_max, self.accel_max]

        self.u_min_v_max = [-self.angular_rate_max, self.accel_min]
        self.u_max_v_max = [self.angular_rate_max, 0]

        super().__init__(x_dim=4,
                         u_min=[-self.angular_rate_max, self.accel_min],
                         u_max=[self.angular_rate_max, self.accel_max],
                         x_limit_min=[-np.inf, -np.inf, -np.inf, self.speed_min],
                         x_limit_max=[np.inf, np.inf, np.inf, self.speed_max])

    def drift(self, time, state):
        dx = torch.zeros_like(state)
        speed = state[3, ...]
        dx[0, ...] = speed * torch.cos(state[2, ...])
        dx[1, ...] = speed * torch.sin(state[2, ...])
        dx[2, ...] = 0.0
        dx[3, ...] = 0.0
        return dx

    def control_jacobian(self, time, state):
        g = torch.zeros(4, 2, dtype=torch.float32)
        g[2, 0] = 1.0
        g[3, 1] = 1.0
        return g

    def _eval_u_opt_with_specific_bound(self, LgV, u_min, u_max, mode):
        assert mode in ["min", "max"]
        u_opt = torch.zeros(self.u_dim, LgV.shape[1])
        for i in range(self.u_dim):
            if mode == "max":
                u_opt[i, LgV[i, ...] >= 0] = u_max[i]
                u_opt[i, LgV[i, ...] < 0] = u_min[i]
            else:
                u_opt[i, LgV[i, ...] >= 0] = u_min[i]
                u_opt[i, LgV[i, ...] < 0] = u_max[i]
        return u_opt

    def optimal_control(self, time, state, gradient, mode):
        assert mode in ["min", "max"]
        convert_to_np = False
        if type(state) == np.ndarray:
            state = torch.tensor(state)
            gradient = torch.tensor(gradient)
            convert_to_np = True
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        assert state.ndim == 2
        assert state.shape[0] == self.x_dim
        if gradient.ndim == 1 and gradient.shape[0] == self.x_dim:
            gradient = gradient.unsqueeze(-1)
        assert gradient.ndim == 2
        assert gradient.shape[0] == self.x_dim
        assert gradient.shape[1] == state.shape[1]

        gx = self.control_jacobian(time, state)
        if gx.ndim == 2:
            if gx.shape[0] != self.x_dim or gx.shape[1] != self.u_dim:
                raise ValueError(f"Invalid control Jacobian shape: {gx.shape}")
            gx = gx.unsqueeze(2).repeat(1, 1, state.shape[1])
        assert gx.ndim == 3
        assert gx.shape[0] == self.x_dim
        assert gx.shape[1] == self.u_dim
        assert gx.shape[2] == state.shape[1]

        LgV = torch.sum(gx * gradient.unsqueeze(1), dim=0)
        assert LgV.ndim == 2
        u_opt_interior = self._eval_u_opt_with_specific_bound(LgV, self.u_min, self.u_max, mode)
        u_opt_v_min = self._eval_u_opt_with_specific_bound(LgV, self.u_min_v_min, self.u_max_v_min, mode)
        u_opt_v_max = self._eval_u_opt_with_specific_bound(LgV, self.u_min_v_max, self.u_max_v_max, mode)
        u_opt = torch.where((state[3, ...] <= self.speed_min), u_opt_v_min, u_opt_interior)
        u_opt = torch.where((state[3, ...] >= self.speed_max), u_opt_v_max, u_opt)

        if convert_to_np:
            u_opt = to_numpy(u_opt)
        return u_opt

class MultipleVehicle(AffineDynamicsWithBoxInputBounds):
    def __init__(self, num_vehicles: int, vehicle_dynamics: Dynamics):
        self.num_vehicles = num_vehicles
        self.single_vehicle = vehicle_dynamics
        assert isinstance(vehicle_dynamics, AffineDynamicsWithBoxInputBounds), "car_dynamics must be an instance of AffineDynamicsWithBoxInputBounds"
        if isinstance(vehicle_dynamics, DubinsCar):
            self.dynamics_type = 'dubins_car'
        elif isinstance(vehicle_dynamics, KinematicVehicle):
            self.dynamics_type = 'kinematic_vehicle'
        else:
            raise ValueError(f"Invalid dynamics type: {type(vehicle_dynamics)}")
        u_min = np.array(vehicle_dynamics.u_min * num_vehicles, dtype=np.float32)
        u_max = np.array(vehicle_dynamics.u_max * num_vehicles, dtype=np.float32)
        if self.single_vehicle.x_limit_min is not None:
            x_limit_min = np.array(vehicle_dynamics.x_limit_min * num_vehicles, dtype=np.float32)
        else:
            x_limit_min = None
        if self.single_vehicle.x_limit_max is not None:
            x_limit_max = np.array(vehicle_dynamics.x_limit_max * num_vehicles, dtype=np.float32)
        else:
            x_limit_max = None
        self.vehicle_x_dim = vehicle_dynamics.x_dim
        self.vehicle_u_dim = vehicle_dynamics.u_dim
        super().__init__(x_dim=self.vehicle_x_dim*num_vehicles, u_min=u_min, u_max=u_max, x_limit_min=x_limit_min, x_limit_max=x_limit_max)
        self.target_offset = [None] * num_vehicles

    def drift(self, time, state):
        dx = torch.zeros_like(state)
        for i in range(self.num_vehicles):
            vehicle_state = state[self.vehicle_x_dim*i:self.vehicle_x_dim*(i+1), ...]
            dx[self.vehicle_x_dim*i:self.vehicle_x_dim*(i+1), ...] = self.single_vehicle.drift(time, vehicle_state)
        return dx

    def control_jacobian(self, time, state):
        g = torch.zeros(self.x_dim, self.u_dim, dtype=torch.float32)
        for i in range(self.num_vehicles):
            vehicle_state = state[self.vehicle_x_dim*i:self.vehicle_x_dim*(i+1), ...]
            g[self.vehicle_x_dim*i:self.vehicle_x_dim*(i+1), self.vehicle_u_dim*i:self.vehicle_u_dim*(i+1)] = self.single_vehicle.control_jacobian(time, vehicle_state)
        return g