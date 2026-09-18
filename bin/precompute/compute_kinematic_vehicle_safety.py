"""Precompute the pairwise avoidance value function (the CBF) in relative kinematic-vehicle dynamics.

Writes ``data/relative_dynamics_cbf.pkl``. Solving on this grid needs roughly 21 GB of GPU memory.

    python bin/precompute/compute_kinematic_vehicle_safety.py
"""
import os
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from hj_reachability_torch.grid import HjTorchGrid, HjGridMetaData, HjData
from hj_reachability_torch.solver import solve_min_backward_reachable_tube

from lib.dubins_car import TwoKinematicVehicleRelative

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
if device == "cuda":
    torch.cuda.set_device(0)
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

SPEED_MIN = 0.03  # km/s
SPEED_MAX = 0.09  # km/s
ACCEL_MIN = -0.001  # km/s^2
ACCEL_MAX = 0.002  # km/s^2
ANGULAR_RATE_MAX = 0.1  # rad/s
SEPARATION_DISTANCE = 0.5  # km

T_MAX = 100.0  # s
DT = 0.1  # s

dynsys = TwoKinematicVehicleRelative(v_min=SPEED_MIN, v_max=SPEED_MAX,
                                     accel_min=ACCEL_MIN, accel_max=ACCEL_MAX,
                                     angular_rate_max=ANGULAR_RATE_MAX)

domain_lo = np.array([-3.0, -3.0, -np.pi, SPEED_MIN, SPEED_MIN])
domain_hi = np.array([3.0, 3.0, np.pi, SPEED_MAX, SPEED_MAX])
grid_shape = [121, 121, 32, 10, 10]
grid = HjTorchGrid(domain_lo, domain_hi, grid_shape, pdims=[2])

distance_between_vehicles = torch.sqrt(grid.states[0, ...] ** 2 + grid.states[1, ...] ** 2)
target_function = distance_between_vehicles - SEPARATION_DISTANCE

values, times = solve_min_backward_reachable_tube(dynsys, grid, target_function, DT, T_MAX,
                                                  return_final=True)

info = {'angular_rate_max': ANGULAR_RATE_MAX,
        'speed_min': SPEED_MIN, 'speed_max': SPEED_MAX,
        'accel_min': ACCEL_MIN, 'accel_max': ACCEL_MAX,
        'separation_distance': SEPARATION_DISTANCE}
grid_meta_data = HjGridMetaData(domain_lo, domain_hi, grid_shape, pdims=[2])
hj_data = HjData(grid_meta_data, target_function, None, times, values, info)

file_name = 'data/relative_dynamics_cbf.pkl'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, 'wb') as file:
    pickle.dump(hj_data, file)
print(f"wrote {file_name}")
