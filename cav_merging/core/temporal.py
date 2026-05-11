from typing import List
from .vehicle import compute_minimum_headway

"""
Computes the terminal crossing times tau[i] for an ordered vehicle sequence.
This ensures that consecutive vehicles in the merging sequence maintain at least the minimum headway h when crossing the merging point.
tau[i] >= tau[i-1] + h    
"""

def compute_terminal_times(
    num_vehicles: int,
    h: float,
    tau_0: float = 0.0
) -> List[float]:

    tau = [0.0] * num_vehicles
    tau[0] = tau_0
    for i in range(1, num_vehicles):
        tau[i] = tau[i - 1] + h
    return tau