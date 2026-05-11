from dataclasses import dataclass, field
from typing import Optional

"""
Represents the instantaneous state of a connected autonomous vehicle (CAV).
"""

@dataclass
class VehicleState:
    vehicle_id: int          
    position: float          
    velocity: float          
    lane_id: int             
    acceleration: float = 0.0  

"""
Computes the minimum safe time headway h between consecutive vehicles.
h = t_gap + (l_veh + s_0) / v_f
"""
def compute_minimum_headway(
    t_gap: float,
    l_veh: float,
    s_0: float,
    v_f: float
) -> float:

    if v_f <= 0:
        raise ValueError("Free-flow speed v_f must be positive.")
    return t_gap + (l_veh + s_0) / v_f