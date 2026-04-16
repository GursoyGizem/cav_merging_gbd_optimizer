from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum, auto

class SolveStatus(Enum):
    OPTIMAL    = auto()   # PS feasible - optimal cut
    INFEASIBLE = auto()   # PS infeasible - feasibility cut
    CONVERGED  = auto()   # UB - LB <= epsilon
    MAX_ITER   = auto()   

@dataclass
class PSResult:
    feasible: bool
    objective_value: float = float('inf')
    
    t_values: dict = field(default_factory=dict)   
    E_values: dict = field(default_factory=dict)   
    a_values: dict = field(default_factory=dict)   

    dual_vars: dict = field(default_factory=dict)

@dataclass
class RMPResult:
    feasible: bool
    lower_bound: float = float('-inf')

    alpha: dict = field(default_factory=dict)
    gamma: dict = field(default_factory=dict)
    beta:  dict = field(default_factory=dict)

    eta: float = 0.0

@dataclass
class GBDIteration:
    iteration: int
    lower_bound: float
    upper_bound: float
    gap: float
    ps_status: SolveStatus
    num_cuts: int