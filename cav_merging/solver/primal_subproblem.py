from typing import List, Dict
import numpy as np
import cvxpy as cp

from ..core.vehicle import VehicleState
from ..core.gbd_results import PSResult, RMPResult

"""
Primal Subproblem — Continuous orbit layer of GBD.

State variables (s ∈ [0, S]):
t_li(s): Transit time at position s [s]
E_li(s): Kinetic energy = 0.5 * v^2 [m²/s²]
a_li(s): Acceleration [m/s²]
Control input:
b_li(s): Jerk (derivative of acceleration) [m/s³]

min sum_i [ w1 * t_li(S) + w2 * integral(b²) + w3 * (v - v_ref)² ]
"""

class PrimalSubproblem:
    W_TERMINAL = 1.0    
    W_JERK     = 0.01   
    W_VELOCITY = 0.1    

    V_MIN = 5.0         
    V_MAX = 40.0         
    A_MIN = -4.0        
    A_MAX = 2.0         
    B_MAX = 2.0         

    def __init__(
        self,
        vehicles: List[VehicleState],
        N: int = 20,        
        S: float = 200.0,   
        v_ref: float = 30.0 
    ):
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.vehicle_ids = [v.vehicle_id for v in vehicles]
        self.N = N
        self.S = S
        self.v_ref = v_ref
        self.ds = S / N   

        self._last_result = None  

    def solve(self, rmp_result: RMPResult, h: float) -> PSResult:
        n_v = len(self.vehicle_ids)
        N   = self.N

        t = [cp.Variable(N + 1, nonneg=True) for _ in range(n_v)]
        E = [cp.Variable(N + 1, nonneg=True) for _ in range(n_v)]
        a = [cp.Variable(N) for _ in range(n_v)]
        b = [cp.Variable(N) for _ in range(n_v)]

        constraints = []
        cost_terms  = []

        for k, vid in enumerate(self.vehicle_ids):
            veh = self.vehicles[vid]

            constraints.append(t[k][0] == 0)
            E0 = 0.5 * veh.velocity ** 2
            constraints.append(E[k][0] == E0)

            for s in range(N):
                E_ref = max(0.5 * veh.velocity ** 2, 1e-3)
                dt_approx = self.ds / (2 * np.sqrt(2 * E_ref)) * (
                    3 - E[k][s] / E_ref
                )
                constraints.append(t[k][s + 1] - t[k][s] == dt_approx)
                constraints.append(E[k][s + 1] == E[k][s] + a[k][s] * self.ds)
                if s < N - 1:
                    constraints.append(a[k][s + 1] == a[k][s] + b[k][s] * self.ds)

            constraints.append(E[k] >= 0.5 * self.V_MIN ** 2)
            constraints.append(E[k] <= 0.5 * self.V_MAX ** 2)
            constraints.append(a[k] >= self.A_MIN)
            constraints.append(a[k] <= self.A_MAX)
            constraints.append(b[k] >= -self.B_MAX)
            constraints.append(b[k] <= self.B_MAX)

            cost_terms.append(self.W_TERMINAL * t[k][N])
            cost_terms.append(self.W_JERK * cp.sum_squares(b[k]))
            v_squared = 2 * E[k]
            v_ref_sq  = self.v_ref ** 2
            cost_terms.append(
                self.W_VELOCITY * cp.sum_squares(v_squared - v_ref_sq) / N
            )

        bridge = self._add_bridge_constraints(t, rmp_result, h)
        if bridge:
            constraints += bridge

        objective = cp.Minimize(sum(cost_terms))
        problem   = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError:
            return PSResult(feasible=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            problem2 = cp.Problem(cp.Minimize(sum(cost_terms)), [c for c in constraints if c not in bridge])
            problem2.solve(solver=cp.ECOS, verbose=False)
            if problem2.status in ("optimal", "optimal_inaccurate"):
                import logging
                logging.getLogger(__name__).warning(
                    "PS infeasible with bridge constraints, but feasible without them. Check the bridge constraint generation logic."
                )
            return PSResult(feasible=False)

        t_vals = {
            vid: t[k].value.tolist()
            for k, vid in enumerate(self.vehicle_ids)
            if t[k].value is not None
        }
        E_vals = {
            vid: E[k].value.tolist()
            for k, vid in enumerate(self.vehicle_ids)
            if E[k].value is not None
        }
        a_vals = {
            vid: a[k].value.tolist()
            for k, vid in enumerate(self.vehicle_ids)
            if a[k].value is not None
        }

        dual_vars = self._extract_duals(constraints, rmp_result)
        self._last_result = PSResult(
            feasible=True,
            objective_value=float(problem.value),
            t_values=t_vals,
            E_values=E_vals,
            a_values=a_vals,
            dual_vars=dual_vars
        )
        return self._last_result

    def _add_bridge_constraints(
        self,
        t: list,
        rmp_result: RMPResult,
        h: float
    ) -> list:
        bridge = []
        id_to_idx = {vid: k for k, vid in enumerate(self.vehicle_ids)}

        for i in self.vehicle_ids:
            for j in self.vehicle_ids:
                if i == j:
                    continue
                alpha_ij = rmp_result.alpha.get(i, {}).get(j, 0)
                if alpha_ij == 1 and i in id_to_idx and j in id_to_idx:
                    ki = id_to_idx[i]
                    kj = id_to_idx[j]
                    
                    bridge.append(
                        t[ki][self.N] >= t[kj][self.N] + h
                    )
        return bridge

    def _extract_duals(self, constraints: list, rmp_result: RMPResult) -> dict:
        dual_vars = {}
        id_to_idx = {vid: k for k, vid in enumerate(self.vehicle_ids)}

        for i in self.vehicle_ids:
            for j in self.vehicle_ids:
                if i == j:
                    continue
                alpha_ij = rmp_result.alpha.get(i, {}).get(j, 0)
                if alpha_ij == 1:
                    dual_vars[(i, j)] = 0.0 

        return dual_vars