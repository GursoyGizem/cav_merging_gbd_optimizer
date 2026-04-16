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
    # Weighting coefficients (default values ​​in the article)
    W_TERMINAL = 1.0    # w1: terminal time weight
    W_JERK     = 0.01   # w2: comfort (jerk minimization)
    W_VELOCITY = 0.1    # w3: reference speed deviation

    # Physical limits
    V_MIN = 5.0         # m/s — minimum speed 
    V_MAX = 40.0        # m/s — maximum speed 
    A_MIN = -4.0        # m/s² — maximum deceleration
    A_MAX = 2.0         # m/s² — maximum acceleration
    B_MAX = 2.0         # m/s³ — maximum jerk

    def __init__(
        self,
        vehicles: List[VehicleState],
        N: int = 20,        # space domain discretization steps
        S: float = 200.0,   # sum distance to merge point
        v_ref: float = 30.0 # reference speed for velocity tracking
    ):
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.vehicle_ids = [v.vehicle_id for v in vehicles]
        self.N = N
        self.S = S
        self.v_ref = v_ref
        self.ds = S / N     

    def solve(self, rmp_result: RMPResult, h: float) -> PSResult:
        n_v = len(self.vehicle_ids)
        N   = self.N

        # t[k][s]: vehicle k's transit time at step s
        t = [cp.Variable(N + 1, nonneg=True) for _ in range(n_v)]
        # E[k][s]: vehicle k's kinetic energy at step s
        E = [cp.Variable(N + 1, nonneg=True) for _ in range(n_v)]
        # a[k][s]: vehicle k's acceleration at step s
        a = [cp.Variable(N,     ) for _ in range(n_v)]
        # b[k][s]: control input 
        b = [cp.Variable(N,     ) for _ in range(n_v)]

        constraints = []
        cost_terms  = []

        for k, vid in enumerate(self.vehicle_ids):
            veh = self.vehicles[vid]

            # Starting conditions at s=0:
            # t[k][0] = 0 
            constraints.append(t[k][0] == 0)
            # E[k][0] = 0.5 * v0^2
            E0 = 0.5 * veh.velocity ** 2
            constraints.append(E[k][0] == E0)

            # Dynamic constraints 
            # Euler discretization:
            #   dt/ds = 1 / v = 1 / sqrt(2*E)  →  t[s+1] = t[s] + ds / sqrt(2*E[s])
            #   t[s+1] - t[s] >= ds / sqrt(2 * E_ref)   
            for s in range(N):
                E_ref = max(0.5 * veh.velocity ** 2, 1e-3)
                dt_approx = self.ds / (2 * np.sqrt(2 * E_ref)) * (
                    3 - E[k][s] / E_ref
                )
                constraints.append(
                    t[k][s + 1] - t[k][s] == dt_approx
                )
                constraints.append(
                    E[k][s + 1] == E[k][s] + a[k][s] * self.ds
                )
                if s < N - 1:
                    constraints.append(
                        a[k][s + 1] == a[k][s] + b[k][s] * self.ds
                    )

            # Physical limits
            # Speed limits: v = sqrt(2E) → E limited means v limited
            constraints.append(E[k] >= 0.5 * self.V_MIN ** 2)
            constraints.append(E[k] <= 0.5 * self.V_MAX ** 2)
            # Acceleration limits
            constraints.append(a[k] >= self.A_MIN)
            constraints.append(a[k] <= self.A_MAX)
            # Jerk limits
            constraints.append(b[k] >= -self.B_MAX)
            constraints.append(b[k] <= self.B_MAX)

            # Objective function terms
            # w1 * terminal time
            cost_terms.append(self.W_TERMINAL * t[k][N])
            # w2 * jerk^2 
            cost_terms.append(self.W_JERK * cp.sum_squares(b[k]))
            # w3 * velocity deviation^2 → v = sqrt(2E) → (v - v_ref)^2 = (sqrt(2E) - v_ref)^2
            v_squared = 2 * E[k]
            v_ref_sq  = self.v_ref ** 2
            cost_terms.append(
                self.W_VELOCITY * cp.sum_squares(v_squared - v_ref_sq) / N
            )

        # Bridge limits
        # tau[i] >= tau[leader] + h 
        constraints += self._add_bridge_constraints(
            t, rmp_result, h
        )

        objective = cp.Minimize(sum(cost_terms))
        problem   = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError:
            return PSResult(feasible=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
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

        return PSResult(
            feasible=True,
            objective_value=float(problem.value),
            t_values=t_vals,
            E_values=E_vals,
            a_values=a_vals,
            dual_vars=dual_vars
        )

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