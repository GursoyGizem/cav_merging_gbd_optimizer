from typing import List, Dict, Tuple
import pulp

from ..core.vehicle import VehicleState
from ..core.decision_variables import BinaryDecisionVariables
from ..core.gbd_results import RMPResult

"""
Relaxed Master Problem — The binary decision layer of GBD.

Formulated as MILP with PuLP. In each iteration:
1. Optimal cut or feasibility cut is added 
2. The problem is solved again → a new binary combination is generated
3. LB (lower bound) is updated

Decision variables:
alpha[i][j] : on-ramp i chooses main-lane j as the leader
gamma[i] : main-lane i changes lanes
beta[i][j] : lane-changing i chooses inner-lane j as the leader
eta : lower bound to the PS optimal value (Benders auxiliary variable)
"""
class MasterProblem:
    def __init__(
        self,
        onramp_ids: List[int],
        mainlane_ids: List[int],
        innerlane_ids: List[int],
        h: float
    ):
        self.onramp_ids   = onramp_ids
        self.mainlane_ids = mainlane_ids
        self.innerlane_ids = innerlane_ids
        self.h = h

        self._cut_count = 0
        self._lower_bound = float('-inf')

        self._build_problem()

    def _build_problem(self):
        self.prob = pulp.LpProblem("RMP", pulp.LpMinimize)

        self.alpha = {
            i: {
                j: pulp.LpVariable(f"alpha_{i}_{j}", cat="Binary")
                for j in self.mainlane_ids
            }
            for i in self.onramp_ids
        }

        self.gamma = {
            i: pulp.LpVariable(f"gamma_{i}", cat="Binary")
            for i in self.mainlane_ids
        }

        self.beta = {
            i: {
                j: pulp.LpVariable(f"beta_{i}_{j}", cat="Binary")
                for j in self.innerlane_ids
            }
            for i in self.mainlane_ids
        }

        self.eta = pulp.LpVariable("eta", lowBound=0)
        self.prob += self.eta, "minimize_eta"
        self._add_structural_constraints()

    def _add_structural_constraints(self):
        # Constraint 1: each on-ramp vehicle selects exactly 1 main-lane vehicle as the leader
        for i in self.onramp_ids:
            self.prob += (
                pulp.lpSum(self.alpha[i][j] for j in self.mainlane_ids) == 1,
                f"merging_sum_{i}"
            )

        # Constraint 2: the sum of beta and gamma must be equal
        for i in self.mainlane_ids:
            self.prob += (
                pulp.lpSum(self.beta[i][j] for j in self.innerlane_ids)
                == self.gamma[i],
                f"lane_change_consistency_{i}"
            )

        # Constraint 3: Same main-lane vehicle can be selected by at most 1 on-ramp
        for j in self.mainlane_ids:
            self.prob += (
                pulp.lpSum(self.alpha[i][j] for i in self.onramp_ids) <= 1,
                f"ordering_col_{j}"
            )

    def add_optimal_cut(self, ps_result, fixed_alpha, fixed_gamma, fixed_beta):
        """
        PS feasible - Optimal cut: eta >= f(x*) + lambda^T * (alpha - alpha*)
        """
        self._cut_count += 1
        cut_name = f"optimal_cut_{self._cut_count}"

        f_star = ps_result.objective_value

        linearization = pulp.lpSum(
            ps_result.dual_vars.get((i, j), 0.0) * (
                self.alpha[i][j] - fixed_alpha.get(i, {}).get(j, 0)
            )
            for i in self.onramp_ids
            for j in self.mainlane_ids
        )

        self.prob += (
            self.eta >= f_star + linearization,
            cut_name
        )

    def add_feasibility_cut(self, ps_result, fixed_alpha, fixed_gamma, fixed_beta):
        """
        PS infeasible - Feasibility cut: sum_{(i,j): alpha*_ij=1} alpha_ij <= (seçilen toplam - 1)
        """
        self._cut_count += 1
        cut_name = f"feasibility_cut_{self._cut_count}"

        active_pairs = [
            (i, j)
            for i in self.onramp_ids
            for j in self.mainlane_ids
            if fixed_alpha.get(i, {}).get(j, 0) >= 0.5
        ]

        if not active_pairs:
            return  

        self.prob += (
            pulp.lpSum(self.alpha[i][j] for i, j in active_pairs)
            <= len(active_pairs) - 1,
            cut_name
        )

    def solve(self) -> RMPResult:
        self.prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if self.prob.status != 1:  
            return RMPResult(feasible=False)

        lb = pulp.value(self.prob.objective)
        self._lower_bound = lb

        alpha_vals = {
            i: {j: int(round(pulp.value(self.alpha[i][j]) or 0))
                for j in self.mainlane_ids}
            for i in self.onramp_ids
        }
        gamma_vals = {
            i: int(round(pulp.value(self.gamma[i]) or 0))
            for i in self.mainlane_ids
        }
        beta_vals = {
            i: {j: int(round(pulp.value(self.beta[i][j]) or 0))
                for j in self.innerlane_ids}
            for i in self.mainlane_ids
        }

        return RMPResult(
            feasible=True,
            lower_bound=lb,
            alpha=alpha_vals,
            gamma=gamma_vals,
            beta=beta_vals,
            eta=pulp.value(self.eta) or 0.0
        )

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def cut_count(self) -> int:
        return self._cut_count