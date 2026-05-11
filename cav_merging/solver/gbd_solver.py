from typing import List
import logging

from ..core.vehicle import VehicleState
from ..core.vehicle import compute_minimum_headway
from ..core.gbd_results import GBDIteration, SolveStatus
from .master_problem import MasterProblem
from .primal_subproblem import PrimalSubproblem

logger = logging.getLogger(__name__)

"""
Algorithm:
1. Solve RMP → binary decisions (alpha, gamma, beta) + LB
2. Solve PS (binary decisions constant) → continuous trajectory + UB
3. If PS is feasible → add optimal cut
If PS is infeasible → add feasibility cut
4. Stop if |UB - LB| <= epsilon
5. Otherwise return to 1
"""

class GBDSolver:
    def __init__(
        self,
        vehicles: List[VehicleState],
        epsilon: float = 1e-4,
        max_iter: int = 50,
        N: int = 20,
        S: float = 200.0,
        v_ref: float = 30.0
    ):
        self.vehicles = vehicles
        self.epsilon  = epsilon
        self.max_iter = max_iter

        self.onramp_ids   = [v.vehicle_id for v in vehicles if v.lane_id == 1]
        self.mainlane_ids = [v.vehicle_id for v in vehicles if v.lane_id == 0]
        self.innerlane_ids = []   

        self.h = compute_minimum_headway(t_gap=1.5, l_veh=4.5, s_0=2.0, v_f=v_ref)

        self.rmp = MasterProblem(
            self.onramp_ids,
            self.mainlane_ids,
            self.innerlane_ids,
            self.h
        )
        self.ps = PrimalSubproblem(vehicles, N=N, S=S, v_ref=v_ref)

        self._upper_bound = float('inf')
        self._lower_bound = float('-inf')

        self.history: List[GBDIteration] = []

    def solve(self) -> List[GBDIteration]:
     
        logger.info("GBD solver:")
        logger.info(f"vehicles: {[v.vehicle_id for v in self.vehicles]}")
        logger.info(f"epsilon={self.epsilon}, max_iter={self.max_iter}")

        for iteration in range(1, self.max_iter + 1):

            # RMP
            rmp_result = self.rmp.solve()

            if not rmp_result.feasible:
                logger.warning(f"  [{iteration}] RMP infeasible.")
                break

            self._lower_bound = rmp_result.lower_bound

            # PS
            ps_result = self.ps.solve(rmp_result, self.h)

            # Cuts
            if ps_result.feasible:
                self._upper_bound = min(
                    self._upper_bound,
                    ps_result.objective_value
                )
                self.rmp.add_optimal_cut(
                    ps_result,
                    rmp_result.alpha,
                    rmp_result.gamma,
                    rmp_result.beta
                )
                ps_status = SolveStatus.OPTIMAL
                logger.info(
                    f"[{iteration}] PS optimal."
                    f"LB={self._lower_bound:.4f}"
                    f"UB={self._upper_bound:.4f}"
                    f"gap={self._gap:.6f}"
                )
            else:
                self.rmp.add_feasibility_cut(
                    ps_result,
                    rmp_result.alpha,
                    rmp_result.gamma,
                    rmp_result.beta
                )
                ps_status = SolveStatus.INFEASIBLE
                logger.info(
                    f"[{iteration}] PS infeasible."
                    f"Feasibility cut added."
                    f"LB={self._lower_bound:.4f}"
                )

            self.history.append(GBDIteration(
                iteration=iteration,
                lower_bound=self._lower_bound,
                upper_bound=self._upper_bound,
                gap=self._gap,
                ps_status=ps_status,
                num_cuts=self.rmp.cut_count
            ))

            if self._gap <= self.epsilon:
                logger.info(
                    f"gap={self._gap:.6f} <= epsilon={self.epsilon}"
                )
                self.history[-1].ps_status = SolveStatus.CONVERGED
                break

        else:
            logger.warning(
                f"Max iter: ({self.max_iter}) reached."
                f"Final gap={self._gap:.4f}"
            )
            if self.history:
                self.history[-1].ps_status = SolveStatus.MAX_ITER

        return self.history

    @property
    def _gap(self) -> float:
        if self._upper_bound == float('inf'):
            return float('inf')
        return max(0.0, self._upper_bound - self._lower_bound)

    @property
    def best_upper_bound(self) -> float:
        return self._upper_bound

    @property
    def best_lower_bound(self) -> float:
        return self._lower_bound