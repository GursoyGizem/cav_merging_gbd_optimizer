from typing import List, Dict, Tuple
from ..core.vehicle import VehicleState
from ..core.decision_variables import BinaryDecisionVariables

"""
Builds and validates the constraint sets for the MINLP formulation.
In the GBD decomposition:
- Merging and ordering constraints feed into the RMP (binary variables).
- Lane-change constraints couple RMP decisions with PS trajectory bounds.
This class operates in 'check mode': each method returns a list of (constraint_name, is_satisfied, violation_amount) tuples so that the Benders cut generator can inspect which constraints are active.
"""

class ConstraintBuilder:
    def __init__(
        self,
        vehicles: List[VehicleState],
        dvars: BinaryDecisionVariables,
        h: float
    ):
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.dvars = dvars
        self.h = h

        self.onramp_ids = [
            v.vehicle_id for v in vehicles if v.lane_id == 1
        ]
        self.mainlane_ids = [
            v.vehicle_id for v in vehicles if v.lane_id == 0
        ]

    def add_merging_constraints(self) -> List[Tuple[str, bool, float]]:
        results = []
        for i in self.onramp_ids:
            if i not in self.dvars.alpha:
                results.append((f"merging_alpha_sum_{i}", False, 1.0))
                continue

            total = sum(self.dvars.alpha[i].values())
            satisfied = (total == 1)
            violation = abs(total - 1)
            results.append((f"merging_alpha_sum_{i}", satisfied, violation))

        return results

    def add_lane_change_constraints(self) -> List[Tuple[str, bool, float]]:
        results = []
        for i in self.mainlane_ids:
            gamma_i = self.dvars.gamma.get(i, 0)
            beta_sum = sum(self.dvars.beta.get(i, {}).values())

            satisfied = (beta_sum == gamma_i)
            violation = abs(beta_sum - gamma_i)
            results.append(
                (f"lane_change_beta_gamma_{i}", satisfied, violation)
            )

        return results

    def add_ordering_constraints(self) -> List[Tuple[str, bool, float]]:
        results = []
        for j in self.mainlane_ids:
            total = sum(
                self.dvars.alpha.get(i, {}).get(j, 0)
                for i in self.onramp_ids
            )
            satisfied = (total <= 1)
            violation = max(0.0, total - 1)
            results.append(
                (f"ordering_alpha_col_{j}", satisfied, float(violation))
            )

        return results

    def check_all(self) -> Dict[str, List[Tuple[str, bool, float]]]:
        return {
            "merging":     self.add_merging_constraints(),
            "lane_change": self.add_lane_change_constraints(),
            "ordering":    self.add_ordering_constraints(),
        }