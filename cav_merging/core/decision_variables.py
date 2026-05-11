from dataclasses import dataclass, field
from typing import Dict

"""
Encapsulates all binary decision variables of the RMP (master problem).
In the GBD framework, the master problem solves exclusively over these binary variables. 
The primal subproblem (PS) then receives a fixed realization of these variables and optimizes the continuous trajectory.
"""

@dataclass
class BinaryDecisionVariables:
    # alpha[i][j]: on-ramp aracı i, main-lane aracı j'yi lider seçer
    alpha: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # gamma[i]: main-lane aracı i şerit değiştirir mi?
    gamma: Dict[int, int] = field(default_factory=dict)

    # beta[i][j]: şerit değiştiren araç i, inner-lane aracı j'yi lider seçer
    beta: Dict[int, Dict[int, int]] = field(default_factory=dict)

    def initialize(
        self,
        onramp_ids: list[int],
        mainlane_ids: list[int],
        innerlane_ids: list[int]
    ) -> None:
        
        # alpha: her on-ramp aracı için, her main-lane aracı için 0
        for i in onramp_ids:
            self.alpha[i] = {j: 0 for j in mainlane_ids}

        # gamma: her main-lane aracı için 0 (şerit değiştirmiyor)
        for i in mainlane_ids:
            self.gamma[i] = 0

        # beta: her main-lane aracı için, her inner-lane aracı için 0 (şerit değiştirmiyor ve inner-lane aracını lider seçmiyor)
        for i in mainlane_ids:
            self.beta[i] = {j: 0 for j in innerlane_ids}