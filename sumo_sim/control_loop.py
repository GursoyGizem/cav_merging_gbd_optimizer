import logging
import pulp
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from cav_merging.core.vehicle import VehicleState
from cav_merging.core.gbd_results import SolveStatus
from cav_merging.solver.gbd_solver import GBDSolver
from sumo_sim.sumo_env import SUMOEnvironment

logger = logging.getLogger(__name__)

@dataclass
class ControlResult:
    time: float
    vehicle_states: List[VehicleState]
    gbd_triggered: bool = False
    gbd_iterations: int = 0
    final_gap: float    = float('inf')
    converged: bool     = False
    speed_commands: Dict[int, float] = field(default_factory=dict)
    lane_commands:  Dict[int, int]   = field(default_factory=dict)

class MergingControlLoop:
  
    CASE_STUDY_1 = {
        "onramp_ids":    [30],        
        "mainlane_ids":  [20, 21],    
        "innerlane_ids": [10, 11],    
    }

    def __init__(self, env: SUMOEnvironment, gbd_epsilon: float = 1e-3, gbd_max_iter: int  = 20):
        self.env          = env
        self.gbd_epsilon  = gbd_epsilon
        self.gbd_max_iter = gbd_max_iter 
        self._gbd_active  = False
        self._last_solver: Optional[GBDSolver] = None
        self.results: List[ControlResult] = []

    def _trigger_gbd(self, vehicles: List[VehicleState], current_time: float) -> ControlResult:
        logger.info(f"[{current_time:.1f}s] Trigerred GBD: "
                    f"{[v.vehicle_id for v in vehicles]}")

        solver = GBDSolver(
            vehicles=vehicles,
            epsilon=self.gbd_epsilon,
            max_iter=self.gbd_max_iter,
            N=15,
            S=150.0,
            v_ref=28.0,
        )
        history = solver.solve()
        self._last_solver = solver
        self._gbd_active  = True

        converged  = any(it.ps_status == SolveStatus.CONVERGED for it in history)
        final_gap  = history[-1].gap if history else float('inf')
        n_iter     = len(history)

        logger.info(f"  GBD completed: {n_iter} iterations, "
                    f"gap={final_gap:.4f}, converged={converged}")

        speed_cmds = self._extract_speed_commands(solver, vehicles)
        lane_cmds  = self._extract_lane_commands(solver)

        return ControlResult(
            time=current_time,
            vehicle_states=vehicles,
            gbd_triggered=True,
            gbd_iterations=n_iter,
            final_gap=final_gap,
            converged=converged,
            speed_commands=speed_cmds,
            lane_commands=lane_cmds,
        )

    def _extract_speed_commands(self, solver: GBDSolver, vehicles: List[VehicleState]) -> Dict[int, float]:
        ps   = solver.ps
        last = getattr(ps, '_last_result', None)

        if last is None or not last.feasible:
            return {v.vehicle_id: v.velocity for v in vehicles}

        cmds = {}
        for vid in solver.vehicle_ids:
            if vid in last.t_values and vid in last.E_values:
                t_list = last.t_values[vid]
                E_list = last.E_values[vid]

                if len(t_list) >= 2 and t_list[-1] > 0:
                    E_terminal = max(E_list[-1], 0)
                    v_terminal = float(np.sqrt(2 * E_terminal))
                    v_terminal = min(max(v_terminal, ps.V_MIN), ps.V_MAX)
                    cmds[vid] = v_terminal
                else:
                    veh = next((v for v in vehicles if v.vehicle_id == vid), None)
                    cmds[vid] = veh.velocity if veh else 25.0
            else:
                veh = next((v for v in vehicles if v.vehicle_id == vid), None)
                cmds[vid] = veh.velocity if veh else 25.0

        return cmds

    def _extract_lane_commands(self, solver: GBDSolver) -> Dict[int, int]:
        cmds = {}
        if self._last_solver is None:
            return cmds

        rmp = solver.rmp
        history = solver.history
        if not history:
            return cmds

        for mainlane_id in solver.mainlane_ids:
            gamma_var = rmp.gamma.get(mainlane_id)
            if gamma_var is not None:
                val = pulp.value(gamma_var)
                if val and round(val) == 1:
                    cmds[mainlane_id] = 1 

        return cmds

    def _apply_commands(self, result: ControlResult):
        for vid, speed in result.speed_commands.items():
            self.env.send_speed_command(vid, speed)
            logger.debug(f"  Hız komutu: araç {vid} → {speed:.2f} m/s")

        for vid, lane in result.lane_commands.items():
            self.env.send_lane_change_command(vid, lane)
            logger.info(f"  Şerit değiştirme: araç {vid} → lane {lane}")

    def run(self, max_steps: int = 300) -> List[ControlResult]:
        self.env.connect()

        try:
            for step in range(max_steps):
                step_result = self.env.step()
                vehicles    = step_result["vehicles"]
                current_time = step_result["time"]
                trigger_events = step_result["trigger_events"]

                if trigger_events and not self._gbd_active:
                    ctrl_result = self._trigger_gbd(vehicles, current_time)
                    self._apply_commands(ctrl_result)
                    self.results.append(ctrl_result)
                else:
                    self.results.append(ControlResult(
                        time=current_time,
                        vehicle_states=vehicles,
                    ))

        finally:
            self.env.close()

        return self.results