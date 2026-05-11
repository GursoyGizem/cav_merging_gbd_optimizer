from typing import List, Optional
from rsu.rsu_trigger import RSUTriggerController, TriggerEvent
from .vehicle import VehicleState
from .decision_variables import BinaryDecisionVariables
from .vehicle import compute_minimum_headway

"""
RSUTriggerController  →  GBDCoordinator  →  (sonraki issue) Solver
"""
class GBDCoordinator:
    def __init__(self, rsu_controller: RSUTriggerController, h: float):
        self.rsu = rsu_controller
        self.h = h
        self._pending_events: List[TriggerEvent] = []

    def step(
        self,
        vehicles: List[VehicleState],
        current_time: float
    ) -> Optional[List[TriggerEvent]]:
    
        results = self.rsu.update_all(vehicles, current_time)

        new_events = [
            event for _, _, event in results if event is not None
        ]
        self._pending_events.extend(new_events)

        active_ids = self.rsu.get_active_vehicles()
        if len(active_ids) >= 2 and self._pending_events:
            events_to_process = list(self._pending_events)
            self._pending_events.clear()
            return events_to_process

        return None

    def status_report(self, vehicles: List[VehicleState]) -> str:
        lines = ["RSU Report:"]
        for v in vehicles:
            state = self.rsu.get_state(v.vehicle_id)
            dist  = abs(self.rsu.config.rsu_position - v.position)
            lines.append(
                f"Vehicle = {v.vehicle_id:>3} | "
                f"Position = {v.position:>7.1f}m | "
                f"RSU distance = {dist:>6.1f}m | "
                f"State = {state.name}"
            )
        return "\n".join(lines)