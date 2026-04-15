from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ..vehicle import VehicleState
from .rsu_config import RSUConfig
from .rsu_state import VehicleZoneState

@dataclass
class TriggerEvent:
    vehicle_id: int
    trigger_x: float
    trigger_time: float
    distance_to_rsu: float

class RSUTriggerController:
    def __init__(self, config: RSUConfig):
        self.config = config
        self._zone_states: Dict[int, VehicleZoneState] = {}
        self._trigger_log: List[TriggerEvent] = []

    def _distance_to_rsu(self, vehicle: VehicleState) -> float:
        return abs(self.config.rsu_position - vehicle.position)

    def _is_in_coverage(self, vehicle: VehicleState) -> bool:
        return self._distance_to_rsu(vehicle) <= self.config.coverage_radius

    def _is_in_trigger_zone(self, vehicle: VehicleState) -> bool:
        return self._distance_to_rsu(vehicle) <= self.config.trigger_distance

    def update(
        self,
        vehicle: VehicleState,
        current_time: float
    ) -> Tuple[VehicleZoneState, Optional[TriggerEvent]]:
    
        vid = vehicle.vehicle_id
        prev_state = self._zone_states.get(vid, VehicleZoneState.OUTSIDE)

        in_coverage    = self._is_in_coverage(vehicle)
        in_trigger     = self._is_in_trigger_zone(vehicle)

        if not in_coverage: 
            new_state = VehicleZoneState.OUTSIDE
        elif in_coverage and in_trigger and prev_state == VehicleZoneState.OUTSIDE: 
            new_state = VehicleZoneState.ENTERING
        elif in_coverage and in_trigger and prev_state in (VehicleZoneState.ENTERING, VehicleZoneState.INSIDE):
            new_state = VehicleZoneState.INSIDE
        elif in_coverage and not in_trigger:
            if prev_state in (VehicleZoneState.INSIDE, VehicleZoneState.ENTERING):
                new_state = VehicleZoneState.LEAVING
            else:
                new_state = VehicleZoneState.OUTSIDE
        else:
            new_state = prev_state

        self._zone_states[vid] = new_state

        trigger_event = None
        if new_state == VehicleZoneState.ENTERING:
            trigger_event = TriggerEvent(
                vehicle_id=vid,
                trigger_x=vehicle.position,
                trigger_time=current_time,
                distance_to_rsu=self._distance_to_rsu(vehicle)
            )
            self._trigger_log.append(trigger_event)

        return new_state, trigger_event

    def update_all(
        self,
        vehicles: List[VehicleState],
        current_time: float
    ) -> List[Tuple[VehicleState, VehicleZoneState, Optional[TriggerEvent]]]:
        results = []
        for v in vehicles:
            state, event = self.update(v, current_time)
            results.append((v, state, event))
        return results

    def get_state(self, vehicle_id: int) -> VehicleZoneState:
        return self._zone_states.get(vehicle_id, VehicleZoneState.OUTSIDE)

    def get_trigger_log(self) -> List[TriggerEvent]:
        return list(self._trigger_log)

    def get_active_vehicles(self) -> List[int]:
        return [
            vid for vid, state in self._zone_states.items()
            if state in (VehicleZoneState.ENTERING, VehicleZoneState.INSIDE)
        ]