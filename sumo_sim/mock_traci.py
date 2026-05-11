"""
SUMO simulation not available, using a mock TraCI interface for testing and development purposes.
"""
import math
from typing import Dict, List

class _VehicleState:
    def __init__(self, vid, x, y, speed, lane_id, lane_index):
        self.vid        = vid
        self.x          = x
        self.y          = y
        self.speed      = speed
        self.lane_id    = lane_id
        self.lane_index = lane_index

class _MockVehicle:
    def __init__(self, states: Dict[str, _VehicleState]):
        self._states = states
        self._speed_overrides: Dict[str, float] = {}
        self._lane_overrides:  Dict[str, int]   = {}

    def getIDList(self) -> List[str]:
        return list(self._states.keys())

    def getPosition(self, vid: str):
        s = self._states[vid]
        return (s.x, s.y)

    def getSpeed(self, vid: str) -> float:
        return self._speed_overrides.get(vid, self._states[vid].speed)

    def getLaneID(self, vid: str) -> str:
        return self._states[vid].lane_id

    def getLaneIndex(self, vid: str) -> int:
        return self._lane_overrides.get(vid, self._states[vid].lane_index)

    def setSpeed(self, vid: str, speed: float):
        self._speed_overrides[vid] = max(0.0, speed)

    def changeLane(self, vid: str, lane_index: int, duration: float):
        self._lane_overrides[vid] = lane_index

    def setSpeedMode(self, vid: str, mode: int):
        pass  

    def step(self):
        dt = 0.1
        for vid, s in self._states.items():
            v = self._speed_overrides.get(vid, s.speed)
            s.x += v * dt
            s.speed = v

class _MockSimulation:
    def __init__(self):
        self._time = 0.0

    def getTime(self) -> float:
        return self._time

    def step(self):
        self._time += 0.1

class MockTraCI:
    def __init__(self):
        initial_states = {
            "inside_1":  _VehicleState("inside_1",  630, 3.5,  28.0, "main_road_0", 0),
            "inside_2":  _VehicleState("inside_2",  590, 7.0,  27.0, "main_road_1", 1),
            "outside_1": _VehicleState("outside_1", 590, 10.5, 28.0, "main_road_2", 2),
            "outside_2": _VehicleState("outside_2", 550, 14.0, 27.0, "main_road_3", 3),
            "ramp_1":    _VehicleState("ramp_1",    600, -40.0, 26.0, "on_ramp_0",  1),
        }
        self.vehicle    = _MockVehicle(initial_states)
        self.simulation = _MockSimulation()

    def step(self):
        self.vehicle.step()
        self.simulation.step()