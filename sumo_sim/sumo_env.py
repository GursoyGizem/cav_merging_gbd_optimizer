import os
import sys
import traci
from typing import List, Optional, Dict
from dataclasses import dataclass
from sumo_sim.mock_traci import MockTraCI
from cav_merging.core.vehicle import VehicleState
from cav_merging.core.rsu.rsu_config import RSUConfig
from cav_merging.core.rsu.rsu_trigger import RSUTriggerController

@dataclass
class SUMOConfig:
    sumocfg_path: str
    rsu_position: float   = 0.0     
    trigger_distance: float = 150.0
    coverage_radius: float  = 200.0
    step_length: float      = 0.1   
    use_gui: bool           = False

"""
SUMO and TraCI connection management, vehicle state reading, and command sending.
"""
class SUMOEnvironment:
    INNER_LANE_INDICES = {0, 1}
    OUTER_LANE_INDICES = {2, 3}
    RAMP_EDGE_PREFIX   = "on_ramp"

    def __init__(self, config: SUMOConfig, use_mock: bool = False):
        self.config    = config
        self.use_mock  = use_mock
        self._traci    = None
        self._time     = 0.0

        rsu_cfg = RSUConfig(
            rsu_position=config.rsu_position,
            coverage_radius=config.coverage_radius,
            trigger_distance=config.trigger_distance,
        )
        self.rsu_controller = RSUTriggerController(rsu_cfg)

        self._cav_group: List[str] = []
        self._car_following_mode: Dict[str, bool] = {}  

    def connect(self):
        if self.use_mock:
            from sumo_sim.mock_traci import MockTraCI
            self._traci = MockTraCI()
            return

        import shutil
        sumo_binary = "sumo-gui" if self.config.use_gui else "sumo"
        
        sumo_path = shutil.which(sumo_binary)
        if sumo_path is None:
            import os
            candidates = [
                r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
            ]
            for c in candidates:
                if os.path.exists(c):
                    sumo_path = c
                    break

        if sumo_path is None:
            raise RuntimeError(
                "SUMO exe dont find. Please install SUMO and ensure it's in your PATH, or update the sumo_path variable in SUMOEnvironment.connect()"
            )

        import traci
        sumo_cmd = [sumo_path, "-c", self.config.sumocfg_path,
                    "--step-length", str(self.config.step_length),
                    "--no-warnings", "true"]
        traci.start(sumo_cmd)
        self._traci = traci

    def close(self):
        if self._traci and not self.use_mock:
            traci.close()

    def _read_vehicle_states(self) -> List[VehicleState]:
        vehicles = []
        for vid in self._traci.vehicle.getIDList():
            pos        = self._traci.vehicle.getPosition(vid)
            speed      = self._traci.vehicle.getSpeed(vid)
            lane_index = self._traci.vehicle.getLaneIndex(vid)
            lane_id_str = self._traci.vehicle.getLaneID(vid)

            if lane_id_str.startswith(self.RAMP_EDGE_PREFIX):
                our_lane_id = 1  
            else:
                our_lane_id = 0  
            vehicles.append(VehicleState(
                vehicle_id=hash(vid) % 10000,  
                position=pos[0],
                velocity=speed,
                lane_id=our_lane_id,
                acceleration=0.0,
            ))

        return vehicles

    def _vid_to_int(self, vid: str) -> int:
        mapping = {
            "inside_1": 10, "inside_2": 11,
            "outside_1": 20, "outside_2": 21,
            "ramp_1": 30,
        }
        return mapping.get(vid, hash(vid) % 10000)

    def _int_to_vid(self, vid_int: int) -> Optional[str]:
        reverse = {10: "inside_1", 11: "inside_2",
                   20: "outside_1", 21: "outside_2",
                   30: "ramp_1"}
        return reverse.get(vid_int)

    def read_vehicles(self) -> List[VehicleState]:
        result = []
        for vid in self._traci.vehicle.getIDList():
            pos         = self._traci.vehicle.getPosition(vid)
            speed       = self._traci.vehicle.getSpeed(vid)
            lane_id_str = self._traci.vehicle.getLaneID(vid)
            our_lane_id = 1 if lane_id_str.startswith(self.RAMP_EDGE_PREFIX) else 0

            result.append(VehicleState(
                vehicle_id=self._vid_to_int(vid),
                position=pos[0],
                velocity=speed,
                lane_id=our_lane_id,
            ))
        return result

    def send_speed_command(self, vehicle_id: int, target_speed: float):
        vid = self._int_to_vid(vehicle_id)
        if vid is None:
            return
        self._traci.vehicle.setSpeedMode(vid, 0)
        self._traci.vehicle.setSpeed(vid, target_speed)
        self._car_following_mode[vid] = False

    def send_lane_change_command(self, vehicle_id: int, target_lane: int, duration: float = 5.0):
        vid = self._int_to_vid(vehicle_id)
        if vid is None:
            return
        self._traci.vehicle.changeLane(vid, target_lane, duration)

    def restore_car_following(self, vehicle_id: int):
        vid = self._int_to_vid(vehicle_id)
        if vid is None:
            return
        self._traci.vehicle.setSpeedMode(vid, 31)
        self._traci.vehicle.setSpeed(vid, -1)  
        self._car_following_mode[vid] = True

    def step(self) -> dict:
        if self._traci is None:
            raise RuntimeError("connect() not called yet")

        vehicles = self.read_vehicles()
        current_time = self._traci.simulation.getTime()

        rsu_results = self.rsu_controller.update_all(vehicles, current_time)
        trigger_events = [ev for _, _, ev in rsu_results if ev is not None]

        if trigger_events:
            active_ids = self.rsu_controller.get_active_vehicles()
            self._cav_group = active_ids

        if self.use_mock:
            self._traci.step()         
        else:
            import traci as _traci
            _traci.simulationStep()    

        self._time += self.config.step_length

        return {
            "time": current_time,
            "vehicles": vehicles,
            "trigger_events": trigger_events,
            "active_cav_group": list(self._cav_group),
        }