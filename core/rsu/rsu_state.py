from enum import Enum, auto

class VehicleZoneState(Enum):
    OUTSIDE  = auto()   
    ENTERING = auto()
    INSIDE   = auto()
    LEAVING  = auto()