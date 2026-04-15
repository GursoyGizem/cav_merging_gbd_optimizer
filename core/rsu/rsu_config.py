from dataclasses import dataclass

@dataclass
class RSUConfig:
    rsu_position: float       
    coverage_radius: float    
    trigger_distance: float   

    def __post_init__(self):
        if self.trigger_distance > self.coverage_radius:
            raise ValueError("Trigger distance cannot exceed coverage radius.")
        
        if self.coverage_radius <= 0 or self.trigger_distance <= 0:
            raise ValueError("Distance values must be positive.")