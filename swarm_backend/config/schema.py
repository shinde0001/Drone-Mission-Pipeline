from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class FormationConfig(BaseModel):
    type: Literal["wedge", "line", "column"]
    spacing_m: float = Field(..., ge=1.0, le=50.0)
    angle_deg: float = Field(135.0, ge=10.0, le=180.0)
    frame: Literal["body_relative", "world_ned"] = "body_relative"

class SafetyConfig(BaseModel):
    collision_radius_m: float = Field(..., ge=0.5, le=10.0)
    min_separation_m: float = Field(..., ge=1.0, le=20.0)
    altitude_separation_m: float = Field(..., ge=0.0, le=10.0)
    avoidance_gain: float = Field(..., ge=0.0, le=5.0)
    max_correction_mps: float = Field(..., ge=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_safety_margins(self) -> 'SafetyConfig':
        if self.min_separation_m < 2 * self.collision_radius_m:
            raise ValueError("min_separation_m must be at least twice the collision_radius_m to avoid constant collision state")
        return self

class AgentConfig(BaseModel):
    id: str
    role: Literal["leader", "wingman_left", "wingman_right", "independent"]
    slot: Optional[int] = None

class ConnectionConfig(BaseModel):
    id: str
    udp: str

class Action(BaseModel):
    type: Literal["takeoff", "goto", "loiter", "land", "return_to_launch"]
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params(self) -> 'Action':
        atype = self.type
        p = self.params
        if atype == "takeoff":
            if "altitude_m" not in p:
                raise ValueError("takeoff action requires altitude_m param")
            if not isinstance(p["altitude_m"], (int, float)) or p["altitude_m"] <= 0:
                raise ValueError("altitude_m must be positive float/int")
        elif atype == "goto":
            for field_name in ("north_m", "east_m", "altitude_m", "speed_mps"):
                if field_name not in p:
                    raise ValueError(f"goto action requires {field_name} param")
                if not isinstance(p[field_name], (int, float)):
                    raise ValueError(f"{field_name} must be numeric")
            if p["altitude_m"] <= 0:
                raise ValueError("altitude_m must be positive")
            if p["speed_mps"] <= 0:
                raise ValueError("speed_mps must be positive")
        elif atype == "loiter":
            if "duration_s" not in p:
                raise ValueError("loiter action requires duration_s param")
            if not isinstance(p["duration_s"], (int, float)) or p["duration_s"] <= 0:
                raise ValueError("duration_s must be positive")
        return self

class LeaderMission(BaseModel):
    actions: List[Action]

    @field_validator("actions")
    @classmethod
    def validate_action_sequence(cls, v: List[Action]) -> List[Action]:
        if not v:
            raise ValueError("Mission must have at least one action")
        if v[0].type != "takeoff":
            raise ValueError("Mission must start with takeoff action")
        if v[-1].type not in ("land", "return_to_launch"):
            raise ValueError("Mission must end with land or return_to_launch action")
        return v

class Mission(BaseModel):
    mission_id: str
    mode: Literal["FORMATION", "INDEPENDENT", "REGROUP"]
    formation: FormationConfig
    safety: SafetyConfig
    agents: List[AgentConfig]
    leader_mission: LeaderMission
    connections: List[ConnectionConfig]

    @model_validator(mode="after")
    def validate_mission_cross_fields(self) -> 'Mission':
        # Check that we have a leader
        leader_agents = [a for a in self.agents if a.role == "leader"]
        if len(leader_agents) != 1:
            raise ValueError("Mission must have exactly one agent with role 'leader'")
        
        # Verify slot numbers for wingmen
        wingmen = [a for a in self.agents if a.role in ("wingman_left", "wingman_right")]
        for w in wingmen:
            if w.slot is None or w.slot <= 0:
                raise ValueError(f"Wingman agent '{w.id}' must have a positive slot assignment")

        # Spacing vs min_separation check
        if self.formation.spacing_m < self.safety.min_separation_m:
            raise ValueError(
                f"Formation spacing ({self.formation.spacing_m}m) must be greater than or equal to safety min_separation_m ({self.safety.min_separation_m}m)"
            )

        # Connection matching check
        agent_ids = {a.id for a in self.agents}
        conn_ids = {c.id for c in self.connections}
        if agent_ids != conn_ids:
            raise ValueError(f"Agent IDs {agent_ids} and Connection IDs {conn_ids} must match exactly")

        return self
