"""
swarm_backend/config/schema.py — Definitive Pydantic v2 contract for drone swarm missions.
Implements the spec (§4) with UserConstraints, typed Action parameters, Geofence validation,
and body-relative formation rules.
"""
from __future__ import annotations
import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, model_validator


class MissionMode(str, Enum):
    SINGLE = "SINGLE"
    FORMATION = "FORMATION"
    INDEPENDENT = "INDEPENDENT"
    MIXED = "MIXED"          # formation transit + independent sub-tasks + regroup
    REGROUP = "REGROUP"      # rally or converge to a single location


class FormationType(str, Enum):
    WEDGE = "wedge"
    LINE_ABREAST = "line_abreast"
    LINE = "line"            # alias/common term for line_abreast
    COLUMN = "column"
    DIAMOND = "diamond"


class AgentRole(str, Enum):
    LEADER = "leader"
    WINGMAN_LEFT = "wingman_left"
    WINGMAN_RIGHT = "wingman_right"
    WINGMAN = "wingman"
    INDEPENDENT = "independent"


class GeofenceCircle(BaseModel):
    shape: Literal["circle"] = "circle"
    center_lat: float
    center_lon: float
    radius_m: float = Field(gt=0)


class GeofenceBox(BaseModel):
    shape: Literal["box"] = "box"          # NED meters relative to mission home/origin
    min_north_m: float
    max_north_m: float
    min_east_m: float
    max_east_m: float

    @model_validator(mode="after")
    def check_order(self) -> GeofenceBox:
        if self.min_north_m >= self.max_north_m:
            raise ValueError("min_north_m must be < max_north_m")
        if self.min_east_m >= self.max_east_m:
            raise ValueError("min_east_m must be < max_east_m")
        return self


Geofence = Union[GeofenceCircle, GeofenceBox]


class UserConstraints(BaseModel):
    """OPERATOR-set constraints loaded from config/mission.json. Never produced by LLM."""
    max_altitude_m: float = Field(default=50.0, gt=0, le=120)
    min_altitude_m: float = Field(default=2.0, gt=0)
    max_speed_mps: float = Field(default=15.0, gt=0, le=20)
    max_agents: int = Field(default=5, ge=1, le=10)
    geofence: Optional[Geofence] = None
    min_separation_m: float = Field(default=3.0, gt=0)
    return_to_launch_on_violation: bool = True

    @model_validator(mode="after")
    def check_altitude_band(self) -> UserConstraints:
        if self.min_altitude_m >= self.max_altitude_m:
            raise ValueError("min_altitude_m must be < max_altitude_m")
        return self


def load_user_constraints(config_path: Optional[Union[str, Path]] = None) -> UserConstraints:
    """Load UserConstraints from config/mission.json, fallback to safety_limits.yaml."""
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "mission.json"
        yaml_path = project_root / "config" / "safety_limits.yaml"
    else:
        config_path = Path(config_path)
        yaml_path = None

    if config_path.exists() and config_path.suffix == ".json":
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserConstraints.model_validate(data)
        except Exception as e:
            pass

    # Fallback to safety_limits.yaml if mission.json not available or failed
    if yaml_path and yaml_path.exists():
        import yaml
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f) or {}
            cdata = {
                "max_altitude_m": float(ydata.get("max_altitude_m", 50.0)),
                "min_altitude_m": float(ydata.get("min_altitude_m", 2.0)),
                "max_speed_mps": float(ydata.get("max_speed_mps", 15.0)),
                "min_separation_m": float(ydata.get("min_waypoint_spacing_m", 3.0)),
            }
            if "geofence_radius_m" in ydata:
                r = float(ydata["geofence_radius_m"])
                cdata["geofence"] = {
                    "shape": "box",
                    "min_north_m": -r, "max_north_m": r,
                    "min_east_m": -r, "max_east_m": r
                }
            return UserConstraints.model_validate(cdata)
        except Exception:
            pass

    return UserConstraints()


class TakeoffParams(BaseModel):
    altitude_m: float = Field(gt=0)


class GotoParams(BaseModel):
    north_m: float
    east_m: float
    altitude_m: float = Field(gt=0)
    speed_mps: float = Field(gt=0)


class LoiterParams(BaseModel):
    duration_s: float = Field(gt=0, default=5.0)


class LandParams(BaseModel):
    pass


class SweepParams(BaseModel):
    corner_a_north_m: float
    corner_a_east_m: float
    corner_b_north_m: float
    corner_b_east_m: float
    altitude_m: float = Field(gt=0)
    lane_spacing_m: float = Field(gt=0, default=5.0)
    speed_mps: float = Field(gt=0)


class RegroupParams(BaseModel):
    timeout_s: float = Field(gt=0, default=60.0)


_PARAMS_BY_TYPE = {
    "takeoff": TakeoffParams,
    "goto": GotoParams,
    "loiter": LoiterParams,
    "land": LandParams,
    "sweep": SweepParams,
    "regroup": RegroupParams,
    "return_to_launch": LandParams
}


class Action(BaseModel):
    type: Literal["takeoff", "goto", "loiter", "land", "return_to_launch", "sweep", "regroup"]
    params: Union[TakeoffParams, GotoParams, LoiterParams, LandParams, SweepParams, RegroupParams]

    @model_validator(mode="before")
    @classmethod
    def coerce_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            t = data.get("type")
            p = data.get("params", {})
            target = _PARAMS_BY_TYPE.get(t)
            if target is not None and isinstance(p, dict):
                data = {**data, "params": target(**p)}
        return data


class LeaderMission(BaseModel):
    actions: List[Action] = Field(min_length=1)

    @model_validator(mode="after")
    def check_takeoff_land(self) -> LeaderMission:
        if not self.actions or self.actions[0].type != "takeoff":
            raise ValueError("Mission must start with takeoff action")
        if self.actions[-1].type not in ("land", "return_to_launch"):
            raise ValueError("Mission must end with land or return_to_launch action")
        return self


class FormationConfig(BaseModel):
    type: FormationType
    spacing_m: float = Field(gt=0)
    angle_deg: float = Field(default=135.0, ge=0, le=360)
    frame: Literal["body_relative"] = "body_relative"  # world-frame is not a valid option per Rule #1


class SafetyConfig(BaseModel):
    collision_radius_m: float = Field(gt=0, default=1.0)
    min_separation_m: float = Field(gt=0, default=3.0)
    altitude_separation_m: float = Field(default=1.0, ge=0)
    avoidance_gain: float = Field(gt=0, default=0.6)
    max_correction_mps: float = Field(gt=0, default=2.0)

    @model_validator(mode="after")
    def sanity(self) -> SafetyConfig:
        if self.min_separation_m < 2.0 * self.collision_radius_m:
            raise ValueError("min_separation_m must be at least twice the collision_radius_m")
        return self


class ConnectionConfig(BaseModel):
    id: str
    udp: str


class AgentConfig(BaseModel):
    id: str
    role: AgentRole
    slot: Optional[int] = None
    udp: Optional[str] = None
    independent_actions: Optional[List[Action]] = None


class Mission(BaseModel):
    mission_id: str
    mode: MissionMode
    formation: Optional[FormationConfig] = None
    safety: SafetyConfig
    agents: List[AgentConfig] = Field(min_length=1)
    leader_mission: LeaderMission
    connections: List[ConnectionConfig] = Field(default_factory=list)
    constraints: UserConstraints = Field(default_factory=load_user_constraints)

    @model_validator(mode="after")
    def cross_checks(self) -> Mission:
        if self.mode in (MissionMode.FORMATION, MissionMode.MIXED) and self.formation is None:
            raise ValueError(f"mode={self.mode} requires a 'formation' block")
        if len(self.agents) > self.constraints.max_agents:
            raise ValueError(f"{len(self.agents)} agents exceeds max_agents={self.constraints.max_agents}")
        leaders = [a for a in self.agents if a.role == AgentRole.LEADER]
        if len(leaders) != 1:
            raise ValueError(f"exactly one agent with role 'leader' required, found {len(leaders)}")
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate agent ids")
        if self.connections:
            conn_ids = [c.id for c in self.connections]
            if set(conn_ids) != set(ids) or len(conn_ids) != len(ids):
                raise ValueError("connections IDs must match exactly the agents IDs")
        if self.formation and self.formation.spacing_m < self.safety.min_separation_m:
            raise ValueError("Formation spacing (formation.spacing_m) must be greater than or equal to safety min_separation_m")
        for actions in [self.leader_mission.actions] + [a.independent_actions for a in self.agents if a.independent_actions]:
            for action in actions:
                self._check_envelope(action)
        return self

    def _check_envelope(self, action: Action) -> None:
        p, c = action.params, self.constraints
        alt = getattr(p, "altitude_m", None)
        if alt is not None:
            if alt > c.max_altitude_m:
                raise ValueError(f"altitude {alt}m exceeds max_altitude_m={c.max_altitude_m}")
            if alt < c.min_altitude_m:
                raise ValueError(f"altitude {alt}m below min_altitude_m={c.min_altitude_m}")
        speed = getattr(p, "speed_mps", None)
        if speed is not None and speed > c.max_speed_mps:
            raise ValueError(f"speed {speed}m/s exceeds max_speed_mps={c.max_speed_mps}")
        north, east = getattr(p, "north_m", None), getattr(p, "east_m", None)
        if north is not None and east is not None and c.geofence is not None:
            if not self._point_in_geofence(north, east):
                raise ValueError(f"target ({north},{east}) is outside the configured geofence")

    def _point_in_geofence(self, north_m: float, east_m: float) -> bool:
        gf = self.constraints.geofence
        if isinstance(gf, GeofenceBox):
            return (gf.min_north_m <= north_m <= gf.max_north_m) and (gf.min_east_m <= east_m <= gf.max_east_m)
        return True  # GeofenceCircle needs home lat/lon projection


def parse_mission(raw: dict) -> Mission:
    return Mission.model_validate(raw)
