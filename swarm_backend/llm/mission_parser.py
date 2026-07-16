import json
import os
import re
import time
import urllib.request
import subprocess
import shutil
from typing import Optional, Dict, Any, Tuple

from swarm_backend.config.schema import Mission

# Prompts optimized for small models (gemini-3.1-flash-lite / llama3.2:1b)
PROMPT_MODE_CLASSIFICATION = """You are a drone swarm command manager. Classify the intent into one of three operational modes:
- FORMATION: Drones fly together in a coordinated shape (wedge, line, column) following a leader's route.
- INDEPENDENT: Drones split up or take on independent tasks/routes.
- REGROUP: Drones rally or converge to a single location.

Output ONLY a valid JSON object:
{"cleaned_prompt": "<clear, concise restatement of the command>", "mode": "FORMATION|INDEPENDENT|REGROUP"}
"""

PROMPT_FORMATION_PARAMS = """You are a formation geometry planner. Determine the formation details:
- type: wedge, line, or column (default: wedge)
- spacing_m: distance between drones in meters (default: 5.0)
- angle_deg: angle of the formation in degrees (default: 135.0)
- frame: body_relative or world_ned (default: body_relative)

Output ONLY a valid JSON object:
{"type": "wedge|line|column", "spacing_m": 5.0, "angle_deg": 135.0, "frame": "body_relative"}
"""

PROMPT_LEADER_MISSION = """You are a drone mission waypoint generator. Generate the actions for the leader drone.
Rules:
1. Every mission must start with a "takeoff" action (params: altitude_m).
2. Every mission must end with a "land" or "return_to_launch" action.
3. Intermediate actions are "goto" (params: north_m, east_m, altitude_m, speed_mps) or "loiter" (params: duration_s).
4. Coordinates: NED offsets in meters from home. north_m=forward, east_m=right. altitude_m is height above ground.
5. Default altitude: 10m. Default speed: 5 m/s.

Output ONLY a JSON object:
{"actions": [
  {"type": "takeoff", "params": {"altitude_m": 15.0}},
  {"type": "goto", "params": {"north_m": 40.0, "east_m": 0.0, "altitude_m": 15.0, "speed_mps": 5.0}},
  {"type": "land", "params": {}}
]}
"""

def _setup_llm_client(ai_engine: str, api_key: Optional[str] = None) -> Tuple[Any, Any, str, bool]:
    """Setup and return (client, gemini_client, model_name, is_local)."""
    if ai_engine == "offline":
        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        is_running = False
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
            is_running = True
        except Exception:
            pass

        if not is_running:
            try:
                ollama_path = shutil.which("ollama")
                if ollama_path:
                    subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(5):
                        time.sleep(1.0)
                        try:
                            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                            is_running = True
                            break
                        except Exception:
                            pass
            except Exception:
                pass

        return None, None, model_name, True

    elif ai_engine.startswith("online"):
        if ai_engine == "online_lite":
            model_name = "gemini-3.1-flash-lite"
        elif ai_engine == "online_pro":
            model_name = "gemini-2.5-pro"
        else:
            model_name = "gemini-3.5-flash"

        from google import genai
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        gemini_client = genai.Client(api_key=actual_api_key)
        return None, gemini_client, model_name, False
    else:
        raise ValueError(f"Unknown ai_engine: {ai_engine}")

def _extract_json(raw_content: str) -> Optional[dict]:
    """Clean and parse JSON from LLM string output."""
    if not raw_content or not raw_content.strip():
        return None
    strategies = [
        raw_content.strip(),
    ]
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw_content, re.DOTALL)
    if fence_match:
        strategies.append(fence_match.group(1).strip())
    brace_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if brace_match:
        strategies.append(brace_match.group(0).strip())

    for cand in strategies:
        try:
            cleaned = re.sub(r'//.*', '', cand)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None

def _call_llm(system_prompt: str, user_prompt: str, client: Any, gemini_client: Any, model_name: str, is_local: bool, ai_engine: str) -> Optional[dict]:
    """Execute LLM request and extract JSON dictionary."""
    for attempt in range(3):
        try:
            if ai_engine.startswith("online"):
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                )
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config
                )
                raw = response.text
            else:
                import urllib.request
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    },
                    "format": "json"
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw = resp_data["message"]["content"]
            
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
        except Exception:
            time.sleep(0.5)
    return None

def parse_to_mission(prompt: str, ai_engine: str = "offline", api_key: Optional[str] = None, num_drones: int = 3) -> Mission:
    """
    Decompose a natural language prompt into a structured, validated Mission object.
    Uses multi-stage calling to ensure success on lower-capability models.
    """
    client, gemini_client, model_name, is_local = _setup_llm_client(ai_engine, api_key)

    # 1. Intent & Mode Classification
    res_mode = _call_llm(PROMPT_MODE_CLASSIFICATION, prompt, client, gemini_client, model_name, is_local, ai_engine)
    if res_mode:
        cleaned_prompt = res_mode.get("cleaned_prompt", prompt)
        mode = res_mode.get("mode", "FORMATION").upper()
    else:
        cleaned_prompt = prompt
        mode = "FORMATION"

    # 2. Formation Parameters
    res_form = _call_llm(PROMPT_FORMATION_PARAMS, cleaned_prompt, client, gemini_client, model_name, is_local, ai_engine)
    if res_form:
        formation_dict = {
            "type": res_form.get("type", "wedge").lower(),
            "spacing_m": float(res_form.get("spacing_m", 5.0)),
            "angle_deg": float(res_form.get("angle_deg", 135.0)),
            "frame": res_form.get("frame", "body_relative")
        }
    else:
        formation_dict = {
            "type": "wedge",
            "spacing_m": 5.0,
            "angle_deg": 135.0,
            "frame": "body_relative"
        }

    # 3. Leader Mission Action Sequence
    res_leader = _call_llm(PROMPT_LEADER_MISSION, cleaned_prompt, client, gemini_client, model_name, is_local, ai_engine)
    if res_leader and "actions" in res_leader:
        actions = res_leader["actions"]
    else:
        # Fallback minimal actions
        actions = [
            {"type": "takeoff", "params": {"altitude_m": 10.0}},
            {"type": "goto", "params": {"north_m": 20.0, "east_m": 0.0, "altitude_m": 10.0, "speed_mps": 5.0}},
            {"type": "land", "params": {}}
        ]

    # Preprocess actions to map synonyms or handle missing fields
    VALID_ACTIONS = {"takeoff", "goto", "loiter", "land", "return_to_launch"}
    cleaned_actions = []
    for act in actions:
        atype = act.get("type", "").lower()
        if atype not in VALID_ACTIONS:
            continue
        # Ensure params dict
        params = act.get("params", {})
        if not isinstance(params, dict):
            params = {}
        cleaned_actions.append({"type": atype, "params": params})

    # Ensure takeoff and landing constraints
    if not cleaned_actions or cleaned_actions[0]["type"] != "takeoff":
        cleaned_actions.insert(0, {"type": "takeoff", "params": {"altitude_m": 10.0}})
    if cleaned_actions[-1]["type"] not in ("land", "return_to_launch"):
        cleaned_actions.append({"type": "land", "params": {}})

    leader_mission_dict = {"actions": cleaned_actions}

    # Default safety values (Pydantic cross-validators require min_separation >= 2 * collision_radius)
    safety_dict = {
        "collision_radius_m": 1.0,
        "min_separation_m": 3.0,
        "altitude_separation_m": 1.0,
        "avoidance_gain": 0.6,
        "max_correction_mps": 2.0
    }

    # Adjust safety defaults if spacing is tighter than min_separation_m
    spacing = formation_dict["spacing_m"]
    if spacing < safety_dict["min_separation_m"]:
        safety_dict["min_separation_m"] = spacing
        safety_dict["collision_radius_m"] = spacing / 2.0

    # Build dynamic agents and connections lists based on num_drones
    agents_list = [{"id": "leader", "role": "leader"}]
    connections_list = [{"id": "leader", "udp": "udp://:14540"}]

    for idx in range(1, num_drones):
        role = "wingman_left" if idx % 2 == 1 else "wingman_right"
        slot_num = (idx + 1) // 2
        drone_id = f"drone_{idx}"
        agents_list.append({"id": drone_id, "role": role, "slot": slot_num})
        connections_list.append({"id": drone_id, "udp": f"udp://:{14540 + idx}"})

    assembled_json = {
        "mission_id": f"mission_{int(time.time())}",
        "mode": mode,
        "formation": formation_dict,
        "safety": safety_dict,
        "agents": agents_list,
        "leader_mission": leader_mission_dict,
        "connections": connections_list
    }

    # Validate using Pydantic
    return Mission(**assembled_json)
