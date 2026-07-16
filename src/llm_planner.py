"""
llm_planner.py — Stage 1: LLM Intent Interpretation → Mission JSON

This module takes a natural-language prompt from the operator and uses
OpenAI GPT-4o to interpret the intent, then emits a structured mission
JSON that conforms to the mission schema.

KEY PRINCIPLE: The LLM *proposes* a plan. It never directly controls the
drone. The output goes through validation (Stage 2) before execution.
"""

import json
import os
import re
import time
import threading
import subprocess
import shutil
from pathlib import Path

from .utils import (
    load_mission_schema,
    load_waypoint_library,
    load_safety_limits,
    print_info,
    print_warning,
    print_error,
    print_success,
    setup_logger,
)
from swarm_backend.config.schema import (
    load_user_constraints,
    Mission,
    MissionMode,
    FormationType,
    AgentRole,
    Action,
)
from swarm_backend.llm.prompts import (
    SWARM_CLASSIFY_PROMPT,
    SWARM_LEADER_ACTIONS_PROMPT,
    SWARM_FORMATION_PARAMS_PROMPT,
    SWARM_FOLLOWER_WAYPOINTS_PROMPT,
    SWARM_REPAIR_PROMPT,
)
from swarm_backend.core.task_manager import generate_lawn_mower_waypoints

logger = setup_logger("llm_planner")

# ── Ollama Auto-Shutdown Watchdog ───────────────────────────────────────────
_last_ollama_use_time = 0
_ollama_watchdog_thread = None

def _ollama_watchdog():
    """Background thread that kills Ollama if inactive for > 60 seconds."""
    global _last_ollama_use_time
    while True:
        time.sleep(10)
        # Check if Ollama was used, and if 60s has passed since last use
        if _last_ollama_use_time > 0 and (time.time() - _last_ollama_use_time) > 60:
            try:
                # Use pkill to shut down the ollama serve process
                subprocess.run(["pkill", "-f", "ollama serve"], check=False)
                logger.info("Ollama inactive for > 60s. Auto-stopped to save CPU/RAM.")
                print_info("Ollama inactive for > 60s. Auto-stopped to save CPU/RAM.")
            except Exception as e:
                logger.error(f"Failed to auto-stop Ollama: {e}")
            _last_ollama_use_time = 0  # Reset so we don't spam kills

def _touch_ollama():
    """Update the last used timestamp and start the watchdog if needed."""
    global _last_ollama_use_time, _ollama_watchdog_thread
    _last_ollama_use_time = time.time()
    
    if _ollama_watchdog_thread is None or not _ollama_watchdog_thread.is_alive():
        _ollama_watchdog_thread = threading.Thread(target=_ollama_watchdog, daemon=True)
        _ollama_watchdog_thread.start()
        logger.info("Started Ollama inactivity watchdog thread.")

# ── System prompt template (full — for capable models like Gemini) ──────
SYSTEM_PROMPT = """You are a drone mission planner AI. Your job is to interpret 
natural-language commands from a human operator and produce a structured mission 
plan in JSON format.

## Your Role
- You PROPOSE a flight plan. You do NOT fly the drone.
- Your output will be validated against safety rules before execution.
- Be conservative: prefer lower altitudes and slower speeds when unspecified.
- Think geometrically: if the user asks to fly a letter shape, a pattern, or a path,
  calculate the waypoints (north_m, east_m) that trace that shape in the NED coordinate plane.

## Available Action Types
- "takeoff": Take off to a specified altitude. Params: altitude_m
- "goto": Fly to a waypoint. Params: north_m, east_m, altitude_m, speed_mps
- "loiter": Hover at current position. Params: duration_s
- "land": Land at current position. Params: (none needed)
- "return_to_launch": Return to the takeoff point and land. Params: (none needed)

## Coordinate System
All positions use NED (North-East-Down) offsets from the home/launch position:
- north_m: positive = north (up on map), negative = south (down on map)
- east_m: positive = east (right on map), negative = west (left on map)
- altitude_m: height above ground in metres (always positive)

## Safety Limits
{safety_limits}

## Mission Structure Rules
1. Every mission MUST start with a "takeoff" action
2. Every mission MUST end with either "land" or "return_to_launch"
3. Include at least one "goto" or "loiter" action between takeoff and landing
4. If the operator says "repeat N times" or "loop N times", set repeat_count=N
5. Default altitude: 10m if not specified
6. Default speed: 5 m/s if not specified
7. For shape/letter/pattern commands: calculate specific (north_m, east_m) waypoints that draw the shape

## Output Format
Return ONLY a valid JSON object. Do NOT include any text, explanation, or markdown fences.
"""

SYSTEM_PROMPT_SWARM = """You are a drone swarm mission planner AI. Your job is to interpret 
natural-language commands for a squad of {num_drones} drones.
You must produce a squad-level mission JSON.

Return ONLY a valid JSON object matching this structure:
{{
  "mission_name": "<descriptive name>",
  "vehicle_type": "swarm",
  "formation": "<wedge, line, or column>",
  "drone_count": {num_drones},
  "repeat_count": <repeat count>,
  "actions": [
    {{"type": "takeoff", "params": {{"altitude_m": <altitude>}}}},
    {{"type": "goto", "params": {{"north_m": <n>, "east_m": <e>, "altitude_m": <a>, "speed_mps": <s>}}}},
    {{"type": "land", "params": {{}}}}
  ]
}}
Do NOT include any text before or after the JSON. Output ONLY the JSON object.
"""

# Old prompt definitions superseded by swarm_backend/llm/prompts.py

# ── Compact system prompt (for small local models like llama3.2:1b) ────
SYSTEM_PROMPT_COMPACT = """You are a drone mission planner. Convert the user command into a JSON mission plan.

Action types: takeoff (altitude_m), goto (north_m, east_m, altitude_m, speed_mps), loiter (duration_s), land (), return_to_launch ().
Coordinates: NED offsets in metres from home. north_m=forward, east_m=right. altitude_m is always positive.

Limits: max altitude {max_alt}m, max speed {max_speed}m/s, geofence {geofence}m.

Rules:
1. Start with takeoff, end with land or return_to_launch. Default altitude=10m, speed=5m/s.
2. For shapes/letters/patterns: calculate (north_m, east_m) waypoints that trace the shape.
3. Return ONLY valid JSON. No text, no explanation.
"""

# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES — These teach the LLM the exact input→output format
# Each example is a (user_prompt, assistant_json_response) pair.
# The LLM sees these as prior conversation turns before the real prompt.
# ══════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES_FULL = [
    # Example 1: Letter shape — demonstrates geometric waypoint calculation
    {
        "user": "fly and patrol by path of letter P at 12 metres",
        "assistant": json.dumps({
            "mission_name": "Letter P Flight Path",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 12}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 0, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 0, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 10, "east_m": 10, "altitude_m": 12, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 10, "east_m": 0, "altitude_m": 12, "speed_mps": 5}},
                {"type": "return_to_launch", "params": {}}
            ]
        }, indent=2)
    },
    # Example 2: Square pattern with repeat — demonstrates multi-loop
    {
        "user": "fly to 15 metres, do a square patrol of 30m size 2 times, then return to launch",
        "assistant": json.dumps({
            "mission_name": "Square Patrol 30m",
            "vehicle_type": "quadcopter",
            "repeat_count": 2,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 15}},
                {"type": "goto", "params": {"north_m": 30, "east_m": 0, "altitude_m": 15, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 30, "east_m": 30, "altitude_m": 15, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 30, "altitude_m": 15, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 0, "altitude_m": 15, "speed_mps": 5}},
                {"type": "return_to_launch", "params": {}}
            ]
        }, indent=2)
    },
    # Example 3: Simple goto + loiter — demonstrates hover/wait
    {
        "user": "take off to 8m, go north 20m and east 10m, hover for 10 seconds, then land",
        "assistant": json.dumps({
            "mission_name": "Goto And Hover",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 8}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 8, "speed_mps": 5}},
                {"type": "loiter", "params": {"duration_s": 10}},
                {"type": "land", "params": {}}
            ]
        }, indent=2)
    },
]

# Compact examples for small local models (shorter to save context window)
FEW_SHOT_EXAMPLES_COMPACT = [
    # Example 1: Letter shape
    {
        "user": "fly path of letter L at 10m",
        "assistant": json.dumps({
            "mission_name": "Letter L Path",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 10}},
                {"type": "goto", "params": {"north_m": 20, "east_m": 0, "altitude_m": 10, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 0, "altitude_m": 10, "speed_mps": 5}},
                {"type": "goto", "params": {"north_m": 0, "east_m": 15, "altitude_m": 10, "speed_mps": 5}},
                {"type": "return_to_launch", "params": {}}
            ]
        })
    },
    # Example 2: Simple movement
    {
        "user": "takeoff 8m, fly north 15m east 10m, hover 5 seconds, land",
        "assistant": json.dumps({
            "mission_name": "Goto Hover Land",
            "vehicle_type": "quadcopter",
            "repeat_count": 1,
            "actions": [
                {"type": "takeoff", "params": {"altitude_m": 8}},
                {"type": "goto", "params": {"north_m": 15, "east_m": 10, "altitude_m": 8, "speed_mps": 5}},
                {"type": "loiter", "params": {"duration_s": 5}},
                {"type": "land", "params": {}}
            ]
        })
    },
]

def _build_system_prompt(compact: bool = False, mode: str = "standard", num_drones: int = 3) -> str:
    """Build the full system prompt with current config data.
    
    Args:
        compact: If True, use the shorter prompt suitable for small
                 local models with limited context windows.
        mode: Operating mode (standard, swarm)
        num_drones: Number of drones in swarm mode
    """
    if mode == "swarm":
        return SYSTEM_PROMPT_SWARM.format(num_drones=num_drones)

    safety = load_safety_limits()

    if compact:
        return SYSTEM_PROMPT_COMPACT.format(
            max_alt=safety['max_altitude_m'],
            max_speed=safety['max_speed_mps'],
            geofence=safety['geofence_radius_m'],
        )

    # Format safety limits for the prompt
    safety_lines = [
        f"- Maximum altitude: {safety['max_altitude_m']}m",
        f"- Maximum speed: {safety['max_speed_mps']} m/s",
        f"- Geofence radius: {safety['geofence_radius_m']}m from home",
        f"- Maximum loiter duration: {safety['max_loiter_duration_s']}s",
        f"- Maximum repeat count: {safety['max_repeat_count']}",
        f"- Maximum actions per mission: {safety['max_actions_per_mission']}",
    ]

    return SYSTEM_PROMPT.format(
        safety_limits="\n".join(safety_lines),
    )


def _build_few_shot_messages(system_prompt: str, user_prompt: str, compact: bool = False, mode: str = "standard", num_drones: int = 3) -> list:
    """Build the full message list with few-shot examples for the LLM.
    
    Structure sent to the LLM:
      1. System message (rules + format)
      2. Example user prompt #1
      3. Example assistant response #1 (correct JSON)
      4. Example user prompt #2
      5. Example assistant response #2 (correct JSON)
      ...
      N. Actual user prompt (the real command)
    
    This teaches the LLM by showing it concrete input→output pairs
    before it processes the real user command.
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    if mode == "swarm":
        examples = [
            {
                "user": f"{num_drones} drones fly in a wedge formation to north 20m, east 10m at altitude 15m, then land",
                "assistant": json.dumps({
                    "mission_name": f"{num_drones} Drone Wedge Formation",
                    "vehicle_type": "swarm",
                    "formation": "wedge",
                    "drone_count": num_drones,
                    "repeat_count": 1,
                    "actions": [
                        {"type": "takeoff", "params": {"altitude_m": 15}},
                        {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 15, "speed_mps": 5}},
                        {"type": "land", "params": {}}
                    ]
                })
            }
        ]
    else:
        examples = FEW_SHOT_EXAMPLES_COMPACT if compact else FEW_SHOT_EXAMPLES_FULL
    
    for example in examples:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    
    # Finally, add the actual user prompt
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def plan_mission(prompt: str, mode: str = "standard", ai_engine: str = "offline", api_key: str = None, num_drones: int = 3) -> dict:
    """
    Stage 1: Interpret a natural-language command and produce mission JSON.

    Args:
        prompt: Natural-language instruction from the operator.
        mode: The operating mode (standard, swarm)
        ai_engine: 'offline' or 'online'
        api_key: Optional API key for online engines.
        num_drones: Number of drones in swarm mode

    Returns:
        dict: Raw mission JSON (not yet validated).

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
        RuntimeError: If the Ollama API call fails or is unavailable.
    """
    # Pre-process the prompt to help the small model map directions to NED coordinates
    # Small models often get confused and put forward distances into altitude_m
    import re
    p_lower = prompt.lower()
    p_lower = re.sub(r'\b(?:forward|front)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'north_m \1', p_lower)
    p_lower = re.sub(r'\b(?:right)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'east_m \1', p_lower)
    p_lower = re.sub(r'\b(?:back|backward|reverse)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'north_m -\1', p_lower)
    p_lower = re.sub(r'\b(?:left)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)?\b', r'east_m -\1', p_lower)
    # Fix common typos and grammar issues that confuse the small model
    p_lower = p_lower.replace("tlenght", "length")
    p_lower = p_lower.replace("lenght", "length")
    prompt = p_lower
    
    if mode == "swarm":
        return plan_swarm_mission(prompt, ai_engine=ai_engine, api_key=api_key, num_drones=num_drones)

    if ai_engine == "offline":
        # Ensure Ollama is running first (attempt to auto-start if not)
        import urllib.request
        import subprocess
        import time
        import shutil

        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

        def is_ollama_running():
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                return True
            except Exception:
                return False

        def is_ollama_model_available(model):
            try:
                req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                data = json.loads(req.read().decode())
                for m in data.get("models", []):
                    if m.get("name") == model or m.get("name") == model + ":latest":
                        return True
                return False
            except Exception:
                return False

        if not is_ollama_running():
            print_warning("Ollama is not running. Attempting to start it automatically...")
            logger.warning("Ollama is not running. Attempting to start it automatically...")
            try:
                ollama_path = shutil.which("ollama")
                if ollama_path:
                    subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for i in range(10):
                        time.sleep(1.0)
                        if is_ollama_running():
                            print_success("Ollama started successfully.")
                            logger.info("Ollama started successfully.")
                            break
                else:
                    logger.error("ollama command not found in PATH")
            except Exception as e:
                logger.error(f"Failed to start Ollama automatically: {e}")

        if not is_ollama_running():
            raise RuntimeError(
                "Ollama server is not running or unreachable on http://127.0.0.1:11434. "
                "Please start it manually using 'ollama serve'."
            )

        if not is_ollama_model_available(model_name):
            print_warning(f"Ollama model '{model_name}' not found locally. Pulling it now... This may take a while.")
            logger.warning(f"Ollama model '{model_name}' not found. Pulling...")
            try:
                ollama_path = shutil.which("ollama")
                subprocess.run([ollama_path, "pull", model_name], check=True)
                print_success(f"Model '{model_name}' pulled successfully.")
                logger.info(f"Model '{model_name}' pulled successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to pull Ollama model '{model_name}': {e}")

    if ai_engine.startswith("online"):
        if ai_engine == "online_lite":
            print_info("Using online LLM via Gemini 3.1 Flash Lite.")
            logger.info("Using online LLM via Gemini 3.1 Flash Lite.")
            model_name = "gemini-3.1-flash-lite"
        elif ai_engine == "online_pro":
            print_info("Using online LLM via Gemini 2.5 Pro.")
            logger.info("Using online LLM via Gemini 2.5 Pro.")
            model_name = "gemini-2.5-pro"
        else:
            print_info("Using online LLM via Gemini 3.5 Flash.")
            logger.info("Using online LLM via Gemini 3.5 Flash.")
            model_name = "gemini-3.5-flash"
            
        from google import genai
        from google.genai import types
        # Prefer the explicitly passed api_key, fallback to environment variable
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please provide it in the UI or set the environment variable.")
        gemini_client = genai.Client(api_key=actual_api_key)
        is_local = False
    else:
        print_info("Using local LLM via Ollama.")
        logger.info("Using local LLM via Ollama.")
        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        is_local = True

    # Use compact prompt for small local models to fit in limited context
    system_prompt = _build_system_prompt(compact=is_local, mode=mode, num_drones=num_drones)
    
    # Build few-shot messages (system + example pairs + real user prompt)
    # This teaches the LLM by showing concrete input→output examples
    few_shot_messages = _build_few_shot_messages(system_prompt, prompt, compact=is_local, mode=mode, num_drones=num_drones)
    
    print_info(f"Sending prompt to {model_name}: \"{prompt}\"")
    logger.info(f"Calling {model_name} for mission planning (with {len(few_shot_messages)//2} few-shot examples)...")

    if is_local:
        _touch_ollama()

    # Retry logic — small local models may need a few attempts
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print_warning(f"Retry attempt {attempt}/{max_attempts}...")
            logger.warning(f"Retry attempt {attempt}/{max_attempts}")

        try:
            if ai_engine.startswith("online"):
                # For Gemini: convert few-shot messages into multi-turn conversation
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                )
                # Build Gemini conversation history from few-shot examples
                if mode == "swarm":
                    examples = [
                        {
                            "user": f"{num_drones} drones fly in a wedge formation to north 20m, east 10m at altitude 15m, then land",
                            "assistant": json.dumps({
                                "mission_name": f"{num_drones} Drone Wedge Formation",
                                "vehicle_type": "swarm",
                                "formation": "wedge",
                                "drone_count": num_drones,
                                "repeat_count": 1,
                                "actions": [
                                    {"type": "takeoff", "params": {"altitude_m": 15}},
                                    {"type": "goto", "params": {"north_m": 20, "east_m": 10, "altitude_m": 15, "speed_mps": 5}},
                                    {"type": "land", "params": {}}
                                ]
                            })
                        }
                    ]
                else:
                    examples = FEW_SHOT_EXAMPLES_FULL
                
                gemini_history = []
                for ex in examples:
                    gemini_history.append({"role": "user", "parts": [{"text": ex["user"]}]})
                    gemini_history.append({"role": "model", "parts": [{"text": ex["assistant"]}]})
                
                chat = gemini_client.chats.create(model=model_name, config=config, history=gemini_history)
                response = chat.send_message(prompt)
                raw_content = response.text
            else:
                # For Ollama: query directly via HTTP POST
                import urllib.request
                payload = {
                    "model": model_name,
                    "messages": few_shot_messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    "format": "json"
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_content = resp_data["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt == max_attempts:
                raise RuntimeError(f"{model_name} API call failed: {e}") from e
            continue

        logger.info(f"Received response from {model_name}")
        logger.debug(f"Raw response: {raw_content[:500]}")
        
        if is_local:
            _touch_ollama() # Touch again after completion so it has a full 60s after generation ends

        # Parse the JSON response — for local models, extract JSON from
        # potentially noisy output (small models often add explanatory text)
        mission = _extract_mission_json(raw_content, is_local, mode=mode)
        if mission is not None:
            logger.info(
                f"Mission planned: '{mission.get('mission_name', 'unnamed')}' "
                f"with {len(mission['actions'])} actions"
            )
            return mission

        last_error = ValueError(
            f"Could not extract valid mission JSON from Ollama response "
            f"(attempt {attempt}): {raw_content[:300]}"
        )
        print_warning(f"Attempt {attempt}: Ollama did not return valid mission JSON.")
        print_info(f"Ollama raw response was:\n{raw_content}")

    raise last_error


def _extract_mission_json(raw_content: str, is_local: bool, mode: str = "standard") -> dict | None:
    """
    Extract a valid mission JSON dict from LLM output.

    Small local models often wrap JSON in explanatory text, markdown
    code fences, or produce minor formatting issues.  This function
    tries multiple strategies to pull out usable JSON.

    Returns the parsed dict if successful, or None if no valid mission
    JSON could be extracted.
    """
    if not raw_content or not raw_content.strip():
        return None

    strategies = []

    # Strategy 1: Direct parse (works for well-behaved models)
    strategies.append(raw_content.strip())

    # Strategy 2: Extract from markdown code fence ```json ... ```
    fence_match = re.search(
        r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw_content, re.DOTALL
    )
    if fence_match:
        strategies.append(fence_match.group(1).strip())

    # Strategy 3: Find the outermost { ... } block
    brace_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if brace_match:
        strategies.append(brace_match.group(0).strip())

    # Strategy 4: Fallback for missing opening brace
    missing_brace_match = re.search(r'"mission_name"\s*:', raw_content)
    if missing_brace_match:
        start_idx = missing_brace_match.start()
        fixed_json = "{\n" + raw_content[start_idx:]
        if not fixed_json.strip().endswith("}"):
            fixed_json = fixed_json.rstrip() + "\n}"
        strategies.append(fixed_json)

    for candidate in strategies:
        try:
            # Clean single-line and multi-line comments
            cleaned = re.sub(r'//.*', '', candidate)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            # Clean trailing commas inside lists or objects
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(parsed, dict):
            continue

        if mode != "standard":
            # For advanced modes, if it has actions, return it without standard cleaning
            if "actions" in parsed and isinstance(parsed["actions"], list):
                return parsed

        # Must have an 'actions' list to be a valid mission
        if "actions" in parsed and isinstance(parsed["actions"], list):
            # Fill in defaults if the small model omitted them
            parsed.setdefault("mission_name", "llm_planned_mission")
            parsed.setdefault("vehicle_type", "quadcopter")
            parsed.setdefault("repeat_count", 1)

            # Post-process actions — fix common small-model mistakes

            # ── Synonym mapping for action types ──
            # Small models often invent action names; map them to valid ones
            ACTION_SYNONYMS = {
                # takeoff synonyms
                "up": "takeoff", "ascend": "takeoff", "launch": "takeoff",
                "take_off": "takeoff", "liftoff": "takeoff", "lift_off": "takeoff",
                "rise": "takeoff",
                # loiter / hover / wait synonyms
                "hover": "loiter", "wait": "loiter", "hold": "loiter",
                "stay": "loiter", "pause": "loiter", "delay": "loiter",
                "stop": "loiter", "idle": "loiter", "remain": "loiter",
                # goto / move / fly synonyms
                "forward": "goto", "move": "goto", "fly": "goto",
                "go": "goto", "travel": "goto", "navigate": "goto",
                "waypoint": "goto", "move_to": "goto", "fly_to": "goto",
                "go_to": "goto", "advance": "goto",
                # return_to_launch synonyms
                "return_to_home": "return_to_launch", "rtl": "return_to_launch",
                "return_home": "return_to_launch", "home": "return_to_launch",
                "rth": "return_to_launch", "go_home": "return_to_launch",
                "return": "return_to_launch",
                # land synonyms
                "descend": "land", "touch_down": "land", "touchdown": "land",
                "down": "land",
            }
            VALID_ACTIONS = {"takeoff", "goto", "loiter", "land", "return_to_launch"}

            for action in parsed["actions"]:
                # Remap action type via synonyms
                raw_type = action.get("type", "").lower().strip()
                if raw_type in ACTION_SYNONYMS:
                    action["type"] = ACTION_SYNONYMS[raw_type]

                # Ensure every action has a 'params' dict
                if "params" not in action:
                    action["params"] = {}

                # Move any top-level keys (other than type/params) into params
                extra_keys = [k for k in action if k not in ("type", "params")]
                for k in extra_keys:
                    if k not in action["params"]:
                        action["params"][k] = action.pop(k)
                    else:
                        action.pop(k)

                # ── Per-action param normalization ──
                atype = action["type"]
                params = action["params"]

                if atype == "loiter":
                    # Normalize duration keys: sec, seconds, time, time_s → duration_s
                    for alias in ("sec", "seconds", "time", "time_s", "wait_s"):
                        if alias in params and "duration_s" not in params:
                            params["duration_s"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    params.setdefault("duration_s", 5)
                    # Schema minimum is 1s; clamp invalid values
                    if params["duration_s"] < 1:
                        params["duration_s"] = 5

                elif atype == "goto":
                    # Normalize distance-only params → north_m
                    for alias in ("m", "distance", "distance_m", "forward_m",
                                  "front_m", "forward", "dist"):
                        if alias in params and "north_m" not in params:
                            params["north_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    # Normalize other direction aliases
                    for alias in ("right_m", "lateral_m"):
                        if alias in params and "east_m" not in params:
                            params["east_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    for alias in ("height_m", "alt_m", "alt", "height"):
                        if alias in params and "altitude_m" not in params:
                            params["altitude_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    for alias in ("speed", "velocity", "speed_m_s"):
                        if alias in params and "speed_mps" not in params:
                            params["speed_mps"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    # Ensure required goto params exist
                    params.setdefault("north_m", 0)
                    params.setdefault("east_m", 0)
                    params.setdefault("altitude_m", 10)
                    params.setdefault("speed_mps", 5)

                elif atype == "takeoff":
                    for alias in ("height_m", "alt_m", "alt", "height"):
                        if alias in params and "altitude_m" not in params:
                            params["altitude_m"] = params.pop(alias)
                        elif alias in params:
                            params.pop(alias)
                    params.setdefault("altitude_m", 10)

            # ── Final sanitization: strip unexpected params ──
            # The schema only allows specific params per action type.
            # Remove anything else so validation never fails on extra keys.
            ALLOWED_PARAMS = {
                "takeoff":          {"altitude_m"},
                "goto":             {"north_m", "east_m", "altitude_m", "speed_mps"},
                "loiter":           {"duration_s"},
                "land":             set(),
                "return_to_launch": set(),
            }
            for action in parsed["actions"]:
                allowed = ALLOWED_PARAMS.get(action["type"], set())
                unexpected = [k for k in action["params"] if k not in allowed]
                for k in unexpected:
                    action["params"].pop(k)

            # Remove duplicate consecutive takeoffs (model sometimes adds one
            # on top of our auto-inserted takeoff)
            cleaned = []
            for i, action in enumerate(parsed["actions"]):
                if action["type"] == "takeoff" and i > 0 and cleaned and cleaned[-1]["type"] == "takeoff":
                    # Keep the one with the higher altitude
                    if action["params"].get("altitude_m", 0) > cleaned[-1]["params"].get("altitude_m", 0):
                        cleaned[-1] = action
                    continue
                # Skip unknown action types that we couldn't remap
                if action["type"] not in VALID_ACTIONS:
                    continue
                cleaned.append(action)
            parsed["actions"] = cleaned

            # If all actions were stripped, this candidate is unusable
            if not parsed["actions"]:
                continue

            # Ensure the mission starts with takeoff
            if parsed["actions"][0].get("type") != "takeoff":
                takeoff_alt = 10
                for action in parsed["actions"]:
                    if "altitude_m" in action.get("params", {}):
                        takeoff_alt = action["params"]["altitude_m"]
                        break
                parsed["actions"].insert(0, {"type": "takeoff", "params": {"altitude_m": takeoff_alt}})

            # Remove any extra land/RTL commands at the end so there is only one
            while len(parsed["actions"]) > 1 and parsed["actions"][-1].get("type") in ("land", "return_to_launch") and parsed["actions"][-2].get("type") in ("land", "return_to_launch"):
                parsed["actions"].pop(-2)
                
            # Ensure the mission ends with land or RTL
            if parsed["actions"][-1].get("type") not in ("land", "return_to_launch"):
                parsed["actions"].append({"type": "land", "params": {}})

            return parsed

    return None


def _setup_llm_client(ai_engine: str, api_key: str = None):
    """Setup and return (client, gemini_client, model_name, is_local)."""
    import urllib.request
    import subprocess
    import time
    import shutil

    if ai_engine == "offline":
        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

        def is_ollama_running():
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                return True
            except Exception:
                return False

        def is_ollama_model_available(model):
            try:
                req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
                data = json.loads(req.read().decode())
                for m in data.get("models", []):
                    if m.get("name") == model or m.get("name") == model + ":latest":
                        return True
                return False
            except Exception:
                return False

        if not is_ollama_running():
            logger.warning("Ollama is not running. Attempting to start it automatically...")
            try:
                ollama_path = shutil.which("ollama")
                if ollama_path:
                    subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(10):
                        time.sleep(1.0)
                        if is_ollama_running():
                            logger.info("Ollama started successfully.")
                            break
            except Exception as e:
                logger.error(f"Failed to start Ollama automatically: {e}")

        if not is_ollama_running():
            raise RuntimeError(
                "Ollama server is not running or unreachable on http://127.0.0.1:11434. "
                "Please start it manually using 'ollama serve'."
            )

        if not is_ollama_model_available(model_name):
            logger.warning(f"Ollama model '{model_name}' not found locally. Pulling it now...")
            try:
                ollama_path = shutil.which("ollama")
                subprocess.run([ollama_path, "pull", model_name], check=True)
                logger.info(f"Model '{model_name}' pulled successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to pull Ollama model '{model_name}': {e}")

        return None, None, model_name, True

    elif ai_engine.startswith("online"):
        if ai_engine == "online_lite":
            model_name = "gemini-3.1-flash-lite"
        elif ai_engine == "online_pro":
            model_name = "gemini-2.5-pro"
        else:
            model_name = "gemini-3.5-flash"

        from google import genai
        from google.genai import types
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please provide it in the UI or set the environment variable.")
        gemini_client = genai.Client(api_key=actual_api_key)
        return None, gemini_client, model_name, False
    else:
        raise ValueError(f"Unknown ai_engine: {ai_engine}")


def _extract_json_dict(raw_content: str) -> dict | None:
    """Extract a general JSON dict from raw string."""
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

    for candidate in strategies:
        try:
            cleaned = re.sub(r'//.*', '', candidate)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _call_llm_json(system_prompt: str, user_prompt: str, few_shot_examples: list, ai_engine: str, model_name: str, client, gemini_client, is_local: bool) -> dict | None:
    """Executes a single LLM call with retries and extracts a JSON dictionary."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if ai_engine.startswith("online"):
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                )
                gemini_history = []
                for ex in few_shot_examples:
                    gemini_history.append({"role": "user", "parts": [{"text": ex["user"]}]})
                    gemini_history.append({"role": "model", "parts": [{"text": ex["assistant"]}]})
                
                chat = gemini_client.chats.create(model=model_name, config=config, history=gemini_history)
                response = chat.send_message(user_prompt)
                raw_content = response.text
            else:
                messages = [{"role": "system", "content": system_prompt}]
                for ex in few_shot_examples:
                    messages.append({"role": "user", "content": ex["user"]})
                    messages.append({"role": "assistant", "content": ex["assistant"]})
                messages.append({"role": "user", "content": user_prompt})

                if is_local:
                    _touch_ollama()

                import urllib.request
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    "format": "json"
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_content = resp_data["message"]["content"]

            if is_local:
                _touch_ollama()

            parsed = _extract_json_dict(raw_content)
            if parsed is not None:
                return parsed
            logger.warning(f"Attempt {attempt}: Could not extract JSON dict from response: {raw_content[:200]}")
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                logger.error(f"All {max_attempts} attempts failed for LLM call.")
    return None


def _expand_sweep_actions(actions: list) -> list:
    """Helper to expand any 'sweep' actions into concrete lawn-mower 'goto' waypoints."""
    expanded = []
    for act in actions:
        if isinstance(act, dict) and act.get("type") == "sweep":
            try:
                from swarm_backend.config.schema import SweepParams
                sweep_obj = SweepParams(**act.get("params", {}))
                goto_list = generate_lawn_mower_waypoints(sweep_obj)
                for g in goto_list:
                    expanded.append(g.model_dump())
            except Exception as e:
                logger.warning(f"Failed to expand sweep action: {e}")
                expanded.append(act)
        else:
            expanded.append(act)
    return expanded


def plan_swarm_mission(prompt: str, ai_engine: str = "offline", api_key: str = None, num_drones: int = 3) -> dict:
    """
    Unified 5-Stage LLM prompting chain with true INDEPENDENT mode waypoints and Pydantic repair loop.
    All calls use concise JSON schemas optimized for both offline/small and online models.
    """
    logger.info(f"Starting Unified 5-Stage Swarm Mission Planning for prompt: '{prompt}'")
    client, gemini_client, model_name, is_local = _setup_llm_client(ai_engine, api_key)
    stages = {"repairs": []}

    # ── Stage 1: Classify Intent ──
    logger.info("Swarm Stage 1/5: Classify Intent...")
    res_classify = _call_llm_json(
        SWARM_CLASSIFY_PROMPT.format(instruction=prompt), prompt, [],
        ai_engine, model_name, client, gemini_client, is_local
    )
    if res_classify and isinstance(res_classify, dict):
        mode = str(res_classify.get("mode", "FORMATION")).upper()
        formation_type = str(res_classify.get("formation_type", "wedge")).lower()
        num_agents = int(res_classify.get("num_agents", num_drones))
        needs_task_split = bool(res_classify.get("needs_task_split", mode in ("INDEPENDENT", "MIXED")))
    else:
        mode, formation_type, num_agents, needs_task_split = "FORMATION", "wedge", num_drones, False
    if mode not in ("FORMATION", "INDEPENDENT", "MIXED", "REGROUP"):
        mode = "FORMATION"
    stages["classify"] = {"mode": mode, "formation_type": formation_type, "num_agents": num_agents, "needs_task_split": needs_task_split}
    logger.info(f"Stage 1 Result: mode={mode}, formation={formation_type}, num_agents={num_agents}, split={needs_task_split}")

    # ── Stage 2: Leader Actions Generator ──
    logger.info("Swarm Stage 2/5: Leader Actions Generator...")
    res_leader = _call_llm_json(
        SWARM_LEADER_ACTIONS_PROMPT.format(instruction=prompt), prompt, [],
        ai_engine, model_name, client, gemini_client, is_local
    )
    if res_leader and isinstance(res_leader, list) and len(res_leader) > 0:
        leader_actions = _expand_sweep_actions(res_leader)
    elif res_leader and isinstance(res_leader, dict) and "actions" in res_leader:
        leader_actions = _expand_sweep_actions(res_leader["actions"])
    else:
        std_res = plan_mission(prompt, mode="standard", ai_engine=ai_engine, api_key=api_key, num_drones=1)
        leader_actions = std_res.get("actions", [{"type": "takeoff", "params": {"altitude_m": 15.0}}, {"type": "land", "params": {}}])
    stages["leader_actions"] = leader_actions
    logger.info(f"Stage 2 Result: Generated {len(leader_actions)} leader actions.")

    # ── Stage 3: Formation Params (Conditional) ──
    spacing_m, angle_deg = 5.0, 135.0
    if mode in ("FORMATION", "MIXED", "REGROUP"):
        logger.info("Swarm Stage 3/5: Extracting Formation Params...")
        res_fparams = _call_llm_json(
            SWARM_FORMATION_PARAMS_PROMPT.format(instruction=prompt), prompt, [],
            ai_engine, model_name, client, gemini_client, is_local
        )
        if res_fparams and isinstance(res_fparams, dict):
            spacing_m = float(res_fparams.get("spacing_m", 5.0))
            angle_deg = float(res_fparams.get("angle_deg", 135.0 if formation_type == "wedge" else 90.0))
        stages["formation_params"] = {"spacing_m": spacing_m, "angle_deg": angle_deg}
        logger.info(f"Stage 3 Result: spacing={spacing_m}m, angle={angle_deg}deg")
    else:
        stages["formation_params"] = None

    # ── Stage 4: Follower Waypoints (True INDEPENDENT Mode) ──
    follower_waypoints_map = {}
    if needs_task_split or mode in ("INDEPENDENT", "MIXED"):
        logger.info("Swarm Stage 4/5: Generating Unique Follower Waypoints...")
        res_fwaypoints = _call_llm_json(
            SWARM_FOLLOWER_WAYPOINTS_PROMPT.format(instruction=prompt, leader_summary=json.dumps(leader_actions)),
            prompt, [], ai_engine, model_name, client, gemini_client, is_local
        )
        if res_fwaypoints and isinstance(res_fwaypoints, dict):
            for k, v_actions in res_fwaypoints.items():
                if isinstance(v_actions, list) and len(v_actions) > 0:
                    follower_waypoints_map[k] = _expand_sweep_actions(v_actions)
        stages["follower_waypoints"] = follower_waypoints_map
        logger.info(f"Stage 4 Result: Generated unique waypoints for {len(follower_waypoints_map)} followers.")
    else:
        stages["follower_waypoints"] = None

    # ── Stage 5: Python Assembly & Pydantic Repair Loop ──
    logger.info("Swarm Stage 5/5: Python Assembly & Pydantic Validation...")
    constraints = load_user_constraints()

    agents_list = []
    agents_list.append({"id": "leader", "role": "leader", "udp": "udp://:14540"})
    for i in range(1, num_agents):
        fid = f"follower_{i}"
        role_str = "wingman_left" if i % 2 != 0 else "wingman_right"
        if mode == "INDEPENDENT":
            role_str = "independent"
        ind_actions = follower_waypoints_map.get(fid) if follower_waypoints_map else None
        agents_list.append({
            "id": fid,
            "role": role_str,
            "slot": i,
            "udp": f"udp://:{14540 + i}",
            "independent_actions": ind_actions
        })

    raw_mission = {
        "mission_id": f"swarm_{int(time.time())}",
        "mode": mode,
        "formation": {"type": formation_type, "spacing_m": spacing_m, "angle_deg": angle_deg, "frame": "body_relative"} if mode in ("FORMATION", "MIXED", "REGROUP") else None,
        "safety": {
            "collision_radius_m": 1.0,
            "min_separation_m": max(3.0, spacing_m if mode in ("FORMATION", "MIXED", "REGROUP") else 3.0),
            "altitude_separation_m": 1.0,
            "avoidance_gain": 0.6,
            "max_correction_mps": 2.0
        },
        "agents": agents_list,
        "leader_mission": {"actions": leader_actions},
        "constraints": constraints.model_dump()
    }

    # Repair Loop (max 2 attempts)
    max_repairs = 2
    for attempt in range(max_repairs + 1):
        try:
            validated_mission = Mission.model_validate(raw_mission)
            logger.info("Pydantic schema validation successful!")
            break
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Validation failure (Attempt {attempt + 1}/{max_repairs + 1}): {err_msg}")
            if attempt == max_repairs:
                logger.error("Max repair attempts reached. Returning best-effort raw mission structure.")
                break
            repair_prompt = SWARM_REPAIR_PROMPT.format(error=err_msg, previous_json=json.dumps(raw_mission))
            res_repair = _call_llm_json(repair_prompt, prompt, [], ai_engine, model_name, client, gemini_client, is_local)
            if res_repair and isinstance(res_repair, dict):
                raw_mission.update(res_repair)
                stages["repairs"].append({"attempt": attempt + 1, "error": err_msg, "fixed": res_repair})

    follower_config_compat = {
        "behavior": "maintain_formation" if mode == "FORMATION" else mode.lower(),
        "formation": formation_type,
        "spacing_m": spacing_m,
        "collision_radius_m": raw_mission["safety"]["collision_radius_m"],
        "altitude_separation_m": raw_mission["safety"]["altitude_separation_m"],
    }
    for ag in agents_list[1:]:
        fid = ag["id"]
        follower_config_compat[fid] = {
            "role": ag["role"],
            "independent_actions": ag.get("independent_actions")
        }
        if fid == "follower_1":
            follower_config_compat["follower_1"]["offset_north_m"] = -spacing_m
            follower_config_compat["follower_1"]["offset_east_m"] = -spacing_m
        elif fid == "follower_2":
            follower_config_compat["follower_2"]["offset_north_m"] = -spacing_m
            follower_config_compat["follower_2"]["offset_east_m"] = spacing_m

    return {
        "mission_id": raw_mission["mission_id"],
        "mission_name": f"Swarm Mission ({mode})",
        "vehicle_type": "swarm",
        "drone_count": num_agents,
        "formation": raw_mission["formation"],
        "formation_type": formation_type,
        "spacing_m": spacing_m,
        "collision_radius_m": raw_mission["safety"]["collision_radius_m"],
        "mode": mode,
        "leader_mission": raw_mission["leader_mission"],
        "follower_config": follower_config_compat,
        "actions": raw_mission["leader_mission"]["actions"],
        "agents": raw_mission["agents"],
        "safety": raw_mission["safety"],
        "constraints": raw_mission["constraints"],
        "llm_stages": stages
    }

