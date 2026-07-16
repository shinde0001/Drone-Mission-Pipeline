"""
swarm_backend/llm/prompts.py — Concise, single-purpose JSON prompt templates
for the 5-stage multi-drone swarm mission planner.
"""

SWARM_CLASSIFY_PROMPT = """You are a drone swarm mission intent classifier. Analyze the operator instruction and output a SINGLE JSON object.
Return STRICT valid JSON only without markdown formatting, backticks, or extra commentary.

JSON structure:
{{
  "mode": "FORMATION" | "INDEPENDENT" | "MIXED" | "REGROUP",
  "formation_type": "wedge" | "line_abreast" | "column" | "diamond",
  "num_agents": integer (total drones, e.g. 3),
  "needs_task_split": boolean (true if followers get unique independent waypoints or tasks)
}}

Rules:
- mode="FORMATION" when all drones fly together in geometric formation following the leader.
- mode="INDEPENDENT" when each drone is assigned a different target, area, or task (e.g. "Alpha checks north, Bravo checks east, Charlie sweeps west").
- mode="MIXED" when drones transit in formation then split up for independent tasks.
- mode="REGROUP" when drones rally/converge to a specific location.
- formation_type defaults to "wedge" unless "line", "column", or "diamond" is requested.
- num_agents defaults to 3 unless specified otherwise.
- needs_task_split is true if mode is INDEPENDENT or MIXED, or if followers require distinct unique waypoints.

Operator Instruction:
"{instruction}"

JSON:
"""

SWARM_LEADER_ACTIONS_PROMPT = """You are a drone flight path planner. Generate the sequence of actions for the leader drone (or overall formation trajectory) based on the operator instruction.
Return STRICT valid JSON only as a list of action objects without markdown formatting, backticks, or commentary.

Allowed action types and parameters:
- "takeoff": {{"altitude_m": float}}
- "goto": {{"north_m": float, "east_m": float, "altitude_m": float, "speed_mps": float}}
- "loiter": {{"duration_s": float}}
- "sweep": {{"corner_a_north_m": float, "corner_a_east_m": float, "corner_b_north_m": float, "corner_b_east_m": float, "altitude_m": float, "lane_spacing_m": float, "speed_mps": float}}
- "land": {{}}
- "return_to_launch": {{}}

Rules:
- Every mission MUST start with a "takeoff" action and end with "land" or "return_to_launch".
- Coordinates are North/East meters relative to launch (North positive forward, East positive right).
- Default altitude_m is 15.0 if unspecified. Default speed_mps is 5.0.

Operator Instruction:
"{instruction}"

JSON Action List:
"""

SWARM_FORMATION_PARAMS_PROMPT = """You are a spatial configuration extractor for drone formations.
Extract the desired spacing between drones and wedge/formation angle from the instruction.
Return STRICT valid JSON only without markdown formatting or commentary.

JSON structure:
{{
  "spacing_m": float (default 5.0),
  "angle_deg": float (default 135.0 for wedge, 90.0 for line_abreast, 180.0 for column)
}}

Operator Instruction:
"{instruction}"

JSON:
"""

SWARM_FOLLOWER_WAYPOINTS_PROMPT = """You are a multi-agent task allocator for autonomous follower drones.
Given the operator instruction and the leader's mission summary, generate UNIQUE, DISTINCT action lists for each follower drone.
Return STRICT valid JSON only without markdown formatting or commentary.

JSON structure:
{{
  "follower_1": [ {{"type": "takeoff", "params": {{"altitude_m": float}}}}, {{"type": "goto", "params": {{...}}}}, ... ],
  "follower_2": [ {{"type": "takeoff", "params": {{"altitude_m": float}}}}, {{"type": "goto", "params": {{...}}}}, ... ]
}}

Rules:
- Each follower must get its own unique waypoints/actions distinct from the leader and distinct from other followers based on the instruction.
- If the instruction splits tasks (e.g. "Bravo checks east 30m, Charlie checks west 30m"), assign those exact targets to follower_1 and follower_2 respectively.
- If the instruction specifies a sweep/grid search across the swarm, assign different sub-areas or unique targets to each follower.
- Every follower list must start with "takeoff" and end with "land" or "return_to_launch".

Operator Instruction:
"{instruction}"

Leader Actions Summary:
{leader_summary}

JSON Follower Action Lists:
"""

SWARM_REPAIR_PROMPT = """You are a JSON schema repair assistant.
The following JSON failed Pydantic schema validation with this exact error:
"{error}"

Original JSON:
{previous_json}

Fix ONLY the offending field to make it valid according to the error message.
Return STRICT valid JSON only without markdown formatting, backticks, or commentary.

Corrected JSON:
"""
