#!/usr/bin/env python3
"""
run_mission.py — Main CLI Entry Point

This is the operator-facing interface for the drone mission pipeline.
It ties together all pipeline stages:

  Prompt → LLM Planner → Validator → Executor → PX4/Gazebo

Usage:
  # Interactive mode (prompted for input)
  python3 run_mission.py

  # Direct command via CLI argument
  python3 run_mission.py --prompt "Patrol the perimeter loop twice at 15 metres"

  # Use a predefined route without calling the LLM
  python3 run_mission.py --route perimeter_loop --altitude 15 --repeats 2

  # Execute a pre-validated JSON file directly
  python3 run_mission.py --mission-file mission.json

  # Dry-run: plan and validate only, don't execute
  python3 run_mission.py --prompt "Fly figure eight" --dry-run
"""

import argparse
import json
import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_planner import plan_mission
from src.mission_validator import validate_mission
from src.executor import execute_mission
from src.utils import (
    print_header,
    print_stage,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_json,
    load_waypoint_library,
    Colours,
    PROJECT_ROOT,
)


def print_banner():
    """Print the application banner."""
    print(f"""
{Colours.BOLD}{Colours.CYAN}
   ╔═══════════════════════════════════════════════════════╗
   ║                                                       ║
   ║   🚁  DRONE MISSION PIPELINE                         ║
   ║                                                       ║
   ║   Prompt → LLM → Validated JSON → Executor → Gazebo  ║
   ║                                                       ║
   ╚═══════════════════════════════════════════════════════╝
{Colours.RESET}""")


def get_prompt_interactive() -> str:
    """Get a mission prompt from the user interactively."""
    routes = load_waypoint_library()
    route_names = list(routes.keys())

    print(f"{Colours.BOLD}Available predefined routes:{Colours.RESET}")
    for name, data in routes.items():
        print(f"  • {Colours.CYAN}{name}{Colours.RESET}: {data['description']}")

    print(f"\n{Colours.BOLD}Example prompts:{Colours.RESET}")
    examples = [
        "Patrol the perimeter loop twice at 15 metres altitude",
        "Fly the figure eight inspection pattern at 12 metres",
        "Take off to 10 metres, hover for 5 seconds, then land",
        "Execute the linear sweep at 8 metres altitude and 3 m/s speed",
        "Fly a square patrol at 20 metres, then return to launch",
    ]
    for ex in examples:
        print(f"  {Colours.DIM}→ \"{ex}\"{Colours.RESET}")

    print()
    prompt = input(f"{Colours.BOLD}{Colours.GREEN}Enter your command: "
                   f"{Colours.RESET}")
    return prompt.strip()


def confirm_execution(mission: dict) -> bool:
    """Ask the operator to confirm mission execution."""
    print(f"\n{Colours.BOLD}{Colours.YELLOW}"
          f"━━━ CONFIRM EXECUTION ━━━{Colours.RESET}")
    print(f"{Colours.YELLOW}The validated mission above will be sent to "
          f"the drone.{Colours.RESET}")
    print(f"{Colours.YELLOW}Make sure PX4 SITL + Gazebo is running "
          f"(bash launch_sim.sh){Colours.RESET}")

    response = input(f"\n{Colours.BOLD}Execute this mission? [y/N]: "
                     f"{Colours.RESET}")
    return response.strip().lower() in ("y", "yes")


def run_pipeline(prompt: str = None, mission_file: str = None,
                 dry_run: bool = False, auto_confirm: bool = False):
    """
    Run the complete pipeline: Plan → Validate → Execute.
    """
    print_banner()
    mission = None

    # ── Stage 1: Get the mission plan ──────────────────────────────
    print_stage(1, "Mission Planning (LLM)")

    if mission_file:
        # Load from file
        print_info(f"Loading mission from file: {mission_file}")
        with open(mission_file, "r") as f:
            mission = json.load(f)
        print_success(f"Loaded mission: {mission.get('mission_name', '?')}")

    else:
        # Use LLM planner
        if not prompt:
            prompt = get_prompt_interactive()
            if not prompt:
                print_error("No prompt provided. Exiting.")
                return 1

        try:
            mission = plan_mission(prompt)
            print_success("LLM generated mission plan")
        except (ValueError, RuntimeError) as e:
            print_error(f"Planning failed: {e}")
            return 1

    # Display the proposed mission
    print_json(mission, "Proposed Mission JSON")

    # Save the raw mission JSON
    raw_path = PROJECT_ROOT / "last_mission_raw.json"
    with open(raw_path, "w") as f:
        json.dump(mission, f, indent=2)
    print_info(f"Raw mission saved to {raw_path}")

    # ── Stage 2: Validate ─────────────────────────────────────────
    print_stage(2, "Validation & Safety Guardrails")

    result = validate_mission(mission)
    result.print_report()

    if not result.valid:
        print_error("\nMission failed validation. Cannot execute.")
        print_info("Please rephrase your command to fix the errors above.")
        return 1

    # Save the validated mission JSON
    validated_path = PROJECT_ROOT / "last_mission_validated.json"
    with open(validated_path, "w") as f:
        json.dump(mission, f, indent=2)
    print_info(f"Validated mission saved to {validated_path}")

    if dry_run:
        print_success("\n🏁 Dry run complete — mission is valid but was "
                      "NOT executed.")
        return 0

    # ── Stage 3: Execute ──────────────────────────────────────────
    if not auto_confirm:
        if not confirm_execution(mission):
            print_info("Execution cancelled by operator.")
            return 0

    print_stage(3, "Deterministic Execution")

    try:
        audit = execute_mission(mission)
        print_success("\n🏁 Mission complete!")

        # Save audit log
        audit_path = audit.save()
        print_info(f"Audit log saved to {audit_path}")

        return 0

    except Exception as e:
        print_error(f"\nMission execution failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="🚁 Drone Mission Pipeline — "
                    "Prompt → LLM → Validated JSON → Executor → Gazebo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "Patrol the perimeter loop twice at 15 metres"
  %(prog)s --mission-file mission.json
  %(prog)s --prompt "Fly figure eight" --dry-run
        """
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--prompt", "-p",
        type=str,
        help="Natural-language mission command"
    )
    input_group.add_argument(
        "--mission-file", "-f",
        type=str,
        help="Path to a pre-built mission JSON file"
    )

    # Options
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Plan and validate only, don't execute"
    )
    parser.add_argument(
        "--auto-confirm", "-y",
        action="store_true",
        help="Skip execution confirmation prompt"
    )

    args = parser.parse_args()

    exit_code = run_pipeline(
        prompt=args.prompt,
        mission_file=args.mission_file,
        dry_run=args.dry_run,
        auto_confirm=args.auto_confirm,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
