"""
utils.py — Logging, configuration loading, and helper functions.

This module provides shared utilities used across all pipeline stages:
- Structured logging with colour-coded output
- YAML/JSON config loaders
- Path resolution for config files
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SCHEMA_PATH = CONFIG_DIR / "mission_schema.json"
SAFETY_PATH = CONFIG_DIR / "safety_limits.yaml"
WAYPOINTS_PATH = CONFIG_DIR / "waypoint_library.yaml"


# ── Colour codes for terminal output ──────────────────────────────────
class Colours:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ── Pretty-print helpers ─────────────────────────────────────────────
def print_header(text: str) -> None:
    """Print a bold, coloured section header."""
    width = max(60, len(text) + 6)
    print(f"\n{Colours.BOLD}{Colours.CYAN}{'═' * width}{Colours.RESET}")
    print(f"{Colours.BOLD}{Colours.CYAN}   {text}{Colours.RESET}")
    print(f"{Colours.BOLD}{Colours.CYAN}{'═' * width}{Colours.RESET}\n")


def print_stage(stage_num: int, title: str) -> None:
    """Print a pipeline stage banner."""
    icons = {1: "🧠", 2: "🛡️", 3: "🚀", 4: "🎮"}
    icon = icons.get(stage_num, "▶")
    print(f"\n{Colours.BOLD}{Colours.BLUE}── Stage {stage_num}: "
          f"{icon} {title} ──{Colours.RESET}\n")


def print_success(text: str) -> None:
    print(f"{Colours.GREEN}✅ {text}{Colours.RESET}")


def print_error(text: str) -> None:
    print(f"{Colours.RED}❌ {text}{Colours.RESET}")


def print_warning(text: str) -> None:
    print(f"{Colours.YELLOW}⚠️  {text}{Colours.RESET}")


def print_info(text: str) -> None:
    print(f"{Colours.CYAN}ℹ️  {text}{Colours.RESET}")


def print_json(data: dict, title: str = "JSON") -> None:
    """Pretty-print a JSON object with syntax colouring."""
    formatted = json.dumps(data, indent=2)
    print(f"{Colours.DIM}── {title} ──{Colours.RESET}")
    for line in formatted.split("\n"):
        # Colour keys vs values
        if '":' in line:
            key_part, _, val_part = line.partition('":')
            print(f"{Colours.CYAN}{key_part}\"{Colours.RESET}:"
                  f"{Colours.GREEN}{val_part}{Colours.RESET}")
        else:
            print(f"{Colours.DIM}{line}{Colours.RESET}")
    print(f"{Colours.DIM}── end ──{Colours.RESET}\n")


# ── Config loaders ────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    """Load and parse a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    """Load and parse a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_mission_schema() -> dict:
    """Load the mission JSON schema."""
    return load_json(SCHEMA_PATH)


def load_safety_limits() -> dict:
    """Load safety limits configuration."""
    return load_yaml(SAFETY_PATH)


def load_waypoint_library() -> dict:
    """Load the waypoint library with predefined routes."""
    return load_yaml(WAYPOINTS_PATH)


# ── Logging setup ────────────────────────────────────────────────────
def setup_logger(name: str = "drone_pipeline",
                 level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger with timestamped output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        f"{Colours.DIM}[%(asctime)s]{Colours.RESET} "
        f"{Colours.BOLD}%(name)s{Colours.RESET} "
        f"%(levelname)s — %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ── Audit log ────────────────────────────────────────────────────────
class AuditLog:
    """
    Records every command issued during mission execution.
    Provides a deterministic, timestamped trace for auditability.
    """

    def __init__(self, callback=None):
        self.entries: list[dict] = []
        self.start_time = datetime.now(timezone.utc)
        self.callback = callback

    def record(self, action: str, details: dict | None = None,
               status: str = "ok") -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": (datetime.now(timezone.utc) -
                          self.start_time).total_seconds(),
            "action": action,
            "status": status,
            "details": details or {},
        }
        self.entries.append(entry)
        if self.callback:
            try:
                self.callback(entry)
            except Exception:
                pass

    def dump(self) -> list[dict]:
        return self.entries

    def summary(self) -> str:
        total = len(self.entries)
        ok = sum(1 for e in self.entries if e["status"] == "ok")
        failed = sum(1 for e in self.entries if e["status"] == "error")
        elapsed = self.entries[-1]["elapsed_s"] if self.entries else 0
        return (f"Audit: {total} actions, {ok} ok, {failed} failed, "
                f"{elapsed:.1f}s total")

    def save(self, path: Path | None = None) -> Path:
        """Save audit log to a JSON file."""
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = PROJECT_ROOT / f"audit_log_{ts}.json"
        with open(path, "w") as f:
            json.dump(self.dump(), f, indent=2)
        return path
