"""
forge_tray.py — The Forge System Tray App (Windows)
=====================================================
Right-click tray icon to control the Forge pipeline.
Runs the WSL2 pipeline in a background thread.

Requirements (install in Windows Python, NOT WSL2):
  pip install pystray pillow

Usage:
  python forge_tray.py
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import pystray
    from pystray import MenuItem as Item, Menu
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install pystray pillow")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────
FORGE_ROOT   = Path(os.environ.get("FORGE_ROOT", r"D:\Irongate\Forge"))
PAUSE_FLAG   = FORGE_ROOT / "pause.flag"
STATUS_FILE  = FORGE_ROOT / "tray_status.json"
VENV_PYTHON  = os.environ.get("FORGE_VENV_PYTHON", "/home/irongate/unsloth-env/bin/python3")
PIPELINE_PY  = os.environ.get("FORGE_PIPELINE_PY", "/mnt/d/Irongate/Forge/pipeline.py")

# Domains available — add more as you create them
DOMAINS = ["godot", "nz_law"]
DEFAULT_DOMAIN = "godot"

# ── State ──────────────────────────────────────────────────────────────────
_state = {
    "running": False,
    "paused": False,
    "domain": DEFAULT_DOMAIN,
    "last_run": None,
    "last_pairs": 0,
    "last_status": "Idle",
}
_pipeline_thread = None
_tray_icon = None


# ── Icon generator ─────────────────────────────────────────────────────────

def make_icon(color: str = "#E8A020") -> Image.Image:
    """
    Generate a simple anvil-style icon.
    color: orange=idle, green=running, yellow=paused, red=error
    """
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Anvil body
    draw.rectangle([8, 28, 56, 48], fill=color)
    # Anvil horn
    draw.polygon([8, 28, 8, 36, 24, 36, 24, 28], fill=color)
    # Anvil top
    draw.rectangle([16, 16, 48, 30], fill=color)
    # Base
    draw.rectangle([12, 46, 52, 56], fill=color)

    # Spark dots when running
    if color == "#20C020":
        draw.ellipse([44, 10, 50, 16], fill="#FFFF40")
        draw.ellipse([50, 6, 54, 10], fill="#FFFF40")
        draw.ellipse([40, 6, 45, 11], fill="#FF8040")

    return img


def update_icon():
    """Update tray icon colour based on state."""
    if _tray_icon is None:
        return
    if _state["paused"]:
        _tray_icon.icon = make_icon("#E8E020")   # yellow
    elif _state["running"]:
        _tray_icon.icon = make_icon("#20C020")   # green
    else:
        _tray_icon.icon = make_icon("#E8A020")   # orange


# ── Pipeline runner ────────────────────────────────────────────────────────

def _run_pipeline(domain: str):
    """Run in background thread — calls WSL2 pipeline via subprocess."""
    global _pipeline_thread

    _state["running"] = True
    _state["paused"] = False
    _state["last_status"] = f"Running ({domain})"
    _state["last_run"] = datetime.now().strftime("%H:%M:%S")
    update_icon()
    _notify(f"Forge started — {domain} domain")

    cmd = [
        "wsl", "-e", "bash", "-c",
        f'source /etc/environment 2>/dev/null; '
        f'export HSA_OVERRIDE_GFX_VERSION=11.0.0; '
        f'export HSA_ENABLE_DXG_DETECTION=1; '
        f'{VENV_PYTHON} {PIPELINE_PY} --domain {domain}'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(FORGE_ROOT)
        )

        if result.returncode == 0:
            # Parse pairs count from status output
            pairs = _parse_pairs_from_output(result.stdout)
            _state["last_pairs"] = pairs
            _state["last_status"] = f"Complete — {pairs} pairs ({domain})"
            _notify(f"Forge complete — {pairs} pairs generated")
        else:
            _state["last_status"] = f"Error ({domain})"
            _notify(f"Forge error — check logs")

    except Exception as e:
        _state["last_status"] = f"Error: {e}"
        _notify(f"Forge error: {e}")

    _state["running"] = False
    _pipeline_thread = None
    update_icon()
    _save_status()


def _parse_pairs_from_output(output: str) -> int:
    """Extract total pairs from pipeline status output."""
    for line in output.splitlines():
        if "Total pairs:" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return 0


# ── Notifications ──────────────────────────────────────────────────────────

def _notify(message: str):
    """Show a Windows toast notification."""
    if _tray_icon:
        try:
            _tray_icon.notify(message, "The Forge")
        except Exception:
            pass  # Notifications not supported on all Windows configs


# ── Status persistence ─────────────────────────────────────────────────────

def _save_status():
    try:
        STATUS_FILE.write_text(
            json.dumps(_state, default=str, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass


def _load_status():
    if STATUS_FILE.exists():
        try:
            saved = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            _state.update(saved)
            _state["running"] = False  # never resume a "running" state on startup
            _state["paused"] = PAUSE_FLAG.exists()
        except Exception:
            pass


# ── Menu actions ───────────────────────────────────────────────────────────

def action_run_now(icon, item):
    global _pipeline_thread
    if _state["running"]:
        _notify("Forge is already running")
        return
    # Remove pause flag if set
    if PAUSE_FLAG.exists():
        PAUSE_FLAG.unlink()
    _state["paused"] = False
    domain = _state["domain"]
    _pipeline_thread = threading.Thread(
        target=_run_pipeline, args=(domain,), daemon=True
    )
    _pipeline_thread.start()


def action_pause(icon, item):
    if not _state["running"]:
        _notify("Forge is not running")
        return
    if _state["paused"]:
        # Resume
        if PAUSE_FLAG.exists():
            PAUSE_FLAG.unlink()
        _state["paused"] = False
        _state["last_status"] = f"Resuming ({_state['domain']})"
        _notify("Forge resumed")
    else:
        # Pause
        PAUSE_FLAG.write_text("paused", encoding="utf-8")
        _state["paused"] = True
        _state["last_status"] = "Paused"
        _notify("Forge pausing after current chunk...")
    update_icon()
    _save_status()


def action_stop(icon, item):
    """Hard stop — sets pause flag, pipeline will stop between chunks."""
    if not _state["running"]:
        _notify("Forge is not running")
        return
    PAUSE_FLAG.write_text("stopped", encoding="utf-8")
    _state["paused"] = True
    _state["last_status"] = "Stopping..."
    _notify("Forge stopping after current chunk...")
    update_icon()


def action_show_status(icon, item):
    """Run pipeline --status and show result in a popup."""
    domain = _state["domain"]
    cmd = [
        "wsl", "-e", "bash", "-c",
        f'{VENV_PYTHON} {PIPELINE_PY} --domain {domain} --status'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()
        # Show first few lines as notification (Windows notifications have char limits)
        lines = [l for l in output.splitlines() if l.strip()]
        summary = "\n".join(lines[1:8]) if len(lines) > 1 else "No status available"
        _notify(summary)
    except Exception as e:
        _notify(f"Status error: {e}")


def action_retry_errors(icon, item):
    if _state["running"]:
        _notify("Stop Forge before retrying errors")
        return
    domain = _state["domain"]
    cmd = [
        "wsl", "-e", "bash", "-c",
        f'{VENV_PYTHON} {PIPELINE_PY} --domain {domain} --retry-errors'
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        _notify(f"Error jobs reset — click Run Now to retry")
    except Exception as e:
        _notify(f"Error: {e}")


def action_open_folder(path: Path):
    def _action(icon, item):
        try:
            subprocess.Popen(["explorer", str(path)])
        except Exception as e:
            _notify(f"Could not open folder: {e}")
    return _action


def action_set_domain(domain: str):
    def _action(icon, item):
        if _state["running"]:
            _notify("Stop Forge before changing domain")
            return
        _state["domain"] = domain
        _save_status()
        _notify(f"Domain set to: {domain}")
        icon.menu = build_menu()
        icon.update_menu()
    return _action


def action_quit(icon, item):
    if _state["running"]:
        # Set pause flag before quitting
        PAUSE_FLAG.write_text("quit", encoding="utf-8")
        time.sleep(1)
    icon.stop()


# ── Menu builder ───────────────────────────────────────────────────────────

def build_menu() -> Menu:
    domain = _state["domain"]
    running = _state["running"]
    paused = _state["paused"]
    status = _state["last_status"]
    last_run = _state.get("last_run", "Never")
    pairs = _state.get("last_pairs", 0)

    pause_label = "Resume" if paused else "Pause"

    domain_items = [
        Item(
            d,
            action_set_domain(d),
            checked=lambda item, d=d: _state["domain"] == d,
            radio=True
        )
        for d in DOMAINS
    ]

    watch_path = FORGE_ROOT / "domains" / domain / "watch"
    datasets_path = FORGE_ROOT / "domains" / domain / "datasets"
    logs_path = FORGE_ROOT / "domains" / domain / "logs"

    return Menu(
        Item(f"The Forge — {status}", None, enabled=False),
        Item(f"Last run: {last_run}  |  Pairs: {pairs}", None, enabled=False),
        Menu.SEPARATOR,
        Item("▶  Run Now", action_run_now, enabled=not running),
        Item(f"⏸  {pause_label}", action_pause, enabled=running),
        Item("⏹  Stop", action_stop, enabled=running),
        Menu.SEPARATOR,
        Item("📊  Show Status", action_show_status),
        Item("🔁  Retry Errors", action_retry_errors, enabled=not running),
        Menu.SEPARATOR,
        Item("🌐  Domain", Menu(*domain_items)),
        Menu.SEPARATOR,
        Item("📁  Open Watch Folder", action_open_folder(watch_path)),
        Item("📁  Open Datasets Folder", action_open_folder(datasets_path)),
        Item("📁  Open Logs", action_open_folder(logs_path)),
        Menu.SEPARATOR,
        Item("✖  Quit", action_quit),
    )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    global _tray_icon

    # Create Forge root if needed
    FORGE_ROOT.mkdir(parents=True, exist_ok=True)

    # Load saved state
    _load_status()

    # Clear stale pause flag from previous crash
    if PAUSE_FLAG.exists():
        PAUSE_FLAG.unlink()
        _state["paused"] = False

    icon_img = make_icon("#E8A020")

    _tray_icon = pystray.Icon(
        name="TheForge",
        icon=icon_img,
        title="The Forge",
        menu=build_menu()
    )

    print("The Forge tray app starting...")
    print(f"Forge root: {FORGE_ROOT}")
    print(f"Default domain: {_state['domain']}")
    print("Right-click the tray icon to control The Forge.")

    _tray_icon.run()


if __name__ == "__main__":
    main()
