"""What the static page generator reads: the team colors, `explain.json`, the model card and the Supabase config."""

import json
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
TEAM_COLORS_PATH = CONFIG_DIR / "team_colors.json"

_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")


def load_team_colors(path: Path = TEAM_COLORS_PATH) -> dict[str, str]:
    """Stored team name (`teams.name`) -> `#RRGGBB`. A malformed file or color raises `ValueError` naming it."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{path.name}: file not found") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path.name}: expected an object of team name -> color")
    for team, color in data.items():
        if not isinstance(color, str) or not _HEX_COLOR.fullmatch(color):
            raise ValueError(f"{path.name}: {team!r} has {color!r}, expected #RRGGBB")
    return data
