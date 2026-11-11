"""What the static page generator reads: the team colors, `explain.json`, the model card and the Supabase config."""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import markdown
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
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


@dataclass(frozen=True)
class SiteConfig:
    """What the browser needs to reach Supabase. Only the public (anon) key: the service key is never read."""

    supabase_url: str
    anon_key: str


@dataclass(frozen=True)
class SiteInputs:
    team_colors: dict[str, str]
    explain: dict  # site/charts/explain.json (SPEC 11); `model_version_id` says which model it explains
    card_html: str  # MODEL_CARD.md rendered to HTML, with the chart links made relative to site/
    config: SiteConfig


def _read_text(path: Path, root: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        raise ValueError(f"{path.relative_to(root)}: file not found") from None


def _read_explain(root: Path) -> dict:
    path = root / "site" / "charts" / "explain.json"
    try:
        data = json.loads(_read_text(path, root))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.relative_to(root)}: not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict) or not isinstance(data.get("model_version_id"), int):
        raise ValueError(f"{path.relative_to(root)}: expected an object with an integer model_version_id")
    return data


def _card_html(root: Path) -> str:
    """The card as HTML. Its charts are linked from the repository root (`site/charts/x.svg`); in the page they sit
    next to `index.html`, so the links become `charts/x.svg`."""
    text = _read_text(root / "MODEL_CARD.md", root).replace("](site/charts/", "](charts/")
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def _read_config(root: Path, environ: Mapping[str, str] | None) -> SiteConfig:
    """URL and anon key from the process environment, then `.env`. Only those two names are looked up."""
    if environ is None:
        environ = {**dotenv_values(root / ".env"), **os.environ}
    names = ("SUPABASE_URL", "SUPABASE_ANON_KEY")
    values = {name: (environ.get(name) or "") for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)}: set it in .env or the environment (see .env.example)")
    return SiteConfig(supabase_url=values["SUPABASE_URL"].rstrip("/"), anon_key=values["SUPABASE_ANON_KEY"])


def read_inputs(root: Path = ROOT, environ: Mapping[str, str] | None = None) -> SiteInputs:
    """Everything the page is built from. A missing file or variable raises `ValueError` naming it."""
    return SiteInputs(
        team_colors=load_team_colors(root / "config" / "team_colors.json"),
        explain=_read_explain(root),
        card_html=_card_html(root),
        config=_read_config(root, environ),
    )
