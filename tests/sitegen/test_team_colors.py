import json

import pytest

from ingestion.reference_config import load_team_aliases
from sitegen.inputs import load_team_colors

# `teams.name` in Supabase (checked 2026-11-10); a new team must be added here and to config/team_colors.json.
TEAM_NAMES = {
    "Alpine",
    "Aston Martin",
    "Audi",
    "Cadillac",
    "Ferrari",
    "Haas F1 Team",
    "Kick Sauber",
    "McLaren",
    "Mercedes",
    "RB",
    "Red Bull Racing",
    "Williams",
}


def write(tmp_path, content):
    p = tmp_path / "team_colors.json"
    p.write_text(content if isinstance(content, str) else json.dumps(content))
    return p


def test_the_repo_file_covers_every_stored_team():
    assert set(load_team_colors()) == TEAM_NAMES


def test_every_alias_target_has_a_color():
    assert set(load_team_aliases().values()) <= set(load_team_colors())


def test_the_loader_returns_the_colors_as_written(tmp_path):
    path = write(tmp_path, {"Ferrari": "#E8002D", "RB": "#6692ff"})
    assert load_team_colors(path) == {"Ferrari": "#E8002D", "RB": "#6692ff"}


@pytest.mark.parametrize("color", ["E8002D", "#E8002", "#E8002DD", "#GGGGGG", "red", 123, None])
def test_a_color_that_is_not_rrggbb_raises_naming_the_team(tmp_path, color):
    with pytest.raises(ValueError, match="Ferrari"):
        load_team_colors(write(tmp_path, {"Ferrari": color}))


@pytest.mark.parametrize("content", ["[]", "{}", '"x"', "not json"])
def test_a_file_that_is_not_a_color_map_raises(tmp_path, content):
    with pytest.raises(ValueError, match="team_colors.json"):
        load_team_colors(write(tmp_path, content))


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_team_colors(tmp_path / "team_colors.json")
