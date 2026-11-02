from pathlib import Path

import pytest

from training.model_card import NOTE_SECTIONS, NOTES_FILE, read_notes

TEXT = """<!-- header comment: ignored -->

## Intended use

First paragraph, as written.

- a bullet
- another bullet

## Known limitations

Limits, **as written**.
  Indented line kept.

## 2026 regulations

The new rules.
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "notes.md"
    path.write_text(text)
    return path


def test_each_section_comes_back_as_written(tmp_path):
    notes = read_notes(write(tmp_path, TEXT))
    assert list(notes) == list(NOTE_SECTIONS)
    assert notes["Intended use"] == "First paragraph, as written.\n\n- a bullet\n- another bullet"
    assert notes["Known limitations"] == "Limits, **as written**.\n  Indented line kept."
    assert notes["2026 regulations"] == "The new rules."


def test_a_missing_file_raises_and_names_it(tmp_path):
    with pytest.raises(ValueError, match="missing.md"):
        read_notes(tmp_path / "missing.md")


@pytest.mark.parametrize("name", NOTE_SECTIONS)
def test_a_missing_section_raises_and_names_it(tmp_path, name):
    text = TEXT.replace(f"## {name}", "## Something else")
    with pytest.raises(ValueError, match=name):
        read_notes(write(tmp_path, text))


@pytest.mark.parametrize("name", NOTE_SECTIONS)
def test_an_empty_section_raises_and_names_it(tmp_path, name):
    lines = TEXT.split(f"## {name}\n")
    after = lines[1].split("\n## ", 1)
    rest = "\n## " + after[1] if len(after) == 2 else ""
    with pytest.raises(ValueError, match=f"'{name}'.*empty"):
        read_notes(write(tmp_path, f"{lines[0]}## {name}\n\n   \n{rest}"))


def test_extra_sections_are_ignored(tmp_path):
    notes = read_notes(write(tmp_path, TEXT + "\n## Extra\n\nNot part of the card.\n"))
    assert list(notes) == list(NOTE_SECTIONS)
    assert "Not part of the card" not in "".join(notes.values())


def test_a_repeated_section_raises(tmp_path):
    with pytest.raises(ValueError, match="twice"):
        read_notes(write(tmp_path, TEXT + "\n## Intended use\n\nAgain.\n"))


def test_the_notes_file_of_the_repo_has_the_three_sections_filled():
    notes = read_notes(Path(__file__).resolve().parents[2] / NOTES_FILE)
    assert list(notes) == list(NOTE_SECTIONS)
    assert all(text.strip() for text in notes.values())
