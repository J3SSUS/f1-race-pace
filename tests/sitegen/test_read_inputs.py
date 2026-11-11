import pytest
from sitegen_helpers import ENV

from sitegen.inputs import SiteConfig, SiteInputs, read_inputs

FAKE_SERVICE_KEY = "service-key-must-never-appear"


class Spy(dict):
    """An environment that records every name looked up."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.looked_up = []

    def get(self, key, default=None):
        self.looked_up.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.looked_up.append(key)
        return super().__getitem__(key)


def test_it_returns_the_four_inputs(root):
    inputs = read_inputs(root, ENV)
    assert isinstance(inputs, SiteInputs)
    assert inputs.team_colors["Ferrari"] == "#E8002D"
    assert inputs.explain["model_version_id"] == 4
    assert "<h1>Card</h1>" in inputs.card_summary_html
    assert "<h2>2. Results</h2>" in inputs.card_details_html
    assert "<table>" in inputs.card_details_html
    assert inputs.config == SiteConfig("https://abc.supabase.co", "anon-key")


def test_the_card_is_split_before_its_second_section(root):
    inputs = read_inputs(root, ENV)
    assert "Intro." in inputs.card_summary_html
    assert "<h2>1. What it predicts</h2>" in inputs.card_summary_html
    assert "The pace." in inputs.card_summary_html
    assert "Results" not in inputs.card_summary_html
    assert "What it predicts" not in inputs.card_details_html


def test_the_card_links_its_charts_next_to_the_pages(root):
    inputs = read_inputs(root, ENV)
    assert 'src="charts/permutation_importance.svg"' in inputs.card_details_html
    assert "site/charts" not in inputs.card_summary_html + inputs.card_details_html


@pytest.mark.parametrize("card", ["# Card\n\nNo sections.\n", "# Card\n\n## 1. Only one\n\nText.\n"])
def test_a_card_with_fewer_than_two_sections_raises(root, card):
    (root / "MODEL_CARD.md").write_text(card)
    with pytest.raises(ValueError, match="MODEL_CARD.md"):
        read_inputs(root, ENV)


def test_the_inputs_are_frozen(root):
    inputs = read_inputs(root, ENV)
    with pytest.raises(AttributeError):
        inputs.card_summary_html = "x"


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("config/team_colors.json", "team_colors.json"),
        ("site/charts/explain.json", "explain.json"),
        ("MODEL_CARD.md", "MODEL_CARD.md"),
    ],
)
def test_a_missing_file_raises_naming_it(root, path, name):
    (root / path).unlink()
    with pytest.raises(ValueError, match=name):
        read_inputs(root, ENV)


@pytest.mark.parametrize("content", ["not json", "[]", '{"importance": []}', '{"model_version_id": "4"}'])
def test_an_explain_file_without_an_integer_model_version_raises(root, content):
    (root / "site" / "charts" / "explain.json").write_text(content)
    with pytest.raises(ValueError, match="explain.json"):
        read_inputs(root, ENV)


@pytest.mark.parametrize("missing", ["SUPABASE_URL", "SUPABASE_ANON_KEY"])
def test_a_missing_variable_raises_naming_it(root, missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        read_inputs(root, env)


def test_an_empty_variable_counts_as_missing(root):
    with pytest.raises(ValueError, match="SUPABASE_ANON_KEY"):
        read_inputs(root, {**ENV, "SUPABASE_ANON_KEY": ""})


def test_it_reads_the_variables_from_dot_env(root, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    (root / ".env").write_text("SUPABASE_URL=https://env.supabase.co\nSUPABASE_ANON_KEY=from-file\n")
    assert read_inputs(root).config == SiteConfig("https://env.supabase.co", "from-file")


def test_the_process_environment_wins_over_dot_env(root, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://process.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "from-process")
    (root / ".env").write_text("SUPABASE_URL=https://env.supabase.co\nSUPABASE_ANON_KEY=from-file\n")
    assert read_inputs(root).config == SiteConfig("https://process.supabase.co", "from-process")


def test_the_service_key_is_never_looked_up_nor_returned(root):
    env = Spy({**ENV, "SUPABASE_SERVICE_KEY": FAKE_SERVICE_KEY})
    inputs = read_inputs(root, env)
    assert "SUPABASE_SERVICE_KEY" not in env.looked_up
    assert FAKE_SERVICE_KEY not in repr(inputs)


def test_the_service_key_in_dot_env_is_not_returned(root, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    (root / ".env").write_text(
        f"SUPABASE_URL=https://env.supabase.co\nSUPABASE_ANON_KEY=from-file\nSUPABASE_SERVICE_KEY={FAKE_SERVICE_KEY}\n"
    )
    assert FAKE_SERVICE_KEY not in repr(read_inputs(root))
