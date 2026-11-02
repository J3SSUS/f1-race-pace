import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_explain_champion import synthetic_X

import training.explain_charts as charts
from features.catalog import load_catalog
from training.explain import Case

FEATURES = [f.name for f in load_catalog().features]
SVG = "{http://www.w3.org/2000/svg}"
VERSION = 4


def texts(path: Path) -> list[str]:
    return ["".join(t.itertext()) for t in ET.parse(path).getroot().iter(f"{SVG}text")]


def shap_features(path: Path) -> list[str]:
    """`shap.plots` writes its rows from the bottom up and some labels twice: top-down and without repeats."""
    return list(dict.fromkeys(only_features(path)))[::-1]


def only_features(path: Path) -> list[str]:
    return [t for t in texts(path) if t in FEATURES]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 60
    X = synthetic_X(n)
    shap = pd.DataFrame(rng.normal(size=(n, 15)) * np.linspace(1, 0.05, 15), columns=FEATURES)  # first is the largest
    oof = pd.DataFrame({"y_true": rng.normal(size=n)})
    oof["y_pred"] = 0.8 * oof["y_true"] + rng.normal(scale=0.2, size=n)
    importance = pd.DataFrame(
        {"feature": FEATURES, "mae_increase_mean": np.linspace(0.3, -0.01, 15), "mae_increase_std": 0.01}
    )
    calibration = {"slope": 1.1, "intercept": -0.02, "n_rows": n, "n_races": 6}
    contributions = dict(zip(FEATURES, rng.normal(scale=0.1, size=15).tolist(), strict=True))
    case = Case("slowest_predicted", 7, 2025, 24, "NOR", 0.5, 0.3, 0.2, 0.1, contributions)
    coefficients = dict(zip(FEATURES[:6], [0.4, -0.3, 0.1, -0.05, 0.02, 0.01], strict=True))
    return {
        "X": X,
        "shap": shap,
        "oof": oof,
        "importance": importance,
        "calibration": calibration,
        "case": case,
        "coefficients": coefficients,
    }


def draw_all(data, folder: Path) -> dict[str, Path]:
    return {
        "coefficients": charts.coefficients_chart(data["coefficients"], VERSION, folder / "coefficients.svg"),
        "permutation": charts.permutation_chart(data["importance"], VERSION, folder / "permutation_importance.svg"),
        "shap": charts.shap_summary_chart(data["shap"], data["X"], VERSION, folder / "shap_summary.svg"),
        "shap_bar": charts.shap_bar_chart(data["shap"], VERSION, folder / "shap_bar.svg"),
        "waterfall": charts.waterfall_chart(data["case"], VERSION, folder / "case_slowest.svg"),
        "pva": charts.predicted_vs_actual_chart(data["oof"], data["calibration"], VERSION, folder / "pva.svg"),
        "residuals": charts.residuals_chart(data["oof"], VERSION, folder / "residuals.svg"),
    }


def test_every_chart_is_a_valid_svg_with_the_model_version_in_its_title(data, tmp_path):
    written = draw_all(data, tmp_path)
    assert len(written) == 7
    for name, path in written.items():
        assert path.exists(), name
        assert ET.parse(path).getroot().tag == f"{SVG}svg", name
        assert f"model version {VERSION}" in " ".join(texts(path)), name


def test_the_folder_is_created_when_it_does_not_exist(data, tmp_path):
    path = charts.residuals_chart(data["oof"], VERSION, tmp_path / "site" / "charts" / "residuals.svg")
    assert path.exists()


def test_the_coefficient_chart_is_written_only_for_a_linear_champion(data, tmp_path):
    path = tmp_path / "coefficients.svg"
    assert charts.coefficients_chart(None, VERSION, path) is None
    assert not path.exists()
    assert charts.coefficients_chart(data["coefficients"], VERSION, path) == path
    assert path.exists()
    assert [t for t in texts(path) if t in data["coefficients"]] == list(data["coefficients"])  # largest first


def test_the_permutation_chart_has_the_15_features_sorted_largest_first_whatever_the_input_order(data, tmp_path):
    shuffled = data["importance"].sample(frac=1, random_state=3)
    path = charts.permutation_chart(shuffled, VERSION, tmp_path / "p.svg")
    assert only_features(path) == FEATURES  # the importance falls from the first to the last feature


def test_the_shap_summary_has_the_15_features_ordered_by_mean_absolute_shap(data, tmp_path):
    path = charts.shap_summary_chart(data["shap"], data["X"], VERSION, tmp_path / "s.svg")
    expected = list(data["shap"].abs().mean().sort_values(ascending=False).index)
    assert shap_features(path) == expected
    assert len(expected) == 15


def test_the_shap_bar_has_the_15_features_ordered_by_mean_absolute_shap(data, tmp_path):
    path = charts.shap_bar_chart(data["shap"], VERSION, tmp_path / "b.svg")
    expected = list(data["shap"].abs().mean().sort_values(ascending=False).index)
    assert only_features(path)[:15] == expected  # top down; the labels are written twice


def test_the_waterfall_shows_the_15_contributions_of_its_case(data, tmp_path):
    case = data["case"]
    path = charts.waterfall_chart(case, VERSION, tmp_path / "w.svg")
    assert sorted(set(only_features(path))) == sorted(FEATURES)
    assert any("NOR" in t and "2025 round 24" in t for t in texts(path))


def test_the_calibration_line_is_in_the_predicted_vs_actual_legend(data, tmp_path):
    path = charts.predicted_vs_actual_chart(data["oof"], data["calibration"], VERSION, tmp_path / "pva.svg")
    legend = " ".join(texts(path))
    assert "slope 1.10" in legend and "intercept" in legend


def test_running_twice_gives_byte_identical_files(data, tmp_path):
    first = draw_all(data, tmp_path / "one")
    second = draw_all(data, tmp_path / "two")
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes(), name


def test_the_files_have_no_date_and_no_random_id(data, tmp_path):
    for name, path in draw_all(data, tmp_path).items():
        content = path.read_text()
        assert "<dc:date>" not in content, name
        assert "Matplotlib v" not in content, name  # the creator is fixed: an upgrade does not rewrite the metadata


def test_drawing_leaves_the_global_matplotlib_settings_alone(data, tmp_path):
    import matplotlib

    before = dict(matplotlib.rcParams)
    draw_all(data, tmp_path)
    assert dict(matplotlib.rcParams) == before
