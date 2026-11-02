import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from training.model_card import NOTE_SECTIONS, TEMPLATE_FILE, CardInputs, SegmentError, render_card

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / TEMPLATE_FILE
NOW = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)
METRICS = {"mae": 0.4123, "rmse": 0.5234, "r2": 0.1345, "pairwise_accuracy": 0.6456}
BASE_METRICS = {"mae": 0.5111, "rmse": 0.6222, "r2": -0.0333, "pairwise_accuracy": 0.5444}


def result(metrics):
    return {
        "metrics": metrics,
        "intervals": {k: [v - 0.0111, v + 0.0222] for k, v in metrics.items()},
        "n_rows": 777,
        "n_races": 13,
    }


def case(label, driver, prediction, actual):
    return {
        "label": label,
        "entry_id": 1,
        "year": 2025,
        "round": 24,
        "driver": driver,
        "prediction": prediction,
        "actual": actual,
        "error": prediction - actual,
        "base_value": 0.0051,
        "contributions": {f"f{i}": 0.1 * i - 0.35 for i in range(8)},
    }


def make_inputs(algorithm="ridge", coefficients=True) -> CardInputs:
    name = "random_forest" if algorithm == "rf" else algorithm
    explain = {
        "base_value": 0.0051,
        "importance": [
            {"feature": "grid_position", "mae_increase_mean": 0.0841, "mae_increase_std": 0.0043},
            {"feature": "team_pace_last3", "mae_increase_mean": 0.0302, "mae_increase_std": 0.0021},
        ],
        "mean_abs_shap": {"grid_position": 0.2718, "team_pace_last3": 0.1618},
        "coefficients": {"grid_position": 0.31, "team_pace_last3": -0.12, "circuit_type_street": 0.02}
        if coefficients
        else None,
        "calibration": {"slope": 1.0743, "intercept": 0.0201, "n_rows": 851, "n_races": 46},
        "cases": [
            case("fastest_predicted", "NOR", -1.0821, -1.5923),
            case("slowest_predicted", "COL", 0.8692, 0.2134),
        ],
    }
    selection = {
        "reference": "ridge",
        "candidate": "ridge",
        "beats_baseline": True,
        "table": [
            {"name": "ridge", "cv_mae": 0.4011, "diff": 0.0, "ci_low": -0.01, "ci_high": 0.01, "not_worse": True},
            {
                "name": "lightgbm",
                "cv_mae": 0.4099,
                "diff": 0.0088,
                "ci_low": 0.001,
                "ci_high": 0.02,
                "not_worse": False,
            },
        ],
        "vs_baseline": {"name": "ridge"},
    }
    return CardInputs(
        model_version_id=4,
        mlflow_version="2",
        algorithm=algorithm,
        feature_set_version="v1",
        trained_through={"year": 2025, "round": 10},
        sampling_run_id="aaaa1111",
        explain_run_id="bbbb2222",
        evaluate_run_id="cccc3333",
        promoted_at="2026-10-18T12:00:00+00:00",
        explain=explain,
        selection=selection,
        test_metrics={name: result(METRICS), "baseline": result(BASE_METRICS)},
        hyperparameters={"alpha": "3.2468", "imputation": "median"},
        tuning={"n_trials": "50", "seed": "42", "n_folds": "5"},
        data_range={"first_year": 2023, "last_year": 2025, "cutoff": None},
        split={
            "test_fraction": 0.2,
            "n_train_races": 46,
            "n_test_races": 13,
            "first_train_race": {"year": 2023, "round": 1},
            "last_train_race": {"year": 2025, "round": 10},
            "first_test_race": {"year": 2025, "round": 11},
            "last_test_race": {"year": 2025, "round": 24},
        },
        cv_folds=5,
    )


ERRORS = [
    SegmentError("team", "Beta", 40, 20, 0.5123, 0.6111, 0.0456, False),
    SegmentError("team", "Alpha", 29, 18, 0.4012, 0.5222, -0.0789, True),
    SegmentError("circuit_type", "street", 90, 12, 0.6234, 0.7333, 0.1011, False),
]
NOTES = {name: f"NOTE-{index} written by hand." for index, name in enumerate(NOTE_SECTIONS)}


def render(inputs=None, errors=None, notes=None, template=TEMPLATE, now=NOW) -> str:
    return render_card(inputs or make_inputs(), ERRORS if errors is None else errors, notes or NOTES, template, now)


def test_the_five_parts_appear_in_order():
    text = render()
    titles = [
        "## 1. What it predicts",
        "## 2. What weighs most",
        "## 3. A concrete case",
        "## 4. Where it fails",
        "## 5. Data and versions",
    ]
    positions = [text.index(title) for title in titles]
    assert positions == sorted(positions)


def test_the_header_has_the_version_the_algorithm_the_date_and_the_three_run_ids():
    header = render().splitlines()[0:3]
    assert header[0] == "# Model card: f1-race-pace, version 4 (ridge)"
    line = header[2]
    assert "2026-11-02" in line
    for run in ("bbbb2222", "cccc3333", "aaaa1111"):
        assert run in line


def test_the_four_test_metrics_of_the_champion_and_the_baseline_appear_with_their_intervals():
    text = render()
    assert "| Metric | ridge | Baseline |" in text
    assert "| MAE | 0.412 [0.401, 0.434] | 0.511 [0.500, 0.533] |" in text
    assert "| RMSE | 0.523 [0.512, 0.546] | 0.622 [0.611, 0.644] |" in text
    assert "| R² | 0.135 [0.123, 0.157] | -0.033 [-0.044, -0.011] |" in text
    assert "| Pairwise accuracy | 0.646 [0.634, 0.668] | 0.544 [0.533, 0.567] |" in text
    assert "777 valid rows of 13 test races" in text


def test_the_importance_table_has_the_numbers_of_the_inputs_largest_first():
    text = render()
    assert "| `grid_position` | +0.084 ± 0.004 | 0.272 |" in text
    assert "| `team_pace_last3` | +0.030 ± 0.002 | 0.162 |" in text
    assert text.index("`grid_position` | +0.084") < text.index("`team_pace_last3` | +0.030")


def test_every_hyperparameter_and_the_tuning_budget_appear():
    text = render()
    assert "`alpha`: 3.247" in text
    assert "`imputation`: median" in text
    assert "50 trials, seed 42, 5 folds" in text


def test_a_negative_zero_never_prints_as_minus_zero():
    inputs = make_inputs()
    inputs.explain["importance"][1]["mae_increase_mean"] = -0.0004
    inputs.explain["importance"][1]["mae_increase_std"] = -0.0001
    text = render(inputs)
    assert "| `team_pace_last3` | +0.000 ± 0.000 |" in text
    assert "-0.000" not in text


def test_a_long_decimal_param_is_shown_with_four_significant_digits_and_the_rest_as_stored():
    inputs = make_inputs()
    inputs.hyperparameters.update({"alpha": "98.47432599131227", "n_estimators": "300", "learning_rate": "1e-05"})
    text = render(inputs)
    assert "`alpha`: 98.47\n" in text
    assert "`n_estimators`: 300\n" in text
    assert "`learning_rate`: 1e-05\n" in text
    assert "`imputation`: median\n" in text
    assert "98.4743" not in text


def test_the_calibration_says_why_its_races_are_fewer_than_the_training_races():
    assert "fewer than the training races" in render()


def test_the_two_cases_show_prediction_actual_and_error_and_their_largest_contributions():
    text = render()
    assert "NOR, 2025 round 24. Prediction -1.082, actual -1.592, error +0.510 (prediction minus actual)." in text
    assert "COL, 2025 round 24. Prediction +0.869, actual +0.213, error +0.656 (prediction minus actual)." in text
    assert text.count("| `f0` | -0.350 |") == 2  # the largest |contribution| first: f0 and f7 are -0.35 and +0.35
    assert "`f3`" not in text  # only the five largest are shown
    assert "extremes of the prediction, not the cases where the model fails" in text


def test_a_linear_champion_links_the_coefficients_chart_and_a_tree_one_does_not():
    linear = render()
    assert "(site/charts/coefficients.svg)" in linear
    assert "| `circuit_type_street` | +0.020 |" in linear
    tree = render(make_inputs("rf", coefficients=False))
    assert "coefficients" not in tree
    assert "# Model card: f1-race-pace, version 4 (rf)" in tree
    assert "| Metric | random_forest | Baseline |" in tree


def test_segments_with_few_rows_are_marked_and_the_baseline_is_beside_them():
    text = render()
    assert "| Alpha | 29 | 18 | 0.401 | 0.522 | -0.079 | few rows |" in text
    assert "| Beta | 40 | 20 | 0.512 | 0.611 | +0.046 |  |" in text
    assert "| street | 90 | 12 | 0.623 | 0.733 | +0.101 |  |" in text
    assert text.count("few rows") == 1


def test_the_calibration_line_and_the_split_and_the_versions_are_printed():
    text = render()
    assert "On 851 out-of-fold rows of 46 races, the actual value is +0.020 + 1.074 × predicted" in text
    assert "seasons 2023 to 2025." in text
    assert "with 20% of the races in test. Train has 46 races (2023 round 1 to 2025 round 10)" in text
    assert "test has 13 (2025 round 11 to 2025 round 24)" in text
    assert "Feature set: v1. The model was trained through 2025 round 10." in text
    assert "Model version: 4 (MLflow version 2), promoted 2026-10-18." in text


def test_the_notes_are_inserted_as_written():
    text = render()
    for note in NOTES.values():
        assert note in text


def test_every_chart_link_is_a_relative_path_in_site_charts():
    links = re.findall(r"!\[[^\]]*\]\(([^)]*)\)", render())
    assert links
    assert all(re.fullmatch(r"site/charts/[a-z_]+\.svg", link) for link in links)


def test_the_template_contains_no_numeric_literal():
    """No digit is left once the names that contain one are removed: the model name, OpenF1, the numbered titles of
    the five parts and the names of the notes sections (a heading of the notes file, not a figure)."""
    template = TEMPLATE.read_text()
    names = [r"f1-race-pace", r"OpenF1", *(re.escape(name) for name in NOTE_SECTIONS)]
    cleaned = re.sub(r"^## \d+\. ", "## ", template, flags=re.MULTILINE)
    cleaned = re.sub("|".join(names), "", cleaned)
    assert not re.findall(r"\d", cleaned)


def test_the_same_inputs_and_date_give_the_same_text_and_another_date_only_changes_the_date():
    assert render() == render()
    other = render(now=datetime(2027, 1, 5, tzinfo=UTC))
    assert other.replace("2027-01-05", "2026-11-02") == render()


def test_an_undefined_variable_raises(tmp_path):
    template = tmp_path / "card.md.j2"
    template.write_text("# {{ not_in_the_context }}\n")
    with pytest.raises(Exception, match="not_in_the_context"):
        render(template=template)


def test_a_case_with_an_unknown_label_raises():
    inputs = make_inputs()
    inputs.explain["cases"][0]["label"] = "mystery"
    with pytest.raises(ValueError, match="mystery"):
        render(inputs)
