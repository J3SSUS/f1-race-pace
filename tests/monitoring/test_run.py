"""Which reports are due and how a season is walked (SPEC 26, step 8), with the report builder and writer stubbed.

The inputs are built by hand: one scored race per row of the table below, with race ids that are not chronological.
"""

import pandas as pd
import pytest
from fake_supabase import FakeSupabase

from monitoring.drift import FeatureDrift
from monitoring.inputs import Champion, DriftError, Inputs, Reference
from monitoring.report import DriftReport, Written
from monitoring.run import run_reports

IDS = [40, 3, 77, 12, 90, 8, 55, 21, 64, 5, 33, 71, 18, 46]  # race id of the n-th race, scrambled


def calendar(*races: tuple[int, int]) -> dict[int, tuple[int, int]]:
    return {IDS[i]: race for i, race in enumerate(races)}


def season(rounds: int, year: int = 2026, first: int = 1) -> list[tuple[int, int]]:
    return [(year, r) for r in range(first, first + rounds)]


def make_inputs(races: list[tuple[int, int]], *, base_race=None, last_report=None, scored=None) -> Inputs:
    """Every race of `races` is scored by the champion unless `scored` (a subset) says otherwise."""
    order = calendar(*races)
    by_value = {v: k for k, v in order.items()}
    scored_ids = [by_value[r] for r in (scored if scored is not None else races)]
    results = pd.DataFrame(
        {
            "race_id": scored_ids,
            "year": [order[i][0] for i in scored_ids],
            "round": [order[i][1] for i in scored_ids],
            "driver": "HAM",
            "abs_error": 0.5,
        }
    )
    champion = Champion(4, "run", "sampling", False, None, pd.Timestamp("2026-01-01", tz="UTC"))
    reference = Reference(0.43, "test", [], by_value[base_race] if base_race else None, pd.DataFrame())
    return Inputs(champion, order, results, pd.DataFrame(), reference, by_value[last_report] if last_report else None)


class Recorder:
    """A stand-in builder and writer: it records what it is asked and returns a report of the window it was given."""

    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.windows: list[list[int]] = []
        self.written: list[dict] = []

    def builder(self, inputs, window, catalog, ids):
        self.windows.append(list(window))
        features = [FeatureDrift("x", 1, "ks", None, 0.5, False)]
        report = DriftReport(4, window[0], window[-1], len(window), 10, 4, 0.5, 0.43, "test", 16.3, 0, True, features)
        return report, "<html/>"

    def writer(self, client, report, html, *, dag_run_id=None, tracking_uri=None):
        self.written.append({"through": report.through_race_id, "dag_run_id": dag_run_id, "uri": tracking_uri})
        return Written(len(self.written), f"run-{len(self.written)}", True, 1)

    def rounds(self, n: int) -> list[int]:
        return [self.inputs.race_order[r][1] for r in self.windows[n]]


def run(inputs, **kwargs):
    recorder = Recorder(inputs)
    client = FakeSupabase({"features": [{"feature_id": 1, "name": "x"}]})
    result = run_reports(client, inputs, builder=recorder.builder, writer=recorder.writer, **kwargs)
    return result, recorder


def test_the_walk_makes_a_report_every_five_races_in_order_and_says_how_many_are_left():
    result, rec = run(make_inputs(season(12)), year=2026)
    assert [rec.rounds(0), rec.rounds(1)] == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert [(m.first, m.last) for m in result.reports] == [((2026, 1), (2026, 5)), ((2026, 6), (2026, 10))]
    assert result.pending == 2  # rounds 11 and 12 wait for three more


def test_four_races_are_not_enough():
    result, rec = run(make_inputs(season(4)), year=2026)
    assert result.reports == [] and result.pending == 4 and rec.windows == []


def test_exactly_five_races_make_one_report_and_leave_none_pending():
    result, rec = run(make_inputs(season(5)), year=2026)
    assert len(result.reports) == 1 and result.pending == 0 and rec.rounds(0) == [1, 2, 3, 4, 5]


def test_a_check_as_of_a_race_ignores_the_later_ones():
    result, rec = run(make_inputs(season(12)), year=2026, round_=5)
    assert len(result.reports) == 1 and rec.rounds(0) == [1, 2, 3, 4, 5] and result.pending == 0


def test_a_check_with_more_than_five_waiting_reports_the_last_five():
    result, rec = run(make_inputs(season(12)), year=2026, round_=7)
    assert rec.rounds(0) == [3, 4, 5, 6, 7] and result.reports[0].last == (2026, 7)


def test_a_check_with_fewer_than_five_waiting_is_not_due_and_builds_nothing():
    result, rec = run(make_inputs(season(12)), year=2026, round_=4)
    assert result.reports == [] and result.pending == 4 and rec.windows == []


def test_the_last_report_of_the_champion_is_where_counting_starts():
    inputs = make_inputs(season(12), last_report=(2026, 5))
    assert run(inputs, year=2026, round_=9)[0].pending == 4  # rounds 6 to 9
    result, rec = run(inputs, year=2026, round_=10)
    assert rec.rounds(0) == [6, 7, 8, 9, 10] and result.pending == 0


def test_the_reports_already_stored_are_not_made_again():
    result, rec = run(make_inputs(season(12), last_report=(2026, 10)), year=2026)
    assert result.reports == [] and result.pending == 2 and rec.windows == [] and rec.written == []


def test_a_check_for_a_race_before_the_last_report_has_nothing_waiting():
    result, _ = run(make_inputs(season(12), last_report=(2026, 10)), year=2026, round_=5)
    assert result.reports == [] and result.pending == 0


def test_races_before_the_base_race_do_not_count():
    result, rec = run(make_inputs(season(12), base_race=(2026, 3)), year=2026)
    assert rec.rounds(0) == [4, 5, 6, 7, 8] and result.pending == 4  # rounds 9 to 12


def test_a_window_can_span_two_seasons():
    races = season(3, year=2026, first=22) + season(4, year=2027)
    result, rec = run(make_inputs(races), year=2027)
    assert [(m.first, m.last) for m in result.reports] == [((2026, 22), (2027, 2))] and result.pending == 2
    assert rec.written[0]["through"] == rec.windows[0][-1]


def test_only_the_scored_races_count():
    inputs = make_inputs(season(8), scored=[(2026, r) for r in (1, 2, 4, 5, 6, 8)])  # rounds 3 and 7 not scored
    result, rec = run(inputs, year=2026)
    assert rec.rounds(0) == [1, 2, 4, 5, 6] and result.pending == 1


def test_the_dag_run_id_and_tracking_uri_reach_the_writer():
    _, rec = run(make_inputs(season(5)), year=2026, dag_run_id="manual__x", tracking_uri="uri")
    assert rec.written == [{"through": rec.windows[0][-1], "dag_run_id": "manual__x", "uri": "uri"}]


def test_a_year_with_no_scored_race_is_an_error():
    with pytest.raises(DriftError, match="no scored prediction in 2027"):
        run(make_inputs(season(5)), year=2027)


def test_a_race_that_does_not_exist_is_an_error():
    with pytest.raises(DriftError, match="race 2026 round 30 is not in races"):
        run(make_inputs(season(5)), year=2026, round_=30)


def test_a_race_that_exists_but_the_champion_did_not_score_counts_what_came_before_it():
    inputs = make_inputs(season(6), scored=[(2026, r) for r in (1, 2, 3, 4, 5)])
    result, rec = run(inputs, year=2026, round_=6)  # round 6 was not scored: the five before it make the report
    assert rec.rounds(0) == [1, 2, 3, 4, 5] and len(result.reports) == 1
