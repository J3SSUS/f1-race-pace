import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { formatPace, teamColor, racesWithPredictions, raceRows, raceHeader, raceKey } = require("../../../docs/site/site.js");

const { champion, baseline, races, metrics, results } = JSON.parse(
  readFileSync(new URL("../sample_snapshot.json", import.meta.url), "utf8"),
);
const DATA = { champion, baseline, races, metrics, results };

test("pace is shown in percent with two decimals, an em dash when missing", () => {
  assert.equal(formatPace(0.3456), "0.35%");
  assert.equal(formatPace(-0.62), "-0.62%");
  assert.equal(formatPace(0), "0.00%");
  assert.equal(formatPace(-0.001), "0.00%");
  for (const value of [null, undefined, NaN]) assert.equal(formatPace(value), "—");
});

test("a team has its color or null, never a color inherited from Object", () => {
  const colors = { Ferrari: "#E8002D" };
  assert.equal(teamColor(colors, "Ferrari"), "#E8002D");
  assert.equal(teamColor(colors, "Brand New Team"), null);
  assert.equal(teamColor(colors, "constructor"), null);
  assert.equal(teamColor(colors, "toString"), null);
});

test("the races with predictions come oldest first with their name and date", () => {
  const list = racesWithPredictions({ ...DATA, results: [...results].reverse() });
  assert.deepEqual(list.map((r) => r.round), [16, 17]);
  assert.equal(list[1].event_name, "Example Grand Prix");
  assert.equal(list[1].race_date, "2026-11-08");
});

test("a race with predictions that is not in the races table still gets an entry", () => {
  const extra = [...results, { ...results[0], round: 18 }];
  const list = racesWithPredictions({ ...DATA, results: extra });
  assert.equal(list[2].round, 18);
  assert.equal(list[2].event_name, null);
  assert.equal(raceHeader(list[2], []).title, "R18 round 18");
});

test("a race has no entry when it has no predictions", () => {
  assert.deepEqual(racesWithPredictions({ ...DATA, results: [] }), []);
});

test("the rows of a race are only that race's, fastest predicted pace first", () => {
  const rows = raceRows(DATA, 2026, 16);
  assert.deepEqual(rows.map((r) => r.driver), ["NOR", "LEC", "ALB"]);
  assert.ok(rows.every((r) => r.round === 16));
  assert.equal(raceRows(DATA, 2026, 99).length, 0);
});

test("ties in the predicted pace are ordered by driver", () => {
  const tied = [
    { year: 2026, round: 1, driver: "ZZZ", predicted_pace_delta_pct: 0.1 },
    { year: 2026, round: 1, driver: "AAA", predicted_pace_delta_pct: 0.1 },
  ];
  assert.deepEqual(raceRows({ results: tied }, 2026, 1).map((r) => r.driver), ["AAA", "ZZZ"]);
});

test("the header marks a race as late only when its predictions are late", () => {
  const [r16, r17] = racesWithPredictions(DATA);
  assert.equal(raceHeader(r16, raceRows(DATA, 2026, 16)).late, false);
  assert.equal(raceHeader(r17, raceRows(DATA, 2026, 17)).late, true);
  assert.equal(raceHeader(r17, raceRows(DATA, 2026, 17)).title, "R17 Example Grand Prix");
});

test("rows keep their own is_late and a row with no actual pace stays in the race", () => {
  const rows = raceRows(DATA, 2026, 17);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((r) => r.is_late), [true, true, true]);
  const unscored = rows.find((r) => r.driver === "SAR");
  assert.equal(unscored.is_valid_row, false);
  assert.equal(unscored.actual_pace_delta_pct, null);
});

test("race keys are year and round", () => {
  assert.equal(raceKey(2026, 17), "2026-17");
});

const { isPending } = require("../../../docs/site/site.js");

const pendingRow = (driver) => ({
  year: 2026,
  round: 21,
  driver,
  team: "Ferrari",
  predicted_pace_delta_pct: 0.1,
  actual_pace_delta_pct: null,
  is_valid_row: null,
  is_late: false,
});

test("a race whose drivers have no result at all is pending, as the view gives it before the race", () => {
  assert.equal(isPending([pendingRow("A"), pendingRow("B")]), true);
  assert.equal(isPending([{ ...pendingRow("A"), actual_pace_delta_pct: undefined, is_valid_row: undefined }]), true);
});

test("a race with any result is not pending, even if some drivers have no reliable pace", () => {
  const scored = { ...pendingRow("B"), actual_pace_delta_pct: 0.3, is_valid_row: true };
  assert.equal(isPending([pendingRow("A"), scored]), false);
  assert.equal(isPending(raceRows(DATA, 2026, 17)), false); // SAR has no pace but the others do
});

test("a race that ran but where no driver has a reliable pace is not pending", () => {
  assert.equal(isPending([{ ...pendingRow("A"), is_valid_row: false }, { ...pendingRow("B"), is_valid_row: false }]), false);
});

test("a race with no rows is not pending, and the header says so", () => {
  assert.equal(isPending([]), false);
  const [r16, r17] = racesWithPredictions(DATA);
  assert.equal(raceHeader(r16, raceRows(DATA, 2026, 16)).pending, false);
  assert.equal(raceHeader(r17, raceRows(DATA, 2026, 17)).pending, false);
  assert.equal(raceHeader({ round: 21, event_name: "X", race_date: "2026-11-15" }, [pendingRow("A")]).pending, true);
});

test("the late and pending marks are independent: a prediction made just after the start is both", () => {
  const header = raceHeader({ round: 21, event_name: "X" }, [{ ...pendingRow("A"), is_late: true }]);
  assert.equal(header.late, true);
  assert.equal(header.pending, true);
});

const { pendingBanner } = require("../../../docs/site/site.js");

test("the banner announces the latest race when it is predicted and has no result yet", () => {
  const data = {
    ...DATA,
    races: [...races, { race_id: 90, year: 2026, round: 21, event_name: "Next Grand Prix", race_date: "2026-11-22" }],
    results: [...results, pendingRow("A"), pendingRow("B")],
  };
  assert.deepEqual(pendingBanner(data), { title: "R21 Next Grand Prix", date: "2026-11-22", key: "2026-21" });
});

test("there is no banner when the latest race already has its result", () => {
  assert.equal(pendingBanner(DATA), null);
  assert.equal(pendingBanner({ ...DATA, results: [] }), null);
});

test("an earlier race that is pending does not get the banner when a later one has its result", () => {
  const earlierPending = { ...DATA, results: [...results.filter((r) => r.round === 17), { ...pendingRow("A"), round: 10 }] };
  assert.equal(pendingBanner(earlierPending), null);
});

test("the banner does not depend on the selector and names the race even when its date is missing", () => {
  const data = { ...DATA, races: [], results: [pendingRow("A")] };
  const banner = pendingBanner(data);
  assert.equal(banner.title, "R21 round 21");
  assert.equal(banner.date, null);
});
