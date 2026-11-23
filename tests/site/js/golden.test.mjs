import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { normalizeLive } = require("../../../docs/site/site.js");

// The same fixture the Python side checks in tests/publishing/test_publishing_snapshot.py: if the two disagree, the
// expected file (the rules of SPEC 16) decides and the wrong side is fixed.
const golden = (name) => JSON.parse(readFileSync(new URL(`../golden/${name}`, import.meta.url), "utf8"));
const PARTS = ["champion", "baseline", "races", "metrics", "results"];

test("normalizeLive on the golden raw rows gives the five parts of the expected snapshot", () => {
  const data = normalizeLive(golden("raw_rows.json"));
  const expected = golden("expected_snapshot.json");
  assert.deepEqual(Object.keys(data).sort(), [...PARTS].sort());
  for (const part of PARTS) {
    assert.deepEqual(data[part], expected[part], `part ${part}`);
  }
});

test("the golden rows keep their nulls for a race that has not run yet", () => {
  const pending = normalizeLive(golden("raw_rows.json")).results.filter((row) => row.round === 18);
  assert.equal(pending.length, 2);
  assert.ok(pending.every((row) => row.actual_pace_delta_pct === null && row.is_valid_row === null));
});

test("a challenger, its baseline and their metrics leave the snapshot as it was (SPEC 24)", () => {
  const raw = golden("raw_rows.json");
  const raceId = raw.metrics.find((m) => m.race_id !== null).race_id;
  const withChallenger = {
    ...raw,
    modelVersions: [
      ...raw.modelVersions,
      { model_version_id: 7, algorithm: "ridge", is_shadow: false },
      { model_version_id: 8, algorithm: "baseline", is_shadow: false }, // newer than the baseline of the champion
    ],
    metrics: [
      ...raw.metrics,
      ...[7, 8].flatMap((id) => [
        { model_version_id: id, race_id: null, metric_name: "mae", value: 9.99 },
        { model_version_id: id, race_id: raceId + 1, metric_name: "mae", value: 9.99 }, // other races than the champion's
      ]),
    ],
  };
  const data = normalizeLive(withChallenger);
  const expected = golden("expected_snapshot.json");
  for (const part of PARTS) assert.deepEqual(data[part], expected[part], `part ${part}`);
});
