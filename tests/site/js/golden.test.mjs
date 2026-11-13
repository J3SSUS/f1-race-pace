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
