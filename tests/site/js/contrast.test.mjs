import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const css = readFileSync(new URL("../../../docs/site/site.css", import.meta.url), "utf8");
const root = css.match(/:root\s*{([^}]*)}/)[1];
const color = Object.fromEntries([...root.matchAll(/--([a-z-]+):\s*(#[0-9a-fA-F]{6})/g)].map((m) => [m[1], m[2]]));

/* WCAG 2.x relative luminance and contrast ratio. */
const channel = (v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => channel(parseInt(hex.slice(i, i + 2), 16) / 255));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

test("the palette defines every color the pairs below use", () => {
  for (const name of ["bg", "panel", "line", "text", "muted", "champion", "good", "flag", "team-unknown"]) {
    assert.match(color[name] ?? "", /^#[0-9a-fA-F]{6}$/, name);
  }
});

// [text color, background, where it is used]
const TEXT_PAIRS = [
  ["text", "bg", "body text"],
  ["text", "panel", "text on a figure card or a hovered row"],
  ["muted", "bg", "secondary text and axis labels"],
  ["muted", "panel", "secondary text on a card"],
  ["champion", "bg", "links and the champion bars"],
  ["champion", "panel", "the champion figures"],
  ["good", "bg", "improvement"],
  ["flag", "bg", "the late mark and the notices in the page"],
  ["flag", "panel", "the late mark on a hovered row"],
  ["bg", "flag", "the banner for a fallback notice"],
];

for (const [fg, bg, where] of TEXT_PAIRS) {
  test(`text contrast is at least 4.5:1: ${fg} on ${bg} (${where})`, () => {
    const value = ratio(color[fg], color[bg]);
    assert.ok(value >= 4.5, `${fg} ${color[fg]} on ${bg} ${color[bg]} is ${value.toFixed(2)}:1`);
  });
}

test("the neutral team color and the grid lines are visible against the background (3:1 for graphics)", () => {
  assert.ok(ratio(color["team-unknown"], color.bg) >= 3);
  assert.ok(ratio(color.muted, color.bg) >= 3);
});
