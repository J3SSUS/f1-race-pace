<!--
Hand-written notes for MODEL_CARD.md (SPEC 12). Each section is inserted into the card as written.
Keep exactly the three H2 headings below. Write no number and no model name here: both change at the next
retraining, and the card prints its own numbers from the stored results.
-->

## Intended use

The model estimates, after Saturday qualifying, how much faster or slower each driver will be than the field on
Sunday's clean laps. It is meant to read the order of the grid, not to name a winner or a podium: the ranking of
drivers by predicted pace is more reliable than the size of any single prediction.

It is a personal MLOps project that runs locally. It is not a betting tool and it is not tuned for one.

## Known limitations

- It only knows what is true at the end of Saturday. Weather changes, strategy, first-lap incidents, safety cars and
  mechanical failures on Sunday are invisible to it. The target removes the laps under a neutralisation, but the
  race that follows them is still shaped by them.
- Out-of-fold errors by team and by circuit type come from cross-validation on the training races, not from the test
  races or from a future season. Segments with few rows are marked, and no interval is computed for a segment, so
  small differences between close segments should not be read as real.
- A team that changed its name across seasons appears under each name, so its rows are split and each part looks
  smaller than the team really is. The card does not merge team lineages.
- Importance and SHAP values describe what this model uses, not what matters in racing: correlated features share
  their importance, and a feature that looks unimportant can still matter in the world.
- The two cases shown are the extremes of the prediction, the driver predicted fastest and the one predicted
  slowest, not the cases where the model fails the most.
- The data starts with the first season OpenF1 serves, so the model has seen a small number of seasons and circuits.

## 2026 regulations

The training data comes from seasons before the 2026 technical regulations. A change of that size can reshuffle the
order of the teams and change what qualifying pace says about race pace, so the relations learned here may not hold.
There is no data from the new regulations in training and the effect is not measured in this card; it is measured
later, in the walk-forward backtest over the 2026 season. Until then, treat predictions for 2026 with extra caution.
