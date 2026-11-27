-- SPEC 26: what a drift report needs to be read and reproduced later. Both tables are empty when this runs
-- (nothing wrote to them before SPEC 26), so the NOT NULL columns need no default. The tables stay private:
-- RLS enabled and no policy, set in 0001; this file does not touch it.

ALTER TABLE drift_reports
    ADD COLUMN from_race_id     integer NOT NULL REFERENCES races,   -- first race of the window
    ADD COLUMN n_races          integer NOT NULL,                    -- races of the window (5, DRIFT_WINDOW)
    ADD COLUMN current_rows     integer NOT NULL,                    -- valid rows of the window
    ADD COLUMN reference_rows   integer NOT NULL,                    -- valid rows the champion trained on
    ADD COLUMN current_mae      double precision NOT NULL,
    ADD COLUMN reference_mae    double precision NOT NULL,
    ADD COLUMN reference_source text NOT NULL CHECK (reference_source IN ('test', 'challenger')),
    ADD COLUMN rule_met         boolean NOT NULL,                    -- MAE up 15% or drift in 3+ features
    ADD COLUMN mlflow_run_id    text NOT NULL,                       -- run of the `drift` experiment with the HTML
    ADD COLUMN generated_at     timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN dag_run_id       text,                                -- NULL when run by hand (backfill)
    ALTER COLUMN mae_change_pct SET NOT NULL,
    ALTER COLUMN n_drifted_features SET NOT NULL,
    ADD UNIQUE (model_version_id, through_race_id);                  -- a rerun writes nothing new

ALTER TABLE drift_feature_results
    ADD COLUMN stattest text NOT NULL CHECK (stattest IN ('ks', 'chisquare'));
