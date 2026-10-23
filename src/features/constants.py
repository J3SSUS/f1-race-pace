"""Thresholds of the target and the features. They live here, not in `config/training.json` (training knobs)."""

MIN_VALID_CLEAN_LAPS = 15  # a row is valid with at least this many clean laps and a qualifying time
MIN_FIELD_CLEAN_LAPS = 10  # a driver enters the field's pooled median with at least this many
QUALI_CAP_PCT = 3.0
FORM_RACES = 3
