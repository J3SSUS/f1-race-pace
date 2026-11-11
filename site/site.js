const SITE_CONFIG = {
  "anonKey": "sb_publishable_fDMkG8QxRWGevhLd0lhSYQ_8Fu-ucLS",
  "championExplain": {
    "importance": [
      {
        "feature": "grid_position",
        "mae_increase_mean": 0.08390666274810789,
        "mae_increase_std": 0.0063082416407766635
      },
      {
        "feature": "driver_pace_last3",
        "mae_increase_mean": 0.035971283082867934,
        "mae_increase_std": 0.0040675112812858984
      },
      {
        "feature": "team_pace_last3",
        "mae_increase_mean": 0.030304436508246956,
        "mae_increase_std": 0.0038685681953447653
      },
      {
        "feature": "quali_delta_pct",
        "mae_increase_mean": 0.022270732770141376,
        "mae_increase_std": 0.003835829335827694
      },
      {
        "feature": "fp_long_run_delta_pct",
        "mae_increase_mean": 0.006092337636500724,
        "mae_increase_std": 0.001986557233083685
      },
      {
        "feature": "team_pace_this_circuit_prev",
        "mae_increase_mean": 0.004067233649064916,
        "mae_increase_std": 0.0019224905990179767
      },
      {
        "feature": "reached_q3",
        "mae_increase_mean": 0.003079095077589642,
        "mae_increase_std": 0.0015503247228706013
      },
      {
        "feature": "teammate_quali_gap",
        "mae_increase_mean": 0.0026391292440303914,
        "mae_increase_std": 0.0010023985539351315
      },
      {
        "feature": "circuit_type",
        "mae_increase_mean": 0.0009056429949035975,
        "mae_increase_std": 0.0006777251029944922
      },
      {
        "feature": "track_temp",
        "mae_increase_mean": 0.0005048558012651269,
        "mae_increase_std": 0.0006564516044253941
      },
      {
        "feature": "rain_flag",
        "mae_increase_mean": 0.0002922388666638409,
        "mae_increase_std": 0.00019787854303129267
      },
      {
        "feature": "fp_tyre_deg",
        "mae_increase_mean": 0.00021520594582604506,
        "mae_increase_std": 0.00043460215146074745
      },
      {
        "feature": "air_temp",
        "mae_increase_mean": 3.6565789835132e-05,
        "mae_increase_std": 0.0004022502559190216
      },
      {
        "feature": "is_sprint_weekend",
        "mae_increase_mean": 1.1456600384059564e-05,
        "mae_increase_std": 0.00011359125526877684
      },
      {
        "feature": "sprint_pace_delta_pct",
        "mae_increase_mean": -7.875082031114089e-05,
        "mae_increase_std": 0.0003259776073445273
      }
    ],
    "model_version_id": 4
  },
  "snapshotUrl": "snapshot.json",
  "supabaseUrl": "https://xbvpofynygkadrtvnxwd.supabase.co",
  "teamColors": {
    "Alpine": "#2293D1",
    "Aston Martin": "#229971",
    "Audi": "#F50537",
    "Cadillac": "#C9A24B",
    "Ferrari": "#E8002D",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "RB": "#6692FF",
    "Red Bull Racing": "#3671C6",
    "Williams": "#64C4FF"
  },
  "timeoutMs": 8000
};

/* Source of site/site.js. The generator puts `const SITE_CONFIG = {...};` before this file (see src/sitegen/build.py).
   SPEC 16 steps 6 to 12 fill in the data access, the sections and the motion. */
"use strict";
