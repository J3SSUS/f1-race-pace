import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "features")
)  # the in-memory Supabase fake and the synthetic races
