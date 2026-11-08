import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.append(str(TESTS / "features"))  # the in-memory Supabase fake and the synthetic races
sys.path.append(str(TESTS / "ingestion"))  # the synthetic OpenF1 tables and the loaders of test_session
