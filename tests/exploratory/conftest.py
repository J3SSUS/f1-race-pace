import sys
from pathlib import Path

# The throwaway module lives with the notebooks, outside `src`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "exploratory"))
