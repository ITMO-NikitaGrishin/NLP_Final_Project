from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

RAW_FILE = DATA_RAW / "winemag-data-130k-v2.csv"

SEED = 42
TEXT_COL = "description"
POINTS_COL = "points"
PRICE_COL = "price"
LOG_PRICE_COL = "log_price"
META_COLS = ["country", "variety", "province", "region_1", "winery", "taster_name"]
