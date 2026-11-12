from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS_FILE = RESULTS / "metrics.csv"

RAW_FILENAME = "winemag-data-130k-v2.csv"
# На Kaggle датасет подключается как input, скачивать не нужно
_KAGGLE_INPUT = Path("/kaggle/input/wine-reviews") / RAW_FILENAME
RAW_FILE = _KAGGLE_INPUT if _KAGGLE_INPUT.exists() else DATA_RAW / RAW_FILENAME

SEED = 42
TEXT_COL = "description"
POINTS_COL = "points"
PRICE_COL = "price"
LOG_PRICE_COL = "log_price"
META_COLS = ["country", "province", "region_1", "variety", "winery", "taster_name"]
