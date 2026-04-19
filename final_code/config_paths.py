"""
Wspólne ścieżki względem katalogu głównego repozytorium (rodzic katalogu final_code).
Dzięki temu skrypty działają po `git clone`, niezależnie od dysku czy systemu.
"""
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = REPO_ROOT / "results"
PLIKI_CSV_DIR: Path = REPO_ROOT / "pliki_csv"
RAW_DETECTIONS_CSV: Path = PLIKI_CSV_DIR / "data.csv"
RAW_PINGS_CSV: Path = PLIKI_CSV_DIR / "pings.csv"
LIST_DEVICES_TXT: Path = Path(__file__).resolve().parent / "list_of_devices.txt"
AI_MODEL_PATH: Path = Path(__file__).resolve().parent / "AI" / "best_model.pth"
