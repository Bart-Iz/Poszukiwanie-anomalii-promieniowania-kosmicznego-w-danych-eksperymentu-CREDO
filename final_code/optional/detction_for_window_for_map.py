import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_paths import REPO_ROOT, RESULTS_DIR

BASE_PATH = RESULTS_DIR

DEVICES = [
"10202","10580","10604","10763","10970","11140","11414","11501","11615","11775",
"11959","11993","12080","12146","12171","12214","12400","12401","12422","12659",
"12661","12833","13046","13087","1312","13157","13239","13263","1328","13294",
"13357","1343","13577","13602","13637","13686","13687","1414","1453","1505",
"4537","4879","6464","7044","7045","7046","7048","7087","7569","7600",
"7723","7927","7928","8323","9429","9451","9848"
]

START = pd.Timestamp("2019-10-21 05:55:00", tz="UTC")
END   = pd.Timestamp("2019-10-21 06:00:00", tz="UTC")
MID   = START + (END - START) / 2

OUT_FILE = REPO_ROOT / "for_map.csv"

rows = []
devices_with_window_dets = 0
devices_with_reference = 0

for dev in DEVICES:
    det_path = BASE_PATH / dev / "data" / "detections_filtered.csv"

    df = None
    if det_path.exists():
        try:
            df = pd.read_csv(det_path, usecols=["timestamp", "latitude", "longitude"])
        except Exception:
            df = None

    ref_lat = np.nan
    ref_lon = np.nan

    # jeśli mamy jakiekolwiek dane -> wybierz najbliższą detekcję do MID i weź z niej lat/lon
    if df is not None and not df.empty:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        if not df.empty:
            # tylko wiersze z sensowną lokacją, jeśli istnieją
            loc_ok = df[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
            ok_mask = np.isfinite(loc_ok["latitude"]) & np.isfinite(loc_ok["longitude"])
            df_loc = df.loc[ok_mask].copy()

            if not df_loc.empty:
                df_loc["time_diff"] = (df_loc["time"] - MID).abs()
                nearest = df_loc.loc[df_loc["time_diff"].idxmin()]
                ref_lat = float(nearest["latitude"])
                ref_lon = float(nearest["longitude"])

            # detekcje w oknie
            sel = df[(df["time"] >= START) & (df["time"] < END)].copy()
            if not sel.empty:
                sel["device_id"] = dev
                rows.append(sel[["device_id", "timestamp", "latitude", "longitude"]])
                devices_with_window_dets += 1

    # ZAWSZE: wiersz referencyjny timestamp=0
    rows.append(pd.DataFrame([{
        "device_id": dev,
        "timestamp": 0,
        "latitude": ref_lat,
        "longitude": ref_lon
    }]))
    devices_with_reference += 1

out = pd.concat(rows, ignore_index=True)
out.to_csv(OUT_FILE, index=False)

print("Zapisano:", OUT_FILE)
print("Wierszy łącznie:", len(out))
print("Urządzeń w pliku (powinno być 57):", out["device_id"].nunique())
print("Wierszy referencyjnych timestamp=0:", devices_with_reference)
print("Urządzeń z detekcjami w oknie:", devices_with_window_dets)