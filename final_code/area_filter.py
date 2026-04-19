from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from config_paths import RESULTS_DIR

# =========================
# KONFIG
# =========================
BASE_RESULTS = RESULTS_DIR
DETECTIONS_FILE = Path("data/detections_filtered.csv")
PINGS_FILE = Path("data/pings.csv")

# raport zapisujemy obok tego pliku .py
SCRIPT_DIR = Path(__file__).resolve().parent
RAPORT_TXT = SCRIPT_DIR / "raport_area_filter.txt"


FREQ = "5min"
MIN_ON_TIME_S = 150.0

# filtr obszaru
RADIUS = 50                 # box: ±RADIUS px w x i y
MIN_DET_IN_WINDOW = 5       # analizujemy tylko okna >= 5 detekcji
CUT_MIN_IN_BOX = 4          # definicja "wycięcia": >= 4 w tym samym obszarze
NEIGHBOR_WINDOWS = 2        # sprawdzamy ±2 okna
NEIGHBOR_MIN_IN_BOX = 4     # dla sąsiadów też >=4

# bezpieczeństwo: nie nadpisuj pliku pustym wynikiem
DO_NOT_OVERWRITE_IF_EMPTY = True


# =========================
# Okna z ekspozycji
# =========================
def count_on_time(pings: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    p = pings.rename(columns={"timestamp": "start", "delta_time": "dt", "on_time": "active_ms"}).copy()

    p["start"] = pd.to_datetime(p["start"], unit="ms", utc=True, errors="coerce")
    p["dt"] = pd.to_timedelta(p["dt"], unit="ms", errors="coerce")
    p["end"] = p["start"] + p["dt"]
    p["active_seconds"] = p["active_ms"] / 1000.0

    p = p.dropna(subset=["start", "end", "dt", "active_seconds"])
    p = p[(p["dt"] > pd.Timedelta(0)) & (p["end"] > p["start"])]

    if p.empty:
        return pd.DataFrame(columns=["window_start", "window_end", "on_time_seconds"])

    p = p.sort_values("start").reset_index(drop=True)
    bin_size = pd.Timedelta(freq)

    grid_start = p["start"].min().floor(freq)
    grid_end = p["end"].max().ceil(freq)

    grid = pd.DataFrame({"window_start": pd.date_range(grid_start, grid_end, freq=freq, tz="UTC")})
    grid["window_end"] = grid["window_start"] + bin_size

    p["bin0"] = p["start"].dt.floor(freq)
    p["bin1"] = (p["end"] - pd.Timedelta(microseconds=1)).dt.floor(freq)

    n_bins = ((p["bin1"] - p["bin0"]) // bin_size + 1).astype(int)
    keep = n_bins > 0
    p = p.loc[keep].copy()
    n_bins = n_bins.loc[keep]

    if p.empty:
        grid["on_time_seconds"] = 0.0
        return grid

    idx = np.repeat(p.index.to_numpy(), n_bins.to_numpy())
    offset = np.concatenate([np.arange(k, dtype=int) for k in n_bins.to_numpy()])

    tmp = p.loc[idx, ["start", "end", "bin0", "active_seconds"]].copy()
    tmp["window_start"] = tmp["bin0"] + offset * bin_size
    tmp["window_end"] = tmp["window_start"] + bin_size

    left = tmp[["start", "window_start"]].max(axis=1)
    right = tmp[["end", "window_end"]].min(axis=1)
    overlap_s = (right - left).dt.total_seconds().clip(lower=0.0)

    interval_s = (tmp["end"] - tmp["start"]).dt.total_seconds()
    interval_s = np.where(interval_s <= 0, np.nan, interval_s)

    tmp["on_time_seconds"] = overlap_s * (tmp["active_seconds"] / interval_s)

    agg = tmp.groupby("window_start")["on_time_seconds"].sum().reset_index()
    out = grid.merge(agg, on="window_start", how="left")
    out["on_time_seconds"] = out["on_time_seconds"].fillna(0.0)
    return out


# =========================
# Hotspot: znajdź box ±R
# Definicja hotspotu: >= CUT_MIN_IN_BOX w boxie
# =========================
def find_hotspot_box_if_any(
    g: pd.DataFrame,
    radius: int,
    cut_min_in_box: int,
) -> Optional[Tuple[float, float, float, float, int]]:
    x = pd.to_numeric(g["x"], errors="coerce")
    y = pd.to_numeric(g["y"], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return None

    xx = x[m].to_numpy()
    yy = y[m].to_numpy()

    # zgrubne binowanie co (2R+1) px, wybierz najgęstszy bin
    cell = 2 * radius + 1
    bx = np.floor(xx / cell).astype(int)
    by = np.floor(yy / cell).astype(int)

    keys = np.stack([bx, by], axis=1)
    uniq, cnt = np.unique(keys, axis=0, return_counts=True)
    i_max = int(np.argmax(cnt))

    # szybki warunek "co najmniej 4 w obszarze"
    if int(cnt[i_max]) < cut_min_in_box:
        return None

    bxp, byp = uniq[i_max]
    in_cell = (bx == bxp) & (by == byp)

    # centrum: mediany w tej komórce
    cx = float(np.median(xx[in_cell]))
    cy = float(np.median(yy[in_cell]))

    xmin, xmax = cx - radius, cx + radius
    ymin, ymax = cy - radius, cy + radius

    in_box = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    n_in_box = int(in_box.sum())

    if n_in_box < cut_min_in_box:
        return None

    return xmin, xmax, ymin, ymax, n_in_box


# =========================
# Filtr 1 urządzenia
# =========================
def apply_area_filter_one_device(det: pd.DataFrame, pings: pd.DataFrame) -> Dict[str, Any]:
    windows = count_on_time(pings, freq=FREQ)
    if windows.empty:
        return {"status": "skip_no_windows"}

    det = det.copy()
    det["time"] = pd.to_datetime(det["timestamp"], unit="ms", utc=True, errors="coerce")
    det = det.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if det.empty:
        return {"status": "skip_empty_det"}

    det["window_start"] = det["time"].dt.floor(FREQ)
    counts = det.groupby("window_start").size().rename("count").reset_index()

    # uodpornienie na konflikt typów object vs datetime64[ns, UTC]
    windows = windows.copy()
    windows["window_start"] = pd.to_datetime(windows["window_start"], utc=True, errors="coerce")
    counts["window_start"] = pd.to_datetime(counts["window_start"], utc=True, errors="coerce")

    windows = windows.merge(counts, on="window_start", how="left")
    windows["count"] = windows["count"].fillna(0).astype(int)

    w = windows[(windows["count"] > 0) & (windows["on_time_seconds"] >= MIN_ON_TIME_S)].copy()
    if w.empty:
        return {"status": "skip_no_windows_with_det"}

    w = w.sort_values("window_start").reset_index(drop=True)
    w["window_id"] = np.arange(len(w), dtype=int)

    det2 = det.merge(w[["window_start", "window_end", "window_id"]], on="window_start", how="inner")
    if det2.empty:
        return {"status": "skip_no_det_after_merge"}

    hotspot_by_wid: Dict[int, Tuple[float, float, float, float, int]] = {}
    for wid, g in det2.groupby("window_id", sort=False):
        if len(g) < MIN_DET_IN_WINDOW:
            continue
        box = find_hotspot_box_if_any(g, radius=RADIUS, cut_min_in_box=CUT_MIN_IN_BOX)
        if box is not None:
            hotspot_by_wid[int(wid)] = box

    if not hotspot_by_wid:
        return {"status": "ok", "det_filtered": det2, "det_removed": det2.iloc[0:0].copy()}

    to_remove = pd.Series(False, index=det2.index)

    idx_by_wid = {int(k): v.index for k, v in det2.groupby("window_id")}
    max_wid = int(det2["window_id"].max())

    for wid, (xmin, xmax, ymin, ymax, _n_in_box) in hotspot_by_wid.items():
        # usuwamy TYLKO w tym samym 5‑minutowym oknie (bez sąsiadów)
        idx0 = idx_by_wid.get(wid)
        if idx0 is None or len(idx0) == 0:
            continue

        g0 = det2.loc[idx0]
        in_box0 = (
            (g0["x"].astype(float) >= xmin) & (g0["x"].astype(float) <= xmax) &
            (g0["y"].astype(float) >= ymin) & (g0["y"].astype(float) <= ymax)
        )
        to_remove.loc[g0.index[in_box0]] = True

    det_removed = det2.loc[to_remove].copy()
    det_filtered = det2.loc[~to_remove].copy()

    return {"status": "ok", "det_filtered": det_filtered, "det_removed": det_removed}


# =========================
# Worker (równolegle) — NADPISUJE ORYGINALNY PLIK
# =========================
def worker(device_id: str) -> Dict[str, Any]:
    det_path = BASE_RESULTS / device_id / DETECTIONS_FILE
    ping_path = BASE_RESULTS / device_id / PINGS_FILE

    if not det_path.exists() or not ping_path.exists():
        return {"device_id": device_id, "status": "skip_missing", "removed": 0, "total": 0}

    try:
        det = pd.read_csv(det_path, usecols=["timestamp", "latitude", "longitude", "frame_content", "x", "y"])
        pings = pd.read_csv(ping_path, usecols=["timestamp", "on_time", "delta_time"])
    except Exception:
        return {"device_id": device_id, "status": "skip_read_error", "removed": 0, "total": 0}

    if det.empty:
        return {"device_id": device_id, "status": "skip_empty_det", "removed": 0, "total": 0}

    out = apply_area_filter_one_device(det, pings)
    if out["status"] != "ok":
        return {"device_id": device_id, "status": out["status"], "removed": 0, "total": int(len(det))}

    det_f = out["det_filtered"]
    det_r = out["det_removed"]

    removed = int(len(det_r))
    total = int(len(det_f) + len(det_r))

    if DO_NOT_OVERWRITE_IF_EMPTY and det_f.empty:
        return {"device_id": device_id, "status": "would_overwrite_empty", "removed": removed, "total": total}

    # NADPISANIE pliku wejściowego (bez dodatkowych plików)
    cols = ["timestamp", "latitude", "longitude", "frame_content", "x", "y"]
    det_f[cols].to_csv(det_path, index=False)

    return {"device_id": device_id, "status": "ok", "removed": removed, "total": total}


# =========================
# MAIN
# =========================
def main() -> None:
    device_ids = sorted([p.name for p in BASE_RESULTS.iterdir() if p.is_dir()])
    total_devices = len(device_ids)

    max_workers = max(1, int((os.cpu_count() or 4) * 0.9))

    rows: List[Dict[str, Any]] = []
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, dev) for dev in device_ids]
        for fut in as_completed(futs):
            done += 1
            if done % 25 == 0:
                print(f"[{done}/{total_devices}] ...")
            rows.append(fut.result())

    df = pd.DataFrame(rows).sort_values(["status", "removed"], ascending=[True, False])


    lines = []
    lines.append(f"Liczba urządzeń w results/: {total_devices}")
    lines.append(f"max_workers: {max_workers}")
    lines.append("")
    lines.append(
        f"Parametry filtra: RADIUS={RADIUS}, CUT_MIN_IN_BOX={CUT_MIN_IN_BOX}, "
        f"MIN_DET_IN_WINDOW={MIN_DET_IN_WINDOW}, NEIGHBOR_WINDOWS={NEIGHBOR_WINDOWS}, "
        f"NEIGHBOR_MIN_IN_BOX={NEIGHBOR_MIN_IN_BOX}, MIN_ON_TIME_S={MIN_ON_TIME_S}"
    )
    lines.append("")
    lines.append("device_id, status, removed, total")
    for r in df.itertuples(index=False):
        lines.append(f"{r.device_id}, {r.status}, {r.removed}, {r.total}")

    RAPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    ok = int((df["status"] == "ok").sum())
    removed_sum = int(df.loc[df["status"] == "ok", "removed"].sum())

    print("\n=== PODSUMOWANIE ===")
    print(f"devices total: {total_devices}")
    print(f"ok:            {ok}")
    print(f"removed sum:   {removed_sum}")
    print(f"raport.txt:    {RAPORT_TXT}")
    print("DONE.")


if __name__ == "__main__":
    main()