"""
Pipeline:

1) Przechodzi po results/* (urządzenia).
2) Dla każdego urządzenia liczy okna 5-min + ekspozycję z pingów (on_time_seconds).
   Bierze tylko okna z on_time_seconds >= MIN_ON_TIME_S.
3) Sprawdza czy urządzenie jest "poissonowskie" testem Fano (Var/Mean ~ 1) na count_eq:
      count_eq = count * (bin_seconds / on_time_seconds)
4) Dla urządzeń poissonowskich wyznacza "nadwyżkowe" okna Poissona:
      rate = sum(count) / sum(on_time_seconds)
      lambda_i = rate * on_time_seconds_i
      p_i = P(X >= count_i | lambda_i) = poisson.sf(count_i - 1, lambda_i)
   i wybiera okna z p_i < P_CUT.
5) Dla każdego takiego okna zapisuje:
   - folder: OUT_WINDOWS/<device_id>/<YYYY-MM-DD>/<HHMMSS>/
   - PNG: siatka max 16 obrazów (4x4) z frame_content
     nad każdym obrazem: odstęp czasu do kolejnej detekcji (ms lub s)
6) Zapisuje zbiorczy CSV: OUT_CSV_ALL z detekcjami z wszystkich wybranych okien:
   kolumny: window_start, window_end, device_id, timestamp_ms, latitude, longitude, frame_content
7) Wykrywa skorelowane okna (to samo window_start) z >=2 urządzeniami:
   - folder: OUT_SAME_TIME/<YYYY-MM-DD>/<HHMMSS>/
   - CSV z danymi okna
   - TXT: liczba urządzeń + czy którekolwiek ma serię >=2 detekcji z gap <= 10 ms

Wymagane pliki w results/<device_id>/:
- data/detections_filtered.csv (timestamp, latitude, longitude, frame_content)
- data/pings.csv (timestamp, on_time, delta_time)
"""

import base64
import os
import sys
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_paths import REPO_ROOT, RESULTS_DIR

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import poisson


# =========================
# KONFIG
# =========================
BASE_PATH = RESULTS_DIR
DETECTIONS_FILE = Path("data/detections_filtered.csv")
PINGS_FILE = Path("data/pings.csv")

FREQ = "5min"
BIN_SECONDS = pd.Timedelta(FREQ).total_seconds()

# filtr ekspozycji
MIN_ON_TIME_S = 150.0  # 2.5 min

# "poissonowskość" (Fano)
MIN_WINDOWS_FOR_TEST = 200
VAR_OVER_MEAN_LOW = 0.8
VAR_OVER_MEAN_HIGH = 1.2

# nadwyżki (jednostronnie)
P_CUT = 0.00135

# seria ultra-gęsta
GAP_SERIES_S = 0.01  # 10 ms

# obrazki
MAX_IMGS = 16
GRID_N = 4

# outputy
OUT_ROOT = REPO_ROOT / "wybrane_okna"
OUT_WINDOWS = OUT_ROOT / "windows"
OUT_SAME_TIME = OUT_ROOT / "same_time"
OUT_CSV_ALL = OUT_ROOT / "selected_detections.csv"


# =========================
# UTIL
# =========================
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _decode_frame_to_array(frame_content: str) -> np.ndarray:
    s = str(frame_content).strip()
    if s.lower().startswith("data:image") and "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s, validate=False)
    img = Image.open(BytesIO(raw))
    return np.array(img)


def poisson_dispersion_stats(x: pd.Series) -> dict:
    x = x.astype(float)
    n = int(x.shape[0])
    mean = float(x.mean()) if n > 0 else np.nan
    var = float(x.var(ddof=1)) if n > 1 else np.nan
    ratio = float(var / mean) if (mean and mean > 0 and np.isfinite(var)) else np.nan
    return {"n_windows": n, "mean": mean, "var": var, "var_over_mean": ratio}


def is_poisson_device(stats: dict) -> bool:
    v = stats.get("var_over_mean", np.nan)
    return (
        stats.get("n_windows", 0) >= MIN_WINDOWS_FOR_TEST
        and np.isfinite(v)
        and (VAR_OVER_MEAN_LOW <= v <= VAR_OVER_MEAN_HIGH)
    )


def fmt_dt(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""
    if seconds < 1.0:
        return f"{seconds*1000.0:.1f} ms"
    return f"{seconds:.3f} s"


# =========================
# EKSPOZYCJA Z PINGÓW
# =========================
def count_on_time(pings: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    on_time_seconds w oknach freq; aktywność rozsmarowana równomiernie na [start,end].
    """
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

    n = ((p["bin1"] - p["bin0"]) // bin_size + 1).astype(int)
    keep = n > 0
    p = p.loc[keep].copy()
    n = n.loc[keep]

    if p.empty:
        grid["on_time_seconds"] = 0.0
        return grid

    idx = np.repeat(p.index.to_numpy(), n.to_numpy())
    offset = np.concatenate([np.arange(k, dtype=int) for k in n.to_numpy()])

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
# ZAPIS PNG 4x4 dla okna
# =========================
def save_window_grid_png(det_win: pd.DataFrame, out_dir: Path, device_id: str, window_start: pd.Timestamp) -> None:
    """
    det_win: detekcje w oknie, posortowane po czasie, ma kolumny: time, frame_content
    Nad każdym obrazem tytuł: dt do kolejnej detekcji.
    """
    _safe_mkdir(out_dir)

    g = det_win.sort_values("time").reset_index(drop=True).copy()
    if g.empty:
        return

    g = g.iloc[:MAX_IMGS].copy()

    # dt do kolejnej detekcji
    dts = g["time"].shift(-1) - g["time"]
    dts_s = dts.dt.total_seconds()

    fig, axes = plt.subplots(GRID_N, GRID_N, figsize=(12, 12))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, row in enumerate(g.itertuples(index=False)):
        ax = axes[i]
        try:
            img = _decode_frame_to_array(row.frame_content)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            dt_txt = fmt_dt(float(dts_s.iloc[i])) if i < len(g) - 1 else "—"
            ax.set_title(dt_txt, fontsize=10)
            ax.axis("off")
        except Exception:
            ax.axis("off")
            ax.set_title("decode error", fontsize=10)

    ws = window_start.strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f"{device_id} | window_start={ws} UTC | N={len(g)} (max {MAX_IMGS})", fontsize=14)
    plt.tight_layout()

    png_path = out_dir / "detections_4x4.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


# =========================
# WORKER per device: wybierz okna nadwyżek, zapisz PNG, zwróć detekcje do CSV
# =========================
def process_device(device_id: str) -> dict:
    device_dir = BASE_PATH / device_id
    det_path = device_dir / DETECTIONS_FILE
    ping_path = device_dir / PINGS_FILE

    if not det_path.exists() or not ping_path.exists():
        return {"device_id": device_id, "status": "skip_missing", "rows": []}

    try:
        detections = pd.read_csv(det_path, usecols=["timestamp", "latitude", "longitude", "frame_content"])
        pings = pd.read_csv(ping_path, usecols=["timestamp", "on_time", "delta_time"])
    except Exception:
        return {"device_id": device_id, "status": "skip_read_error", "rows": []}

    if detections.empty or pings.empty:
        return {"device_id": device_id, "status": "skip_empty", "rows": []}

    detections["time"] = pd.to_datetime(detections["timestamp"], unit="ms", utc=True, errors="coerce")
    detections = detections.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if detections.empty:
        return {"device_id": device_id, "status": "skip_empty_after_time_parse", "rows": []}

    windows = count_on_time(pings, freq=FREQ)
    if windows.empty:
        return {"device_id": device_id, "status": "skip_no_windows", "rows": []}

    detections["window_start"] = detections["time"].dt.floor(FREQ)
    counts = detections.groupby("window_start").size().rename("count").reset_index()

    windows = windows.merge(counts, on="window_start", how="left")
    windows["count"] = windows["count"].fillna(0).astype(int)

    # tylko okna z sensowną ekspozycją
    w = windows[windows["on_time_seconds"] >= MIN_ON_TIME_S].copy()
    if w.empty:
        return {"device_id": device_id, "status": "skip_no_exposure_ge_2p5min", "rows": []}

    # poissonowskość urządzenia po Fano na count_eq
    w["count_eq"] = w["count"].astype(float) * (BIN_SECONDS / w["on_time_seconds"].astype(float))
    stats = poisson_dispersion_stats(w["count_eq"])
    if not is_poisson_device(stats):
        return {
            "device_id": device_id,
            "status": "not_poisson_device",
            "rows": [],
            "var_over_mean": stats.get("var_over_mean", np.nan),
            "n_windows": stats.get("n_windows", 0),
        }

    # nadwyżkowe okna Poissona
    rate = w["count"].sum() / w["on_time_seconds"].sum()
    w["lambda"] = rate * w["on_time_seconds"]
    w["p"] = poisson.sf(w["count"] - 1, w["lambda"])
    cand = w[w["p"] < P_CUT].copy()
    if cand.empty:
        return {"device_id": device_id, "status": "ok_no_candidates", "rows": []}

    rows_out = []
    # zapis PNG + zbieranie detekcji do CSV
    for r in cand.itertuples(index=False):
        ws = pd.Timestamp(r.window_start)  # tz-aware UTC
        we = pd.Timestamp(r.window_end)

        det_win = detections[(detections["time"] >= ws) & (detections["time"] < we)].copy()
        if det_win.empty:
            continue

        # struktura folderów: OUT_WINDOWS/<device>/<YYYY-MM-DD>/<HHMMSS>/
        date_dir = ws.strftime("%Y-%m-%d")
        time_dir = ws.strftime("%H%M%S")
        out_dir = OUT_WINDOWS / str(device_id) / date_dir / time_dir
        save_window_grid_png(det_win, out_dir, str(device_id), ws)

        # zbiorczy CSV: detekcje + frame_content
        for d in det_win.itertuples(index=False):
            rows_out.append({
                "window_start": ws,
                "window_end": we,
                "device_id": str(device_id),
                "timestamp_ms": int(d.timestamp),
                "latitude": getattr(d, "latitude", np.nan),
                "longitude": getattr(d, "longitude", np.nan),
                "frame_content": getattr(d, "frame_content", ""),
            })

    return {"device_id": device_id, "status": "ok", "rows": rows_out}


# =========================
# same_time: grupowanie skorelowanych okien
# =========================
def save_detections_per_window_histogram(df_all: pd.DataFrame, out_png: Path) -> None:
    if df_all.empty:
        return
    counts = df_all.groupby(["device_id", "window_start"], sort=False).size().to_numpy(dtype=int)
    if counts.size == 0:
        return
    cmax = int(counts.max())
    bins = np.arange(0, cmax + 2) - 0.5

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=bins)
    ax.set_xlabel("Liczba detekcji w oknie", fontsize=15)
    ax.set_ylabel("Liczba okien", fontsize=15)
    ax.set_title("Rozkład liczby detekcji na wybrane okno", fontsize=17)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.35)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def write_same_time_groups(df_all: pd.DataFrame) -> None:
    if df_all.empty:
        return

    # ujednolicenie typów
    df = df_all.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)

    # okna z >=2 urządzeniami
    g = df.groupby("window_start")["device_id"].nunique()
    correlated_starts = g[g >= 2].index

    for ws in correlated_starts:
        sub = df[df["window_start"] == ws].copy()
        if sub.empty:
            continue

        ws = pd.Timestamp(ws)
        date_dir = ws.strftime("%Y-%m-%d")
        time_dir = ws.strftime("%H%M%S")
        out_dir = OUT_SAME_TIME / date_dir / time_dir
        _safe_mkdir(out_dir)

        # CSV danych okna
        csv_path = out_dir / "detections.csv"
        sub.to_csv(csv_path, index=False)

        # TXT: ile urządzeń + czy jest seria <=10ms na którymkolwiek urządzeniu
        n_devices = int(sub["device_id"].nunique())
        has_fast_series = False

        for dev, sdev in sub.groupby("device_id"):
            ts = np.sort(sdev["timestamp_ms"].astype(np.int64).to_numpy())
            if ts.size >= 2:
                dt_ms = np.diff(ts)
                if np.any(dt_ms <= int(GAP_SERIES_S * 1000.0)):
                    has_fast_series = True
                    break

        txt = (
            f"window_start_utc: {ws.isoformat()}\n"
            f"n_devices: {n_devices}\n"
            f"any_series_gap_le_10ms: {has_fast_series}\n"
        )
        (out_dir / "summary.txt").write_text(txt, encoding="utf-8")


# =========================
# MAIN
# =========================
def main() -> None:
    _safe_mkdir(OUT_ROOT)
    _safe_mkdir(OUT_WINDOWS)
    _safe_mkdir(OUT_SAME_TIME)

    device_ids = sorted([p.name for p in BASE_PATH.iterdir() if p.is_dir()])
    if not device_ids:
        print("Brak urządzeń w results/*")
        return

    workers = max(1, (os.cpu_count() or 2) - 1)

    all_rows = []
    status_counts = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_device, dev) for dev in device_ids]
        for f in as_completed(futs):
            res = f.result()
            st = res.get("status", "unknown")
            status_counts[st] = status_counts.get(st, 0) + 1
            all_rows.extend(res.get("rows", []))

    # zbiorczy CSV
    df_all = pd.DataFrame(all_rows)
    if not df_all.empty:
        df_all.to_csv(OUT_CSV_ALL, index=False)
        write_same_time_groups(df_all)
        save_detections_per_window_histogram(df_all, OUT_ROOT / "hist_detections_per_window.png")

    # krótki print
    print("Statusy:")
    for k in sorted(status_counts):
        print(f"  {k}: {status_counts[k]}")
    print(f"Zapisano detekcje do: {OUT_CSV_ALL}")
    print(f"Okna per-device: {OUT_WINDOWS}")
    print(f"Skorelowane okna: {OUT_SAME_TIME}")


if __name__ == "__main__":
    main()
