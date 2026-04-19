"""
PIPELINE: anomalies (z obsługą 3. kolumny w became_poisson_k_le_10)
+ POPRAWKA: dla showerów zapis/rysunek obejmuje TYLKO detekcje należące do showera (a nie całe okno)

NOWE ŚCIEŻKI:
- became_poisson_k_le_10.txt: wyniki/became_poisson_k_le_10.txt   (format linii bez zmian)
- foldery urządzeń poisson: wyniki_poisson/<device_id>/
- dropped okna: wyniki_poisson/<device_id>/dropped_windows.csv (preferowane)
"""

from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import poisson

from config_paths import REPO_ROOT, RESULTS_DIR

# =========================
# USTAWIENIA
# =========================
BASE_RESULTS = RESULTS_DIR

POISSON_DIR = REPO_ROOT / "wyniki_poisson"

SUMMARY_DIR = REPO_ROOT / "wyniki"
BECAME_FILE = SUMMARY_DIR / "turn_poisson.txt"

ANOMALIES_DIR = REPO_ROOT / "anomalies"

DETECTIONS_FILE = Path("data/detections_filtered.csv")
PINGS_FILE = Path("data/pings.csv")

FREQ = "5min"
MIN_ON_TIME_S = 150.0
P_CUT = 0.00135  # 3 sigma jednostronnie

GAP_SHOWER_MS = 50.0

MAX_IMGS_PER_PNG = 16  # 4x4

OUT_WINDOWS_CSV = ANOMALIES_DIR / "windows.csv"
OUT_SHOWERS_CSV = ANOMALIES_DIR / "showers_report.csv"
OUT_SHOWERS_TXT = ANOMALIES_DIR / "showers_report.txt"


# =========================
# POMOCNICZE
# =========================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def decode_frame_content_to_array(frame_content: str) -> np.ndarray:
    s = str(frame_content).strip()
    if s.lower().startswith("data:image") and "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s, validate=False)
    img = Image.open(BytesIO(raw))
    return np.array(img)


def fmt_dt(dt_s: float) -> str:
    if not np.isfinite(dt_s):
        return ""
    ms = dt_s * 1000.0
    if ms < 1000:
        return f"Δt={ms:.1f}ms"
    return f"Δt={dt_s:.3f}s"


# =========================
# EKSPOZYCJA Z PINGÓW
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
# INPUT: urządzenia + flagi (+ Nmax)
# =========================
def read_devices_from_poisson_dir(poisson_dir: Path) -> List[str]:
    if not poisson_dir.exists():
        return []
    return sorted([p.name for p in poisson_dir.iterdir() if p.is_dir()])


def read_became_flags(
    became_file: Path,
) -> Dict[str, Tuple[int, Optional[int], Optional[float], Optional[float], List[Dict[str, object]]]]:
    """
    Format linii (rozszerzony):
      <device_id> <flag> [Nmax] [mean] [var]
    gdzie:
      - flag: 1 jeśli urządzenie stało się poissonowskie,
      - Nmax: opcjonalny parametr (historycznie: próg/limit),
      - mean, var: opcjonalnie średnia i wariancja po "poissonizacji".
    """
    out: Dict[str, Tuple[int, Optional[int], Optional[float], Optional[float], List[Dict[str, object]]]] = {}
    if not became_file.exists():
        return out

    for line in became_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        dev = parts[0].strip()

        flag = 0
        nmax: Optional[int] = None
        mean_val: Optional[float] = None
        var_val: Optional[float] = None
        dropped_windows: List[Dict[str, object]] = []

        if len(parts) >= 2:
            try:
                flag = 1 if int(parts[1]) != 0 else 0
            except Exception:
                flag = 0

        if len(parts) >= 3:
            try:
                nmax = int(parts[2])
            except Exception:
                nmax = None

        if len(parts) >= 4:
            try:
                mean_val = float(parts[3])
            except Exception:
                mean_val = None

        if len(parts) >= 5:
            try:
                var_val = float(parts[4])
            except Exception:
                var_val = None

        # opcjonalnie: dropped=ws|count|lambda|p;ws|...
        # UWAGA: może być jako osobny token (bez spacji w środku).
        for tok in parts[5:]:
            if not tok.startswith("dropped="):
                continue
            blob = tok.split("=", 1)[1].strip()
            if not blob:
                continue
            for item in blob.split(";"):
                item = item.strip()
                if not item:
                    continue
                cols = item.split("|")
                if len(cols) < 4:
                    continue
                ws_txt = cols[0].strip()
                try:
                    c = int(cols[1])
                except Exception:
                    c = 0
                try:
                    lam = float(cols[2])
                except Exception:
                    lam = float("nan")
                try:
                    pval = float(cols[3])
                except Exception:
                    pval = float("nan")
                dropped_windows.append({"window_start": ws_txt, "count": c, "lambda": lam, "p": pval})

        out[dev] = (flag, nmax, mean_val, var_val, dropped_windows)

    return out


# =========================
# DROPPED WINDOWS: z turn_poisson.txt (dropped=...)
# =========================
DROPPED_PNG_RE = re.compile(r"dropped_(\d{8})_(\d{6})", re.IGNORECASE)


def get_dropped_window_starts_from_flags(
    device_id: str,
    flag: int,
    dropped_windows: List[Dict[str, object]],
) -> set[pd.Timestamp]:
    """
    Zwraca set window_start dla okien "dropped" danego urządzenia.
    """
    if flag != 1:
        return set()

    if not dropped_windows:
        return set()

    out: set[pd.Timestamp] = set()
    for d in dropped_windows:
        ws_txt = str(d.get("window_start", "")).strip()
        if not ws_txt:
            continue
        ts = pd.to_datetime(ws_txt, utc=True, errors="coerce")
        if pd.notna(ts):
            out.add(pd.Timestamp(ts))
    return out


# =========================
# RYSOWANIE DETEKCJI Z OKNA/SHOWERA
# =========================
def save_window_png(device_id: str, window_start: pd.Timestamp, det_subset: pd.DataFrame, out_png: Path) -> None:

    if det_subset is None or det_subset.empty:
        return

    d = det_subset.sort_values("time").reset_index(drop=True).copy()
    if len(d) > MAX_IMGS_PER_PNG:
        d = d.iloc[:MAX_IMGS_PER_PNG].copy()

    ws = pd.Timestamp(window_start).tz_convert("UTC") if pd.Timestamp(window_start).tzinfo else pd.Timestamp(window_start).tz_localize("UTC")
    we = ws + pd.Timedelta(FREQ)  # koniec okna (np. +5min)

    # format zakresu czasu jak w przykładzie użytkownika
    start_txt = ws.strftime("%Y-%m-%d %H:%M")
    if ws.minute == 55:
        end_txt = we.strftime("%H:%M")          # np. 21:55-22:00
    else:
        end_txt = we.strftime("%M")             # np. 21:50-55

    title = f"Detekcje dla urządzenia {device_id} otrzymane {start_txt}-{end_txt}"

    n = len(d)
    fig_w = max(6, n * 2.2)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4))
    if n == 1:
        axes = [axes]

    for ax in axes:
        ax.axis("off")

    for i, row in enumerate(d.itertuples(index=False)):
        ax = axes[i]
        try:
            img = decode_frame_content_to_array(row.frame_content)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
        except Exception:
            ax.axis("off")

    fig.suptitle(title, fontsize=12, y=0.84)  # tytuł niżej (bliżej obrazków)

    # zostaw miejsce na suptitle, ale minimalne
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.88), pad=0.1)

    # dociśnij jeszcze marginesy (mniej "powietrza" góra/dół)
    fig.subplots_adjust(top=0.88, bottom=0.04, wspace=0.05)

    safe_mkdir(out_png.parent)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_shower_png(
    device_id: str,
    det_shower: pd.DataFrame,
    out_png: Path,
) -> None:
    """
    Rysuje detekcje showera w JEDNYM RZĘDZIE.
    Tytuł nad obrazkami: "Czas między detekcjami Δt = {lista czasów w ms}".
    """
    if det_shower.empty:
        return

    d = det_shower.sort_values("time").reset_index(drop=True).copy()
    n = len(d)

    # czasy między kolejnymi detekcjami w ms
    d["dt_next_s"] = d["time"].shift(-1).sub(d["time"]).dt.total_seconds()
    dt_ms = (d["dt_next_s"].iloc[:-1] * 1000.0).tolist()
    dt_ms_str = ", ".join(f"{t:.1f}" for t in dt_ms) if dt_ms else "—"
    title = f"Czas między detekcjami $\\Delta t$ = {dt_ms_str} ms"

    fig, axes = plt.subplots(1, n, figsize=(max(4, n * 2.5), 4))
    if n == 1:
        axes = [axes]
    for ax in axes:
        ax.axis("off")

    for i, row in enumerate(d.itertuples(index=False)):
        ax = axes[i]
        try:
            img = decode_frame_content_to_array(row.frame_content)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
        except Exception:
            ax.axis("off")

    fig.suptitle(title, fontsize=12, y=0.84)  # tytuł niżej (bliżej obrazków)

    # zostaw miejsce na suptitle, ale minimalne
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.88), pad=0.1)

    # dociśnij jeszcze marginesy (mniej "powietrza" góra/dół)
    fig.subplots_adjust(top=0.88, bottom=0.04, wspace=0.05)

    safe_mkdir(out_png.parent)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Wielki krąg (haversine) w km.
    Jeśli któryś argument jest NaN, zwraca NaN.
    """
    vals = [lat1, lon1, lat2, lon2]
    if any(not np.isfinite(v) for v in vals):
        return float("nan")

    R = 6371.0  # promień Ziemi w km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# =========================
# SHOWERS + ANALIZA OKIEN
# =========================
@dataclass
class ShowerEvent:
    device_id: str
    window_start: pd.Timestamp
    shower_start: pd.Timestamp
    shower_end: pd.Timestamp
    n: int
    duration_ms: float
    correlated: bool
    suspicious_window: bool
    n_det_window: int


def find_showers_in_window(det_win: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, int, float]]:
    """
    Shower = spójna sekwencja >=2 detekcji, gdzie dla KAŻDEJ pary kolejnych detekcji:
      - odstęp czasu < GAP_SHOWER_MS
      - (x,y) są różne (nie może być identycznego (x,y) jak w poprzedniej detekcji)

    Zwraca listę krotek: (start, end, n, duration_ms)
    """
    if det_win is None or det_win.empty:
        return []

    d = det_win.sort_values("time").reset_index(drop=True).copy()

    # wymagane kolumny
    if "time" not in d.columns:
        return []
    if "x" not in d.columns or "y" not in d.columns:
        return []

    # upewnij się, że time/x/y są w sensownych typach
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")

    d = d.dropna(subset=["time", "x", "y"]).reset_index(drop=True)
    if d.empty:
        return []

    t = d["time"]
    dt_ms = t.diff().dt.total_seconds().fillna(np.inf) * 1000.0

    # TRUE jeśli (x,y) różne względem poprzedniej detekcji
    diff_xy_as_prev = ~(d["x"].eq(d["x"].shift(1)) & d["y"].eq(d["y"].shift(1)))
    diff_xy_as_prev = diff_xy_as_prev.fillna(False)

    # kontynuujemy shower tylko jeśli oba warunki spełnione
    continue_shower = (dt_ms < GAP_SHOWER_MS) & diff_xy_as_prev

    # start nowego showera, gdy nie kontynuujemy
    new_shower = ~continue_shower
    shower_id = new_shower.cumsum()

    out: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
    for _, g in d.groupby(shower_id, sort=True):
        if len(g) >= 2:
            start = pd.Timestamp(g["time"].iloc[0])
            end = pd.Timestamp(g["time"].iloc[-1])
            dur_ms = float((end - start).total_seconds() * 1000.0)
            out.append((start, end, int(len(g)), dur_ms))

    return out


def build_windows_and_anomalies_for_device(
    device_id: str,
    flag: int,
    nmax: Optional[int],
    mean_val: Optional[float],
    var_val: Optional[float],
    dropped_windows: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    det_path = BASE_RESULTS / device_id / DETECTIONS_FILE
    ping_path = BASE_RESULTS / device_id / PINGS_FILE

    if not det_path.exists() or not ping_path.exists():
        return (pd.DataFrame(), pd.DataFrame())

    try:
        det = pd.read_csv(det_path, usecols=["timestamp", "latitude", "longitude", "x", "y", "frame_content"],)
        pings = pd.read_csv(ping_path, usecols=["timestamp", "on_time", "delta_time"])
    except Exception:
        return (pd.DataFrame(), pd.DataFrame())

    if det.empty or pings.empty:
        return (pd.DataFrame(), pd.DataFrame())

    det["time"] = pd.to_datetime(det["timestamp"], unit="ms", utc=True, errors="coerce")
    det = det.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if det.empty:
        return (pd.DataFrame(), pd.DataFrame())

    windows = count_on_time(pings, freq=FREQ)
    if windows.empty:
        return (pd.DataFrame(), pd.DataFrame())

    det["window_start"] = det["time"].dt.floor(FREQ)
    counts = det.groupby("window_start").size().rename("count").reset_index()

    windows = windows.merge(counts, on="window_start", how="left")
    windows["count"] = windows["count"].fillna(0).astype(int)

    w = windows[windows["on_time_seconds"] >= MIN_ON_TIME_S].copy()
    if w.empty:
        return (pd.DataFrame(), pd.DataFrame())

    w["device_id"] = device_id
    # jeśli znamy nową średnią / wariancję z etapu poisson,
    # dołącz je do każdego okna tego urządzenia
    w["mean_eq"] = float(mean_val) if mean_val is not None else np.nan
    w["var_eq"] = float(var_val) if var_val is not None else np.nan

    dropped = get_dropped_window_starts_from_flags(device_id, flag, dropped_windows)

    w["is_dropped"] = False
    if dropped:
        w.loc[w["window_start"].isin(list(dropped)), "is_dropped"] = True

    # nmax steruje tylko tym, co pomijamy przy estymacji rate
    if flag == 1 and nmax is not None:
        w["drop_for_rate"] = w["is_dropped"] & (w["count"] > int(nmax))
    elif flag == 1:
        w["drop_for_rate"] = w["is_dropped"]
    else:
        w["drop_for_rate"] = False

    if flag == 1:
        w_rate = w[~w["drop_for_rate"]].copy()
    else:
        w_rate = w.copy()

    w["used_for_rate"] = False
    if not w_rate.empty:
        w.loc[w.index.isin(w_rate.index), "used_for_rate"] = True

    denom = float(w_rate["on_time_seconds"].sum())
    if denom <= 0:
        return (pd.DataFrame(), pd.DataFrame())

    rate = float(w_rate["count"].sum()) / denom
    w["lambda"] = rate * w["on_time_seconds"].astype(float)

    w["p"] = poisson.sf(w["count"].astype(int) - 1, w["lambda"].astype(float))

    anom_windows = w[w["p"] < P_CUT].copy()
    if anom_windows.empty:
        return (w, pd.DataFrame())

    anom_keys = set(anom_windows["window_start"].tolist())
    det_anom = det[det["window_start"].isin(list(anom_keys))].copy()
    det_anom["device_id"] = device_id

    det_anom = det_anom.merge(
        anom_windows[
            [
                "window_start",
                "window_end",
                "on_time_seconds",
                "count",
                "lambda",
                "p",
                "is_dropped",
                "drop_for_rate",
                "used_for_rate",
            ]
        ],
        on="window_start",
        how="left",
    )

    return (w, det_anom)


def add_relative_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje kolumny czasu względnego:
    - t_rel_s: sekundy od window_start (t=0 na początku okna 5-min)
    Wymaga: window_start + time (Timestamp).
    """
    if df is None or df.empty:
        return df
    if "time" not in df.columns or "window_start" not in df.columns:
        return df

    out = df.copy()
    out["window_start"] = pd.to_datetime(out["window_start"], utc=True, errors="coerce")
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["t_rel_s"] = (out["time"] - out["window_start"]).dt.total_seconds()
    return out


def write_window_poisson_txt(out_dir: Path, k: int, lam: float, p_tail: float) -> None:
    lines = []
    lines.append(f"k (liczba detekcji w oknie): {int(k)}")
    lam_s = "nan" if not np.isfinite(lam) else f"{float(lam):.6g}"
    p_s = "nan" if not np.isfinite(p_tail) else f"{float(p_tail):.6g}"
    lines.append(f"lambda: {lam_s}")
    lines.append(f"p_tail = P(X >= k | Poisson(lambda)): {p_s}")
    (out_dir / "window_poisson.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# MAIN
# =========================
def main() -> None:
    safe_mkdir(ANOMALIES_DIR)

    devices_from_dir = set(read_devices_from_poisson_dir(POISSON_DIR))
    flags = read_became_flags(BECAME_FILE)  # dev -> (flag, nmax, mean, var, dropped_windows)
    devices_from_file = set(flags.keys())

    all_devices = sorted(list(devices_from_dir.union(devices_from_file)))
    if not all_devices:
        print("Brak urządzeń: sprawdź wyniki_poisson/ oraz wyniki/became_poisson_k_le_10.txt")
        return

    def get_flag_nmax(dev: str) -> Tuple[int, Optional[int], Optional[float], Optional[float]]:
        v = flags.get(dev)
        if v is None:
            return (0, None, None, None)
        return (v[0], v[1], v[2], v[3])

    def get_dropped_windows(dev: str) -> List[Dict[str, object]]:
        v = flags.get(dev)
        if v is None or len(v) < 5:
            return []
        return list(v[4] or [])

    all_det_anom = []
    window_to_devices: Dict[pd.Timestamp, List[str]] = {}
    cache_det_by_devwin: Dict[Tuple[str, pd.Timestamp], pd.DataFrame] = {}
    shower_candidates: List[ShowerEvent] = []
    # dodatkowo: które urządzenia były aktywne (miały ekspozycję) w danym oknie
    active_by_window: Dict[pd.Timestamp, List[str]] = {}

    for i, dev in enumerate(all_devices, 1):
        flag, nmax, mean_val, var_val = get_flag_nmax(dev)
        dropped_windows = get_dropped_windows(dev)
        w_dev, det_anom = build_windows_and_anomalies_for_device(dev, flag, nmax, mean_val, var_val, dropped_windows)

        if i % 25 == 0:
            print(f"[{i}/{len(all_devices)}] ...")

        # z okien ekspozycji (spełniających MIN_ON_TIME_S) zbierz, które urządzenia były aktywne
        if w_dev is not None and not w_dev.empty:
            for ws in w_dev["window_start"]:
                ws_ts = pd.Timestamp(ws)
                active_by_window.setdefault(ws_ts, []).append(dev)

        if det_anom is None or det_anom.empty:
            continue

        all_det_anom.append(det_anom)

        for ws, g in det_anom.groupby("window_start"):
            ws = pd.Timestamp(ws)
            window_to_devices.setdefault(ws, []).append(dev)
            cache_det_by_devwin[(dev, ws)] = g.copy()

            # showers wyznaczamy na DETEKCJACH Z ANOMALNEGO OKNA (nie na całym urządzeniu)
            showers = find_showers_in_window(g)

            # flaga: czy okno jest "podejrzane" (do wycięcia) na podstawie drop_for_rate / is_dropped
            is_suspicious_window = False
            if "drop_for_rate" in g.columns:
                is_suspicious_window = bool(g["drop_for_rate"].astype(bool).any())
            elif "is_dropped" in g.columns:
                is_suspicious_window = bool(g["is_dropped"].astype(bool).any())

            n_det_window = int(len(g))

            for (s0, s1, n, dur_ms) in showers:
                shower_candidates.append(
                    ShowerEvent(
                        device_id=dev,
                        window_start=ws,
                        shower_start=s0,
                        shower_end=s1,
                        n=n,
                        duration_ms=dur_ms,
                        correlated=False,
                        suspicious_window=is_suspicious_window,
                        n_det_window=n_det_window,
                    )
                )

    # zapis wszystkich anomalnych detekcji
    if not all_det_anom:
        print("Brak anomalnych okien (p < 3 sigma). Kończę.")
        return

    df_all = pd.concat(all_det_anom, ignore_index=True)
    cols = [
        "device_id",
        "window_start",
        "window_end",
        "p",
        "lambda",
        "count",
        "on_time_seconds",
        "is_dropped",
        "drop_for_rate",
        "used_for_rate",
        "mean_eq",
        "var_eq",
        "timestamp",
        "time",
        "latitude",
        "longitude",
        "x",
        "y",
        "frame_content",
    ]
    cols = [c for c in cols if c in df_all.columns]
    df_all = df_all[cols]
    df_all = add_relative_time_columns(df_all)
    df_all.to_csv(OUT_WINDOWS_CSV, index=False)
    print(f"Zapisano anomalne okna: {OUT_WINDOWS_CSV} (wierszy: {len(df_all)})")

    # korelacje okien
    correlated_windows = {ws for ws, devs in window_to_devices.items() if len(set(devs)) >= 2}

    # foldery korelacji
    for ws in sorted(correlated_windows):
        # zbuduj dane dla tego okna
        parts = []
        for dev in sorted(set(window_to_devices.get(ws, []))):
            g = cache_det_by_devwin.get((dev, ws))
            if g is not None and not g.empty:
                parts.append(g)

        if not parts:
            continue

        df = pd.concat(parts, ignore_index=True)
        df = add_relative_time_columns(df)

        # flaga: czy to okno jest "podejrzane" (jakiekolwiek detekcje z drop_for_rate / is_dropped)
        suspicious_mask = None
        if "drop_for_rate" in df.columns:
            suspicious_mask = df["drop_for_rate"].astype(bool)
        elif "is_dropped" in df.columns:
            suspicious_mask = df["is_dropped"].astype(bool)

        is_suspicious_window = bool(suspicious_mask.any()) if suspicious_mask is not None else False

        # czytelna nazwa folderu + flaga (dokładność do minuty – jak okna)
        base_name = pd.Timestamp(ws).strftime("win_%Y-%m-%d_%H-%M")
        suffix = "_SUSP" if is_suspicious_window else "_OK"
        tname = base_name + suffix

        out_dir = ANOMALIES_DIR / tname
        safe_mkdir(out_dir)

        df.to_csv(out_dir / "windows.csv", index=False)

        # dodatkowy raport tekstowy dla skorelowanego okna
        active_devs = set(active_by_window.get(ws, []))
        corr_devs = sorted(set(df["device_id"].astype(str).tolist()))

        lines_txt = []
        lines_txt.append(f"Window start (UTC): {pd.Timestamp(ws)}")
        lines_txt.append(f"Flaga podejrzanego okna (drop_for_rate / is_dropped): {'TAK' if is_suspicious_window else 'NIE'}")
        total_det = int(len(df))
        suspicious_det = int(suspicious_mask.sum()) if suspicious_mask is not None else 0
        lines_txt.append(f"Liczba detekcji w tym skorelowanym oknie (łącznie): {total_det}")
        if is_suspicious_window:
            lines_txt.append(f"Liczba detekcji oznaczonych jako 'do wycięcia': {suspicious_det}")
        lines_txt.append("")
        lines_txt.append("Poisson per urządzenie (k, lambda, p_tail=P(X>=k|lambda)):")
        for dev in sorted(set(df["device_id"].astype(str).tolist())):
            gd = df[df["device_id"].astype(str) == str(dev)]
            if gd.empty:
                continue
            try:
                k = int(pd.to_numeric(gd.get("count"), errors="coerce").iloc[0])
            except Exception:
                k = int(len(gd))
            try:
                lam = float(pd.to_numeric(gd.get("lambda"), errors="coerce").iloc[0])
            except Exception:
                lam = float("nan")
            try:
                p_tail = float(pd.to_numeric(gd.get("p"), errors="coerce").iloc[0])
            except Exception:
                p_tail = float("nan")
            lam_s = "nan" if not np.isfinite(lam) else f"{lam:.6g}"
            p_s = "nan" if not np.isfinite(p_tail) else f"{p_tail:.6g}"
            lines_txt.append(f"  {dev}: k={k}, lambda={lam_s}, p_tail={p_s}")
        lines_txt.append("")
        lines_txt.append(f"Urządzeń w dalszej analizie (łącznie): {len(all_devices)}")
        lines_txt.append(f"Urządzeń aktywnych w tym oknie: {len(active_devs)}")
        lines_txt.append(f"Urządzeń ze skorelowanymi anomaliami w tym oknie: {len(corr_devs)}")
        if active_devs:
            lines_txt.append("Aktywne urządzenia w tym oknie: " + ", ".join(sorted(active_devs)))
        lines_txt.append("")


        # średnie współrzędne per urządzenie i dystanse
        coords: Dict[str, Optional[Tuple[float, float]]] = {}
        for dev in corr_devs:
            gd = df[df["device_id"].astype(str) == str(dev)]
            lat = pd.to_numeric(gd.get("latitude"), errors="coerce")
            lon = pd.to_numeric(gd.get("longitude"), errors="coerce")
            lat = lat[np.isfinite(lat)]
            lon = lon[np.isfinite(lon)]
            if lat.empty or lon.empty:
                coords[dev] = None
            else:
                coords[dev] = (float(lat.mean()), float(lon.mean()))

        lines_txt.append("Średnie współrzędne urządzeń (jeśli dostępne):")
        for dev, c in coords.items():
            if c is None:
                lines_txt.append(f"  {dev}: brak lokacji")
            else:
                lines_txt.append(f"  {dev}: lat={c[0]:.6f}, lon={c[1]:.6f}")

        # dystanse między parami urządzeń z dostępną lokalizacją
        valid_devs = [d for d, c in coords.items() if c is not None]
        if len(valid_devs) >= 2:
            lines_txt.append("")
            lines_txt.append("Odległości między parami urządzeń z poprawnymi współrzędnymi [km]:")
            for i in range(len(valid_devs)):
                for j in range(i + 1, len(valid_devs)):
                    d1, d2 = valid_devs[i], valid_devs[j]
                    (lat1, lon1) = coords[d1]
                    (lat2, lon2) = coords[d2]
                    dist_km = haversine_km(lat1, lon1, lat2, lon2)
                    if np.isfinite(dist_km):
                        lines_txt.append(f"  {d1} - {d2}: {dist_km:.3f} km")
                    else:
                        lines_txt.append(f"  {d1} - {d2}: brak lokacji")
        else:
            lines_txt.append("")
            lines_txt.append("brak lokacji (mniej niż 2 urządzenia z poprawnymi współrzędnymi).")

        (out_dir / "summary.txt").write_text("\n".join(lines_txt), encoding="utf-8")

        for dev in sorted(set(df["device_id"].astype(str).tolist())):
            gd = df[df["device_id"].astype(str) == str(dev)][["time", "frame_content"]].copy()
            if gd.empty:
                continue
            save_window_png(str(dev), pd.Timestamp(ws), gd, out_dir / f"{dev}.png")

    # showers + zapis: TYLKO shower (nie całe okno)
    showers_rows = []
    for ev in shower_candidates:
        ev.correlated = (ev.window_start in correlated_windows)

        showers_rows.append(
            {
                "device_id": ev.device_id,
                "window_start": ev.window_start,
                "shower_start": ev.shower_start,
                "shower_end": ev.shower_end,
                "n": ev.n,
                "duration_ms": ev.duration_ms,
                "correlated": ev.correlated,
                "suspicious_window": ev.suspicious_window,
                "n_det_window": ev.n_det_window,
            }
        )

        if not ev.correlated:
            # nazwa okna tylko do minuty (jak podział okien); ok = bez odrzucenia, sus = okno do odrzucenia
            wname = pd.Timestamp(ev.window_start).strftime("%Y-%m-%d_%H-%M")
            sname = pd.Timestamp(ev.shower_start).strftime("%H-%M-%S-%f")[:-3]  # do ms
            subdir = "sus" if ev.suspicious_window else "ok"
            folder_name = f"{wname}_{ev.device_id}_sh{sname}"
            out_dir = ANOMALIES_DIR / "shower" / subdir / folder_name
            safe_mkdir(out_dir)

            g = cache_det_by_devwin.get((ev.device_id, ev.window_start))
            if g is None or g.empty:
                continue

            # CSV z detekcjami w całym okienku 5-min
            g2 = add_relative_time_columns(g)
            g2.to_csv(out_dir / "window_detections.csv", index=False)

            # raport tekstowy: k + prawdopodobieństwo w Poissonie
            try:
                k = int(len(g))
            except Exception:
                k = int(ev.n_det_window)
            try:
                lam = float(pd.to_numeric(g.get("lambda"), errors="coerce").iloc[0])
            except Exception:
                lam = float("nan")
            try:
                p_tail = float(pd.to_numeric(g.get("p"), errors="coerce").iloc[0])
            except Exception:
                p_tail = float("nan")
            write_window_poisson_txt(out_dir, k=k, lam=lam, p_tail=p_tail)

            # WYCIĘCIE: tylko detekcje należące do showera
            gs = g[(g["time"] >= ev.shower_start) & (g["time"] <= ev.shower_end)].copy()
            if gs.empty:
                continue

            gs2 = add_relative_time_columns(gs)
            gs2.to_csv(out_dir / "shower_detections.csv", index=False)
            # PNG: tytuł z Δt w ms, jeden rząd
            png_name = f"{ev.device_id}.png"
            save_shower_png(
                ev.device_id,
                gs[["time", "frame_content"]].copy(),
                out_dir / png_name,
            )

    df_sh = pd.DataFrame(showers_rows)
    if not df_sh.empty:
        df_sh.to_csv(OUT_SHOWERS_CSV, index=False)

        n_total = int(len(df_sh))
        n_corr = int(df_sh["correlated"].fillna(False).sum())
        n_non = n_total - n_corr

        OUT_SHOWERS_TXT.write_text(
            "\n".join(
                [
                    f"Showers total: {n_total}",
                    f"Showers correlated (window shared with >=2 devices): {n_corr}",
                    f"Showers non-correlated: {n_non}",
                    "",
                    (
                        "Columns: device_id, window_start, shower_start, shower_end, "
                        "n, duration_ms, correlated, suspicious_window, n_det_window"
                    ),
                ]
            ),
            encoding="utf-8",
        )
        print(f"Showers report: {OUT_SHOWERS_CSV} + {OUT_SHOWERS_TXT}")
    else:
        OUT_SHOWERS_TXT.write_text("Showers total: 0\n", encoding="utf-8")
        print("Brak showerów (<10ms).")

    print("DONE.")


if __name__ == "__main__":
    main()