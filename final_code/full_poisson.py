from __future__ import annotations

import base64
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import poisson

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from concurrent.futures import ProcessPoolExecutor, as_completed

from config_paths import REPO_ROOT, RESULTS_DIR

# =========================
# KONFIG
# =========================
BASE_RESULTS = RESULTS_DIR
DETECTIONS_FILE = Path("data/detections_filtered.csv")
PINGS_FILE = Path("data/pings.csv")

OUT_POISSON = REPO_ROOT / "wyniki_poisson"
OUT_NON_POISSON = REPO_ROOT / "wyniki_non_poisson"
OUT_ROOT = REPO_ROOT / "wyniki"

FREQ = "5min"
BIN_SECONDS = pd.Timedelta(FREQ).total_seconds()

# okna z ekspozycją < 2.5 min wyrzucamy
MIN_ON_TIME_S = 150.0

# anomalne okna (p-value): ~3 sigma jednostronnie
P_CUT = 0.00135

# test poissonowości: Fano var/mean na count_eq
MIN_WINDOWS_FOR_TEST = 200
VAR_OVER_MEAN_LOW = 0.8
VAR_OVER_MEAN_HIGH = 1.2

# drop do 10 dla non-poisson
K_MAX = 10

# rysunki okien (dla suspicious)
WINDOW_IMG_DPI = 200


# =========================
# Frame decode
# =========================
def decode_frame_content_to_array(frame_content: str) -> np.ndarray:
    s = str(frame_content).strip()
    if s.lower().startswith("data:image") and "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s, validate=False)
    img = Image.open(BytesIO(raw))
    return np.array(img)


# =========================
# Staty: Fano
# =========================
def poisson_dispersion_stats(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x[np.isfinite(x)]
    n = int(x.shape[0])
    mean = float(x.mean()) if n > 0 else np.nan
    var = float(x.var(ddof=1)) if n > 1 else np.nan
    ratio = float(var / mean) if (np.isfinite(var) and mean > 0) else np.nan
    return {"n_windows": n, "mean": mean, "var": var, "var_over_mean": ratio}


def is_poisson_by_stats(stats: Dict[str, float]) -> bool:
    v = stats.get("var_over_mean", np.nan)
    n = int(stats.get("n_windows", 0))
    return (
        n >= MIN_WINDOWS_FOR_TEST
        and np.isfinite(v)
        and (VAR_OVER_MEAN_LOW <= v <= VAR_OVER_MEAN_HIGH)
    )


# =========================
# Ekspozycja z pingów
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
# Histogram
# =========================
def save_histogram_counts(w: pd.DataFrame, out_dir: Path, device_id: str, var_over_mean: float) -> None:
    if w.empty:
        return
    ww = w[w["on_time_seconds"] > 0].copy()
    if ww.empty:
        return

    ww["count_eq_int"] = np.floor(ww["count_eq"].astype(float) + 0.5).astype(int)
    data = ww["count_eq_int"]
    if data.empty:
        return

    bins = np.arange(0, data.max() + 2) - 0.5

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(data, bins=bins)
    ax.set_xlabel(f"Liczba detekcji ({FREQ})", fontsize=15)
    ax.set_ylabel("Liczba okien", fontsize=15)
    ax.set_title(f"{device_id} | War/Śr={var_over_mean:.3f}", fontsize=17)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.35)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"histogram_okien_{FREQ}.png", dpi=250)
    plt.close(fig)


# =========================
# Rysunek okna: dynamiczna siatka NxN (WSZYSTKIE detekcje)
# =========================
def save_window_images_grid(device_id: str, window_start: pd.Timestamp, det_win: pd.DataFrame, out_png: Path) -> None:
    if det_win.empty:
        return

    d = det_win.sort_values("time").reset_index(drop=True).copy()
    n = len(d)

    side = int(math.ceil(math.sqrt(n)))
    side = max(side, 1)

    # żeby kafelki nie były mikroskopijne przy dużej siatce
    tile_inch = 2.2
    fig_w = max(6.0, side * tile_inch)
    fig_h = max(6.0, side * tile_inch)

    fig, axes = plt.subplots(side, side, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)
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
            ax.set_title("decode error", fontsize=9)

    title = f"{device_id} | okno {window_start.strftime('%Y-%m-%d %H:%M:%S')} UTC | N={n}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=WINDOW_IMG_DPI)
    plt.close(fig)


# =========================
# Drop windows: minimalne k (po największym count_eq)
# =========================
def min_k_and_dropped_windows(w: pd.DataFrame, k_max: int = 10) -> Tuple[Optional[int], pd.DataFrame, Dict[str, float]]:
    if w is None or w.empty or "count_eq" not in w.columns:
        return None, pd.DataFrame(), {}

    w_sorted = w.sort_values("count_eq", ascending=False).reset_index(drop=True)
    x = w_sorted["count_eq"].astype(float)

    if x.shape[0] < MIN_WINDOWS_FOR_TEST:
        return None, pd.DataFrame(), {}

    for k in range(1, k_max + 1):
        if x.shape[0] - k < MIN_WINDOWS_FOR_TEST:
            break
        stats_k = poisson_dispersion_stats(x.iloc[k:])
        if is_poisson_by_stats(stats_k):
            # k: liczba odrzuconych okien (na początku),
            # stats_k: statystyki dla reszty (po "poissonizacji")
            return k, w_sorted.iloc[:k].copy(), stats_k

    return None, pd.DataFrame(), {}


# =========================
# Składanie CSV dla detekcji z listy okien
# =========================
def build_detections_csv(device_id: str, det_all: pd.DataFrame, windows_sel: pd.DataFrame) -> pd.DataFrame:
    """
    device_id, timestamp, latitude, longitude, x, y, frame_content, window_start, t_rel_s
    """
    cols = ["device_id", "timestamp", "latitude", "longitude", "x", "y", "frame_content", "window_start", "t_rel_s"]

    if det_all.empty or windows_sel.empty:
        return pd.DataFrame(columns=cols)

    intervals = windows_sel[["window_start", "window_end"]].copy()
    intervals["window_start"] = pd.to_datetime(intervals["window_start"], utc=True, errors="coerce")
    intervals["window_end"] = pd.to_datetime(intervals["window_end"], utc=True, errors="coerce")
    intervals = intervals.dropna(subset=["window_start", "window_end"])
    if intervals.empty:
        return pd.DataFrame(columns=cols)

    det = det_all.copy()
    det["window_start"] = det["time"].dt.floor(FREQ)

    sel_starts = set(pd.Timestamp(x) for x in intervals["window_start"].tolist())
    det_sel = det[det["window_start"].isin(list(sel_starts))].copy()
    if det_sel.empty:
        return pd.DataFrame(columns=cols)

    out = det_sel[["timestamp", "latitude", "longitude", "x", "y", "frame_content", "window_start"]].copy()
    out.insert(0, "device_id", device_id)
    out["window_start"] = pd.to_datetime(out["window_start"], utc=True, errors="coerce")
    # czas względny: t=0 na początku okna (window_start)
    t = pd.to_datetime(out["timestamp"], unit="ms", utc=True, errors="coerce")
    out["t_rel_s"] = (t - out["window_start"]).dt.total_seconds()
    return out


# =========================
# Jedno urządzenie: compute + zapis
# =========================
def process_one_device(device_id: str) -> Dict[str, Any]:
    det_path = BASE_RESULTS / device_id / DETECTIONS_FILE
    ping_path = BASE_RESULTS / device_id / PINGS_FILE

    if not det_path.exists() or not ping_path.exists():
        return {"device_id": device_id, "status": "skip_missing"}

    try:
        det = pd.read_csv(det_path, usecols=["timestamp", "latitude", "longitude", "frame_content", "x", "y"])
        pings = pd.read_csv(ping_path, usecols=["timestamp", "on_time", "delta_time"])
    except Exception:
        return {"device_id": device_id, "status": "skip_read_error"}

    if det.empty or pings.empty:
        return {"device_id": device_id, "status": "skip_empty"}

    det["time"] = pd.to_datetime(det["timestamp"], unit="ms", utc=True, errors="coerce")
    det = det.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if det.empty:
        return {"device_id": device_id, "status": "skip_empty_after_time_parse"}

    windows = count_on_time(pings, freq=FREQ)
    if windows.empty:
        return {"device_id": device_id, "status": "skip_no_windows"}

    det["window_start"] = det["time"].dt.floor(FREQ)
    counts = det.groupby("window_start").size().rename("count").reset_index()

    windows = windows.merge(counts, on="window_start", how="left")
    windows["count"] = windows["count"].fillna(0).astype(int)

    w = windows[windows["on_time_seconds"] >= MIN_ON_TIME_S].copy()
    if w.empty:
        return {"device_id": device_id, "status": "skip_no_exposure_ge_2p5min"}

    w["count_eq"] = w["count"].astype(float) * (BIN_SECONDS / w["on_time_seconds"].astype(float))

    denom = float(w["on_time_seconds"].sum())
    if denom <= 0:
        return {"device_id": device_id, "status": "skip_bad_exposure_sum"}

    rate = float(w["count"].sum()) / denom

    # !!! ZAMIANA: 'lambda' -> 'lam' (bezpieczna nazwa kolumny)
    w["lam"] = rate * w["on_time_seconds"].astype(float)
    w["p"] = poisson.sf(w["count"].astype(int) - 1, w["lam"].astype(float))

    stats = poisson_dispersion_stats(w["count_eq"])
    is_poi = is_poisson_by_stats(stats)

    cand_w = pd.DataFrame()
    if is_poi:
        cand_w = w[w["p"] < P_CUT].copy().sort_values("p").reset_index(drop=True)

    k_drop = None
    dropped_w = pd.DataFrame()
    stats_after = None
    became_poisson = False
    if not is_poi:
        k_drop, dropped_w, stats_after = min_k_and_dropped_windows(w, k_max=K_MAX)
        became_poisson = (k_drop is not None) and (not dropped_w.empty)

    stats_poisson = stats
    if stats_after is not None and became_poisson:
        stats_poisson = stats_after

    return {
        "device_id": device_id,
        "status": "ok",
        "is_poisson": bool(is_poi),
        "became_poisson": bool(became_poisson),
        "k_drop": int(k_drop) if k_drop is not None else None,
        "stats": stats,
        "stats_poisson": stats_poisson,
        "rate_per_s": float(rate),
        "windows": w,
        "cand_windows": cand_w,
        "dropped_windows": dropped_w,
        "det_all": det,
    }


def write_device_outputs(r: Dict[str, Any]) -> None:
    dev = r["device_id"]
    cls = "poisson" if r["is_poisson"] else "non_poisson"
    out_root = OUT_POISSON if cls == "poisson" else OUT_NON_POISSON
    out_dir = out_root / dev
    out_dir.mkdir(parents=True, exist_ok=True)

    v = float(r["stats"]["var_over_mean"]) if np.isfinite(r["stats"]["var_over_mean"]) else np.nan
    save_histogram_counts(r["windows"], out_dir, dev, v)

    if cls == "poisson":
        cand_w = r["cand_windows"]
        if isinstance(cand_w, pd.DataFrame) and not cand_w.empty:
            cand_det_csv = build_detections_csv(dev, r["det_all"], cand_w)
            if not cand_det_csv.empty:
                cand_det_csv.to_csv(out_dir / "candidates.csv", index=False)

    if cls == "non_poisson" and r["became_poisson"]:
        dropped_w = r["dropped_windows"]
        if isinstance(dropped_w, pd.DataFrame) and not dropped_w.empty:
            susp_csv = build_detections_csv(dev, r["det_all"], dropped_w)
            if not susp_csv.empty:
                susp_csv.to_csv(out_dir / "suspicious.csv", index=False)

            img_dir = out_dir / "suspicious_windows"
            img_dir.mkdir(parents=True, exist_ok=True)

            for rr in dropped_w.itertuples(index=False):
                ws = pd.Timestamp(rr.window_start)
                we = pd.Timestamp(rr.window_end)

                det_win = r["det_all"][
                    (r["det_all"]["time"] >= ws) & (r["det_all"]["time"] < we)
                ][["time", "frame_content"]].copy()

                if det_win.empty:
                    continue

                fname = f"okno_{ws.strftime('%Y%m%d_%H%M%S')}_UTC.png"
                save_window_images_grid(dev, ws, det_win, img_dir / fname)


def save_var_over_mean_histograms(summary_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows).copy()
    if df.empty or "var_over_mean" not in df.columns:
        return

    vals = pd.to_numeric(df["var_over_mean"], errors="coerce")
    vals = vals[np.isfinite(vals)]

    if vals.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    bin_width = 0.1
    bins = np.arange(vals.min(), vals.max() + 1, 1)

    # ===== histogram pełny =====
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=bins)
    ax.axvline(1, linestyle="--")
    ax.set_xlabel("Var / Śr", fontsize=15)
    ax.set_ylabel("Liczba urządzeń", fontsize=15)
    ax.set_title("Histogram rozkładu Var/Śr (pełny zakres)", fontsize=17)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.35)
    fig.tight_layout()

    fig.savefig(out_dir / "hist_var_over_mean_full.png", dpi=250)
    plt.close(fig)

    # ===== histogram przycięty =====
    vals_zoom = vals[vals <= 5]

    if len(vals_zoom) > 0:
        bins_zoom = np.arange(0, 5 + bin_width, bin_width)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.hist(vals_zoom, bins=bins_zoom)
        ax.axvline(1, linestyle="--")
        ax.set_xlabel("Var / Śr", fontsize=15)
        ax.set_ylabel("Liczba urządzeń", fontsize=15)
        ax.set_title("Histogram rozkładu Var/Śr (zakres 0–5)", fontsize=17)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.35)
        fig.tight_layout()

        fig.savefig(out_dir / "hist_var_over_mean_zoom.png", dpi=250)
        plt.close(fig)

# =========================
# Worker równoległy
# =========================
def worker(dev: str) -> Dict[str, Any]:
    r = process_one_device(dev)
    if r["status"] != "ok":
        return {
            "device_id": dev,
            "status": r["status"],
            "is_poisson": False,
            "became_poisson": False,
            "var_over_mean": np.nan,
        }

    write_device_outputs(r)

    stats_p = r.get("stats_poisson", {}) or {}
    mean_p = float(stats_p.get("mean", float("nan")))
    var_p = float(stats_p.get("var", float("nan")))

    stats0 = r.get("stats", {}) or {}
    var_over_mean0 = float(stats0.get("var_over_mean", float("nan")))

    dropped_info: List[Dict[str, Any]] = []
    if r.get("became_poisson") and isinstance(r.get("dropped_windows"), pd.DataFrame):
        dw = r["dropped_windows"]
        if not dw.empty:
            for rr in dw.itertuples(index=False):
                dct = rr._asdict()

                try:
                    ws = pd.Timestamp(dct.get("window_start"))
                    ws_txt = ws.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    ws_txt = ""

                try:
                    c = int(dct.get("count"))
                except Exception:
                    c = 0

                lam_val = dct.get("lam", None)
                if lam_val is None:
                    lam_val = dct.get("lambda", None)
                if lam_val is None:
                    lam_val = dct.get("lambda_", None)
                if lam_val is None:
                    lam_val = dct.get("_lambda", None)

                try:
                    lam_f = float(lam_val)
                except Exception:
                    lam_f = float("nan")

                try:
                    p_f = float(dct.get("p"))
                except Exception:
                    p_f = float("nan")

                dropped_info.append(
                    {
                        "window_start": ws_txt,
                        "count": c,
                        "lambda": lam_f,
                        "p": p_f,
                    }
                )

    return {
        "device_id": dev,
        "status": "ok",
        "is_poisson": bool(r["is_poisson"]),
        "became_poisson": bool(r["became_poisson"]),
        "k_drop": r.get("k_drop"),
        "mean_poisson": mean_p,
        "var_poisson": var_p,
        "var_over_mean": var_over_mean0,
        "dropped_windows": dropped_info,
    }

# =========================
# MAIN
# =========================
def main() -> None:
    summary_rows: List[Dict[str, Any]] = []
    OUT_POISSON.mkdir(parents=True, exist_ok=True)
    OUT_NON_POISSON.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device_ids = sorted([p.name for p in BASE_RESULTS.iterdir() if p.is_dir()])
    total = len(device_ids)

    max_workers = max(1, int((os.cpu_count() or 4) * 0.9))

    processed_ok = 0
    poisson_ok = 0
    skipped = 0
    turn_poisson_rows: List[Tuple[str, Optional[int], float, float, List[Dict[str, Any]]]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, dev): dev for dev in device_ids}

        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 25 == 0:
                print(f"[{done}/{total}] ...")

            res = fut.result()
            summary_rows.append(res)

            if res["status"] != "ok":
                skipped += 1
                continue

            processed_ok += 1
            if res["is_poisson"]:
                poisson_ok += 1
            elif res["became_poisson"]:
                turn_poisson_rows.append(
                    (
                        res["device_id"],
                        res.get("k_drop"),
                        res.get("mean_poisson", float("nan")),
                        res.get("var_poisson", float("nan")),
                        res.get("dropped_windows", []) or [],
                    )
                )

    # zapis urządzeń, które STAŁY SIĘ poissonowskie:
    # format: device_id flag nmax mean var dropped=<ws|count|lambda|p;...>
    # (flag=1, nmax=k_drop lub 0 jeśli brak)
    turn_poisson_rows_sorted = sorted(turn_poisson_rows, key=lambda x: str(x[0]))
    lines = []
    for dev_id, k_drop, mean_p, var_p, dropped in turn_poisson_rows_sorted:
        flag = 1
        nmax = int(k_drop) if k_drop is not None else 0
        mean_s = "nan" if not np.isfinite(mean_p) else f"{float(mean_p):.6f}"
        var_s = "nan" if not np.isfinite(var_p) else f"{float(var_p):.6f}"
        dropped_parts = []
        for d in dropped or []:
            ws_txt = str(d.get("window_start", "")).strip()
            if not ws_txt:
                continue
            try:
                c = int(d.get("count", 0))
            except Exception:
                c = 0
            try:
                lam = float(d.get("lambda", float("nan")))
            except Exception:
                lam = float("nan")
            try:
                pval = float(d.get("p", float("nan")))
            except Exception:
                pval = float("nan")
            lam_s = "nan" if not np.isfinite(lam) else f"{lam:.6g}"
            p_s = "nan" if not np.isfinite(pval) else f"{pval:.6g}"
            dropped_parts.append(f"{ws_txt}|{c}|{lam_s}|{p_s}")

        dropped_blob = ";".join(dropped_parts)
        dropped_field = f" dropped={dropped_blob}" if dropped_blob else ""
        lines.append(f"{dev_id} {flag} {nmax} {mean_s} {var_s}{dropped_field}")

    (OUT_ROOT / "turn_poisson.txt").write_text("\n".join(lines), encoding="utf-8")

    turn_poisson_ids = sorted({dev for dev, _, _, _, _ in turn_poisson_rows})

    raport = (
        f"Przetworzone urządzenia (OK): {processed_ok}\n"
        f"Pominięte (braki / błędy / za mało danych): {skipped}\n"
        f"Poissonowskie (Var/Śr w [{VAR_OVER_MEAN_LOW}, {VAR_OVER_MEAN_HIGH}] i min {MIN_WINDOWS_FOR_TEST} okien): {poisson_ok}\n"
        f"Stały się poissonowskie po odrzuceniu <= {K_MAX} okien: {len(turn_poisson_ids)}\n"
        f"max_workers: {max_workers}\n"
    )
    (OUT_ROOT / "raport.txt").write_text(raport, encoding="utf-8")
    save_var_over_mean_histograms(summary_rows, OUT_ROOT)

    print("\n=== PODSUMOWANIE ===")
    print(f"results/*:                  {total}")
    print(f"OK:                        {processed_ok}")
    print(f"SKIPPED:                   {skipped}")
    print(f"poisson (OK):              {poisson_ok}")
    print(f"turn_poisson (<= {K_MAX}):  {len(turn_poisson_ids)}   ({OUT_ROOT / 'turn_poisson.txt'})")
    print(f"raport:                    {OUT_ROOT / 'raport.txt'}")
    print("DONE.")


if __name__ == "__main__":
    main()