import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def decode_frame_content(frame_content: str) -> Image.Image:
    s = str(frame_content).strip()
    if s.lower().startswith("data:image") and "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s, validate=False)
    return Image.open(BytesIO(raw)).convert("RGB")


def to_gray_uint8(img_rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(img_rgb, dtype=np.uint8)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def paste_with_subpixel_shift_to_canvas(
    img: Image.Image,
    canvas_size: tuple[int, int],
    offset_x: float,
    offset_y: float,
    fill=(0, 0, 0),
) -> Image.Image:
    """
    Zwraca obraz o rozmiarze canvas_size, na który img jest wstawiony z przesunięciem (offset_x, offset_y)
    względem lewego górnego rogu canvasa. offset_x/y mogą być float.

    PIL.Image.transform używa mapowania output->input:
      x_in = a*x_out + b*y_out + c
      y_in = d*x_out + e*y_out + f
    Dla translacji w prawo o +offset_x: x_in = x_out - offset_x => c = -offset_x
    """
    Wc, Hc = canvas_size
    a, b, c = 1.0, 0.0, -float(offset_x)
    d, e, f = 0.0, 1.0, -float(offset_y)

    return img.transform(
        (Wc, Hc),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR,
        fillcolor=fill,
    )


def crop_to_content_rgb(
    out_rgb: np.ndarray,
    margin: int = 0,     # zmniejsz z 2 → 0 albo nawet -2
    thr: int = 15,       # zwiększ z 5 → 10–30
    force_square: bool = False,
    square_size: int = 32,
) -> np.ndarray:
    """
    Bardziej agresywne przycinanie.
    margin:
        0   = bardzo ciasno
       -2   = jeszcze ciaśniej
    thr:
       większy = ignoruje słabe piksele
    force_square:
       jeśli True → wytnie kwadrat square_size x square_size wokół środka
    """

    mask = (out_rgb[..., 0] > thr) | (out_rgb[..., 2] > thr)

    if not mask.any():
        return out_rgb

    ys, xs = np.where(mask)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # margin może być ujemny
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(out_rgb.shape[0] - 1, y1 + margin)
    x1 = min(out_rgb.shape[1] - 1, x1 + margin)

    cropped = out_rgb[y0:y1+1, x0:x1+1]

    if force_square:

        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2

        half = square_size // 2

        y0 = max(0, cy - half)
        x0 = max(0, cx - half)

        y1 = y0 + square_size
        x1 = x0 + square_size

        cropped = out_rgb[y0:y1, x0:x1]

    return cropped


def overlay_from_csv_align_xy(
    csv_path: str | Path,
    row1: int,
    row2: int,
    out_png: str | Path = "overlay_aligned.png",
    pad: int = 50,
    crop_margin: int = -20,
    crop_thr: int = 5,
) -> None:
    """
    row1 -> czerwony (R), row2 -> niebieski (B)
    Wyrównanie: środki (x,y) z CSV mają się pokryć.
    Na końcu przycina puste marginesy.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie ma takiego pliku: {csv_path}")

    df = pd.read_csv(csv_path)

    # numery wierszy (0..N-1) -> iloc
    r1 = df.iloc[int(row1)]
    r2 = df.iloc[int(row2)]

    x1 = float(r1["x"])
    y1 = float(r1["y"])
    x2 = float(r2["x"])
    y2 = float(r2["y"])

    img1 = decode_frame_content(r1["frame_content"])
    img2 = decode_frame_content(r2["frame_content"])

    # jeśli rozmiary różne -> dopasuj img2 do img1
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, resample=Image.BILINEAR)

    W, H = img1.size

    # przesunięcie img2, żeby (x2,y2) trafiło w (x1,y1)
    dx = x1 - x2
    dy = y1 - y2

    # canvas z zapasem
    Wc = int(np.ceil(W + abs(dx) + 2 * pad))
    Hc = int(np.ceil(H + abs(dy) + 2 * pad))

    # pozycja img1 na canvasie
    off1x = pad + max(0.0, -dx)
    off1y = pad + max(0.0, -dy)

    # pozycja img2 na canvasie
    off2x = off1x + dx
    off2y = off1y + dy

    c1 = paste_with_subpixel_shift_to_canvas(img1, (Wc, Hc), off1x, off1y, fill=(0, 0, 0))
    c2 = paste_with_subpixel_shift_to_canvas(img2, (Wc, Hc), off2x, off2y, fill=(0, 0, 0))

    g1 = to_gray_uint8(c1)
    g2 = to_gray_uint8(c2)

    out = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    out[..., 0] = g1  # R
    out[..., 2] = g2  # B

    # PRZYTNij puste marginesy
    out = crop_to_content_rgb(out, margin=crop_margin, thr=crop_thr)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(out_png)

    print("Zapisano:", out_png.resolve())
    print(f"dx={dx:.3f}px, dy={dy:.3f}px (img2 przesunięty względem img1)")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_paths import REPO_ROOT

    # Przykład: podmień ścieżkę na własny plik shower_detections.csv z anomalies/shower/...
    overlay_from_csv_align_xy(
        csv_path=REPO_ROOT / "anomalies/shower/ok/2018-10-30_16-40_6096_sh16-43-45-152/shower_detections.csv",
        row1=0,
        row2=2,
        out_png=REPO_ROOT / "wynik_overlay.png",
        pad=50,
    )