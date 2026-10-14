"""Stages 3–5: binarisation, grid-line extraction and a regular cell lattice."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GridGeometry:
    x_lines: np.ndarray  # cols + 1 vertical line positions
    y_lines: np.ndarray  # rows + 1 horizontal line positions

    @property
    def rows(self) -> int:
        return len(self.y_lines) - 1

    @property
    def cols(self) -> int:
        return len(self.x_lines) - 1

    def cell_bounds(self, r: int, c: int, pad: int = 0) -> tuple[int, int, int, int]:
        return (
            int(self.x_lines[c]) + pad,
            int(self.y_lines[r]) + pad,
            int(self.x_lines[c + 1]) - pad,
            int(self.y_lines[r + 1]) - pad,
        )

    def cell_center(self, r: int, c: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self.cell_bounds(r, c)
        return (x0 + x1) // 2, (y0 + y1) // 2


# ---- Stage 3: CamScanner-style binarisation ---------------------------------------
def white_balance(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def xdog(
    gray: np.ndarray,
    gamma: float = 0.98,
    sigma: float = 0.8,
    k: float = 2.0,
    eps: float = -0.1,
    phi: float = 200,
) -> np.ndarray:
    g = gray.astype(np.float32)
    d = cv2.GaussianBlur(g, (0, 0), sigma) - gamma * cv2.GaussianBlur(g, (0, 0), sigma * k)
    x = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def sauvola(gray: np.ndarray, w: int = 25, k: float = 0.34, r: float = 128) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (w, w))
    sqmean = cv2.boxFilter(g * g, cv2.CV_32F, (w, w))
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    thresh = mean * (1 + k * ((std / r) - 1))
    return (g < thresh).astype(np.uint8) * 255


def binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(white_balance(img), cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return sauvola(xdog(gray, sigma=0.7, k=1.6), w=25, k=0.36)


def line_masks(bw: np.ndarray, divisor: int = 18) -> tuple[np.ndarray, np.ndarray]:
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // divisor), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // divisor)))
    return (
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk, iterations=2),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk, iterations=2),
    )


# ---- Stage 3.1: line positions -------------------------------------------------------
def detect_line_positions(bin_img: np.ndarray, vertical: bool) -> list[int]:
    lines = cv2.HoughLinesP(
        bin_img, 1, np.pi / 180, threshold=50, minLineLength=bin_img.shape[1] // 3, maxLineGap=30
    )
    if lines is None:
        return []
    out = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if vertical and abs(x1 - x2) < 10:
            out.append(int((x1 + x2) // 2))
        elif not vertical and abs(y1 - y2) < 10:
            out.append(int((y1 + y2) // 2))
    return out


def merge_close(values: list[int], thresh: int = 20) -> list[int]:
    if not values:
        return []
    vals = sorted(values)
    merged = [vals[0]]
    for v in vals[1:]:
        if abs(v - merged[-1]) < thresh:
            merged[-1] = (merged[-1] + v) // 2
        else:
            merged.append(v)
    return merged


def force_uniform(lines: list[int], expected_cells: int, limit_max: int) -> np.ndarray:
    """Regular lattice starting at the first detected line with the median spacing."""
    arr = np.array(sorted(lines))
    if len(arr) < 2:
        step = limit_max // expected_cells
        start = 0
    else:
        step = int(np.median(np.diff(arr)))
        start = int(arr[0])
    out = [start + i * step for i in range(expected_cells + 1)]
    out[-1] = min(limit_max - 1, max(out[-1], out[-2] + 1))
    return np.array(out, dtype=int)


def fit_grid(img_masked: np.ndarray, rows: int, cols: int) -> tuple[GridGeometry, dict[str, np.ndarray]]:
    """Detect grid lines on the (solution-box-masked) image and return a rows × cols lattice."""
    bw = binarize(img_masked)
    h_bin, v_bin = line_masks(bw)
    grid_mask = cv2.morphologyEx(cv2.bitwise_or(h_bin, v_bin), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    xs = merge_close(detect_line_positions(v_bin, vertical=True), 20)
    ys = merge_close(detect_line_positions(h_bin, vertical=False), 20)
    # keep only intersections that lie on the real grid
    pts = [
        (x, y)
        for y in ys
        for x in xs
        if 0 <= y < grid_mask.shape[0] and 0 <= x < grid_mask.shape[1] and grid_mask[y, x] > 0
    ]
    xs = sorted({p[0] for p in pts}) or xs
    ys = sorted({p[1] for p in pts}) or ys
    xs, ys = merge_close(xs, 6), merge_close(ys, 6)

    H, W = img_masked.shape[:2]
    x_lines = np.array(xs) if len(xs) == cols + 1 else force_uniform(xs, cols, W)
    y_lines = np.array(ys) if len(ys) == rows + 1 else force_uniform(ys, rows, H)
    return GridGeometry(x_lines, y_lines), {"binary": bw, "grid_mask": grid_mask}


def mask_cells(img: np.ndarray, geo: GridGeometry, region: tuple[int, int, int, int]) -> np.ndarray:
    """Grey out a rectangular block of cells (r0, c0, r1, c1 inclusive) — e.g. the solution box."""
    r0, c0, r1, c1 = region
    r1, c1 = min(r1, geo.rows - 1), min(c1, geo.cols - 1)
    x1, y1 = int(geo.x_lines[c0]), int(geo.y_lines[r0])
    x2, y2 = int(geo.x_lines[c1 + 1]), int(geo.y_lines[r1 + 1])
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), -1)
    out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (35, 35), 0)
    return out
