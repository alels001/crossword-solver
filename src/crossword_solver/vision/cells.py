"""Stage 6: classify cells as clue ("question") cells using ink and stroke statistics."""

from __future__ import annotations

import cv2
import numpy as np

from .grid import GridGeometry


def _remove_long_strokes(bw: np.ndarray, frac: float) -> np.ndarray:
    h, w = bw.shape
    kx, ky = max(5, int(w * frac)), max(5, int(h * frac))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    cleaned = cv2.subtract(bw, cv2.bitwise_or(hor, ver))
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def prep_bin(bgr: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))


def text_score(roi_bgr: np.ndarray) -> tuple[float, float, int]:
    """Weighted mix of ink ratio, number of letter-sized components and edge density."""
    g = cv2.normalize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX)
    h, w = g.shape
    area = float(h * w)
    bw = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    b = max(1, int(0.06 * min(h, w)))
    bw[:b, :] = 0
    bw[-b:, :] = 0
    bw[:, :b] = 0
    bw[:, -b:] = 0
    bw2 = _remove_long_strokes(bw, 0.55)
    ink = bw2.sum() / (255.0 * area)
    cnts, _ = cv2.findContours(bw2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    small = sum(1 for c in cnts if area * 0.0006 <= cv2.contourArea(c) <= area * 0.08)
    sx = cv2.Sobel(g, cv2.CV_16S, 1, 0)
    sy = cv2.Sobel(g, cv2.CV_16S, 0, 1)
    edge_frac = (cv2.convertScaleAbs(np.abs(sx) + np.abs(sy)) > 40).sum() / area
    score = 0.55 * min(1.0, ink / 0.05) + 0.25 * min(1.0, small / 8.0) + 0.20 * min(1.0, edge_frac / 0.08)
    return float(score), float(ink), int(small)


def _center_box_has_text(
    roi: np.ndarray,
    center_ratio: float = 0.44,
    ink_ratio_th: float = 0.035,
    min_comp_area: int = 18,
    min_spread: float = 0.15,
) -> bool:
    """Stage 6.1 recovery: heavy, spread-out ink in the cell centre (anti-noise checks)."""
    bw = prep_bin(roi)
    if bw.sum() == 0:
        return False
    bw = _remove_long_strokes(bw, 0.5)
    h, w = bw.shape
    box = bw[
        int(h * center_ratio) : int(h * (1 - center_ratio)),
        int(w * center_ratio) : int(w * (1 - center_ratio)),
    ]
    if box.size == 0 or np.count_nonzero(box) / box.size < ink_ratio_th:
        return False
    ys, xs = np.where(box > 0)
    if len(xs) < 10:
        return False
    if np.std(xs) / max(1.0, box.shape[1]) < min_spread or np.std(ys) / max(1.0, box.shape[0]) < min_spread:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(box, connectivity=8)
    return any(
        stats[i, cv2.CC_STAT_AREA] >= min_comp_area
        and stats[i, cv2.CC_STAT_WIDTH] >= 4
        and stats[i, cv2.CC_STAT_HEIGHT] >= 4
        for i in range(1, n)
    )


def detect_clue_cells(
    img: np.ndarray, geo: GridGeometry, excluded: set[tuple[int, int]] | None = None
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return clue cells (row, col) and the per-cell score map."""
    excluded = excluded or set()
    scores = np.zeros((geo.rows, geo.cols), np.float32)
    inks = np.zeros_like(scores)
    for r in range(geo.rows):
        for c in range(geo.cols):
            if (r, c) in excluded:
                continue
            x0, y0, x1, y1 = geo.cell_bounds(r, c)
            roi = img[y0:y1, x0:x1]
            if roi.size:
                scores[r, c], inks[r, c], _ = text_score(roi)

    threshold = max(0.30, float(np.percentile(scores, 80)))
    cells = {
        (r, c)
        for r in range(geo.rows)
        for c in range(geo.cols)
        if scores[r, c] >= threshold and (r, c) not in excluded
    }

    # Top-left cell is a clue in practically every Schwedenrätsel; accept weaker evidence there.
    if (0, 0) not in cells and (0, 0) not in excluded:
        x0, y0, x1, y1 = geo.cell_bounds(0, 0)
        _, ink0, small0 = text_score(img[y0:y1, x0:x1])
        mean_ink = float(np.mean(inks[inks > 0])) if np.any(inks > 0) else 0.0
        if (small0 >= 2 and ink0 > 0.4 * mean_ink) or ink0 > 0.015:
            cells.add((0, 0))

    # Stage 6.1: centre-box recovery for cells the global threshold missed.
    for r in range(geo.rows):
        for c in range(geo.cols):
            if (r, c) in cells or (r, c) in excluded:
                continue
            x0, y0, x1, y1 = geo.cell_bounds(r, c)
            roi = img[y0:y1, x0:x1]
            if roi.size and _center_box_has_text(roi):
                cells.add((r, c))
    return sorted(cells), scores
