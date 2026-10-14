"""Stage 7: arrow-head detection (template matching) and direction assignment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .cells import prep_bin
from .grid import GridGeometry

_SIZES = (10, 12, 14, 16, 18, 20, 22, 24, 26, 28)


def tri_template(size: int, direction: str) -> np.ndarray:
    img = np.zeros((size, size), np.uint8)
    if direction == "right":
        pts = np.array([[2, 2], [2, size - 2], [size - 2, size // 2]], np.int32)
    else:
        pts = np.array([[2, 2], [size - 2, 2], [size // 2, size - 2]], np.int32)
    cv2.fillConvexPoly(img, pts, 255)
    return img


TEMPLATES = {d: [tri_template(s, d) for s in _SIZES] for d in ("right", "down")}


@dataclass(frozen=True)
class Arrow:
    row: int
    col: int
    direction: str  # "right" | "down"
    kind: str  # "large" | "small"
    score: float = 0.0


def remove_small_components(bw: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return bw
    out = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def tri_match_score(bw: np.ndarray, direction: str, include_full: bool = False) -> float:
    h, w = bw.shape
    stripes = (
        [bw[:, int(w * 0.45) :], bw[:, int(w * 0.60) :]]
        if direction == "right"
        else [bw[int(h * 0.45) :, :], bw[int(h * 0.60) :, :]]
    )
    if include_full:
        stripes.append(bw)
    best = 0.0
    for stripe in stripes:
        if stripe.size == 0:
            continue
        for t in TEMPLATES[direction]:
            if stripe.shape[0] < t.shape[0] or stripe.shape[1] < t.shape[1]:
                continue
            best = max(best, float(cv2.matchTemplate(stripe, t, cv2.TM_CCOEFF_NORMED).max()))
    return best


def detect_arrows(
    img: np.ndarray,
    geo: GridGeometry,
    clue_cells: set[tuple[int, int]],
    excluded: set[tuple[int, int]] | None = None,
    *,
    large_th: float = 0.38,
    small_th: float = 0.45,
    ink_max: float = 0.40,
    ink_clean_range: tuple[float, float] = (0.010, 0.220),
    right_strip: float = 0.50,
    bottom_strip: float = 0.60,
) -> list[Arrow]:
    """Find answer-start cells marked with an arrow and decide each arrow's direction."""
    excluded = excluded or set()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_full = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    arrows: list[Arrow] = []
    large_cells: set[tuple[int, int]] = set()
    for r in range(geo.rows):
        for c in range(geo.cols):
            if (r, c) in clue_cells or (r, c) in excluded:
                continue
            x0, y0, x1, y1 = geo.cell_bounds(r, c)
            if y1 <= y0 or x1 <= x0 or bw_full[y0:y1, x0:x1].mean() / 255.0 > ink_max:
                continue
            bw = prep_bin(img[y0:y1, x0:x1])
            clean = remove_small_components(bw, max(18, int(0.002 * bw.size)))
            ink = clean.mean() / 255.0
            if not ink_clean_range[0] <= ink <= ink_clean_range[1]:
                continue
            s_r, s_d = tri_match_score(clean, "right"), tri_match_score(clean, "down")
            if max(s_r, s_d) < large_th:
                continue
            # Direction by where the ink sits (validated on the dataset): ink on the right edge
            # means a bent arrow coming from above → answer runs down, otherwise it runs right.
            h, w = bw.shape
            ink_r = float(np.mean(bw[:, int(w * (1 - right_strip)) :] == 255))
            ink_b = float(np.mean(bw[int(h * (1 - bottom_strip)) :, :] == 255))
            arrows.append(Arrow(r, c, "down" if ink_r > ink_b else "right", "large", max(s_r, s_d)))
            large_cells.add((r, c))

    # Small arrow heads (can carry two directions in one cell).
    for r in range(geo.rows):
        for c in range(geo.cols):
            if (r, c) in clue_cells or (r, c) in excluded or (r, c) in large_cells:
                continue
            x0, y0, x1, y1 = geo.cell_bounds(r, c)
            if y1 <= y0 or x1 <= x0 or bw_full[y0:y1, x0:x1].mean() / 255.0 > ink_max:
                continue
            bw = prep_bin(img[y0:y1, x0:x1])
            s_r = tri_match_score(bw, "right", include_full=True)
            s_d = tri_match_score(bw, "down", include_full=True)
            if s_r >= small_th:
                arrows.append(Arrow(r, c, "right", "small", s_r))
            if s_d >= small_th:
                arrows.append(Arrow(r, c, "down", "small", s_d))
    return arrows
