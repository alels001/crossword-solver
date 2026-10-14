"""Stages 0–2: resize, deskew, grid-aligned crop, perspective rectification, solution-box masking.

Ported from the research notebook (stages 0, 1, 2.1, 2.2, 2.3); file I/O and plotting
removed so every function is pure ``ndarray -> ndarray`` and unit-testable.
"""

from __future__ import annotations

import cv2
import imutils
import numpy as np


def ensure_max_side(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)


# ---- Stage 1: deskew --------------------------------------------------------------
def compute_angle_hough(gray: np.ndarray) -> float | None:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=20)
    if lines is None:
        return None
    angles = [
        float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
        if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 45
    ]
    return float(np.median(angles)) if angles else None


def compute_angle_fallback(gray: np.ndarray) -> float:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    angle = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[-1]
    if angle < -45:
        angle += 90
    return float(angle)


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = compute_angle_hough(gray)
    if angle is None:
        angle = compute_angle_fallback(gray)
    return imutils.rotate_bound(img, -angle), angle


# ---- Stage 2.1: grid-based alignment + crop ---------------------------------------
def grid_line_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    h, w = bw.shape
    vertical = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 20))), iterations=2
    )
    horizontal = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 20), 1)), iterations=2
    )
    grid = cv2.bitwise_or(vertical, horizontal)
    return cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)


def compute_grid_angle(mask: np.ndarray) -> float:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=200, minLineLength=150, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(a) < 30:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def align_and_crop_grid(img: np.ndarray) -> np.ndarray:
    angle = compute_grid_angle(grid_line_mask(img))
    aligned = imutils.rotate_bound(img, -angle)
    cnts, _ = cv2.findContours(grid_line_mask(aligned), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return aligned
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return aligned[y : y + h, x : x + w]


# ---- Stage 2.2: perspective rectification ----------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32"
    )


def rectify_grid(img: np.ndarray) -> np.ndarray:
    """Warp the outer grid quadrilateral to the full image rectangle (no-op if not found)."""
    h, w = img.shape[:2]
    cnts, _ = cv2.findContours(grid_line_mask(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) > 4:
        hull = cv2.convexHull(approx)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        return img
    src = order_points(approx.reshape(4, 2))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


# ---- Stage 2.3: hide the dark solution box (bottom-right) for line detection -------
def mask_solution_box(img: np.ndarray, roi_start: float = 0.55, pad: int = 25) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = gray.shape
    oy, ox = int(h * roi_start), int(w * roi_start)
    cnts, _ = cv2.findContours(th[oy:, ox:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x1, y1 = max(0, ox + x - pad), max(0, oy + y - pad)
    x2, y2 = min(w, ox + x + bw + pad), min(h, oy + y + bh + pad)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), -1)
    out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (35, 35), 0)
    return out
