"""Photo → :class:`~crossword_solver.models.PuzzleLayout` (clue texts filled in later by OCR)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..models import Direction, Entry, PuzzleLayout
from . import arrows as arrows_mod
from .cells import detect_clue_cells
from .grid import GridGeometry, fit_grid, mask_cells
from .preprocess import align_and_crop_grid, deskew, ensure_max_side, mask_solution_box, rectify_grid
from .structure import build_entries

log = logging.getLogger(__name__)


@dataclass
class VisionResult:
    layout: PuzzleLayout
    image: np.ndarray  # rectified grid image (unmasked), used for OCR crops
    geometry: GridGeometry
    debug: dict[str, np.ndarray] = field(default_factory=dict)

    def clue_crop(self, cell: tuple[int, int], pad: int = 4) -> np.ndarray:
        x0, y0, x1, y1 = self.geometry.cell_bounds(*cell, pad=pad)
        return self.image[y0:y1, x0:x1]


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image (supported: PNG, JPEG, WebP, BMP, TIFF)")
    return img


def analyse_image(
    img: np.ndarray,
    *,
    rows: int = 13,
    cols: int = 10,
    mask_region: tuple[int, int, int, int] | None = (8, 6, 12, 9),
    keep_debug: bool = True,
) -> VisionResult:
    debug: dict[str, np.ndarray] = {}
    img0 = ensure_max_side(img)
    img1, angle = deskew(img0)
    cropped = align_and_crop_grid(img1)
    rect = rectify_grid(cropped)
    prepped = mask_solution_box(rect)
    geo, aux = fit_grid(prepped, rows, cols)

    blocked: set[tuple[int, int]] = set()
    work = rect
    if mask_region is not None:
        r0, c0, r1, c1 = mask_region
        blocked = {(r, c) for r in range(r0, min(r1, rows - 1) + 1) for c in range(c0, min(c1, cols - 1) + 1)}
        work = mask_cells(rect, geo, mask_region)

    clue_cells, _scores = detect_clue_cells(work, geo, excluded=blocked)
    found = arrows_mod.detect_arrows(work, geo, set(clue_cells), excluded=blocked)
    linked = build_entries(clue_cells, found, rows, cols, blocked)

    entries = [
        Entry(
            id=f"E{i + 1:02d}",
            clue="",
            start=le.start,
            direction=Direction(le.direction),
            length=le.length,
            clue_cell=le.clue_cell,
        )
        for i, le in enumerate(linked)
    ]
    layout = PuzzleLayout(
        rows=rows, cols=cols, entries=entries, clue_cells=clue_cells, blocked_cells=sorted(blocked)
    )
    log.info(
        "vision_done",
        extra={
            "deskew_deg": round(angle, 2),
            "clue_cells": len(clue_cells),
            "arrows": len(found),
            "entries": len(entries),
        },
    )

    if keep_debug:
        debug["binary"] = aux["binary"]
        vis = work.copy()
        for r, c in clue_cells:
            x0, y0, x1, y1 = geo.cell_bounds(r, c)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        for e in entries:
            for r, c in e.cells:
                cx, cy = geo.cell_center(r, c)
                cv2.circle(vis, (cx, cy), 4, (0, 165, 255), -1)
            cx, cy = geo.cell_center(*e.start)
            cv2.putText(
                vis, f"{e.id}:{e.length}", (cx - 18, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
            )
        debug["structure"] = vis
    return VisionResult(layout=layout, image=rect, geometry=geo, debug=debug)
