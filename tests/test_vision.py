"""Computer-vision regression tests on a synthetic, slightly rotated Schwedenrätsel."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from crossword_solver.models import PuzzleLayout
from crossword_solver.vision import analyse_image, decode_image
from crossword_solver.vision.arrows import Arrow
from crossword_solver.vision.structure import answer_length, build_entries

CS, MARGIN = 64, 40


def render(layout: PuzzleLayout, angle: float = 1.5) -> np.ndarray:
    h, w = layout.rows * CS + 2 * MARGIN, layout.cols * CS + 2 * MARGIN
    img = np.full((h, w, 3), 235, np.uint8)
    for r, c in layout.blocked_cells:
        x0, y0 = MARGIN + c * CS, MARGIN + r * CS
        cv2.rectangle(img, (x0, y0), (x0 + CS, y0 + CS), (120, 120, 120), -1)
    for e in layout.entries:
        r, c = e.clue_cell
        x0, y0 = MARGIN + c * CS, MARGIN + r * CS
        lines, cur = [], ""
        for word in e.clue.encode("ascii", "ignore").decode().split():
            if cur and len(cur) + len(word) + 1 > 8:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        lines.append(cur)
        for i, line in enumerate(lines[:4]):
            cv2.putText(
                img,
                line[:9],
                (x0 + 3, y0 + 14 + i * 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        sr, sc = e.start
        sx, sy = MARGIN + sc * CS, MARGIN + sr * CS
        if e.direction.value == "right":
            pts = np.array(
                [[sx + 3, sy + CS // 2 - 7], [sx + 3, sy + CS // 2 + 7], [sx + 15, sy + CS // 2]], np.int32
            )
        else:
            pts = np.array(
                [[sx + CS // 2 - 7, sy + 3], [sx + CS // 2 + 7, sy + 3], [sx + CS // 2, sy + 15]], np.int32
            )
        cv2.fillConvexPoly(img, pts, (20, 20, 20))
    for i in range(layout.rows + 1):
        cv2.line(
            img, (MARGIN, MARGIN + i * CS), (MARGIN + layout.cols * CS, MARGIN + i * CS), (30, 30, 30), 2
        )
    for j in range(layout.cols + 1):
        cv2.line(
            img, (MARGIN + j * CS, MARGIN), (MARGIN + j * CS, MARGIN + layout.rows * CS), (30, 30, 30), 2
        )
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, rot, (w, h), borderValue=(250, 250, 250))


@pytest.fixture(scope="module")
def vision_result(layout: PuzzleLayout):
    ok, buf = cv2.imencode(".png", render(layout))
    assert ok
    return analyse_image(decode_image(buf.tobytes()))


def test_grid_geometry_recovered(vision_result) -> None:
    geo = vision_result.geometry
    assert (geo.rows, geo.cols) == (13, 10)
    widths = np.diff(geo.x_lines)
    assert widths.std() / widths.mean() < 0.05  # regular lattice after deskew + rectification


def test_clue_cells_detected(vision_result, layout: PuzzleLayout) -> None:
    truth = {tuple(c) for c in layout.clue_cells}
    found = {tuple(c) for c in vision_result.layout.clue_cells}
    assert len(found & truth) / len(truth) >= 0.95
    assert len(found - truth) <= 2


def test_entries_detected_with_high_precision(vision_result, layout: PuzzleLayout) -> None:
    truth = {(tuple(e.start), e.direction.value, e.length) for e in layout.entries}
    found = {(tuple(e.start), e.direction.value, e.length) for e in vision_result.layout.entries}
    # Baseline of the ported research pipeline: simple arrows are found reliably; bent arrows
    # (clue above/beside the start cell) and double arrows in one cell are known gaps.
    assert len(found & truth) >= 15
    assert len(found - truth) <= 1
    assert vision_result.clue_crop((0, 0)).size > 0


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        decode_image(b"not an image")


def test_linking_and_lengths_unit() -> None:
    # 1 clue at (0,0) with a small right arrow at (0,1); a clue at (2,0) with a large down arrow at (3,0)
    clue_cells = [(0, 0), (2, 0)]
    arrows = [Arrow(0, 1, "right", "small"), Arrow(3, 0, "down", "large")]
    entries = build_entries(clue_cells, arrows, rows=6, cols=5, blocked={(5, 0)})
    got = {(e.clue_cell, e.start, e.direction, e.length) for e in entries}
    assert got == {((0, 0), (0, 1), "right", 4), ((2, 0), (3, 0), "down", 2)}
    assert answer_length((0, 0), "down", 4, 4, stop_cells={(2, 0)}) == 2
