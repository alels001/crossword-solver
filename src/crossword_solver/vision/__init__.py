"""Computer-vision front end: crossword photo → grid structure (clue cells, arrows, entries)."""

from .grid import GridGeometry
from .pipeline import VisionResult, analyse_image, decode_image

__all__ = ["GridGeometry", "VisionResult", "analyse_image", "decode_image"]
