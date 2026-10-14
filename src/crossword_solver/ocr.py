"""Clue OCR with a vision LLM (OpenAI or Anthropic), parallel and cached."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .knowledge.normalize import clean_clue
from .llm.base import ImagePart, LLMClient, Message
from .llm.prompts import load_prompt

log = logging.getLogger(__name__)


def encode_png(img: np.ndarray, upscale: float = 2.5) -> bytes:
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


class ClueOCR:
    def __init__(self, llm: LLMClient, max_workers: int = 6) -> None:
        self.llm = llm
        self.prompt = load_prompt("ocr_clue")
        self.max_workers = max_workers

    def read(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return ""
        system, user = self.prompt.render()
        resp = self.llm.chat(
            [Message.user(user, images=[ImagePart(encode_png(crop))])],
            system=system,
            max_tokens=120,
            temperature=0.0,
        )
        return clean_clue(resp.text)

    def read_many(self, crops: list[np.ndarray]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.read, crops))
