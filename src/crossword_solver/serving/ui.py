"""Streamlit front end: RAG chat assistant, single-clue helper and full puzzle solving.

Run with ``streamlit run apps/streamlit_app.py`` (or ``crossword ui``).
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from crossword_solver.config import get_settings
from crossword_solver.models import PuzzleLayout
from crossword_solver.pipeline import CrosswordSolver


@st.cache_resource
def solver() -> CrosswordSolver:
    return CrosswordSolver()


def main() -> None:
    st.set_page_config(page_title="Kreuzworträtsel-Assistent", page_icon="🧩", layout="wide")
    settings = get_settings()
    s = solver()

    with st.sidebar:
        st.header("🧩 Crossword Solver")
        st.caption(
            f"LLM provider: **{settings.llm_provider}** · KB: {len(s.store)} clues · retriever: {settings.retriever}"
        )
        if settings.llm_provider == "fake":
            st.info(
                "Offline mode: no LLM key configured. Set CWS_LLM_PROVIDER=openai or anthropic for full quality."
            )

    tab_chat, tab_clue, tab_solve = st.tabs(["💬 Chat", "🔎 Clue helper", "🧩 Solve a puzzle"])

    with tab_chat:
        st.subheader("Kreuzworträtsel-Assistent")
        if "history" not in st.session_state:
            st.session_state.history = []
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
        if prompt := st.chat_input("z. B. „sizilianischer Vulkan, 5 Buchstaben, A _ _ N A“"):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"), st.spinner("Denke nach…"):
                reply = s.chat(prompt, st.session_state.history[-10:])
                st.markdown(reply)
            st.session_state.history += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]

    with tab_clue:
        c1, c2, c3 = st.columns([3, 1, 2])
        clue = c1.text_input("Clue", "sizilianischer Vulkan")
        length = c2.number_input("Length", 1, 30, 5)
        pattern = c3.text_input("Known letters ('_' = unknown)", "_" * int(length))
        if st.button("Suggest answers", type="primary"):
            with st.spinner("Retrieving and generating…"):
                try:
                    cands = s.suggest(clue, int(length), pattern or None)
                except ValueError as e:
                    st.error(str(e))
                    cands = []
            if cands:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "answer": c.word,
                                "score": c.score,
                                "source": c.source.value,
                                "evidence": c.evidence,
                            }
                            for c in cands
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning("No candidates found.")

    with tab_solve:
        mode = st.radio("Input", ["Demo puzzle (benchmark)", "Upload a photo"], horizontal=True)
        use_agent = st.toggle("Use agent for open entries", value=True)
        report = None
        progress = st.empty()
        if mode.startswith("Demo"):
            bench = json.loads((settings.benchmark_dir / "puzzle_01.json").read_text(encoding="utf-8"))
            layout = PuzzleLayout.model_validate(bench["layout"])
            st.caption(f"{len(layout.entries)} clues, {layout.rows}×{layout.cols} grid")
            if st.button("Solve demo", type="primary"):
                report = s.solve_layout(layout, use_agent=use_agent, on_progress=progress.info)
        else:
            up = st.file_uploader("Crossword photo", type=["png", "jpg", "jpeg", "webp"])
            if up and st.button("Solve photo", type="primary"):
                report = s.solve_image(up.getvalue(), use_agent=use_agent, on_progress=progress.info)

        if report is not None:
            progress.empty()
            sol = report.solution
            left, right = st.columns([1, 2])
            with left:
                st.markdown("**Grid**")
                st.code(sol.as_text(), language=None)
                st.json(
                    {"usage": sol.usage.model_dump(), "stats": sol.stats, "timings_ms": report.timings_ms},
                    expanded=False,
                )
            with right:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "id": e.entry_id,
                                "clue": e.clue,
                                "answer": e.answer,
                                "confidence": e.confidence,
                                "source": e.source.value if e.source else "",
                            }
                            for e in sol.entries
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if report.agent_transcript:
                with st.expander(f"Agent transcript ({len(report.agent_transcript)} tool calls)"):
                    st.json(report.agent_transcript)
