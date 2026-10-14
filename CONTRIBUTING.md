# Contributing

Guidelines for anyone working on this repository — people and AI coding assistants alike.

## Commands
- Install: `pip install -e ".[all,dev]"`
- Check everything before opening a pull request: `ruff check src tests eval tools && ruff format --check src tests eval tools && mypy && pytest`
- Offline evaluation: `python -m eval.run_eval` (no API key needed with `CWS_LLM_PROVIDER=fake`)

## Conventions
- Business logic lives in `src/crossword_solver/`; `serving/` only adapts it to HTTP/MCP/UI.
- Never call an LLM SDK directly — use `LLMClient` from `crossword_solver.llm` (`build_llm`, `TrackedLLM`).
- Prompts are files in `llm/prompts/<name>.v<N>.md`. Change a prompt by adding a new version, never by editing an old one, and attach the eval report.
- Validate every model output in code (`is_valid_answer`, pydantic schemas). Never trust lengths or patterns from the model.
- Tests must not hit the network: use `FakeLLM`, `ScriptedLLM` or recorded payloads. Mark real-API tests with `@pytest.mark.live`.
- Answers are upper-case A–Z with Ä→AE, Ö→OE, Ü→UE, ß→SS (`normalize_answer`).

## Layout
See README.md → "Repository layout" and docs/architecture.md.
