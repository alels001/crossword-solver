# ADR 0001 — Provider-agnostic LLM layer

**Status:** accepted

## Context
The notebook called the OpenAI SDK directly from several cells, with two code paths for different SDK versions.
Switching vendors, testing without a key, or adding cost tracking would have touched every call site.

## Decision
Define one small `LLMClient` protocol (`chat(messages, system, tools, json_mode)`) with a neutral message format
(text, images, tool calls, tool results). Adapters translate to OpenAI Chat Completions and the Anthropic Messages API.
Cross-cutting concerns are decorators: `CachedLLM`, `TrackedLLM`. Tests use `FakeLLM` / `ScriptedLLM`.

## Consequences
+ Provider is a configuration value; the agent loop is written once for both vendors.
+ Unit tests cover adapter payloads with recorded fakes; no network in CI.
− Provider-specific features (e.g. native structured outputs) are only used where they fit the common interface;
  JSON validity is enforced by our own validation + repair turn instead.
