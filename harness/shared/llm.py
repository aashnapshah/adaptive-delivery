"""Unified LLM interface shared by all demos.

One function, `chat(...)`, dispatches to a configurable backend so every demo
can run against a real model OR fully offline:

  - "openrouter": any model via OpenRouter's OpenAI-compatible API.
                  Needs OPENROUTER_API_KEY. Set the model with OPENROUTER_MODEL
                  (default: a small open model) or the `model=` arg.
  - "gemini":     Google's Gemini models via their OpenAI-compatible endpoint.
                  Needs GEMINI_API_KEY. Set the model with GEMINI_MODEL
                  (default: "gemini-2.5-pro") or the `model=` arg.
  - "ollama":     a local model served by Ollama (needs `ollama serve` running).
                  Set the model with OLLAMA_MODEL (default: "llama3.2:3b").
  - "stub":       deterministic, no network/deps. Each demo can register a
                  stage-appropriate handler via `set_stub_handler(...)` so
                  offline runs still produce sensible output.

Backend is chosen by DEMO_LLM_BACKEND, else auto-detected:
    Gemini key present -> "gemini"
    else OpenRouter key present -> "openrouter"
    else local ollama importable -> "ollama"
    else -> "stub"

Auto-detect prefers Gemini: it's the default model for the harness right now. Other models
(for benchmarking) are selected explicitly via DEMO_LLM_BACKEND / OPENROUTER_MODEL.

Keys/config are read from the environment; a `.env` next to this file (or in the repo root)
is loaded if present.
"""

from __future__ import annotations

import contextlib
import os
import random
import threading
import time
from typing import Callable, Optional

# Transient failures worth retrying: rate limits and upstream hiccups. A long sweep makes ~thousands
# of calls, so hitting one of these is expected, not exceptional.
_MAX_TRIES = 5
_TRANSIENT = ("429", "rate limit", "quota", "resource_exhausted", "503", "500", "502", "504",
              "unavailable", "overloaded", "timeout", "timed out", "connection")


def _is_transient(exc) -> bool:
    return any(t in str(exc).lower() for t in _TRANSIENT)

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

# --- per-run telemetry: token usage + latency ------------------------------
# Every chat() call reports its usage into the active capture() block (if any),
# so a caller can measure a whole run without threading counters through code.
# Thread-local so the threaded Flask server doesn't cross-contaminate sessions.
_telemetry = threading.local()


def _note_usage(prompt_tokens, completion_tokens, total_tokens, latency_s: float) -> None:
    cap = getattr(_telemetry, "cap", None)
    if cap is None:
        return
    p = int(prompt_tokens or 0)
    c = int(completion_tokens or 0)
    cap["calls"] += 1
    cap["prompt_tokens"] += p
    cap["completion_tokens"] += c
    cap["total_tokens"] += int(total_tokens or (p + c))
    cap["latency_s"] += latency_s


@contextlib.contextmanager
def capture():
    """Accumulate token usage + latency for every chat() call inside the block.

    Yields a dict {calls, prompt_tokens, completion_tokens, total_tokens, latency_s}.
    Restores any outer capture on exit (nesting is scoped, not additive).
    """
    prev = getattr(_telemetry, "cap", None)
    cap = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_s": 0.0}
    _telemetry.cap = cap
    try:
        yield cap
    finally:
        _telemetry.cap = prev

_DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
_DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Optional per-demo offline handler: (messages) -> str
_stub_handler: Optional[Callable[[list[Message]], str]] = None


def _load_dotenv() -> None:
    """Best-effort load of the repo-root .env into os.environ (no hard dependency)."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    here = os.path.dirname(os.path.abspath(__file__))            # harness/shared
    root = os.path.abspath(os.path.join(here, "..", ".."))       # repo root
    for env_path in (os.path.join(root, ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return


def set_stub_handler(handler: Callable[[list[Message]], str]) -> None:
    """Register a deterministic offline response generator for the 'stub' backend."""
    global _stub_handler
    _stub_handler = handler


def detect_backend() -> str:
    """Resolve which backend will be used (without making a call)."""
    _load_dotenv()   # before reading DEMO_LLM_BACKEND, else a .env-only setting is ignored
    explicit = os.environ.get("DEMO_LLM_BACKEND")
    if explicit:
        return explicit.lower()
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if _ollama_running():
        return "ollama"
    return "stub"


def _ollama_running() -> bool:
    """True only if the Ollama package is importable AND a server actually responds.

    Importability alone is not enough: the pip package can be installed with no
    server running, in which case we must fall back to the offline stub.
    """
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def chat(
    messages: list[Message],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    backend: Optional[str] = None,
    max_tokens: int = 512,
    think: Optional[bool] = None,
) -> str:
    """Send a chat request and return the assistant's text.

    Deterministic by default (temperature=0) so demos reproduce exactly.
    `think` controls reasoning-model thinking on the ollama backend: True/False force it,
    None uses the model default. Pass think=False for cheap extraction calls (gatekeeper,
    judge) so a reasoning model doesn't spend ~60x the time/tokens on a lookup.
    """
    _load_dotenv()
    backend = (backend or detect_backend()).lower()

    def once():
        if backend == "openrouter":
            return _chat_openrouter(messages, model, temperature, max_tokens)
        if backend == "gemini":
            return _chat_gemini(messages, model, temperature, max_tokens, think)
        if backend == "ollama":
            return _chat_ollama(messages, model, temperature, max_tokens, think)
        if backend == "stub":
            return _chat_stub(messages)
        raise ValueError(f"Unknown DEMO_LLM_BACKEND: {backend!r}")

    # Rate limits and 5xx are transient: retry with exponential backoff rather than losing a long
    # sweep to one blip. Anything else (bad key, bad model, hit max_tokens) fails immediately.
    for attempt in range(_MAX_TRIES):
        try:
            return once()
        except Exception as exc:
            if attempt == _MAX_TRIES - 1 or not _is_transient(exc):
                raise
            time.sleep(2 ** attempt + random.random())


def _chat_openai_compatible(url, key, label, messages, model, temperature, max_tokens,
                            extra_body=None) -> str:
    # Call an OpenAI-compatible REST endpoint with the standard library, so we don't
    # depend on the `openai` SDK (which clashes with some httpx versions). Both
    # OpenRouter and Gemini speak this shape, including the `usage` block.
    import json
    import urllib.request

    body = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    body.update(extra_body or {})
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{label} HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}") from None
    usage = data.get("usage") or {}
    _note_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("total_tokens"), time.time() - t0)

    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    finish = choice.get("finish_reason")
    # A reasoning model can burn the whole max_tokens budget on hidden thinking and return
    # finish_reason="length" with no content at all. Returning "" here would look to the
    # caller like a model that failed to answer, silently corrupting a benchmark run --
    # so make the operator fix the budget instead.
    if not content and finish == "length":
        raise RuntimeError(
            f"{label}: {model} hit max_tokens={max_tokens} without emitting an answer "
            f"(completion_tokens={usage.get('completion_tokens')}). Hidden reasoning tokens "
            f"count against max_tokens. Raise the budget or disable thinking."
        )
    return content or ""


def _chat_openrouter(messages, model, temperature, max_tokens) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy demo/.env.example to demo/.env and add your key, "
            "or set DEMO_LLM_BACKEND=ollama / stub."
        )
    model = model or os.environ.get("OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL)
    return _chat_openai_compatible(_OPENROUTER_URL, key, "OpenRouter",
                                   messages, model, temperature, max_tokens)


def _thinking_body(budget: int) -> dict:
    return {"extra_body": {"google": {"thinking_config": {"thinking_budget": budget}}}}


def _chat_gemini(messages, model, temperature, max_tokens, think=None) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to demo/.env, or set DEMO_LLM_BACKEND=ollama / stub."
        )
    model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)

    # Callers tune max_tokens as an *answer* budget (sized against non-thinking models).
    # Gemini charges hidden reasoning tokens to the same ceiling, so grant headroom on top
    # rather than editing every call site -- that keeps the ollama/OpenRouter baselines
    # byte-for-byte reproducible while giving Gemini room to think.
    headroom = int(os.environ.get("GEMINI_THINKING_HEADROOM", "6144"))

    if think is False:
        # 2.5-pro cannot disable thinking (rejects budget 0); 3.x can. Try off, then floor.
        try:
            return _chat_openai_compatible(_GEMINI_URL, key, "Gemini", messages, model,
                                           temperature, max_tokens, _thinking_body(0))
        except RuntimeError as e:
            if "Budget 0 is invalid" not in str(e):
                raise
        return _chat_openai_compatible(_GEMINI_URL, key, "Gemini", messages, model,
                                       temperature, max_tokens + headroom, _thinking_body(128))

    return _chat_openai_compatible(_GEMINI_URL, key, "Gemini",
                                   messages, model, temperature, max_tokens + headroom)


def _chat_ollama(messages, model, temperature, max_tokens: int = 512, think=None) -> str:
    import ollama

    model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
    opts = {"temperature": temperature, "num_predict": max_tokens}
    # think=None -> model default; True/False -> force. Reasoning models (qwen3) otherwise emit
    # long <think> traces (~60x slower); callers pass think=False for cheap extraction.
    kw = {} if think is None else {"think": think}
    t0 = time.time()
    try:
        resp = ollama.chat(model=model, messages=messages, options=opts, **kw)
    except (TypeError, ollama.ResponseError):        # model/version doesn't accept `think`
        resp = ollama.chat(model=model, messages=messages, options=opts)
    _note_usage(resp.get("prompt_eval_count"), resp.get("eval_count"), None, time.time() - t0)
    return resp["message"]["content"]


def _chat_stub(messages) -> str:
    t0 = time.time()
    if _stub_handler is not None:
        out = _stub_handler(messages)
        _note_usage(0, 0, 0, time.time() - t0)   # offline: no real token accounting
        return out
    return (
        "[stub LLM] No backend configured and no stub handler registered. "
        "Set OPENROUTER_API_KEY in demo/.env (any model) or run a local Ollama model."
    )


def light_model(backend: Optional[str] = None) -> Optional[str]:
    """A cheap/fast model for extraction + scoring calls - the gatekeeper, the diagnosis readout, and the
    judges - reserving the default (strong) model for the recommender's actual reasoning. Choose it with
    <BACKEND>_LIGHT_MODEL (e.g. GEMINI_LIGHT_MODEL=gemini-2.5-flash-lite); unset returns None, so chat()
    uses the normal default model and nothing changes until you opt in."""
    backend = (backend or detect_backend()).lower()
    env = {"gemini": "GEMINI_LIGHT_MODEL", "openrouter": "OPENROUTER_LIGHT_MODEL",
           "ollama": "OLLAMA_LIGHT_MODEL"}.get(backend)
    return os.environ.get(env) if env else None


def active_model(backend: Optional[str] = None) -> str:
    """Human-readable description of what will actually answer (for demo headers)."""
    backend = (backend or detect_backend()).lower()
    if backend == "openrouter":
        return f"openrouter:{os.environ.get('OPENROUTER_MODEL', _DEFAULT_OPENROUTER_MODEL)}"
    if backend == "gemini":
        return f"gemini:{os.environ.get('GEMINI_MODEL', _DEFAULT_GEMINI_MODEL)}"
    if backend == "ollama":
        return f"ollama:{os.environ.get('OLLAMA_MODEL', _DEFAULT_OLLAMA_MODEL)}"
    return "stub (deterministic, offline)"
