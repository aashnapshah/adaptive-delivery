"""Unified LLM interface shared by all demos.

One function, `chat(...)`, dispatches to a configurable backend so every demo
can run against a real model OR fully offline:

  - "openrouter": any model via OpenRouter's OpenAI-compatible API.
                  Needs OPENROUTER_API_KEY. Set the model with OPENROUTER_MODEL
                  (default: a small open model) or the `model=` arg.
  - "ollama":     a local model served by Ollama (needs `ollama serve` running).
                  Set the model with OLLAMA_MODEL (default: "llama3.2:3b").
  - "stub":       deterministic, no network/deps. Each demo can register a
                  stage-appropriate handler via `set_stub_handler(...)` so
                  offline runs still produce sensible output.

Backend is chosen by DEMO_LLM_BACKEND, else auto-detected:
    OpenRouter key present -> "openrouter"
    else local ollama importable -> "ollama"
    else -> "stub"

Keys/config are read from the environment; `demo/.env` is loaded if present.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Callable, Optional

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
_DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Optional per-demo offline handler: (messages) -> str
_stub_handler: Optional[Callable[[list[Message]], str]] = None


def _load_dotenv() -> None:
    """Best-effort load of demo/.env into os.environ (no hard dependency)."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, "..", ".env")
    load_dotenv(env_path)


def set_stub_handler(handler: Callable[[list[Message]], str]) -> None:
    """Register a deterministic offline response generator for the 'stub' backend."""
    global _stub_handler
    _stub_handler = handler


def detect_backend() -> str:
    """Resolve which backend will be used (without making a call)."""
    explicit = os.environ.get("DEMO_LLM_BACKEND")
    if explicit:
        return explicit.lower()
    _load_dotenv()
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

    if backend == "openrouter":
        return _chat_openrouter(messages, model, temperature, max_tokens)
    if backend == "ollama":
        return _chat_ollama(messages, model, temperature, max_tokens, think)
    if backend == "stub":
        return _chat_stub(messages)
    raise ValueError(f"Unknown DEMO_LLM_BACKEND: {backend!r}")


def _chat_openrouter(messages, model, temperature, max_tokens) -> str:
    # Call OpenRouter's OpenAI-compatible REST endpoint with the standard library,
    # so we don't depend on the `openai` SDK (which clashes with some httpx versions).
    import json
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy demo/.env.example to demo/.env and add your key, "
            "or set DEMO_LLM_BACKEND=ollama / stub."
        )
    model = model or os.environ.get("OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL)
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}") from None
    usage = data.get("usage") or {}
    _note_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("total_tokens"), time.time() - t0)
    return data["choices"][0]["message"]["content"] or ""


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


def active_model(backend: Optional[str] = None) -> str:
    """Human-readable description of what will actually answer (for demo headers)."""
    backend = (backend or detect_backend()).lower()
    if backend == "openrouter":
        return f"openrouter:{os.environ.get('OPENROUTER_MODEL', _DEFAULT_OPENROUTER_MODEL)}"
    if backend == "ollama":
        return f"ollama:{os.environ.get('OLLAMA_MODEL', _DEFAULT_OLLAMA_MODEL)}"
    return "stub (deterministic, offline)"
