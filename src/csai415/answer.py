"""Task 3 (D3 Phase A) — Answer generation + citations. See briefs/D3_D4_TASKS.md.

Owner: WAFIQ. Turn the ranked chunks into a grounded answer with inline ``[n]`` citation
markers, where ``[n]`` resolves to ``citations[n-1]`` (1-based). ``citations[i]`` and
``contexts[i]`` are aligned: ``contexts[i]`` is the chunk text for ``citations[i]``.

**The CONTRACT is frozen at H0** — the ``generate_answer(query, citations, contexts)``
signature and the ``(answer_text, used_idxs)`` return. Everything below is body the owner
fills in.

Design (decided in the D3 kickoff discussion):
  * **One OpenAI-compatible client, backend chosen by env** — Ollama (local Qwen-3B) and
    Groq (the free 70B ceiling row) both speak the OpenAI API, so zero-shot vs tuned
    (D4) and Qwen vs Groq are a ``CSAI415_ANSWERER`` swap, never a code change.
  * **Numbered-source prompting** is the shipped default: contexts go in as ``[1]..[k]``,
    the model cites ``[n]`` inline, we parse + range-validate the markers.
  * **Offline-safe fallback** — with no backend configured (e.g. the contract test, or a
    laptop with no GPU/key) it degrades to a deterministic extractive answer that still
    emits ``[1]``, so the pipeline and ``tests/test_graphrag_contract.py`` stay green.
  * **Citation strategy is a knob** (``CSAI415_CITE_MODE``: numbered | hybrid | posthoc)
    so the deep-dive comparison flips strategies without touching the signature. Only
    ``numbered`` is implemented here; ``hybrid`` (cite→verify→repair) and ``posthoc``
    are the owner's build — left as marked seams.

Env:
  CSAI415_ANSWERER  model id. ``groq:<model>`` / ``groq/<model>`` → Groq; any other
                    non-empty value → Ollama with that model
                    (``qwen2.5:3b-instruct`` zero-shot, ``qwen2.5-3b-csai415`` tuned);
                    unset → extractive fallback.
  CSAI415_CITE_MODE numbered (default) | hybrid | posthoc.
  GROQ_API_KEY      required when the backend is Groq.
  OLLAMA_BASE_URL   defaults to ``http://localhost:11434/v1``.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with graphrag
    from csai415.graphrag import Citation

REFUSAL = "I could not find relevant information in the retrieved context."

# Latency/quality knobs — kept tight so a 3B model stays under the p95 ≤ 2s target.
_MAX_CTX_CHARS = 1200   # truncate each source before it enters the prompt
_MAX_TOKENS = 320       # cap generation length
_TEMPERATURE = 0.0      # deterministic answers (so quality is reproducible)
_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"

# Hybrid verifier knobs (read from env at call time so the notebook can sweep them):
#   CSAI415_VERIFY_TIER       t1 (lexical) | t2 (embedding cosine) | t3 (cross-encoder)
#   CSAI415_VERIFY_THRESHOLD  support score below which a citation is treated as unsupported
#   CSAI415_REPAIR_POLICY     drop | reattribute | flag
_DEFAULT_VERIFY_THRESHOLD = {"t1": 0.30, "t2": 0.50, "t3": 0.50}
_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None  # lazy SentenceTransformer cache (T2)

_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY the numbered "
    "sources provided. After each claim, cite the source you used by writing its number in "
    "square brackets, for example [1] or [2]. Always use the actual digit of the source — "
    "never write the literal letter n or 'n.1'. Keep the answer concise. If the sources do "
    f'not contain the answer, reply with exactly "{REFUSAL}" and nothing else. Do not use '
    "any outside knowledge."
)


# --- Backend resolution ----------------------------------------------------------------
def resolve_backend() -> "tuple[object, str, str] | None":
    """Return ``(client, model, kind)`` for the configured backend, or ``None`` if offline.

    ``kind`` is "groq" or "ollama" (handy for the comparison notebook). ``openai`` is
    imported lazily so importing this module never requires the SDK or a network/key —
    that is what keeps the contract test runnable with no env set.
    """
    spec = (os.getenv("CSAI415_ANSWERER") or "").strip()
    if not spec:
        return None

    from openai import OpenAI  # lazy — only when a backend is actually configured

    low = spec.lower()
    if low.startswith("groq:") or low.startswith("groq/"):
        model = spec.split(":", 1)[-1].split("/", 1)[-1] or _DEFAULT_GROQ_MODEL
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("CSAI415_ANSWERER=groq but GROQ_API_KEY is not set.")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        return client, model, "groq"
    if low == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("CSAI415_ANSWERER=groq but GROQ_API_KEY is not set.")
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key), _DEFAULT_GROQ_MODEL, "groq"

    # anything else → a local Ollama model (zero-shot or the D4-tuned GGUF)
    base = os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE)
    client = OpenAI(base_url=base, api_key="ollama")  # Ollama ignores the key
    return client, spec, "ollama"


def current_backend() -> str:
    """One-line description of the active backend (for notebook display / logging)."""
    try:
        b = resolve_backend()
    except RuntimeError as e:
        return f"misconfigured ({e})"
    if b is None:
        return "extractive-fallback (no CSAI415_ANSWERER set)"
    _, model, kind = b
    return f"{kind}:{model}  [cite_mode={os.getenv('CSAI415_CITE_MODE', 'numbered')}]"


# --- Public contract -------------------------------------------------------------------
def generate_answer(
    query: str,
    citations: "list[Citation]",
    contexts: list[str],
) -> tuple[str, list[int]]:
    """Produce a grounded answer with inline ``[n]`` markers. Returns ``(text, used_idxs)``.

    ``used_idxs`` are the 1-based citation numbers actually cited (range-validated against
    ``contexts``). Refuses cleanly when there is no context. Falls back to a deterministic
    extractive answer when no backend is configured, so the pipeline always runs.
    """
    if not citations or not contexts:
        return (REFUSAL, [])

    backend = resolve_backend()
    if backend is None:
        return _extractive(contexts)  # offline / contract path — still emits [1]

    client, model, _kind = backend
    mode = (os.getenv("CSAI415_CITE_MODE") or "numbered").strip().lower()
    if mode == "numbered":
        return _numbered_source(client, model, query, contexts)
    if mode == "hybrid":
        return _hybrid(client, model, query, citations, contexts)
    if mode == "posthoc":
        return _posthoc(client, model, query, citations, contexts)
    raise ValueError(f"unknown CSAI415_CITE_MODE={mode!r}; expected numbered|hybrid|posthoc")


# --- Strategy 1: numbered-source (shipped default) -------------------------------------
def _numbered_source(client, model, query: str, contexts: list[str]) -> tuple[str, list[int]]:
    """Single generation call; the model cites ``[n]`` inline. Fast, abstractive-constrained."""
    sources = "\n".join(f"[{i}] {(c or '')[:_MAX_CTX_CHARS]}" for i, c in enumerate(contexts, 1))
    user = (
        f"Sources:\n{sources}\n\nQuestion: {query}\n\n"
        "Answer (cite each claim with its source's number in square brackets, e.g. [1]):"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text, _parse_citations(text, len(contexts))


# --- Strategy 2 & 3: the owner's deep-dive (cite→verify→repair / post-hoc attribution) --
# These are the comparison arms WAFIQ builds and benchmarks against `_numbered_source` on
# (faithfulness × p95). Scaffolded as seams so the notebook can already select them; until
# implemented they delegate to numbered-source so nothing in the pipeline breaks.
def _hybrid(client, model, query, citations, contexts) -> tuple[str, list[int]]:
    """cite → verify → repair.

    Generate with numbered-source (one fast call), then check every emitted ``[n]``: does the
    sentence it sits in actually align with ``contexts[n-1]``? A citation that scores below the
    tier's threshold is *unsupported* — the model cited the wrong source or invented the number.
    Repair policy decides what happens to it (``drop`` / ``reattribute`` / ``flag``). This buys
    numbered-source's latency with post-hoc-style faithfulness; tier + threshold + policy are the
    knobs you sweep against the pure ``numbered`` arm.
    """
    text, _ = _numbered_source(client, model, query, contexts)
    if not text or text.strip() == REFUSAL:
        return text, []

    tier = (os.getenv("CSAI415_VERIFY_TIER") or "t1").strip().lower()
    policy = (os.getenv("CSAI415_REPAIR_POLICY") or "drop").strip().lower()
    thr_env = os.getenv("CSAI415_VERIFY_THRESHOLD")
    threshold = float(thr_env) if thr_env else _DEFAULT_VERIFY_THRESHOLD.get(tier, 0.3)

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    repaired = [_repair_sentence(s, contexts, tier, threshold, policy) for s in sentences]
    new_text = re.sub(r"\s+([.,;])", r"\1", " ".join(p for p in repaired if p.strip()))
    return new_text.strip(), _parse_citations(new_text, len(contexts))


def _repair_sentence(sentence: str, contexts, tier: str, threshold: float, policy: str) -> str:
    """Verify each ``[n]`` in one sentence against its context; apply the repair policy."""
    claim = re.sub(r"\[\d+\]", "", sentence).strip()  # the sentence text, markers stripped
    if not claim:
        return sentence

    def fix(m: "re.Match") -> str:
        n = int(m.group(1))
        if not (1 <= n <= len(contexts)):
            return ""  # out-of-range marker is always a hallucination → drop
        if _support(claim, contexts[n - 1], tier) >= threshold:
            return m.group(0)  # supported — keep as-is
        if policy == "reattribute":
            best, score = _best_context(claim, contexts, tier)
            return f"[{best + 1}]" if score >= threshold else ""
        if policy == "flag":
            return f"{m.group(0)}?"  # keep but mark unsupported (for inspection), ASCII-safe
        return ""  # policy == "drop" (default)

    return re.sub(r"\[(\d+)\]", fix, sentence)


# --- Tiered support scorer (the verifier) ----------------------------------------------
def _support(claim: str, context: str, tier: str) -> float:
    """How well ``context`` supports ``claim``, in [0, 1]. Higher tier = stronger (slower) signal."""
    if tier == "t1":
        return _lexical_overlap(claim, context)
    if tier == "t2":
        return _embed_cosine(claim, context)
    if tier == "t3":
        # TODO(WAFIQ): wire a cross-encoder NLI/relevance model (reuse Task 2's reranker model
        # family, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) and map its score to [0, 1]. This is
        # the strongest/slowest tier — the high-faithfulness, high-latency end of the comparison.
        raise NotImplementedError("verify tier t3 (cross-encoder) is a wired seam — implement before use")
    raise ValueError(f"unknown CSAI415_VERIFY_TIER={tier!r}; expected t1|t2|t3")


def _best_context(claim: str, contexts, tier: str) -> tuple[int, float]:
    """Index + score of the context that best supports the claim (for reattribution)."""
    scores = [_support(claim, c or "", tier) for c in contexts]
    j = max(range(len(scores)), key=scores.__getitem__)
    return j, scores[j]


def _lexical_overlap(a: str, b: str) -> float:
    """T1: fraction of the claim's content words found in the context (cheap, no model)."""
    aw = set(re.findall(r"[a-z]{3,}", (a or "").lower()))
    bw = set(re.findall(r"[a-z]{3,}", (b or "").lower()))
    return len(aw & bw) / len(aw) if aw else 0.0


def _embed_cosine(a: str, b: str) -> float:
    """T2: cosine of sentence embeddings (lazy-loads a small SentenceTransformer once)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # lazy — only when tier=t2

        _embedder = SentenceTransformer(_EMBEDDER_MODEL)
    va, vb = _embedder.encode([a or "", b or ""], normalize_embeddings=True)
    return max(0.0, float(va @ vb))  # clamp the [-1,1] cosine to [0,1]


def _posthoc(client, model, query, citations, contexts) -> tuple[str, list[int]]:
    # TODO(WAFIQ): generate the answer WITHOUT citations, then attribute each sentence to its
    # best-matching context (embedding cosine / cross-encoder) and inject [n]. Slower, more
    # faithful — the high-latency end of the comparison table.
    return _numbered_source(client, model, query, contexts)


# --- Helpers ---------------------------------------------------------------------------
def _parse_citations(text: str, n: int) -> list[int]:
    """Extract in-range citation numbers (1..n) from bracket groups, deduped in first-seen order.

    Lenient on purpose: small models malform the marker (``[B1]``, ``[n.1]``, ``[Source 1]``,
    ``[1, 2]``) often enough that a strict ``[\\d+]`` match silently drops real citations. We pull
    any digit-run *inside* a ``[...]`` group and keep those in range — the 1..n bound still rejects
    junk like a stray ``[2024]``. (``[n]`` with no digit yields nothing, so the template artifact
    is ignored rather than miscounted.)
    """
    seen: list[int] = []
    for inner in re.findall(r"\[([^\[\]]*)\]", text):
        for d in re.findall(r"\d+", inner):
            idx = int(d)
            if 1 <= idx <= n and idx not in seen:
                seen.append(idx)
    return seen


def _extractive(contexts: list[str]) -> tuple[str, list[int]]:
    """Deterministic, model-free fallback: lead sentence of the top two contexts + [n]."""
    sentences, used = [], []
    for i, ctx in enumerate(contexts[:2], start=1):
        lead = re.split(r"(?<=[.!?])\s+", (ctx or "").strip())
        if lead and lead[0]:
            sentences.append(f"{lead[0]} [{i}]")
            used.append(i)
    if not sentences:
        return (REFUSAL, [])
    return (" ".join(sentences), used)
