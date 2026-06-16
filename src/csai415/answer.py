"""D3-4 — Answer generation + citations + page ranges. See briefs/D3_TASKS.md (Task D3-4).

Owner: Task D3-4. Turn the ranked chunks into a grounded answer with inline ``[n]``
citation markers, where ``[n]`` resolves to ``citations[n-1]`` (1-based).

**The CONTRACT is frozen at H0** — the ``generate_answer`` signature and the
``(answer_text, used_idxs)`` return. The body is a deterministic *extractive* stub so
the executor and ``/ask`` run end-to-end without loading any model. D3-4 owner swaps in
the graded answerer (``Qwen2.5-3B-Instruct``, the D4 QLoRA target) and the optional
Groq ``llama-3.3-70b`` ceiling row, and compares numbered-source prompting vs post-hoc
citation attribution. Load the model lazily inside the function.

``citations[i]`` and ``contexts[i]`` are aligned: ``contexts[i]`` is the chunk text for
``citations[i]``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with graphrag
    from csai415.graphrag import Citation


def generate_answer(
    query: str,
    citations: "list[Citation]",
    contexts: list[str],
) -> tuple[str, list[int]]:
    """Produce a grounded answer with inline ``[n]`` markers.

    Returns ``(answer_text, used_idxs)`` where ``used_idxs`` are the 1-based citation
    numbers actually cited. H0 STUB: extractive — stitches the lead sentence of the top
    two contexts and cites them. Refuses cleanly when there is no context (this refusal
    path is what D3-7's citation-faithfulness ideas build on).
    """
    if not citations or not contexts:
        return ("I could not find relevant information in the retrieved context.", [])

    sentences: list[str] = []
    used: list[int] = []
    for i, ctx in enumerate(contexts[:2], start=1):
        lead = re.split(r"(?<=[.!?])\s+", (ctx or "").strip())
        if lead and lead[0]:
            sentences.append(f"{lead[0]} [{i}]")
            used.append(i)

    if not sentences:
        return ("I could not find relevant information in the retrieved context.", [])
    return (" ".join(sentences), used)
