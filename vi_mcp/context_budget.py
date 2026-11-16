"""Context budget: which chunks of the result objects reach the reader (0.67.0, the collect mode).

Ported unchanged in logic from the IT-AIKB knowledge base (`backend/app/services/context_budget.py`), where it was measured: on 48
questions with a blind judge, about 10k tokens chosen by the reranker answered as well as the full context (1.75 vs 1.77, the interval
includes 0) in 53 s instead of 137 s. The unit is a chunk of `vi_mcp/chunking.py` (it starts with its heading path).

How (design from the IT-AIKB research of 2026-10-15):
  * the unit is the indexed chunk: it starts with its heading path, so a kept chunk
    still says which page section (host, site, product) it belongs to — cutting fresh
    windows out of the page lost that line and caused a wrong host attribution,
  * the reranker scores every chunk against the question; chosen best first, shown in
    document order with "[…]" for gaps (OP-RAG: selection by score, order by document),
  * a chunk brings its direct neighbours along only within the same section (a section
    split for size continues there; the next section is a different topic),
  * at most `per_entry_core` best chunks per page before other pages get their turn (Onyx),
  * chunks naming a host/FQDN of the question go first (they are what the question is about),
  * adaptive amount: cut at the largest drop of the reranker scores (Adaptive-k), but never
    below `min_tokens` and never above `max_tokens`; reranker logits are not calibrated
    across questions, so a fixed score threshold would not transfer.
Every piece that is not kept has a named reason in the returned counts (logged by the caller).
"""
from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 3.5  # German technical text with qwen tokenizer, measured on the eval prompts


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def section_of(chunk_text: str) -> str:
    """The heading path a chunk starts with ("Title > H2 > H3\\n\\nbody"), '' if none."""
    head, sep, _ = chunk_text.partition('\n\n')
    return head.strip() if sep and '\n' not in head.strip() else ''


@dataclass(frozen=True)
class Piece:
    entry: int      # position of the context entry (page or image) in the prompt
    position: int   # position inside the entry, in document order
    text: str
    section: str = ''


@dataclass
class Selection:
    keep: dict[int, set[int]]
    tokens: int
    cutoff: float | None
    # Why the rest is missing: below_cutoff | page_cap | over_budget
    dropped: dict[str, int] = field(default_factory=lambda: {'below_cutoff': 0, 'page_cap': 0, 'over_budget': 0})


def score_gap_cutoff(scores: list[float], top_fraction: float = 0.9) -> float | None:
    """Adaptive-k: the score just above the largest drop within the top 90 % of candidates."""
    ordered = sorted(scores, reverse=True)
    if len(ordered) < 3:
        return None
    span = max(2, int(len(ordered) * top_fraction))
    _gap, at = max((ordered[i] - ordered[i + 1], i) for i in range(span - 1))
    return ordered[at]


def select(
    pieces: list[Piece],
    scores: list[float],
    header_tokens: dict[int, int],
    *,
    min_tokens: int,
    max_tokens: int,
    per_entry_core: int,
    forced: set[int] | None = None,
) -> Selection:
    """Choose pieces (indices into `pieces`, scored by `scores`) for the context.

    header_tokens: cost of an entry's header lines (title, URL, …), paid once when its
    first piece is kept. forced: piece indices that go first (still bound by cap and budget).
    """
    forced = forced or set()
    by_slot = {(p.entry, p.position): i for i, p in enumerate(pieces)}
    cutoff = score_gap_cutoff(scores)
    order = sorted(range(len(pieces)), key=lambda i: (i not in forced, -scores[i]))

    keep: dict[int, set[int]] = {}
    cores: dict[int, int] = {}
    used = 0
    selection = Selection(keep=keep, tokens=0, cutoff=cutoff)
    for rank, i in enumerate(order):
        piece = pieces[i]
        if piece.position in keep.get(piece.entry, set()):
            continue  # already in as a neighbour
        if i not in forced and cutoff is not None and scores[i] < cutoff and used >= min_tokens:
            selection.dropped['below_cutoff'] += sum(
                1 for j in order[rank:] if pieces[j].position not in keep.get(pieces[j].entry, set()))
            break
        if cores.get(piece.entry, 0) >= per_entry_core:
            selection.dropped['page_cap'] += 1
            continue
        neighbours = [
            by_slot[slot] for slot in ((piece.entry, piece.position - 1), (piece.entry, piece.position + 1))
            if slot in by_slot and piece.section and pieces[by_slot[slot]].section == piece.section
            and slot[1] not in keep.get(piece.entry, set())
        ]
        header = header_tokens.get(piece.entry, 0) if piece.entry not in keep else 0
        with_neighbours = [i, *neighbours]
        cost = header + sum(estimate_tokens(pieces[j].text) for j in with_neighbours)
        if used + cost > max_tokens:
            with_neighbours = [i]
            cost = header + estimate_tokens(piece.text)
            if used + cost > max_tokens:
                selection.dropped['over_budget'] += 1
                continue
        keep.setdefault(piece.entry, set()).update(pieces[j].position for j in with_neighbours)
        cores[piece.entry] = cores.get(piece.entry, 0) + 1
        used += cost
    selection.tokens = used
    return selection


def render(texts: list[str], kept: set[int]) -> str:
    """Kept pieces of one entry in document order; '[…]' where something was left out."""
    parts: list[str] = []
    previous: int | None = None
    for position in sorted(kept):
        if previous is not None and position != previous + 1:
            parts.append('[…]')
        elif previous is None and position > 0:
            parts.append('[…]')
        parts.append(texts[position])
        previous = position
    if previous is not None and previous < len(texts) - 1:
        parts.append('[…]')
    return '\n\n'.join(parts)
