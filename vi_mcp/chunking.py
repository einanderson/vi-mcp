"""Structure-aware chunking of long content: the SEARCH unit of a large document (0.67.0).

Ported from the IT-AIKB knowledge base (its chunker v2, `backend/app/services/chunking.py`, measured there), with one change: the
memory stores content as Markdown, so the blocks come from Markdown (headings, fenced code, tables, lists, paragraphs) instead of
Confluence XHTML, and no HTML library is needed.

Why: fixed windows cut through sections, tables and code and carry no hint of where they come from, so full text and the embedding
see anonymous fragments. Structure-aware paragraph grouping retrieves better than fixed-size splitting (arXiv 2603.06976: nDCG@5 0.459
vs 0.244, bge-m3 included), and prefixing each chunk with its context cuts retrieval failures (contextual retrieval).

So the content is split at its headings into sections; the blocks of a section are grouped into chunks of at most ``max_chars``;
tables and code blocks stay whole where possible (an oversized table repeats its header row); every chunk starts with its heading
path ("Title > Section > Subsection").

Chunks are the SEARCH unit only. The object keeps its whole content (AGENTS.md: knowledge is stored in full); what a reader gets is
assembled from the chunks by `context_budget`, never shortened here.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Bump whenever chunk boundaries or heading paths change: the stored hash carries it ("c<version>:<digest>"), so content an older
# chunker split is re-chunked instead of skipped as unchanged.
CHUNKER_VERSION = 1

DEFAULT_MAX_CHARS = 1600  # ~400 tokens for German technical text (bge-m3/Qwen tokenizers: ~3.5–4 characters per token)
DEFAULT_MIN_CHARS = 200   # a trailing chunk shorter than this is merged into its predecessor (same section only)
MAX_HEADING_CHARS = 120   # longer heading lines are content: the heading path gets only their start
_CHARS_PER_TOKEN = 4


def versioned_hash(digest: str) -> str:
    return f"c{CHUNKER_VERSION}:{digest}"


def hash_digest(stored: str | None) -> str | None:
    """The content digest of a stored hash, with or without the chunker version prefix."""
    if not stored:
        return None
    prefix, sep, digest = stored.partition(":")
    return digest if sep and prefix.startswith("c") and prefix[1:].isdigit() else stored


@dataclass(frozen=True)
class Chunk:
    header_path: str
    body: str

    @property
    def text(self) -> str:
        """Content as indexed and embedded: heading path first, then the body."""
        if not self.body:
            return self.header_path
        return f"{self.header_path}\n\n{self.body}" if self.header_path else self.body

    @property
    def token_estimate(self) -> int:
        return math.ceil(len(self.text) / _CHARS_PER_TOKEN)


@dataclass(frozen=True)
class _Block:
    kind: str  # 'heading' | 'text' | 'table' | 'code'
    text: str
    level: int = 0
    table_header: str = ""


# ── Markdown → blocks ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def markdown_blocks(text: str) -> list[_Block]:
    """Blocks of a Markdown text: headings, fenced code (verbatim), tables (header row kept apart), lists and paragraphs."""
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[_Block] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            joined = "\n".join(para).strip()
            if joined:
                blocks.append(_Block("text", joined))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush_para()
            mark, code = fence.group(1), [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                if lines[i].strip().startswith(mark):
                    i += 1
                    break
                i += 1
            blocks.append(_Block("code", "\n".join(code)))
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_para()
            title = heading.group(2).strip()
            if len(title) > MAX_HEADING_CHARS:
                # A "heading" that is really a paragraph (seen: 1,583 characters) would make every path longer than a chunk. Its text
                # stays content; the path gets its start only.
                blocks.append(_Block("heading", title[:MAX_HEADING_CHARS].rstrip() + " …", level=len(heading.group(1))))
                blocks.append(_Block("text", title))
            elif title:
                blocks.append(_Block("heading", title, level=len(heading.group(1))))
            i += 1
            continue
        if _is_table_line(line):
            flush_para()
            rows = []
            while i < len(lines) and _is_table_line(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            header = ""
            if len(rows) >= 2 and _TABLE_RULE.match(rows[1]):
                header = rows[0] + "\n" + rows[1]
            blocks.append(_Block("table", "\n".join(rows), table_header=header))
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        if _LIST_ITEM.match(line) and para and not _LIST_ITEM.match(para[-1]) and not para[-1].startswith(" "):
            flush_para()  # a list after a paragraph is its own block
        para.append(line.rstrip())
        i += 1
    flush_para()
    return blocks


# ── blocks → chunks (unchanged from the IT-AIKB chunker v2) ────────────────────────────────────────────────────────────────────────

def _split_oversized(block: _Block, max_chars: int) -> list[str]:
    """Split one block larger than max_chars at its natural seams — never drops text."""
    if block.kind == "table":
        rows = block.text.split("\n")
        header = block.table_header
        head_rows = header.split("\n") if header else []
        body_rows = rows[len(head_rows):] if head_rows and rows[:len(head_rows)] == head_rows else rows
        # The header is repeated only when it is short and there is a body to repeat it over. Without this guard a table that is only
        # an oversized header row lost all its text (the IT-AIKB original has the same gap; found on a production copy, 16.11.2026).
        if not header or not body_rows or len(header) > max_chars // 2:
            return _pack(rows, max_chars, sep="\n")
        return _pack(body_rows, max_chars, sep="\n", prefix=f"{header}\n")
    if block.kind == "code":
        return _pack(block.text.split("\n"), max_chars, sep="\n")
    sentences = re.split(r"(?<=[.!?:;])\s+|\n", block.text)
    return _pack([s for s in sentences if s.strip()], max_chars, sep=" ")


def _pack(pieces: list[str], max_chars: int, *, sep: str, prefix: str = "") -> list[str]:
    parts: list[str] = []
    current = ""
    budget = max(max_chars - len(prefix), 1)
    for piece in pieces:
        while len(piece) > budget:  # a single line or sentence longer than the budget
            if current:
                parts.append(prefix + current)
                current = ""
            parts.append(prefix + piece[:budget])
            piece = piece[budget:]
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) > budget and current:
            parts.append(prefix + current)
            current = piece
        else:
            current = candidate
    if current:
        parts.append(prefix + current)
    return parts


def _section_bodies(blocks: list[_Block], max_chars: int, min_chars: int) -> list[str]:
    bodies: list[str] = []
    current = ""
    for block in blocks:
        text = block.text
        if len(text) > max_chars:
            if current:
                bodies.append(current)
                current = ""
            bodies.extend(_split_oversized(block, max_chars))
            continue
        candidate = f"{current}\n\n{text}" if current else text
        if len(candidate) > max_chars and current:
            bodies.append(current)
            current = text
        else:
            current = candidate
    if current:
        if bodies and len(current) < min_chars and len(bodies[-1]) + len(current) + 2 <= max_chars:
            bodies[-1] = f"{bodies[-1]}\n\n{current}"
        else:
            bodies.append(current)
    return bodies


def chunk_blocks(title: str, blocks: list[_Block], *, max_chars: int = DEFAULT_MAX_CHARS,
                 min_chars: int = DEFAULT_MIN_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[float, str]] = []
    section: list[_Block] = []

    def flush() -> None:
        if not section:
            return
        path = " > ".join([title, *(text for _, text in stack)]) if title else " > ".join(t for _, t in stack)
        budget = max(max_chars - len(path) - 2, max_chars // 2)
        for body in _section_bodies(section, budget, min_chars):
            chunks.append(Chunk(header_path=path, body=body))
        section.clear()

    # Manual pages (Unix man style) repeat the same subheadings for every entry — NAME, Synopsis, Options, Description — and carry the
    # entry's name only as the text under NAME. The name becomes a path level above those subheadings (level − 0.5, so sibling
    # subheadings keep it) until the next NAME or a higher heading.
    name_level: int | None = None
    for block in blocks:
        if block.kind == "heading":
            flush()
            is_name = block.text.strip().upper() == "NAME"
            floor = block.level - 0.5 if is_name else block.level
            while stack and stack[-1][0] >= floor:
                stack.pop()
            if is_name:
                name_level = block.level
            else:
                name_level = None
                stack.append((block.level, block.text))
        else:
            if name_level is not None and block.kind == "text" and len(block.text) <= 200 and "\n" not in block.text.strip():
                entry = re.split(r"\s+[-–—]\s+", block.text.strip(), maxsplit=1)[0].strip()
                if entry and len(entry.split()) <= 4 and len(entry) <= 80:
                    stack.append((name_level - 0.5, entry))
            name_level = None
            section.append(block)
    flush()
    return chunks


def chunk_markdown(title: str, text: str, *, max_chars: int = DEFAULT_MAX_CHARS, min_chars: int = DEFAULT_MIN_CHARS) -> list[Chunk]:
    """Chunk Markdown content along its heading structure."""
    if not text or not text.strip():
        return []
    return chunk_blocks(title, markdown_blocks(text), max_chars=max_chars, min_chars=min_chars)


def chunk_plain_text(header: str, text: str, *, max_chars: int = DEFAULT_MAX_CHARS, min_chars: int = DEFAULT_MIN_CHARS) -> list[Chunk]:
    """Chunk text without markup by paragraph groups; ``header`` names the source and prefixes every chunk."""
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks = [_Block("text", re.sub(r"\s+", " ", p).strip()) for p in paragraphs]
    budget = max(max_chars - len(header) - 2, max_chars // 2)
    return [Chunk(header_path=header, body=body) for body in _section_bodies(blocks, budget, min_chars)]
