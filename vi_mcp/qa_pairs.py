"""Grounded question-answer pairs for knowledge — the training form of a VI's knowledge (0.65.0).

ONE definition, used in two places (standard library only, so the fine-tuning kit carries a copy of this file):
- the server stores pairs on each knowledge object while it sleeps (classify.run_pairs, data.qa) — knowledge is training-ready from
  the moment it is learned, nobody has to synthesise 23,000 documents before a run;
- the kit's synth_qa.py fills in what the server has not stored yet (dataset/documents.jsonl).

Why pairs (14.11.2026): training "question = the document's own text, answer = the document" taught Qwen3-0.6B to echo and loop and
answered no real question better. Several varied grounded pairs per fact did inject knowledge: run 3 (500 documents x ~5 pairs, LoRA
rank 64, 4 epochs) answered 7 correct / 9 partial of 60 held-out probe questions about trained documents against the base model's 0 / 5
(blind judgment, sign-flip p = 0.008). Paraphrases per fact are what makes knowledge extractable (Allen-Zhu & Li, arXiv 2309.14316;
Yang et al., arXiv 2409.07431; Ovadia et al., arXiv 2312.05934).

A pair is kept only if it is grounded: the question is self-contained (names its subject, never "the document"), is not copied from the
document's beginning, and at least GROUNDED of the answer's words occur in the document. One kept pair per object becomes its PROBE:
never trained, it asks after training whether the model learned that object (a knowledge probe; a split by object can only show
generalisation to unseen objects, which fine-tuning does not give). Pairs too close to the probe (word Jaccard >= HOLD_OUT_JACCARD) are
dropped, so the probe stays a new question about a trained fact.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA_VERSION = "qa-v4"
#: qa-v2 (15.11.2026): grounding counts content words only, and a document's compound tokens also match their parts. With qa-v1, real
#: answers of Qwen3.6-35B to six short German findings kept 1 of 25 pairs: "Die Domain www.bad-shop.example wurde auf dem Host dns01
#: erstmals sinkholed." failed on "die, wurde, auf, dem" and on "host"/"dns01" hidden in "host:dns01.example.org". Invented hosts,
#: numbers and paths are still missing from the document and still rejected.
#: qa-v3 (15.11.2026, Codex's review): the question must name the document's subject; the content hash is exact bytes. Known limit of
#: any word-overlap check, reproduced by Codex: a wrong port that also occurs in the document, or a reversed negation, still passes —
#: grounding is lexical, not entailment. Numbers are checked exactly (every number of the answer must occur in the document).
#: qa-v4 (15.11.2026, user: "Die Standardsprache sollte Englisch sein"): questions and answers are written in English whatever the
#: document's language. Word overlap cannot ground an English answer in a German page, so grounding is now two-step:
#:   1. fixed rule — every name the answer carries (a token with a digit, dot, colon, slash, underscore or hyphen; a number; a quoted
#:      term; a capitalised word inside a sentence) occurs in the document, and the answer carries at least one such name, copies at
#:      least three consecutive words of the document (a command, a setting) or shares GROUNDED of its content words with it
#:      (same-language answers); otherwise it cannot be shown and is rejected;
#:   2. a language model checks each kept pair against the document (SUPPORT_SYSTEM) and may only REJECT (project rule: language-model
#:      verdicts only tighten). This also catches what Codex reproduced against lexical grounding: a reversed negation, a wrong
#:      relation between names that both occur in the document.
PAIRS = 5
DOC_CHARS = 6000
GROUNDED = 0.8
HOLD_OUT_JACCARD = 0.5
SYSTEM = ("You write training questions for a company knowledge base. You get one document. Write up to 3 questions that an employee "
          "would really ask and that this document answers, each with a short answer of one to three sentences. Vary the facts and the "
          "wording; when asked for more than 3, ask about the same facts in different words too. Use only facts stated "
          "in the document; copy names, paths, hosts, commands and numbers exactly. Each question must make sense without seeing the "
          "document: name the system, host or topic, never write 'the document', 'this page' or 'the text'. Write questions and answers "
          "in English, whatever the language of the document (English is the memory's standard language); keep names, paths, hosts, "
          "commands, numbers and quoted terms exactly as the document writes them. If the document states no answerable fact (an empty "
          "table, only a title), return an empty list. "
          'Reply with JSON only: {"pairs": [{"question": "...", "answer": "..."}]}')
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pairs"],
          "properties": {"pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                              "required": ["question", "answer"],
                                                              "properties": {"question": {"type": "string"}, "answer": {"type": "string"}}}}}}
META = re.compile(r"\b(the|this|that) (document|page|text|article)\b|\b(dieses?|das|der|die|im) (dokument|seite|text|artikel)\b", re.I)
WORD = re.compile(r"[\w./:-]{3,}")
PART = re.compile(r"[^\W_]{3,}")
STOP = frozenset("""
the and are was were been being has have had for with from this that these those its which who whom what when where how why not
only also can could must should will would than then there their they them you your our out into onto over under about after before
der die das den dem des ein eine einen einem einer eines und oder ist sind wird werden wurde wurden war waren hat haben hatte auf aus
bei mit von vom zum zur fuer für als auch nicht nur noch wie was wer wo wann warum sich sie ihr ihre ihren sein seine seinen dieser
diese dieses diesem diesen lautet heisst heißt über ueber unter nach vor durch bis kann können koennen muss soll sollte man dort hier
dabei damit dafür dazu sowie bzw jedoch aber wenn dann denn weil dass daß ob ja nein es er
""".split())

SUPPORT_SYSTEM = ("You check training data for a company knowledge base. You get one document and numbered question-answer pairs "
                  "written about it, possibly in another language than the document. For each pair decide: supported = the document states "
                  "what the answer says, as an answer to that question. Not supported = the answer adds a fact, changes a value, name or "
                  "relation, reverses a statement (e.g. 'must not' becomes 'can'), or the document does not answer the question. Judge only "
                  "against the document, not against general knowledge. "
                  'Reply with JSON only: {"verdicts": [{"i": 0, "supported": true, "reason": "..."}]}')
SUPPORT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
                  "properties": {"verdicts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                                         "required": ["i", "supported", "reason"],
                                                                         "properties": {"i": {"type": "integer"}, "supported": {"type": "boolean"},
                                                                                        "reason": {"type": "string"}}}}}}


def support_user(document: str, pairs: list[dict]) -> str:
    return (f"Document:\n{document[:DOC_CHARS]}\n\nPairs:\n"
            + "\n".join(f"{i}. Q: {p['question']}\n   A: {p['answer']}" for i, p in enumerate(pairs)))


def apply_support(pairs: list[dict], text: str) -> tuple[list[dict], dict, bool]:
    """(pairs the model did not reject, rejected counts, check complete). The model may only reject: a pair without a verdict is
    neither kept nor silently passed — the whole check counts as incomplete and the caller retries later."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        verdicts = json.loads(m.group(0)).get("verdicts") if m else None
    except (ValueError, AttributeError):
        verdicts = None
    if not isinstance(verdicts, list):
        return [], {}, False
    by_i = {v.get("i"): v for v in verdicts if isinstance(v, dict)}
    if any(i not in by_i for i in range(len(pairs))):
        return [], {}, False
    kept = [p for i, p in enumerate(pairs) if by_i[i].get("supported") is True]
    rejected = len(pairs) - len(kept)
    return kept, ({"not supported by the document (model check)": rejected} if rejected else {}), True


def system_for(pairs: int = PAIRS) -> str:
    return SYSTEM.replace("Write up to 3 questions", f"Write up to {int(pairs)} questions")


def prompt_sha256(pairs: int = PAIRS) -> str:
    """Identity of the prompt — part of every pair's version: another prompt regenerates instead of mixing."""
    return hashlib.sha256((system_for(pairs) + SUPPORT_SYSTEM + SCHEMA_VERSION).encode()).hexdigest()


def content_sha256(document: str) -> str:
    """Identity of the document the pairs were made from: changed knowledge makes its pairs stale. Exact bytes — whitespace is
    content in code and tables (Codex's review 15.11.2026: an indentation change kept the old hash)."""
    return hashlib.sha256((document or "").encode()).hexdigest()


def words(text: str) -> set[str]:
    return {w.lower().strip(".:/-") for w in WORD.findall(text or "")} - {""}


def content_words(text: str) -> set[str]:
    """The words that carry facts: function words of German and English do not count for grounding."""
    return words(text) - STOP


def document_words(document: str) -> set[str]:
    """The document's words plus the parts of compound tokens: host:dns01.example.org also grounds "host" and "dns01"."""
    w = words(document)
    return w | {x.lower() for t in w for x in PART.findall(t)}


NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


def numbers(text: str) -> set[str]:
    """Every number, also short ones: a port, a version, a time. The 80 % word rule ignored tokens under three characters, so "port 22"
    passed for a document saying "port 80" (Codex's review 15.11.2026). Every number of an answer must occur in the document."""
    return set(NUMBER.findall(text or ""))


QUOTED = re.compile(r"[`\"“„']([^`\"”“'\n]{2,80})[`\"”“']")
CAPITAL = re.compile(r"(?<=[a-z0-9,;:)] )[A-Z][\w-]{2,}")


def names(text: str) -> set[str]:
    """The language-independent names an answer carries: tokens with a digit, dot, colon, slash, underscore or hyphen (hosts, paths,
    versions, commands, options), quoted terms and capitalised words inside a sentence (products, systems, people). Lower-cased."""
    out = {w for w in words(text) if re.search(r"[\d./:_-]", w)}
    out |= {x.lower().strip() for q in QUOTED.findall(text or "") for x in [q] if x.strip()}
    out |= {m.lower() for m in CAPITAL.findall(text or "")}
    return out


TOKEN = re.compile(r"[\w./:-]+")


def copied_phrase(answer: str, document: str, n: int = 3) -> bool:
    """The answer copies at least n consecutive words of the document verbatim — a command ("systemctl restart postfix"), a setting,
    a name of several words. Language-independent evidence like a name."""
    a = [t.lower() for t in TOKEN.findall(answer or "")]
    d = " " + " ".join(t.lower() for t in TOKEN.findall(document or "")) + " "
    return any(" " + " ".join(a[i:i + n]) + " " in d for i in range(len(a) - n + 1))


def subject_named(question: str, document: str) -> bool:
    """The question names something specific to the document: a content word of its title (first line), or one of its tokens that
    carries a name (digit, dot, colon, slash, underscore or hyphen: hosts, paths, versions, commands). "What is the port?" names
    nothing and teaches the model a question without its subject (Codex's review 15.11.2026)."""
    title = document.split("\n", 1)[0]
    specific = {w for w in words(document) if re.search(r"[\d./:_-]", w)}
    specific |= {x.lower() for t in specific for x in PART.findall(t) if not x.isdigit()}
    return bool(content_words(question) & (content_words(title) | specific))


def jaccard(a: str, b: str) -> float:
    x, y = words(a), words(b)
    return len(x & y) / max(1, len(x | y))


def key(question: str) -> str:
    return " ".join(sorted(words(question)))


def check(pair, document: str) -> str | None:
    """None if the pair is kept, else the named reason."""
    if not isinstance(pair, dict):
        return "format"
    q, a = str(pair.get("question") or "").strip(), str(pair.get("answer") or "").strip()
    if not 15 <= len(q) <= 300 or not 2 <= len(a) <= 800:
        return "length"
    if META.search(q):
        return "refers to the document instead of its subject"
    if not subject_named(q, document):
        return "names no subject of the document"
    head = re.sub(r"\s+", " ", document[:200]).lower()
    if re.sub(r"\s+", " ", q).lower()[:60] in head:
        return "copied from the document's beginning"
    aw, dw = content_words(a), document_words(document)
    if not numbers(a) <= numbers(document):
        return "a number the document does not state"
    nm = names(a)
    low_doc = (document or "").lower()
    missing = {x for x in nm if x not in dw and x not in low_doc}
    if missing:
        return "a name the document does not state"
    overlap = len(aw & dw) / len(aw) if aw else 0.0
    if not nm and overlap < GROUNDED and not copied_phrase(a, document):  # nothing could show the answer is true
        return "not grounded in the document"
    return None


def parse(text: str):
    """The pairs list from a model answer (JSON, possibly in a code fence), or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        pairs = json.loads(m.group(0)).get("pairs")
    except (ValueError, AttributeError):
        return None
    return pairs if isinstance(pairs, list) else None


def select(pairs, document: str, held: list[str] | None = None, seen: set[str] | None = None) -> tuple[list[dict], dict]:
    """(kept pairs, rejected reason -> count). held = questions kept out of training (probes, evaluation); seen = keys already used."""
    seen = set() if seen is None else seen
    kept, rejected = [], {}
    for p in pairs or []:
        why = check(p, document)
        if not why and any(jaccard(p["question"], h) >= HOLD_OUT_JACCARD for h in held or []):
            why = "too close to a held-out evaluation question"
        if not why and key(p["question"]) in seen:
            why = "duplicate question"
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        seen.add(key(p["question"]))
        kept.append({"question": p["question"].strip(), "answer": p["answer"].strip()})
    return kept, rejected


def split_probe(object_id: str, kept: list[dict]) -> tuple[list[dict], dict | None, int]:
    """(training pairs, probe or None, dropped as too close to the probe). With fewer than two kept pairs there is no probe: one pair
    is worth more as training than as a measurement. The choice depends only on the object id (reproducible)."""
    if len(kept) < 2:
        return kept, None, 0
    i = hashlib.sha256(object_id.encode()).digest()[0] % len(kept)
    probe = kept[i]
    train = [p for j, p in enumerate(kept) if j != i]
    close = [p for p in train if jaccard(p["question"], probe["question"]) >= HOLD_OUT_JACCARD]
    if len(close) == len(train):  # nothing left to learn the fact from: a probe would measure an untrained object
        return kept, None, 0
    return [p for p in train if p not in close], probe, len(close)
