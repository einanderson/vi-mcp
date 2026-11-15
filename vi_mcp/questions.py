"""Questions between instances — instances learn from each other by asking each other.

A question is an object `art:frage` (geteilt) with `daten`:
  von, an (instance name or "alle"), text, reize, zustand, zugestellt_an [instances], antworten [{instanz, text,
  belege, zeit}], gemeldet (answers the asker has already seen).
States (exhaustive): `offen` (not yet shown to any target instance) · `zugestellt` (shown to at least one) ·
`beantwortet` (at least one answer) · `abgelaufen` (after EXPIRY_DAYS without an answer; set by the loop).
Delivery: at the target instance's next `vi_before` as a hint (once per instance). Answers reach the
asker likewise at its next `vi_before` — or via `vi_questions`.

Between conversations of ONE instance (Unreleased): all Claude Code sessions of a machine share the instance `claude-code`, so a
question to the own instance goes to its other conversations. The question records its origin `von_gespraech` (the asker's
conversation; a hook's conversation counts as its agent's, session.canonical) and optionally a target `an_gespraech`
(`to="claude-code#<conversation>"`). Such a question is delivered once per other conversation (`zugestellt_gespraeche`), never to the
asking one; its answers go back to the asking conversation — to any conversation of the instance only once the asking one has ended
(said in the hint). Questions between different instances keep their behaviour: delivered once per instance.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from . import memory as g

STATES = ("offen", "zugestellt", "beantwortet", "abgelaufen")
EXPIRY_DAYS = 7


def _set_state(o: g.MemoryObject, state: str) -> None:
    o.data["zustand"] = state
    o.hashtags = [t for t in o.hashtags if not t.startswith("zustand:")] + [f"zustand:{state}"]


ANSWER_MAX = 20000
EVIDENCE_MAX = 2000


CONVERSATION = re.compile(r"\d{11,14}")


def split_target(to: str) -> tuple[str, str | None]:
    """'claude-code#20261115004' → ('claude-code', '20261115004'); without '#' the conversation is None. Pure."""
    inst, _, conv = (to or "").partition("#")
    return inst, (conv or None)


def own_conversation() -> str | None:
    """The asking conversation as other conversations know it: a hook's conversation stands for its agent's."""
    from . import session
    return session.canonical(g.current_conversation.get())


def ask(from_: str, to: str, text: str, triggers: list[str], source: str, conversation: str | None = None) -> g.MemoryObject:
    if len((text or "").strip()) < 10:
        raise ValueError("text (the question, one sentence) is required")
    to, target = split_target(to)
    if not to or not (all(c.isalnum() or c in "-_." for c in to) or re.fullmatch(r"netz:[a-z][a-z0-9-]{1,30}", to)):
        raise ValueError("an: name of an instance, 'alle', or netz:<network> (0.43.0: a question to the network); "
                         "<instance>#<conversation> asks one conversation")
    if target is not None and (not CONVERSATION.fullmatch(target) or to in ("alle",) or to.startswith("netz:")):
        raise ValueError(f"#{target}: a conversation is its serial number (e.g. claude-code#20261115004), and only after an instance name")
    if to == from_ and not conversation:
        raise ValueError("a question to your own instance needs a conversation to come from (an MCP session) — otherwise use vi_recall")
    if to == from_ and target is not None and target == conversation:
        raise ValueError("a question to your own conversation is no question — use vi_recall")
    tags = [t.strip() for t in triggers or [] if isinstance(t, str) and g.TAG.match(t.strip()) and not t.startswith(g.ADMIN_PREFIXES)]
    text = text.strip()
    if len(text) > ANSWER_MAX:
        raise ValueError(f"a question has at most {ANSWER_MAX} characters — put longer material into an object (vi_learn) and name its id")
    # 14.11.2026: the question used to be cut to 1000 characters without notice; the whole text is now kept in inhalt
    o = g.new_object("frage", text[:1000], tags + [f"an:{to}", f"von:{from_}", "zustand:offen"],
                       {"von": from_, "an": to, "zustand": "offen", "quelle": (source or "").strip()[:500],
                        "zugestellt_an": [], "antworten": [], "gemeldet": 0, **({"inhalt": text} if len(text) > 1000 else {}),
                        **({"von_gespraech": conversation} if conversation else {}), **({"an_gespraech": target} if target else {}),
                        **({"zugestellt_gespraeche": []} if to == from_ or target else {})},
                       importance=2, who=f"ki@{from_}", reason="vi_ask", instance=from_)
    return o


def _per_conversation(d: dict) -> bool:
    """Delivered per conversation (a question within one instance, or one aimed at a conversation) — else once per instance."""
    return d["von"] == d["an"] or bool(d.get("an_gespraech"))


def _who(d: dict, a: dict | None = None) -> str:
    if a is not None:
        return a["instanz"] + (f"#{a['gespraech']}" if a.get("gespraech") else "")
    return d["von"] + (f"#{d['von_gespraech']}" if d.get("von_gespraech") else "")


def _due_here(d: dict, instance: str, conv: str | None) -> bool:
    """Is this question to be shown in this conversation now? Exhaustive over the two delivery kinds."""
    if d["zustand"] == "abgelaufen" or d["an"] not in (instance, "alle"):
        return False
    if _per_conversation(d):
        if not conv or (d["von"] == instance and d.get("von_gespraech") == conv):
            return False  # no conversation to deliver to, or the asker's own
        if d.get("an_gespraech") and d["an_gespraech"] != conv:
            return False
        from . import session
        done = set(d.get("zugestellt_gespraeche", []))
        return conv not in done and session._hook_of.get(conv) not in done  # shown to its hook before the pairing: shown
    return d["von"] != instance and instance not in d["zugestellt_an"]


def _answers_due_here(d: dict, instance: str, conv: str | None) -> bool:
    """New answers go to the asking conversation; to any of the instance's conversations only once it has ended (or never had one)."""
    from . import session
    if d["von"] != instance or len(d["antworten"]) <= d.get("gemeldet", 0):
        return False
    asker = d.get("von_gespraech")
    return not asker or asker == conv or not session.is_open(asker)


def deliver(instance: str | None, conversation: str | None = None) -> list[str]:
    """Hints for `vi_before`: new questions to this instance (or this conversation) and new answers to its questions. Marks them.
    conversation: the calling conversation, canonical (a hook's stands for its agent's); default: the current one."""
    if not instance:
        return []
    conv = conversation if conversation is not None else own_conversation()
    hints = []
    with g.lock:
        ids = [r[0] for r in g.db().execute(
            "SELECT id FROM objekt WHERE art = 'frage' AND json_extract(daten, '$.zustand') IN ('offen', 'zugestellt', 'beantwortet') "
            "AND modus NOT IN ('quarantaene', 'widerlegt') "  # 0.43.0: a question from the network only after review
            "AND (json_extract(daten, '$.an') IN (?, 'alle') OR json_extract(daten, '$.von') = ?) ORDER BY zeit", (instance, instance))]
        for oid in ids:
            o = g.load(oid)
            d = o.data
            if _due_here(d, instance, conv):
                full = str(d.get("inhalt") or o.text)
                more = f" … (complete: vi_questions(id={o.id}))" if len(full) > 400 else ""
                hints.append(f"Question from {_who(d)} [{o.id}]: {full[:400]}{more} — answer with vi_answer(id, text, belege)")
                if _per_conversation(d):
                    d["zugestellt_gespraeche"] = d.get("zugestellt_gespraeche", []) + [conv]
                    where = f"{instance}#{conv}"
                else:
                    d["zugestellt_an"] = d["zugestellt_an"] + [instance]
                    where = instance
                if d["zustand"] == "offen":
                    _set_state(o, "zugestellt")
                o.change("vi", f"delivered to {where}", "next vi_before of the target")
                g.save(o)
            if _answers_due_here(d, instance, conv):  # not elif: another conversation of the asker may get the question and its answers
                new = d["antworten"][d.get("gemeldet", 0):]
                ended = (f" (the asking conversation {d['von_gespraech']} has ended — shown here so it is not lost)"
                         if d.get("von_gespraech") and d["von_gespraech"] != conv else "")
                hints += [f"Answer from {_who(d, a)} to your question [{o.id}] \"{o.text[:80]}\"{ended}: {a['text'][:400]}"
                          + (f" … (complete: vi_questions(id={o.id}))" if len(a["text"]) > 400 or len(a.get("belege") or "") > 160 else "")
                             + (f" [Evidence: {a['belege'][:160]}]" if a.get("belege") else "") for a in new]
                d["gemeldet"] = len(d["antworten"])
                g.save(o)
    return hints


def answer_question(oid: str, instance: str, text: str, evidence_list: str, conversation: str | None = None) -> g.MemoryObject:
    o = g.load(oid)
    if not o or o.kind != "frage":
        raise ValueError(f"question {oid} unknown")
    d = o.data
    if d["zustand"] == "abgelaufen":
        raise ValueError(f"question {oid} has expired ({EXPIRY_DAYS} days without an answer)")
    if d["von"] == instance and (not d.get("von_gespraech") or d["von_gespraech"] == conversation):
        raise ValueError("you do not answer your own question — use vi_learn if you found it out yourself")
    if d["an"] not in (instance, "alle"):
        raise ValueError(f"the question went to {d['an']}, not to {instance}")
    if d.get("an_gespraech") and d["an_gespraech"] != conversation:
        raise ValueError(f"the question went to conversation {d['an']}#{d['an_gespraech']}, not to this one ({conversation or 'none'})")
    if len((text or "").strip()) < 10 or len((evidence_list or "").strip()) < 8:
        raise ValueError("text (the answer) and belege (command/URL/measurement) are required")
    # 14.11.2026: answers were cut to 2000 and evidence to 500 characters without notice (a review ended mid-sentence). Stored whole
    # now; longer is refused with the reason — nothing disappears silently.
    if len(text.strip()) > ANSWER_MAX or len(evidence_list.strip()) > EVIDENCE_MAX:
        raise ValueError(f"an answer has at most {ANSWER_MAX} characters and its evidence at most {EVIDENCE_MAX} — put longer material "
                         "into an object (vi_learn) and name its id in the answer")
    entry = {"instanz": instance, "text": text.strip(), "belege": evidence_list.strip(), "zeit": g.now(),
             **({"gespraech": conversation} if conversation else {})}
    # The object's data has a size limit, past which the generic shortening drops the oldest half of the longest list — here the
    # answers (Codex's review, 14.11.2026). Refuse before that happens; nothing already answered may vanish.
    if len(json.dumps({**d, "antworten": d["antworten"] + [entry]}, ensure_ascii=False)) > g.DATA_MAX - 2000:
        raise ValueError(f"this answer does not fit: the question already holds {len(d['antworten'])} answers near the size limit "
                         f"({g.DATA_MAX} characters) — ask a new question or put the material into an object (vi_learn) and name its id")
    d["antworten"] = d["antworten"] + [entry]
    _set_state(o, "beantwortet")
    o.change(f"ki@{instance}", "answered", "vi_answer")
    return g.save(o)


def listing(instance: str, conversation: str | None = None) -> tuple[list[g.MemoryObject], list[g.MemoryObject]]:
    """(questions to me, offen/zugestellt), (my questions, all states). Within one instance, "me" is this conversation: questions of
    the instance's other conversations are to me, and my own are those this conversation asked (and questions asked without one)."""
    with g.lock:
        to_me = [r[0] for r in g.db().execute(
            "SELECT id FROM objekt WHERE art = 'frage' AND json_extract(daten, '$.zustand') IN ('offen', 'zugestellt') AND modus NOT IN ('quarantaene', 'widerlegt') "
            "AND json_extract(daten, '$.an') IN (?, 'alle') ORDER BY zeit DESC LIMIT 200", (instance,))]
        mine = [r[0] for r in g.db().execute(
            "SELECT id FROM objekt WHERE art = 'frage' AND json_extract(daten, '$.von') = ? ORDER BY zeit DESC LIMIT 200", (instance,))]

    def to_this(d: dict) -> bool:
        if d.get("an_gespraech") and d["an_gespraech"] != conversation:
            return False
        if d["von"] != instance:
            return True
        return bool(d.get("von_gespraech")) and d["von_gespraech"] != conversation and bool(conversation)

    def from_this(d: dict) -> bool:
        return not d.get("von_gespraech") or not conversation or d["von_gespraech"] == conversation

    return ([o for o in (g.load(i) for i in to_me) if o and to_this(o.data)][:50],
            [o for o in (g.load(i) for i in mine) if o and from_this(o.data)][:50])


def let_expire() -> int:
    bound = (datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS)).isoformat(timespec="seconds")
    with g.lock:
        ids = [r[0] for r in g.db().execute(
            "SELECT id FROM objekt WHERE art = 'frage' AND zeit < ? AND json_extract(daten, '$.zustand') IN ('offen', 'zugestellt')", (bound,))]
        for oid in ids:
            o = g.load(oid)
            _set_state(o, "abgelaufen")
            o.change("vi", "expired", f"{EXPIRY_DAYS} days without an answer")
            g.save(o)
    return len(ids)
