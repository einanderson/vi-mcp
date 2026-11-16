"""Skills — solution paths that arise when something counts as solved (stage 6; procedural memory, docs/FORSCHUNG.md A3).

A skill is a shared object `skill:<name>`:
  Problem    triggers (tool:, skill:, host:, thema:, error:) at which it appears before an action
  Solution   steps (tool sequence) + one sentence of instructions — at first text shown to the conscious session;
             a skill becomes executable only via the reviewed reflex path (code in memory), never by itself
  Versions   every evolution with time, author, reason, evidence (complete in daten.versionen)
  Counters   helped/did not help after being shown (bandit) — a skill that fails three times and never helps becomes `strittig`
Origin:
  vi_solved      an instance reports: problem X was solved like this (steps, reason, evidence)
  sleep 4c        automatically: an action succeeded after errors of the same tool in the same conversation
                  (`surprise:recovered`) — the tool sequence from the first error to the success is the solution path
Merging:
  vi_skill_merge(a, b, reason) new skill from both (triggers united, steps in sequence, evidence united);
                  a and b get mode `aufgegangen` with a reference — they do not vanish, they are merged.
  sleep           same problem key (tool + error signature) or vector proximity ≥ 0.95 → `fusion:proposal` + edge;
                  merging is done only by an instance, never by the server.
"""
from __future__ import annotations

import re

from . import memory as g

BAD_FROM = 3
#: 0.53.1: at most this many versions are kept (the first and the newest); measured on production 11.11.2026: one automatic skill collected
#: 221 versions in one conversation, its data passed 400,000 characters and was replaced — sleep then failed on it six times
VERSIONS_MAX = 30


def usable(o: g.MemoryObject) -> bool:
    """A skill whose problem and solution are readable. A damaged one (e.g. data replaced for size before 0.53.1) is skipped, named."""
    p, s = o.data.get("problem"), o.data.get("solution")
    return isinstance(p, dict) and isinstance(p.get("triggers"), list) and isinstance(s, dict) and isinstance(s.get("steps"), list)
BAD_RATIO = 0.3

#: Tools that occur in almost every conversation. A solution path made only of them says nothing about what solved the problem.
GENERAL_TOOLS = {"Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "AskUserQuestion", "ToolSearch", "TodoWrite", "WebSearch", "WebFetch",
                        "Agent", "Task", "Skill", "SendMessage", "NotebookEdit", "BashOutput", "KillShell", "Monitor", "ScheduleWakeup", "ListAgents"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")[:40]


def _triggers_ok(triggers: list[str]) -> list[str]:
    return sorted({t.strip() for t in triggers if isinstance(t, str) and g.TAG.match(t.strip()) and not t.startswith(g.ADMIN_PREFIXES)})


def _steps_ok(steps) -> list[dict]:
    outputs = []
    for s in steps or []:
        if isinstance(s, str) and s.strip():
            outputs.append({"tool": s.strip()[:80]})
        elif isinstance(s, dict) and str(s.get("tool", "")).strip():
            outputs.append({"tool": str(s["tool"]).strip()[:80], **({"hinweis": str(s["hinweis"])[:200]} if s.get("hinweis") else {})})
    return outputs[:20]


def _content(instructions: str) -> dict:
    """Detailed instructions (commands, code) are kept in full; the hint shows only the beginning."""
    a = (instructions or "").strip()
    if len(a) > g.CONTENT_MAX:
        raise ValueError(f"instructions have {len(a)} characters, allowed are {g.CONTENT_MAX}")
    return {"inhalt": a} if len(a) > 600 else {}


def new(name: str, triggers: list[str], steps, instructions: str, reason: str, evidence_list: list[str], instance: str,
        source: str = "", oid: str | None = None, who: str | None = None) -> g.MemoryObject:
    n = _slug(name)
    if len(n) < 2:
        raise ValueError("name: at least 2 characters a-z, 0-9, -")
    rz = _triggers_ok(triggers)
    if not rz:
        raise ValueError("at least one trigger (hashtag kind:value) at which the skill appears")
    st = _steps_ok(steps)
    if not st:
        raise ValueError("at least one step (tool name)")
    if len((instructions or "").strip()) < 10 or len((reason or "").strip()) < 5:
        raise ValueError("instructions (the solution path, at least 10 characters) and reason are required")
    oid = oid or f"skill:{n}"
    old = g.load(oid)
    if old and old.kind == "skill":
        return evolve(old, st, instructions, reason, evidence_list, instance, who, triggers=rz)
    o = g.new_object("skill", instructions.strip()[:600], rz + [f"skill:{n}", "mode:test"],
                       {"name": n, "problem": {"triggers": rz}, "solution": {"steps": st, "text": instructions.strip()[:600]}, **_content(instructions),
                        "version": 1, "versions": [{"version": 1, "ts": g.now(), "who": who or f"ki@{instance}", "reason": reason.strip()[:300],
                                                     "evidence": list(evidence_list or [])[:50]}],
                        "evidence": list(evidence_list or [])[:200], "source": (source or reason).strip()[:300]},
                       importance=2, who=who or f"ki@{instance}", oid=oid, mode="test", reason=reason, instance=instance)
    for b in evidence_list or []:
        o.link(b, "Beleg (solution)")
    return o


def evolve(o: g.MemoryObject, steps, instructions: str, reason: str, evidence_list: list[str], instance: str, who: str | None = None,
                     triggers: list[str] | None = None) -> g.MemoryObject:
    """New version: missing steps are appended; instructions are replaced, the old ones are kept in the version.
    0.20.11: New triggers replace the old ones (before, they stayed — a too general trigger like tool:Bash could never be removed again);
    the old ones are kept in the version."""
    st = _steps_ok(steps)
    if not usable(o):
        raise ValueError(f"{o.id} is damaged (no readable problem or solution) — not evolved")
    triggers_before = list(o.data["problem"]["triggers"])
    present = {s["tool"] for s in o.data["solution"]["steps"]}
    new_steps = [s for s in st if s["tool"] not in present]
    new_text = (instructions or "").strip()[:600]
    same_reason = any(v.get("reason") == (reason or "").strip()[:300] for v in o.data.get("versions", []))
    if same_reason and not new_steps and (not triggers or sorted(triggers) == sorted(triggers_before)) and (not new_text or new_text == o.data["solution"]["text"]):
        # 0.53.1: nothing changed and this reason (for a recovery: the same conversation) is already a version — the evidence counts, but no
        # new version. One version per recovery made 221 versions in one conversation. Another conversation is independent: a version.
        o.data["evidence"] = (o.data.get("evidence", []) + [b for b in evidence_list or [] if b not in o.data.get("evidence", [])])[-200:]
        o.data["confirmed_unchanged"] = int(o.data.get("confirmed_unchanged") or 0) + 1
        g.save(o)
        return o
    if triggers and sorted(triggers) != sorted(triggers_before):
        base = [t for t in o.hashtags if t not in triggers_before]
        o.hashtags = base + [t for t in triggers if t not in base]  # check against the cleaned list, otherwise remaining triggers get lost (0.20.12)
        o.data["problem"]["triggers"] = list(triggers)
    o.data["solution"]["steps"] += new_steps
    o.data["versions"].append({"version": o.data["version"] + 1, "ts": g.now(), "who": who or f"ki@{instance}", "reason": reason.strip()[:300],
                                 "belege": list(evidence_list or [])[:50], "text_before": o.data["solution"]["text"],
                                 **({"triggers_before": triggers_before} if o.data["problem"]["triggers"] != triggers_before else {})})
    o.data["version"] += 1
    if len(o.data["versions"]) > VERSIONS_MAX:  # the first version and the newest stay; how many went is named
        cut = len(o.data["versions"]) - VERSIONS_MAX
        o.data["versions"] = o.data["versions"][:1] + o.data["versions"][1 + cut:]
        o.data["versions_trimmed"] = int(o.data.get("versions_trimmed") or 0) + cut
    if (instructions or "").strip():
        o.data["solution"]["text"] = instructions.strip()[:600]
        o.text = o.data["solution"]["text"]
        o.data.pop("content", None)
        o.data.update(_content(instructions))
    o.data["evidence"] = (o.data.get("evidence", []) + [b for b in evidence_list or [] if b not in o.data.get("evidence", [])])[:200]
    o.change(who or f"ki@{instance}", f"Version {o.data['version']}", reason)
    g.save(o)
    for b in evidence_list or []:
        o.link(b, "Beleg (solution)")
    return o


def fusion(a_id: str, b_id: str, name: str, reason: str, instance: str) -> g.MemoryObject:
    a, b = g.load(a_id), g.load(b_id)
    for x, xid in ((a, a_id), (b, b_id)):
        if not x or x.kind != "skill":
            raise ValueError(f"{xid} is not a skill")
        if x.mode == "merged":
            raise ValueError(f"{xid} is already merged into {x.data.get('merged_into')}")
        if x.mode == "quarantine":
            raise ValueError(f"{xid} is in quarantine")
    if a.id == b.id:
        raise ValueError("a skill cannot be merged with itself")
    triggers = sorted(set(a.data["problem"]["triggers"]) | set(b.data["problem"]["triggers"]))
    steps = list(a.data["solution"]["steps"]) + [s for s in b.data["solution"]["steps"] if s["tool"] not in {x["tool"] for x in a.data["solution"]["steps"]}]
    instructions = f"{a.data['solution']['text']} — {b.data['solution']['text']}"[:600]
    evidence_list = sorted(set(a.data.get("evidence", [])) | set(b.data.get("evidence", [])))
    new_o = new(name, triggers, steps, instructions, f"merge of {a.id} and {b.id}: {reason}", evidence_list, instance,
                source=f"merge {a.id} + {b.id}")
    new_o.data["merged_from"] = [a.id, b.id]
    if "origin:p2p" in a.hashtags or "origin:p2p" in b.hashtags:  # the origin survives the merge
        new_o.hashtags = sorted(set(new_o.hashtags) | {"origin:p2p"})
    g.save(new_o)
    for x in (a, b):
        x.mode = "merged"
        x.data["merged_into"] = new_o.id
        x.hashtags = [t for t in x.hashtags if not t.startswith(("mode:", "fusion:"))] + ["mode:merged"]
        x.change(f"ki@{instance}", f"merged into {new_o.id}", reason)
        g.save(x)
        new_o.link(x.id, "merged from")
    return new_o


def line(o: g.MemoryObject) -> str:
    from . import bandit
    steps = " → ".join(s["tool"] for s in o.data["solution"]["steps"][:6])
    return (f"Skill \"{o.data['name']}\" v{o.data['version']} ({o.mode}, from {o.instance}, {bandit.line(o)}): {o.text[:200]}"
            f" — steps: {steps}")


# ── Sleep steps ───────────────────────────────────────────────────────────────

def from_recovery(since: str) -> dict:
    """4c: success after errors of the same tool in the same conversation → skill (test) or new version."""
    new_n = evolved_n = 0
    damaged: set[str] = set()
    with g.lock:
        recovered = g.db().execute(
            "SELECT o.id FROM object o JOIN tag t ON t.object_id = o.id AND t.tag = 'surprise:recovered' "
            "WHERE o.kind = 'experience' AND o.ts > ? AND o.conversation IS NOT NULL ORDER BY o.ts", (since,)).fetchall()
    for (eid,) in recovered:
        e = g.load(eid)
        tool = next((t[5:] for t in e.hashtags if t.startswith("tool:")), None)
        if not tool or not e.conversation:
            continue
        with g.lock:
            rows = g.db().execute("SELECT id FROM object WHERE kind = 'experience' AND conversation = ? AND ts < ? ORDER BY ts", (e.conversation, e.ts)).fetchall()
        episode = [g.load(r[0]) for r in rows]
        error_idx = [i for i, x in enumerate(episode) if x and f"tool:{tool}" in x.hashtags and "ok:no" in x.hashtags]
        if not error_idx:
            continue
        remove = [x for x in episode[error_idx[0]:] if x] + [e]
        steps = []
        for x in remove:
            t = next((tg[5:] for tg in x.hashtags if tg.startswith("tool:")), None)
            if t and (not steps or steps[-1] != t):
                steps.append(t)
        sig = next((tg for tg in episode[error_idx[0]].hashtags if tg.startswith("error:")), None)
        triggers = sorted({tg for x in remove for tg in x.hashtags if tg.startswith(("host:", "topic:"))} | {f"tool:{tool}", f"skill:{tool.split('_')[0]}"} | ({sig} if sig else set()))
        if not _has_content(tool, steps):
            continue  # "Bash failed and then succeeded" without another step is not knowledge, just the normal case
        name = f"auto-{g.hash_value(tool, sig or '', 'erholung')[:12]}"
        instructions = (f"{tool} failed ({sig[7:] if sig else 'error'}) and then succeeded in the same conversation — solution path: "
                     + " → ".join(steps))
        before = g.load(f"skill:{name}")
        if before is not None and not usable(before):
            damaged.add(before.id)  # 0.53.1: skipped and named instead of failing the whole sleep step
            continue
        o = new(name, triggers, steps, instructions, f"sleep: recovery in conversation {e.conversation}", [x.id for x in remove], g.SERVER_INSTANCE,
                source=f"sleep: experiences {remove[0].id} … {e.id}", who="vi")
        if before is None:
            new_n += 1
        else:
            evolved_n += 1
        o.link(e.id, "recovery (sleep)")
    return {"neu": new_n, "weiterentwickelt": evolved_n, **({"beschaedigt": sorted(damaged)} if damaged else {})}


def _has_content(tool: str, steps) -> bool:
    """A recovery path says something if another step lay between error and success. Bash only counts if Bash itself did not fail
    (then something was done on the system); reading, writing, asking back occur in almost every conversation."""
    general = GENERAL_TOOLS - ({"Bash"} if tool != "Bash" else set())
    return bool(set(steps) - general - {tool})


def content_empty(o: g.MemoryObject) -> bool:
    """An automatic recovery skill without a meaningful step."""
    if not o.data.get("name", "").startswith("auto-"):
        return False
    tool = next((r[5:] for r in (o.data.get("problem") or {}).get("triggers", []) if r.startswith("tool:")), "")
    return not _has_content(tool, [s.get("tool") for s in (o.data.get("solution") or {}).get("steps", [])])


def rate() -> dict:
    """Skills that fail repeatedly and never help → strittig (named), not deleted. Automatic skills without content → zurueckgezogen
    (0.67.0: the server withdraws its own skill; widerlegt is reserved for two independent foreign families)."""
    from . import bandit
    disputed = 0
    for _, o, _g in g.recall("kind:skill (mode:test OR mode:active OR mode:disputed)", limit=500, viewer=None):
        if content_empty(o):
            o.mode = "withdrawn"
            o.hashtags = [t for t in o.hashtags if not t.startswith("mode:")] + ["mode:withdrawn"]
            o.change("vi", "withdrawn", "sleep: solution path consists only of general tools; \"helped\" only counted that the tool usually succeeds anyway")
            g.save(o)
    for _, o, _g in g.recall("kind:skill (mode:test OR mode:active)", limit=500, viewer=None):
        yes, no = bandit.counter(o)
        if bandit.shown(o) >= BAD_FROM and bandit.expectation(o) < BAD_RATIO and o.mode != "disputed":
            o.mode = "disputed"
            o.hashtags = [t for t in o.hashtags if not t.startswith("mode:")] + ["mode:disputed"]
            o.change("vi", "disputed", f"sleep: helped {yes}, did not help {no} — check or refute the solution path")
            g.save(o)
            disputed += 1
    return {"strittig": disputed}


def fusion_proposals() -> dict:
    """Same problem key (tool + error signature) for two skills → fusion:proposal + edge. An instance decides."""
    skills = [o for _, o, _g in g.recall("kind:skill !mode:merged", limit=500, viewer=None)]
    by_key: dict[tuple, list[g.MemoryObject]] = {}
    for o in skills:
        if not usable(o):
            continue
        tools = tuple(sorted(t for t in o.data["problem"]["triggers"] if t.startswith("tool:")))
        error = tuple(sorted(t for t in o.data["problem"]["triggers"] if t.startswith("error:")))
        if tools:
            by_key.setdefault((tools, error), []).append(o)
    n = 0
    for group in by_key.values():
        if len(group) < 2:
            continue
        for o in group:
            others = [x.id for x in group if x.id != o.id]
            if "fusion:proposal" not in o.hashtags:
                o.hashtags = o.hashtags + ["fusion:proposal"]
                o.change("vi", "fusion:proposal", f"sleep: same problem as {', '.join(others)} — an instance decides with vi_skill_merge")
                g.save(o)
                n += 1
            for a in others:
                o.link(a, "same problem (merge proposal)")
    return {"vorschlaege": n}
