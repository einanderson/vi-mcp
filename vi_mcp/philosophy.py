"""vi-mcp side of philosophy packages (0.66.0): the command line, installing a checked package into the VI, the tool vi_philosophy.

The technique itself is clingo_object.philosophy (standard library plus clingo, usable without a server). The user (15.11.2026): "Die
Philosophie-Technik muss als Skill vorhanden sein, damit man eine KI mit irgendeiner Philosophie ausstatten kann."
"""
from __future__ import annotations

import json

from clingo_object import philosophy as ph

from . import memory as g

TAG = "thema:philosophie"
#: a binding is its own object (kind regel, visibility intern), not a config entry: it has an author, a change history and a tombstone
#: when it ends (mode aus), it reaches every server process and both databases without a restart, and an instance can read which
#: philosophy applies to it. It stores the name and the HASH it was bound to — a newer install does not silently take over.
BIND_TAG = "thema:philosophie-bindung"


def check_lines(r: dict) -> list[str]:
    head = "Package passes every check." if r["ok"] else f"Package refused: {len(r['errors'])} problem(s)."
    out = [head + (f" Hash {r['hash'][:16]}…" if r.get("hash") else "")]
    out += [f"  ✗ {e}" for e in r["errors"]] + [f"  ! {w}" for w in r["warnings"]]
    if r.get("worlds"):
        out.append(f"  worlds: {sum(1 for w in r['worlds'] if w['ok'])}/{len(r['worlds'])} as expected")
    for pred, cov in (r.get("coverage") or {}).items():
        out.append(f"  {pred}: " + " ".join(f"{v}{'✓' if ok else '✗'}" for v, ok in cov.items()))
    return out


def show_lines(pkg: dict) -> list[str]:
    out = [f"Philosophy {pkg['name']} {pkg['version']} — {pkg.get('description', '')}", f"Author: {pkg['author']} · profile "
           f"{pkg['profile'].get('semantics')} · priority {pkg['profile'].get('priority', 'none')} · hash {ph.package_hash(pkg)[:16]}…", "", "Principles:"]
    out += [f"  {p['id']} {p.get('title', '')}: {p['text']}" for p in ph.principles(pkg)]
    out += ["", "Inputs the world gives:"] + [f"  {k}/{v['arity']}: {v.get('doc', '')}" for k, v in ph.vocabulary(pkg).items() if v["kind"] == "input"]
    iface = [f"  {k}/{v['arity']}: {v.get('doc', '')}" for k, v in ph.vocabulary(pkg).items() if v["kind"] == "interface"]
    out += (["", "Read by the priority profile (given by the world or derived by the seed from evidence):"] + iface) if iface else []
    out += ["", "Answers:"] + [f"  {k}: {', '.join(v['values'])}" for k, v in pkg["answers"].items()]
    out += ["", "Seed (clingo-object notation):", pkg["seed"]]
    return out


def installed() -> list[g.MemoryObject]:
    """The newest installed object per philosophy name."""
    newest: dict[str, g.MemoryObject] = {}
    for _s, o, _w in g.recall(TAG, limit=1000, viewer=None):
        p = o.data.get("philosophie") or {}
        if p.get("name") and o.mode not in ("widerlegt", "archiviert") and (p["name"] not in newest or o.ts > newest[p["name"]].ts):
            newest[p["name"]] = o
    return sorted(newest.values(), key=lambda o: o.data["philosophie"]["name"])


def install(ref: str, instance: str, who: str | None = None, visibility: str | None = None) -> g.MemoryObject:
    """Check, then store as a rule set object (vi_logic runs it) carrying the manifest and hash. Refused with the reasons if any check fails."""
    from . import logic
    pkg = ph.load(ref)
    r = ph.check(pkg)
    if not r["ok"]:
        raise ValueError("package refused: " + "; ".join(r["errors"][:5]))
    o = logic.save_rule_set(f"philosophie-{pkg['name']}", pkg["seed"], f"philosophy {pkg['name']} {pkg['version']}: {pkg.get('description', '')}"[:300],
                            visibility, instance, who)
    o.hashtags = sorted(set(o.hashtags) | {TAG, f"philosophie:{pkg['name']}"})
    o.data = {**o.data, "philosophie": {**{k: v for k, v in pkg.items() if k != "seed"}, "hash": r["hash"],
                                         "check": {"worlds": len(r["worlds"]), "coverage": r["coverage"]}}}
    o.change(f"ki@{who or instance}", "philosophy installed", f"{pkg['name']} {pkg['version']} hash {r['hash'][:16]} — every check passed")
    return g.save(o)


def _installed_all(name: str) -> list[g.MemoryObject]:
    """Every installed object of one philosophy (all versions), newest first."""
    objs = [o for _s, o, _w in g.recall(TAG, limit=1000, viewer=None)
            if (o.data.get("philosophie") or {}).get("name") == name and o.mode not in ("widerlegt", "archiviert")]
    return sorted(objs, key=lambda o: o.ts, reverse=True)


def _package_of(o: g.MemoryObject) -> dict:
    """The package an installed object holds (manifest + seed), as it was checked."""
    pkg = {k: v for k, v in o.data["philosophie"].items() if k not in ("hash", "check")}
    pkg["seed"] = o.data["logik"]["quelle"]
    return pkg


def _target(instance: str | None, index: str | None) -> tuple[str, str]:
    from . import indexes
    if bool(instance) == bool(index):
        raise ValueError("name exactly one target: an instance or an index")
    return ("instance", str(instance)) if instance else ("index", indexes.normalize(index))


def _active_bindings() -> list[g.MemoryObject]:
    return [o for _s, o, _w in g.recall(BIND_TAG, limit=1000, viewer=None)
            if o.mode == "aktiv" and (o.data.get("philosophie_bindung") or {}).get("name")]


def bind(name: str, who: str, instance: str | None = None, index: str | None = None, off: bool = False) -> str:
    """Bind an installed philosophy (its current hash) to an instance or an index, or end the binding (off: mode aus, kept as a
    tombstone with the reason). Several philosophies may apply to one target; their verdicts stay apart (evaluate_bound)."""
    kind, target = _target(instance, index)
    current = [o for o in _active_bindings() if o.data["philosophie_bindung"]["name"] == name
               and (o.data["philosophie_bindung"]["target_kind"], o.data["philosophie_bindung"]["target"]) == (kind, target)]
    if off:
        if not current:
            return f"no active binding of {name} to {kind} {target} — nothing to end"
        for o in current:
            o.mode = "aus"
            o.change(f"ki@{who}", "philosophy binding ended", f"{name} no longer applies to {kind} {target}")
            g.save(o)
        return f"ended: {name} no longer applies to {kind} {target} ({', '.join(o.id for o in current)} kept with mode aus)"
    objs = _installed_all(name)
    if not objs:
        raise ValueError(f"{name} is not installed — install it first (vi_philosophy action install, or vi-mcp philosophy install {name})")
    inst = objs[0]
    h = inst.data["philosophie"]["hash"]
    for o in current:
        if o.data["philosophie_bindung"]["hash"] == h:
            return f"already bound: {name} {inst.data['philosophie']['version']} (hash {h[:16]}…) applies to {kind} {target} — {o.id}"
        o.mode = "aus"
        o.change(f"ki@{who}", "philosophy binding replaced", f"rebound to hash {h[:16]}")
        g.save(o)
    meta = inst.data["philosophie"]
    o = g.new_object("regel", f"Philosophy binding: {name} {meta['version']} applies to {kind} {target}",
                     [BIND_TAG, f"philosophie:{name}"],
                     {"philosophie_bindung": {"name": name, "version": meta["version"], "hash": h, "installed": inst.id,
                                              "profile": meta.get("profile"), "target_kind": kind, "target": target}},
                     importance=1, who=f"ki@{who}", reason="vi_philosophy use", instance=who,
                     visibility="intern")
    return f"bound {o.id}: {name} {meta['version']} (hash {h[:16]}…) applies to {kind} {target}"


def bindings(instance: str | None = None, index: str | None = None) -> list[dict]:
    """The philosophies that apply to an instance and/or an index, each with the hash it was bound to and a named state:
    current (the installed package has that hash) · stale (a newer install differs — the bound hash still decides) · missing."""
    out = []
    for o in _active_bindings():
        b = o.data["philosophie_bindung"]
        if not ((b["target_kind"] == "instance" and b["target"] == instance) or (b["target_kind"] == "index" and index and b["target"] == index)):
            continue
        objs = _installed_all(b["name"])
        hashes = [x.data["philosophie"]["hash"] for x in objs]
        state = "missing" if b["hash"] not in hashes else ("current" if hashes[0] == b["hash"] else "stale")
        out.append({**b, "binding": o.id, "state": state})
    return sorted(out, key=lambda b: (b["name"], b["target_kind"]))


def evaluate_bound(facts: str, instance: str | None = None, index: str | None = None) -> dict:
    """The verdict of every philosophy that applies, NAMESPACED by philosophy and never merged (Codex: bind each query to one
    profile or return the verdicts of both apart). Each runs the package with the bound hash; a missing one gives no verdict."""
    out: dict[str, dict] = {}
    for b in bindings(instance, index):
        key = b["name"] if b["name"] not in out else f"{b['name']}@{b['target_kind']}"
        if b["state"] == "missing":
            out[key] = {"hash": b["hash"], "state": "missing", "verdict": None}
            continue
        o = next(x for x in _installed_all(b["name"]) if x.data["philosophie"]["hash"] == b["hash"])
        pkg = _package_of(o)
        if ph.package_hash(pkg) != b["hash"]:  # the stored package must still be the one that was checked and bound
            out[key] = {"hash": b["hash"], "state": "hash_mismatch", "verdict": None}
            continue
        out[key] = {"hash": b["hash"], "state": b["state"], "applies_to": f"{b['target_kind']} {b['target']}", "verdict": ph.evaluate(pkg, facts)}
    return out


def binding_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No philosophy is bound."]
    return [f"  {b['name']} {b['version']} → {b['target_kind']} {b['target']}: hash {b['hash'][:16]}… ({b['state']}) [{b['binding']}]" for b in rows]


def tool(args: dict, instance: str) -> str:
    """vi_philosophy: guide (default) · list · show · check · evaluate · install · use · bound."""
    action = args.get("action") or "guide"
    if action == "guide":
        return ph.GUIDE
    if action == "list":
        inst = installed()
        lines = ["Built in: " + ", ".join(sorted(ph.BUILTIN))]
        lines += ["Installed: " + (", ".join(f"{o.data['philosophie']['name']} {o.data['philosophie']['version']} ({o.id})" for o in inst) or "none")]
        lines += ["Bound to you:"] + binding_lines(bindings(instance, args.get("index")))
        return "\n".join(lines)
    if action == "bound":
        return "\n".join([f"Philosophies that apply to instance {instance}" + (f" and index {args['index']}" if args.get("index") else "") + ":"]
                         + binding_lines(bindings(instance, args.get("index"))))
    if action == "evaluate" and not args.get("name"):
        return json.dumps(evaluate_bound(str(args.get("facts") or ""), instance, args.get("index")), ensure_ascii=False, indent=1)
    ref = str(args.get("name") or "")
    if not ref:
        raise ValueError(f"{action} needs name — a built-in philosophy ({', '.join(sorted(ph.BUILTIN))}) or a package directory on the server")
    pkg = ph.load(ref)
    if action == "show":
        return "\n".join(show_lines(pkg))
    if action == "check":
        return "\n".join(check_lines(ph.check(pkg)))
    if action == "evaluate":
        return json.dumps(ph.evaluate(pkg, str(args.get("facts") or "")), ensure_ascii=False, indent=1)
    if action == "install":
        o = install(ref, instance, None, args.get("visibility") or "intern")
        return f"installed {o.id} — philosophy {pkg['name']} {pkg['version']}, hash {o.data['philosophie']['hash'][:16]}…"
    if action == "use":
        target = args.get("instance")
        if target and target != instance:  # a seed never overrides tool rights: an instance binds itself; an administrator binds others
            raise ValueError(f"an instance binds only itself — bind {target} with: vi-mcp philosophy use {pkg['name']} --instance {target}")
        idx = args.get("index")
        return bind(pkg["name"], instance, None if idx else instance, idx, bool(args.get("off")))
    raise ValueError(f"unknown action {action}")


def main(argv: list[str]) -> int:
    """vi-mcp philosophy guide | list | show REF | check REF | new DIR --name N [--author A] | evaluate REF --facts FILE | install REF
    | use NAME --instance I | --index X [--off] | bound --instance I | --index X"""
    import argparse
    a = argparse.ArgumentParser(prog="vi-mcp philosophy", description="equip an AI with a philosophy: packages of principles, seed and test worlds")
    u = a.add_subparsers(dest="cmd", required=True)
    u.add_parser("guide", help="the technique, step by step")
    u.add_parser("list", help="built-in and installed philosophies")
    for c in ("show", "check"):
        u.add_parser(c).add_argument("ref", help="built-in name or package directory")
    nw = u.add_parser("new", help="write a minimal valid package to start from")
    nw.add_argument("directory")
    nw.add_argument("--name", required=True)
    nw.add_argument("--author", default="unknown")
    ev = u.add_parser("evaluate", help="the seed's verdict on facts (clingo-object notation)")
    ev.add_argument("ref")
    ev.add_argument("--facts", required=True, help="file with facts")
    ins = u.add_parser("install", help="check and store in the VI as a rule set")
    ins.add_argument("ref")
    ins.add_argument("--visibility", choices=["shared", "internal"], default="internal")
    us = u.add_parser("use", help="bind an installed philosophy (its current hash) to an instance or an index")
    us.add_argument("name")
    for c in (us, u.add_parser("bound", help="which philosophies and hashes apply to an instance or an index")):
        t = c.add_mutually_exclusive_group(required=True)
        t.add_argument("--instance")
        t.add_argument("--index")
    us.add_argument("--off", action="store_true", help="end the binding (kept with mode aus)")
    o = a.parse_args(argv)
    if o.cmd == "guide":
        print(ph.GUIDE)
        return 0
    if o.cmd == "new":
        d = ph.write(ph.template(o.name, o.author), o.directory)
        print(f"written {d} — edit philosophy.json, seed.co and worlds.jsonl, then: vi-mcp philosophy check {d}")
        return 0
    if o.cmd == "show":
        print("\n".join(show_lines(ph.load(o.ref))))
        return 0
    if o.cmd == "check":
        r = ph.check(ph.load(o.ref))
        print("\n".join(check_lines(r)))
        return 0 if r["ok"] else 1
    if o.cmd == "evaluate":
        from pathlib import Path
        print(json.dumps(ph.evaluate(ph.load(o.ref), Path(o.facts).read_text(encoding="utf-8")), ensure_ascii=False, indent=1))
        return 0
    from . import config
    g.open_db(config.load().db)
    try:
        if o.cmd == "list":
            print(tool({"action": "list"}, "admin"))
        elif o.cmd == "install":
            ob = install(o.ref, "admin", "admin", {"shared": "geteilt", "internal": "intern"}[o.visibility])
            print(f"installed {ob.id} (hash {ob.data['philosophie']['hash'][:16]}…)")
        elif o.cmd == "use":
            print(bind(o.name, "admin", o.instance, o.index, o.off))
        elif o.cmd == "bound":
            print("\n".join(binding_lines(bindings(o.instance, o.index))))
    finally:
        g.close_db()
    return 0
