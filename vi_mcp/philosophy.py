"""vi-mcp side of philosophy packages (0.66.0): the command line, installing a checked package into the VI, the tool vi_philosophy.

The technique itself is clingo_object.philosophy (standard library plus clingo, usable without a server). The user (15.11.2026): "Die
Philosophie-Technik muss als Skill vorhanden sein, damit man eine KI mit irgendeiner Philosophie ausstatten kann."
"""
from __future__ import annotations

import json

from clingo_object import philosophy as ph

from . import memory as g

TAG = "thema:philosophie"


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
           f"{pkg['profile'].get('semantics')} · hash {ph.package_hash(pkg)[:16]}…", "", "Principles:"]
    out += [f"  {p['id']} {p.get('title', '')}: {p['text']}" for p in pkg["principles"]]
    out += ["", "Inputs the world gives:"] + [f"  {k}/{v['arity']}: {v.get('doc', '')}" for k, v in pkg["vocabulary"].items() if v["kind"] == "input"]
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


def tool(args: dict, instance: str) -> str:
    """vi_philosophy: guide (default) · list · show · check · evaluate · install."""
    action = args.get("action") or "guide"
    if action == "guide":
        return ph.GUIDE
    if action == "list":
        inst = installed()
        lines = ["Built in: " + ", ".join(sorted(ph.BUILTIN))]
        lines += ["Installed: " + (", ".join(f"{o.data['philosophie']['name']} {o.data['philosophie']['version']} ({o.id})" for o in inst) or "none")]
        return "\n".join(lines)
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
    raise ValueError(f"unknown action {action}")


def main(argv: list[str]) -> int:
    """vi-mcp philosophy guide | list | show REF | check REF | new DIR --name N [--author A] | evaluate REF --facts FILE | install REF"""
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
    finally:
        g.close_db()
    return 0
