"""A persistent, private virtual intelligence machine executing VIL and clingo-object.

The LLM is a replaceable program writer. The runtime owns the world, execution,
verification, revisions and snapshots. No instruction invokes a shell or a foreign tool.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from clingo_object import agent, notation as n
from clingo_object.filters import object_data
from . import memory as g, vil, machine_runtime as mr

CONTRACT = "vil-machine-v1"
MAX_OBJECTS = 500
MAX_HISTORY = 16
MAX_SNAPSHOTS = 16
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
OBJECT_NAME = re.compile(r"[a-z0-9][a-z0-9_:.-]{0,127}")


def _name(name):
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError("name must be lower case, digits, - or _ (1–64 characters)")
    return name


def _id(owner, name, snapshot=False):
    return "maschine:" + g.hash_value(owner, "snapshot" if snapshot else "machine", _name(name))


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _load(owner, name, snapshot=False):
    o = g.load(_id(owner, name, snapshot))
    if o is None or o.instance != owner or o.kind != "maschine" or o.mode != "aktiv":
        raise ValueError("unknown machine or snapshot")
    if o.data.get("vertrag") != CONTRACT:
        raise ValueError("unsupported machine contract")
    return o


def _save(o, reason, *, commit=True):
    # Refuse secrets instead of silently changing a program or snapshot during scrubbing.
    from . import anonymize
    if anonymize.without_secrets(o.data) != o.data:
        raise ValueError("machine state contains a credential pattern")
    # Receipts carry copies of the world they read: with a large world a handful of them filled the object and every further
    # instruction — a deletion too — was refused. The oldest receipts give way first, counted in history_dropped; the newest
    # stays so it can still be rated. Checked examples keep their own copy and never give way.
    while len(json.dumps(o.data, ensure_ascii=False)) > g.DATA_MAX - 2000 and len(o.data["history"]) > 1:
        o.data["history"].pop(0)
        o.data["history_dropped"] += 1
    if len(json.dumps(o.data, ensure_ascii=False)) > g.DATA_MAX - 2000:
        raise ValueError("machine state limit reached; remove objects, drop checked examples by teaching them, or start another machine")
    o.data["revision"] += 1
    o.last_at = g.now()
    o.change(o.instance, reason, "explicit machine instruction")
    g.save(o, commit=commit)


def _evidence(o):
    return copy.deepcopy({k: o.data[k] for k in ("history", "history_dropped", "checks")})


def _new(owner, name, state=None, parent=None, snapshot=False, *, commit=True, evidence=None):
    oid = _id(owner, name, snapshot)
    if g.load(oid) is not None:
        raise ValueError("machine or snapshot already exists")
    state = mr.validate_world(state if state is not None else {"objects": {}, "rules": {}}, MAX_OBJECTS)
    evidence = copy.deepcopy(evidence if evidence is not None else {"history": [], "history_dropped": 0, "checks": []})
    if not isinstance(evidence, dict) or set(evidence) != {"history", "history_dropped", "checks"} or type(evidence["history_dropped"]) is not int or evidence["history_dropped"] < 0:
        raise ValueError("invalid instance execution evidence")
    if any(not isinstance(evidence[k], list) or len(evidence[k]) > MAX_HISTORY for k in ("history", "checks")):
        raise ValueError("invalid or oversized execution evidence")
    for receipt in evidence["history"]:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("id"), str) or not isinstance(receipt.get("execution"), dict):
            raise ValueError("invalid execution receipt")
        if "source" in receipt["execution"]:
            if not isinstance(receipt["execution"]["source"], str):
                raise ValueError("invalid frozen program source")
            n.translate(receipt["execution"]["source"])
    for check in evidence["checks"]:
        if not isinstance(check, dict) or not isinstance(check.get("receipt"), str) or type(check.get("correct")) is not bool or not isinstance(check.get("execution"), dict):
            raise ValueError("invalid independent reference evidence")
        if not isinstance(check["execution"].get("source"), str) or not isinstance(check.get("expected"), list) or not isinstance(check.get("reference"), str) or not isinstance(check.get("question"), str):
            raise ValueError("incomplete frozen reference evidence")
        n.translate(check["execution"].get("source", ""))
    data = {"vertrag": CONTRACT, "revision": 0, "zustand": state, **evidence,
            "snapshots": [], "parent": parent, "snapshot": snapshot}
    from . import anonymize
    if anonymize.without_secrets(data) != data or len(json.dumps(data)) > g.DATA_MAX - 2000:
        raise ValueError("machine state exceeds limits or contains a credential pattern")
    return g.new_object("maschine", "Snapshot " + name if snapshot else "Virtual intelligence machine " + name,
                        ["thema:vi-machine"], data, oid=oid, visibility="privat", instance=owner, link=False,
                        reason="explicit snapshot" if snapshot else "explicit machine creation", commit=commit)


def _status(o):
    s = o.data["zustand"]
    return {"contract": CONTRACT, "revision": o.data["revision"], "state_hash": state_hash(s), "snapshot": o.data["snapshot"],
            "objects": sum(not x.get("deleted") for x in s["objects"].values()), "rules": len(s["rules"]),
            "history": len(o.data["history"]), "history_dropped": o.data["history_dropped"], "checks": len(o.data["checks"]),
            "snapshots": list(o.data["snapshots"]), "parent": o.data["parent"],
            "context_messages": len(s.get("context", [])), "context_dropped": s.get("context_dropped", 0),
            "model": s.get("runtime", mr.defaults()["runtime"])["model"],
            "permissions": s.get("runtime", mr.defaults()["runtime"])["permissions"],
            "loop_status": s.get("loop", mr.idle_loop())["status"],
            "pending_device": bool(s.get("loop", {}).get("pending"))}




class World:
    """Read-only clingo-object memory adapter over one frozen machine state; never reads the global knowledge base."""
    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.source_prefix = "\n".join(self.state["rules"].values())

    def objects(self, filters, edges=True):
        objs = [o for o in self.state["objects"].values() if not o.get("deleted")]
        data = object_data(objs, filters)
        data["counts"]["excluded"] = sum(o.get("deleted", False) for o in self.state["objects"].values())
        data["limit"] = MAX_OBJECTS
        return data

    def call(self, tool, args):
        if tool != "vi_vocabulary":
            raise ValueError("machine frontend may only inspect its own vocabulary")
        kinds = {}
        for o in self.state["objects"].values():
            if not o.get("deleted"):
                for k, vals in o["tags"].items():
                    kinds.setdefault(k, set()).update(vals)
        if args.get("word_kind"):
            k = args["word_kind"]
            return "\n".join(f"  {k}:{v} (known)" for v in sorted(kinds.get(k, []))[:args.get("limit", 30)])
        return "\n".join(f"  {k}: ({len(v)} values)" for k, v in sorted(kinds.items())[:args.get("limit", 30)])


def _run(state, source):
    from . import logic
    if not logic.available():
        raise ValueError("clingo is unavailable; install it with vi-mcp logic install")
    source = "\n".join([*state["rules"].values(), source])
    program, stmts, filters = n.translate_full(source)
    world = World(state).objects(filters, edges="vi__edge(" in program)
    if "vi__edge(" in program:
        raise ValueError("machine contract v1 has no edges; use object values for joins")
    raw = n.run_program("\n".join(n.facts(world)) + "\n" + program, stmts)
    result = n.check_result(raw)
    projections = [{a for a in m if a == "ans" or a.startswith("ans(")} for m in raw["map"]]
    complete = bool(result.get("vollstaendig")) and not raw.get("timeout")
    certain = bool(projections) and all(p == projections[0] for p in projections)
    answers = sorted(projections[0]) if certain else []
    context = [copy.deepcopy(o) for o in state["objects"].values() if not o.get("deleted")]
    # The receipt keeps the input objects once: the filtered view is recomputed from them (training_eval, dataset), so it is stored
    # as ids, members and counts only. Two full copies per receipt made a machine at its object limit unable to run one query.
    world_ref = {**{k: v for k, v in world.items() if k != "objects"}, "object_ids": [x["id"] for x in world["objects"]]}
    return {"state": result["zustand"], "complete": complete, "determinate": certain, "answers": answers, "result": result,
            "source": source, "world": world_ref, "input_objects": context, "source_prefix": "\n".join(state["rules"].values()),
            "state_hash": state_hash(state), "input_context": copy.deepcopy(state.get("context", [])), "evidence": "inferred"}


def _receipt(o, instruction, result):
    rid = g.hash_value(o.id, str(o.data["revision"]), instruction, g.now())
    o.data["history"].append({"id": rid, "instruction": instruction, "execution": result})
    if len(o.data["history"]) > MAX_HISTORY:
        o.data["history"].pop(0)
        o.data["history_dropped"] += 1
    return rid


def call(owner: str, args: dict) -> dict:
    """Atomic machine transition. expected_revision prevents stale writers; all named states are private to the owner."""
    if not owner or owner == g.OPEN_INSTANCE:
        raise ValueError("a machine needs an authenticated instance")
    name = _name(args.get("name") or "default")
    action = args.get("action") or "status"
    with g.lock:
        if action == "create":
            payload = args.get("payload") or {}
            if not isinstance(payload, dict) or set(payload) - {"world", "evidence"}:
                raise ValueError("create accepts only an optional complete world")
            return _status(_new(owner, name, payload.get("world"), evidence=payload.get("evidence")))
        o = _load(owner, name)
        if action == "status":
            return _status(o)
        if args.get("expected_revision") is not None and args["expected_revision"] != o.data["revision"]:
            raise ValueError("stale machine revision; read status and retry")
        if action == "inspect":
            obj = _load(owner, args["snapshot"], True) if args.get("snapshot") else o
            world = copy.deepcopy(obj.data["zustand"])
            for key, value in mr.defaults().items():
                world.setdefault(key, value)
            return {**_status(obj), "world": world, "evidence": _evidence(obj)}
        pending = o.data["zustand"].get("loop", {}).get("pending")
        if pending and action not in ("observe", "configure", "loop"):
            raise ValueError("pending device action prevents state changes and snapshots; observe or resolve it first")
        if action in ("configure", "context", "device", "observe", "loop"):
            payload = args.get("payload") or {}
            if not isinstance(payload, dict):
                raise ValueError("instance operands must be an object")
            result = mr.transition(o.data["zustand"], action, payload)
            _save(o, "instance " + action)
            return {**result, "revision": o.data["revision"], "state_hash": state_hash(o.data["zustand"]), "contract": CONTRACT}
        if action == "snapshot":
            snap = _name(args.get("snapshot"))
            if len(o.data["snapshots"]) >= MAX_SNAPSHOTS:
                raise ValueError("snapshot limit reached")
            # One transaction: a rejected/full parent must not leave an orphaned immutable snapshot.
            try:
                frozen = _new(owner, snap, o.data["zustand"], _status(o), True, commit=False, evidence=_evidence(o))
                o.data["snapshots"].append(snap)
                _save(o, "snapshot created", commit=False)
                g.db().commit()
            except Exception:
                g.db().rollback()
                raise
            return {**_status(frozen), "name": snap}
        if action == "fork":
            base = _load(owner, args["snapshot"], True) if args.get("snapshot") else o
            return _status(_new(owner, _name(args.get("target")), base.data["zustand"], _status(base)))
        if action == "restore":
            frozen = _load(owner, _name(args.get("snapshot")), True)
            o.data["zustand"] = copy.deepcopy(frozen.data["zustand"])
            _save(o, "snapshot restored")
            return _status(o)
        if action == "import":
            from . import logic
            filters = args.get("filters")
            if not isinstance(filters, list) or not 1 <= len(filters) <= logic.MAX_FILTERS:
                raise ValueError("import needs a bounded list of object filters")
            data = logic.object_data(filters, owner, edges=False)
            if data["counts"]["cut"] or len(data["objects"]) > MAX_OBJECTS:
                raise ValueError("import exceeds machine object limit; narrow the filters")
            state = o.data["zustand"]
            mount = state.setdefault("imports", {})
            key = state_hash(sorted(filters))
            ids = {x["id"] for x in data["objects"]}
            for oid in set(mount.get(key, [])) - ids:
                state["objects"][oid] = {"id": oid, "deleted": True, "reason": "object no longer matches imported filters"}
            for obj in data["objects"]:
                state["objects"][obj["id"]] = obj
            if len(state["objects"]) > MAX_OBJECTS:
                raise ValueError("import exceeds total object limit")
            mount[key] = sorted(ids)
            _save(o, "visible VI objects imported or refreshed")
            return {**_status(o), "imported": len(ids), "excluded": data["counts"]["excluded"], "not_visible": data["counts"]["not_visible"]}
        if action == "teach":
            checked = next((x for x in o.data["checks"] if x["receipt"] == args.get("receipt")), None)
            if not checked or not checked["question"].strip():
                raise ValueError("teach needs a checked receipt with a question")
            # The published example carries the frozen execution (its input objects, world and context). Imported objects keep their
            # memory visibility: a private one (the owner sees its own) or one that is not knowledge (experience, conversation) must not
            # become internal training data through teach. Objects written into the machine itself are the owner's explicit content.
            ex = checked["execution"]
            inputs = ex.get("input_objects") or list((ex.get("world") or {}).get("objects") or [])
            from .dataset import KINDS
            blocked = sorted({x.get("id", "?") for x in inputs if isinstance(x, dict) and "instance" in x
                              and (x.get("visibility") == "privat" or x.get("kind") not in KINDS)})
            if blocked:
                raise ValueError(f"teach refused: the checked execution read {len(blocked)} imported private or non-knowledge "
                                 "object(s); delete them from the machine world or import narrower filters, rerun and rate again")
            if ex.get("input_context") and args.get("include_context") is not True:
                raise ValueError("teach refused: the execution carries private instance context; pass include_context=true to "
                                 "publish it internally with the example, or clear the context, rerun and rate again")
            oid ="regel:" + g.hash_value("machine-example", owner, o.id, checked["receipt"])
            existing = g.load(oid)
            example = {**copy.deepcopy(checked), "contract": CONTRACT}
            if existing is not None:
                if existing.data.get("machine_example") == example:
                    return {"state": "already taught", "id": oid}
                existing.data.update(machine_example=example, quelle=checked["reference"])
                existing.text = checked["question"]
                existing.change(owner, "training reference corrected", "explicit teach after independent recheck")
                g.save(existing)
                return {"state": "teaching corrected", "id": oid, "correct": checked["correct"], "visibility": "internal"}
            published = g.new_object("regel", checked["question"], ["thema:machine-training"],
                                     {"machine_example": example, "quelle": checked["reference"]}, oid=oid, instance=owner,
                                     visibility="intern", link=False, reason="machine owner explicitly published checked training example")
            return {"state": "taught", "id": published.id, "correct": checked["correct"], "visibility": "internal"}
        if action not in ("execute", "answer"):
            raise ValueError("action must be create, status, execute, answer, snapshot, restore, fork or teach")
        state = o.data["zustand"]
        payload = args.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        instruction = str(args.get("instruction") or "")
        if action == "answer":
            from . import logic
            if not logic.available():
                raise ValueError("clingo is unavailable; install it with vi-mcp logic install")
            if state.get("runtime", {}).get("model", {}).get("binding", "server-classify") != "server-classify":
                raise ValueError("this model binding runs through the trusted instance controller, not the server frontend")
            q = str(args.get("question") or "").strip()
            if not q or len(q) > 2000:
                raise ValueError("answer needs a question of 1–2000 characters")
            # Model calls run without holding the global database lock. Commit only if the world stayed unchanged.
            revision = o.data["revision"]
            world = World(state)
        else:
            u = vil.parse(instruction)
            head = vil._HEAD.match(instruction)
            if not head:
                raise ValueError("machine instructions must start with a VIL act")
            subtype = (head.group(2) or "").lower()
            body = instruction[head.end():].strip()
            if u.act == "tell":
                if subtype not in ("fact", "object"):
                    raise ValueError("tell.fact or tell.object is required")
                if not OBJECT_NAME.fullmatch(body):
                    raise ValueError("invalid object id")
                key = body
                if payload.get("deleted"):
                    if key not in state["objects"]:
                        raise ValueError("unknown object")
                    state["objects"][key] = {"id": key, "deleted": True, "reason": str(payload.get("reason") or "explicit deletion")}
                else:
                    if key not in state["objects"] and len(state["objects"]) >= MAX_OBJECTS:
                        raise ValueError("object limit reached (including tombstones)")
                    tags = {}
                    for t in payload.get("hashtags", []):
                        if not isinstance(t, str) or not g.TAG.fullmatch(t) or t.startswith(("instanz:", "modus:")):
                            raise ValueError("invalid machine hashtag")
                        k, v = t.split(":", 1)
                        tags.setdefault(k, []).append(v)
                    kind = str(payload.get("kind") or "erkenntnis")
                    if kind not in ("erkenntnis", "regel", "skill"):
                        raise ValueError("object kind must be erkenntnis, regel or skill")
                    tags["art"] = [kind]
                    if not str(payload.get("source") or "").strip():
                        raise ValueError("a fact needs its source")
                    state["objects"][key] = {"id": key, "kind": kind, "mode": "aktiv", "visibility": "privat", "index": name,
                        "text": str(payload.get("text") or ""), "source": str(payload["source"]), "tags": tags,
                        "age": 0, "importance": 1, "confirmed": 0, "refuted": 0, "accesses": 0}
                result = {"state": "stored", "object": key}
            elif u.act == "def":
                if subtype != "rule":
                    raise ValueError("def.rule is required")
                key = _name(body)
                source = str(payload.get("source") or "")
                n.translate(source)
                if not source.strip():
                    raise ValueError("a rule needs source")
                state["rules"][key] = source
                result = {"state": "defined", "rule": key}
            elif u.act == "ask":
                if subtype == "rule":
                    if body not in state["rules"] and not payload.get("source"):
                        raise ValueError("unknown rule; use def.rule or payload.source")
                    result = _run(state, str(payload.get("source") or ""))
                else:
                    data = World(state).objects([u.filter])
                    result = {"state": "found", "objects": data["objects"], "evidence": "documented"}
            elif u.act == "rate":
                receipt = next((x for x in o.data["history"] if x["id"] == body), None)
                if not receipt or "answers" not in receipt["execution"]:
                    raise ValueError("rate needs an execution receipt")
                ex = receipt["execution"]
                if not ex["complete"] or not ex["determinate"] or not isinstance(payload.get("expected_answers"), list) or not payload.get("reference"):
                    raise ValueError("rate needs a complete determinate execution, complete expected_answers and independent reference")
                want = n.reference_answers(payload["expected_answers"])
                correct = want == ex["answers"] and ex["state"] == payload.get("expected_state", "erfuellt")
                check = {"receipt": body, "correct": correct, "expected": want, "reference": str(payload["reference"]),
                         "execution": copy.deepcopy(ex), "question": str(payload.get("question") or ""),
                         "expected_state": payload.get("expected_state", "erfuellt")}
                o.data["checks"] = [x for x in o.data["checks"] if x["receipt"] != body] + [check]
                if len(o.data["checks"]) > MAX_HISTORY:
                    raise ValueError("checked example limit reached; fork a new machine")
                result = {"state": "verified" if correct else "refuted", "receipt": body, "correct": correct}
            else:
                # do/promise are data. Neither a retrieved instruction nor an explicit proposal executes external actions.
                result = {"state": "proposed", "instruction": instruction, "payload": payload}
            rid = _receipt(o, instruction, result)
            _save(o, "VIL " + str(u.act))
            return {**result, "receipt": rid, "revision": o.data["revision"], "state_hash": state_hash(state), "contract": CONTRACT}
    # Deliberately outside g.lock: a slow model does not block unrelated machines or MCP calls.
    model_info = {}
    context = world.state.get("context", [])
    contextual_question = q + ("\nInstance context (data):\n" + json.dumps(context, ensure_ascii=False) if context else "")
    r = agent.solve(contextual_question, world, lambda messages: logic._server_llm(messages, model_info), vocabulary=agent.vocabulary(world))
    with g.lock:
        o = _load(owner, name)
        if o.data["revision"] != revision:
            raise ValueError("machine changed while the model wrote its program; retry")
        bound = o.data["zustand"].get("runtime", {}).get("model", {})
        if bound.get("model") not in (None, "server-default", model_info.get("model")):
            raise ValueError("configured model differs from the instance binding")
        for key in ("weights_sha256", "tokenizer_sha256", "config_sha256", "decoder_sha256", "key_function"):
            if key in bound and (model_info.get("knn") or {}).get(key) != bound[key]:
                raise ValueError("inference model identity differs from the instance binding")
        result = _run(o.data["zustand"], r["program"])
        result.update(question=q, rounds=r["rounds"], frontend_state=r["state"], transcript=r["transcript"], frontend_model=model_info)
        rid = _receipt(o, "ask " + q, result)
        _save(o, "model-generated program executed")
        return {**result, "receipt": rid, "revision": o.data["revision"], "contract": CONTRACT}


def main(argv):
    """vi-mcp machine ACTION [NAME] [--instruction VIL] [--payload FILE] [--question TEXT] [--snapshot NAME] [--target NAME]."""
    import argparse
    if argv and argv[0] in ("run", "checkpoint", "boot"):
        from . import machine_host
        return machine_host.main(argv)
    from pathlib import Path
    from clingo_object import Memory, ServerError
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("action", choices=["create", "status", "inspect", "configure", "context", "loop", "device", "observe", "execute", "answer", "snapshot", "restore", "fork", "import", "teach"])
    p.add_argument("name", nargs="?", default="default")
    for field in ("instruction", "question", "snapshot", "target", "receipt", "url", "token-file"):
        p.add_argument("--" + field)
    p.add_argument("--expected-revision", type=int)
    p.add_argument("--filter", action="append", dest="filters")
    p.add_argument("--payload", help="JSON file with structured operands")
    opts = vars(p.parse_args(argv))
    url, token_file = opts.pop("url"), opts.pop("token_file")
    try:
        if opts["payload"]:
            opts["payload"] = json.loads(Path(opts["payload"]).read_text(encoding="utf-8"))
        args = {k: v for k, v in opts.items() if v is not None}
        print(Memory(url=url, token_file=token_file).call("vi_machine", args))
        return 0
    except (OSError, ValueError, ServerError) as e:
        print(f"vi-mcp machine: {e}")
        return 2
