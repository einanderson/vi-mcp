"""Declarative instance state and device reservation protocol. No external device runs in the memory server."""
from __future__ import annotations

import copy
import json
import uuid

from clingo_object import notation
from clingo_object.runtime import canonical, context_messages, default_runtime, fingerprint, validate_arguments, validate_runtime

MAX_EVENTS = 16


def idle_loop():
    return {"status": "idle", "goal": "", "steps": 0, "pending": None, "events": [], "events_dropped": 0}


def defaults():
    return {"context": [], "context_dropped": 0, "runtime": default_runtime(), "loop": idle_loop()}


def validate_world(world, max_objects):
    if not isinstance(world, dict) or not {"objects", "rules"}.issubset(world) or set(world) - {
            "objects", "rules", "imports", "context", "context_dropped", "runtime", "loop"}:
        raise ValueError("invalid complete instance world")
    world = copy.deepcopy(world)
    for key, value in defaults().items():
        world.setdefault(key, value)
    if not isinstance(world["objects"], dict) or len(world["objects"]) > max_objects or not isinstance(world["rules"], dict):
        raise ValueError("invalid or oversized instance objects/rules")
    for key, obj in world["objects"].items():
        if not isinstance(key, str) or not isinstance(obj, dict) or obj.get("id") != key:
            raise ValueError("invalid frozen object identity")
        if not obj.get("deleted"):
            if not isinstance(obj.get("tags"), dict) or any(not isinstance(v, list) or any(not isinstance(t, str) for t in v) for v in obj["tags"].values()):
                raise ValueError("invalid frozen object tags")
    for source in world["rules"].values():
        if not isinstance(source, str):
            raise ValueError("a frozen rule needs notation source")
        notation.translate(source)
    world["runtime"] = validate_runtime(world["runtime"])
    world["context"] = context_messages(world["context"])
    if type(world["context_dropped"]) is not int or world["context_dropped"] < 0:
        raise ValueError("invalid context compaction counter")
    if "imports" in world and (not isinstance(world["imports"], dict) or any(not isinstance(v, list)
            or any(not isinstance(oid, str) for oid in v) for v in world["imports"].values())):
        raise ValueError("invalid imported object membership")
    loop = world["loop"]
    if not isinstance(loop, dict) or set(loop) != set(idle_loop()) or loop.get("pending") or loop.get("status") not in ("idle", "running", "paused", "completed"):
        raise ValueError("cannot boot an in-flight or invalid loop")
    if not isinstance(loop["events"], list) or len(loop["events"]) > MAX_EVENTS or type(loop["steps"]) is not int or loop["steps"] < 0 or not isinstance(loop["goal"], str):
        raise ValueError("invalid loop state")
    if type(loop["events_dropped"]) is not int or loop["events_dropped"] < 0:
        raise ValueError("invalid loop history counter")
    canonical(world)
    return world


def context(state, messages, *, replace=False, reason=None):
    messages = context_messages(messages)
    if type(replace) is not bool:
        raise ValueError("replace must be boolean")
    before = state.get("context", [])
    if replace and before and not reason:
        raise ValueError("replacing context needs an explicit compaction/deletion reason")
    after = context_messages(messages if replace else before + messages)
    if replace:
        state["context_dropped"] = state.get("context_dropped", 0) + len(before)
    state["context"] = after


def _event(state, event):
    loop = state["loop"]
    loop["events"].append(event)
    if len(loop["events"]) > MAX_EVENTS:
        loop["events"].pop(0)
        loop["events_dropped"] += 1


def _observation(value):
    from . import anonymize
    value = anonymize.without_secrets(value)
    text = canonical(value)
    if len(text) > 3000:
        return {"prefix": text[:2500], "truncated": True, "characters": len(text), "sha256": fingerprint(value)}
    return value


def transition(state, action, payload):
    """Edit the caller's detached world. The enclosing machine commits only after validation and size checks."""
    for key, value in defaults().items():
        state.setdefault(key, value)
    loop = state["loop"]
    if action == "configure":
        state["runtime"] = validate_runtime(payload)
        return {"state": "configured", "runtime": state["runtime"]}
    if action == "context":
        context(state, payload.get("messages", []), replace=payload.get("replace", False), reason=payload.get("reason"))
        return {"state": "context stored", "messages": len(state["context"]), "dropped": state["context_dropped"]}
    if action == "device":
        if loop["status"] != "running" or loop["pending"]:
            raise ValueError("device dispatch needs a running loop without a pending action")
        name, arguments = payload.get("device"), payload.get("arguments")
        runtime = validate_runtime(state["runtime"])
        if name not in runtime["permissions"]:
            raise ValueError("device is not authorized")
        descriptor = runtime["devices"][name]
        validate_arguments(arguments, descriptor["schema"])
        # Reserve space for the bounded observation before any external action is allowed to start.
        context_messages(state["context"] + [{"role": "tool", "content": "x" * 4000}])
        pending = {"id": uuid.uuid4().hex, "device": name, "arguments": copy.deepcopy(arguments), "descriptor": descriptor,
                   "status": "prepared", "policy_sha256": fingerprint(runtime)}
        loop["pending"] = pending
        return {"state": "prepared", "reservation": pending["id"], "descriptor": descriptor}
    if action == "observe":
        pending = loop["pending"]
        if not pending or pending["id"] != payload.get("reservation"):
            raise ValueError("unknown or already completed device reservation")
        if pending["status"] != "prepared":
            raise ValueError("uncertain reservation needs an independent resolve, not a repeated observation")
        if type(payload.get("ok")) is not bool:
            raise ValueError("device observation needs boolean ok")
        result = _observation(payload.get("result"))
        uncertain = bool(payload.get("uncertain"))
        event = {"device": pending["device"], "reservation": pending["id"], "ok": payload["ok"], "result": result, "uncertain": uncertain}
        # If context is full, this refuses and leaves the persisted reservation pending. No automatic replay follows.
        context(state, [{"role": "tool", "content": canonical(event), "source": {"device": pending["device"], "reservation": pending["id"]}}])
        _event(state, event)
        loop["steps"] += 1
        if uncertain:
            pending["status"], loop["status"] = "uncertain", "paused"
        else:
            loop["pending"] = None
        return {"state": "uncertain" if uncertain else "observed", "reservation": pending["id"], "ok": payload["ok"]}
    if action != "loop":
        raise ValueError("unknown instance transition")
    operation = payload.get("operation")
    if operation == "resolve":
        pending = loop["pending"]
        if not pending or payload.get("reservation") != pending["id"] or not payload.get("reference") or payload.get("outcome") not in ("completed", "not_run", "failed"):
            raise ValueError("resolve needs pending reservation, independently checked outcome and reference")
        _event(state, {"reservation": pending["id"], "resolution": payload["outcome"], "reference": str(payload["reference"])})
        loop["pending"], loop["status"] = None, "paused"
        return {"state": "resolved", "outcome": payload["outcome"]}
    if loop["pending"]:
        raise ValueError("pending device action needs explicit resolution")
    if operation in ("start", "resume"):
        if operation == "start":
            goal = payload.get("goal")
            if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
                raise ValueError("loop goal must be 1–4000 characters")
            context(state, [{"role": "user", "content": goal}])
            loop["goal"], loop["steps"] = goal, 0
        elif loop["status"] not in ("running", "paused"):
            raise ValueError("no unfinished loop to resume")
        loop["status"] = "running"
        return {"state": "running"}
    if operation == "pause":
        loop["status"] = "paused"
        return {"state": "paused", "reason": str(payload.get("reason") or "explicit pause"), "steps": loop["steps"]}
    if loop["status"] != "running":
        raise ValueError("loop is not running")
    if operation == "finish":
        answer = payload.get("answer")
        if not isinstance(answer, str) or len(answer) > 8000:
            raise ValueError("finish needs an answer of at most 8000 characters")
        context(state, [{"role": "assistant", "content": answer}])
        loop["status"] = "completed"
        _event(state, {"operation": "finish", "answer": answer})
        return {"state": "completed", "answer": answer, "steps": loop["steps"], "evidence": "model_report", "verified": False}
    if operation in ("record", "error"):
        event = _observation({"plan": payload.get("plan"), "result": payload.get("result")} if operation == "record" else {"refused": str(payload.get("error"))})
        context(state, [{"role": "tool", "content": canonical(event), "source": {"runtime": "VIL"}}])
        _event(state, event)
        loop["steps"] += 1
        return {"state": "recorded", "steps": loop["steps"]}
    raise ValueError("unknown loop operation")
