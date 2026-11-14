"""Trusted inference/client host for VI instances and coherent checkpoints. Talks to the memory only through MCP."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

from clingo_object import Memory, ServerError
from clingo_object.runtime import Device, Instance, OpenAIProgrammer, default_runtime, fingerprint, validate_runtime


class KNNProgrammer:
    """One frozen neural core plus this instance's mutable datastore and greedy decoder settings."""
    def __init__(self, lm, steps=256):
        if type(steps) is not int or not 1 <= steps <= 4096:
            raise ValueError("invalid decoder step budget")
        self.lm, self.steps = lm, steps
        self.parameters = {"weight": lm.weight, "neighbors": lm.k, "temperature": lm.temperature, "steps": steps}
        self.identity = {"binding": "local-knn", **{k: lm.core.identity[k] for k in
            ("model", "weights_sha256", "tokenizer_sha256", "config_sha256", "key_function") if k in lm.core.identity},
                         "decoder_sha256": fingerprint(self.parameters)}

    def __call__(self, messages):
        result = self.lm.generate(self.lm.core.chat_prompt(messages), steps=self.steps)
        if result["finish_reason"] != "stop":
            raise ValueError("incomplete local model output; check thinking, context and decoder resource budget")
        return result["text"]

    @contextlib.contextmanager
    def freeze(self, destination):
        # The datastore writer lock remains held until the machine has captured its world at this same boundary.
        with self.lm.store.frozen_snapshot(destination) as frozen:
            yield {"kind": "knn", "identity": self.identity, "parameters": self.parameters,
                   "datastore_sha256": fingerprint(frozen.view())}

    @classmethod
    def boot(cls, core, bundle_path, destination):
        from .knn_lm import Datastore, KNNLanguageModel
        bundle_path = Path(bundle_path)
        if (bundle_path / "incomplete.json").exists():
            raise ValueError("cannot boot an incomplete inference checkpoint")
        bundle = json.loads((bundle_path / "instance.json").read_text())
        backend = bundle["model"]
        if backend["kind"] != "knn":
            raise ValueError("checkpoint does not contain a local kNN backend")
        frozen = Datastore(bundle_path / "inference", core.identity)
        if fingerprint(frozen.view()) != backend["datastore_sha256"]:
            raise ValueError("inference checkpoint manifest is corrupt")
        parameters = backend["parameters"]
        # Check the model/decoder before making any new datastore or VI object.
        probe = cls(KNNLanguageModel(core, frozen, weight=parameters["weight"], k=parameters["neighbors"],
                                    temperature=parameters["temperature"]), parameters["steps"])
        if probe.identity != backend["identity"]:
            raise ValueError("checkpoint inference identity differs from the current core")
        live = frozen.snapshot(destination)
        return cls(KNNLanguageModel(core, live, weight=parameters["weight"], k=parameters["neighbors"],
                                   temperature=parameters["temperature"]), parameters["steps"])


def bindings(path, *, boot_path=None):
    """An operator's private JSON file defines transports and an explicit capability allow-list. No Python modules from a file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = data["model"]
    if model["kind"] == "knn":
        from .knn_lm import Datastore, KNNLanguageModel, TransformersCore
        core = TransformersCore.load(model["model"], model.get("revision"), model.get("device", "cpu"))
        if boot_path:
            programmer = KNNProgrammer.boot(core, boot_path, model["store"])
        else:
            lm = KNNLanguageModel(core, Datastore(model["store"], core.identity), weight=model.get("weight", .5),
                                  k=model.get("neighbors", 32), temperature=model.get("temperature", 1.))
            programmer = KNNProgrammer(lm, model.get("steps", 256))
    elif model["kind"] == "remote":
        programmer = OpenAIProgrammer(model["url"], model["model"], binding=model["binding"], token_file=model.get("token_file"),
                                       identity=model.get("identity"))
    else:
        raise ValueError("model kind must be remote or knn")
    devices = {}
    for name, d in data.get("devices", {}).items():
        # A foreign MCP device must never inherit the VI server's discovered default credential.
        devices[name] = Device.mcp(Memory(url=d["url"], token_file=d.get("token_file") or os.devnull), d["tool"], binding=d["binding"],
                                   schema=d["schema"], description=d["description"], effect=d["effect"])
    runtime = {**default_runtime(), "model": programmer.identity, "devices": {k: v.descriptor for k, v in devices.items()},
               "permissions": data.get("permissions", []), "max_steps": data.get("max_steps", 32)}
    runtime = validate_runtime(runtime)
    # A restored snapshot cannot restore a capability removed from today's host profile allow-list.
    return programmer, {name: d for name, d in devices.items() if name in runtime["permissions"]}, runtime


def main(argv):
    """vi-mcp machine run|checkpoint|boot NAME --bindings FILE [--goal TEXT] [--directory DIR] [--snapshot NAME] [--create]."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("action", choices=["run", "checkpoint", "boot"])
    p.add_argument("name")
    p.add_argument("--bindings", required=True, help="private JSON: model transport, registered MCP devices and explicit permissions")
    p.add_argument("--goal")
    p.add_argument("--steps", type=int)
    p.add_argument("--directory")
    p.add_argument("--snapshot")
    p.add_argument("--create", action="store_true", help="create and configure a new private instance from this binding file")
    p.add_argument("--url")
    p.add_argument("--token-file")
    a = p.parse_args(argv)
    try:
        if a.action in ("boot", "checkpoint") and not a.directory:
            raise ValueError("boot/checkpoint need --directory")
        if a.action == "checkpoint" and not a.snapshot:
            raise ValueError("checkpoint needs --snapshot")
        model, devices, runtime = bindings(a.bindings, boot_path=a.directory if a.action == "boot" else None)
        memory = Memory(url=a.url, token_file=a.token_file)
        if a.action == "boot":
            instance = Instance.boot(memory, a.name, model, a.directory, devices=devices)
            result = instance.inspect()
        else:
            instance = Instance(memory, a.name, model, identity=model.identity, devices=devices)
            if a.create:
                instance.call("create")
                instance.call("configure", payload=runtime, expected_revision=0)
            result = instance.run(a.goal, steps=a.steps) if a.action == "run" else instance.checkpoint(a.directory, a.snapshot)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ServerError) as e:
        print(f"vi-mcp machine host: {type(e).__name__}: {e}")
        return 2
