"""Reproduce mutable-world execution, snapshots, isolation and checked training export.

    .venv/bin/python scripts/demo_machine.py [--output DIR] [--llm-url URL --model MODEL]

Default: a labelled scripted program writer, no network and no training.
An explicit --llm-url uses a real OpenAI-compatible model as the program writer.
The world and database are temporary; --output keeps the synthetic training export.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vi_mcp import dataset, evaluation, logic, machine, memory as g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    p.add_argument("--llm-url")
    p.add_argument("--model", default="default")
    opts = p.parse_args()
    if opts.llm_url:
        from clingo_object.agent import openai_chat
        logic.agent_llm = openai_chat(opts.llm_url, opts.model)
    else:
        logic.agent_llm = lambda messages: 'ans(P) <= (O in objects("service:api")) & (O.port == P)'
    with tempfile.TemporaryDirectory(prefix="vi-machine-demo-") as td:
        base = Path(td)
        evaluation.EVAL_DIR = base / "eval"
        g.open_db(base / "demo.db")
        try:
            def call(action, **kw):
                return machine.call("demo", {"action": action, "name": "operations", **kw})
            def fact(port):
                return call("execute", instruction="tell.fact api", payload={"text": "API endpoint", "hashtags": ["service:api", f"port:{port}"],
                            "source": "independent synthetic inventory"})
            def check(ex, port):
                result = call("execute", instruction="rate " + ex["receipt"], payload={"question": "Which port does the API use?",
                              "expected_answers": [f'ans("{port}")'], "reference": "Python fixture expected_port"})
                call("teach", receipt=ex["receipt"])
                return result["correct"]
            call("create")
            fact(80)
            a = call("answer", question="Which port does the API use?")
            assert check(a, 80)
            call("snapshot", snapshot="initial")
            call("fork", target="independent", snapshot="initial")
            fact(8080)
            b = call("answer", question="Which port does the API use?")
            assert check(b, 8080)
            fork = machine.call("demo", {"action": "execute", "name": "independent", "instruction": "ask.rule", "payload": {"source": a["source"]}})
            call("restore", snapshot="initial")
            restored = call("execute", instruction="ask.rule", payload={"source": a["source"]})
            assert fork["answers"] == restored["answers"] == ['ans("80")']
            wrong = call("execute", instruction="ask.rule", payload={"source": 'ans("wrong")'})
            assert not check(wrong, 80)
            manifest = dataset.export(Path(opts.output) if opts.output else base / "dataset")
            print(json.dumps({"frontend": "real LLM" if opts.llm_url else "scripted reference program (not a trained model)",
                              "before": a["answers"], "after": b["answers"], "fork": fork["answers"], "restored": restored["answers"],
                              "same_generated_program": a["source"] == b["source"], "training_counts": manifest["counts"],
                              "limits": "one task family; no fine-tuning or general improvement demonstrated"}, indent=2))
        finally:
            g.close_db()


if __name__ == "__main__":
    main()
