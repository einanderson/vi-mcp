"""The exported kit evaluates independently of the checkout and refuses incomplete or mismatched evidence."""
import json
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vi_mcp import finetune
from tests.test_finetune import vi  # noqa: F401


def run_script(directory, name, *args):
    # -I excludes the project and cwd; only the exported kit and installed dependencies are made available.
    code = "import sys,runpy; sys.path.insert(0," + repr(str(directory)) + "); sys.argv=" + repr([name, *args]) + "; runpy.run_path(" + repr(str(directory / name)) + ",run_name='__main__')"
    return subprocess.run([sys.executable, "-I", "-c", code], cwd=directory, text=True, capture_output=True)


@pytest.fixture()
def exported(vi):
    directory = vi / "kit"
    finetune.kit(str(directory), base_model="fixture-model", force=True)
    case = {"id": "one", "group": "independent-fixture", "question": "Which value?", "reference": "independent fixture",
            "case": {"expected": ['ans("safe")']}}
    (directory / "program_eval.jsonl").write_text(json.dumps(case) + "\n")
    return directory


def test_program_evaluator_runs_standalone_and_requires_exact_offline_coverage(exported):
    pytest.importorskip("clingo")
    predictions = exported / "cached.json"
    predictions.write_text(json.dumps([{"id": "one", "base": 'ans("wrong")', "tuned": 'ans("safe")'}]))
    result = run_script(exported, "program_evaluate.py", "--base", "unused", "--tuned", "unused", "--predictions", str(predictions))
    assert result.returncode == 0, result.stderr
    report = json.loads((exported / "program_eval_result.json").read_text())
    assert report["base"] == 0 and report["tuned"] == 1 and report["lower_95"] < 0 and not report["decision"]
    predictions.write_text("[]")
    result = run_script(exported, "program_evaluate.py", "--base", "do-not-contact", "--tuned", "do-not-contact", "--predictions", str(predictions))
    assert result.returncode != 0 and "cover exactly" in result.stderr


@pytest.fixture()
def truncated_model():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": 'ans("safe")'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cut_model_program_does_not_certify_success(exported, truncated_model):
    result = run_script(exported, "program_evaluate.py", "--base", truncated_model, "--tuned", truncated_model)
    assert result.returncode == 0, result.stderr
    report = json.loads((exported / "program_eval_result.json").read_text())
    assert report["base"] == report["tuned"] == 0
    assert report["details"][0]["tuned"]["error"] == "incomplete_model_output"


def test_independent_judgments_cannot_override_cut_predictions_or_another_test_file(exported, truncated_model):
    result = run_script(exported, "evaluate.py", "--base", truncated_model, "--tuned", truncated_model)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (exported / "split/test.jsonl").read_text().splitlines()]
    judgments = [{"id": r["meta"]["id"] + ":" + str(i), "base_correct": True, "tuned_correct": True, "reference": "independent fixture"}
                 for i, r in enumerate(rows)]
    path = exported / "judgments.json"
    path.write_text(json.dumps(judgments))
    result = run_script(exported, "evaluate.py", "--base", "unused", "--tuned", "unused", "--judgments", str(path))
    assert result.returncode == 0, result.stderr
    report = json.loads((exported / "eval.json").read_text())
    assert report["base"] == report["tuned"] == 0 and report["incomplete_predictions"]["base"] == len(rows)
    with (exported / "split/test.jsonl").open("a") as f:
        f.write("\n")  # even a different byte stream must be recollected before judgments apply
    result = run_script(exported, "evaluate.py", "--base", "unused", "--tuned", "unused", "--judgments", str(path))
    assert result.returncode != 0 and "do not belong" in result.stderr
