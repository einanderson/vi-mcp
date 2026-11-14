"""0.62.0: fine-tuning an AI with what a VI learned — the VI knows how; vi-mcp writes the kit and records the run, it trains nothing."""
import json
import py_compile

import pytest

from vi_mcp import evaluation, finetune, tools
from vi_mcp import memory as g


@pytest.fixture()
def vi(gd, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    for i in range(12):
        g.new_object("erkenntnis", f"knowledge {i}: how to deploy service {i}", ["thema:deploy"],
                     {"quelle": "t", "inhalt": f"step one, step two for {i}", "klasse": {"fragen": [f"how to deploy {i}?", f"deploy {i} steps?"]}},
                     visibility="intern", instance="a")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "recall.json").write_text(json.dumps([{"target": "erkenntnis:none", "group": "g", "stratum": "s",
                                                                  "frage": "how is the database backed up?", "stichworte": "backup database"}]))
    return tmp_path


def test_plan_names_an_immature_vi_and_the_missing_base_model(vi):
    p = finetune.plan()
    assert p["zustand"] == "im_inkubator" and p["fehlt"] and p["grundmodell"] is None
    text = "\n".join(finetune.plan_lines(p))
    assert "still in the incubator" in text and "trial run" in text and "--force" in text
    out = tools.call_tool("vi_finetune", {}, "a", False)
    assert "still in the incubator" in out and "Steps:" in out
    assert "LoRA" in tools.call_tool("vi_finetune", {"action": "guide"}, "a", False)
    with pytest.raises(tools.Rejected):
        tools.call_tool("vi_finetune", {}, "offen", False)  # needs a token
    assert "call vi_finetune" in tools.RULES  # "I want to fine-tune" is enough for an AI on vi-mcp


def test_kit_writes_data_split_and_valid_scripts_without_running_anything(vi):
    with pytest.raises(ValueError, match="no base model"):
        finetune.kit(str(vi / "k0"), force=True)
    with pytest.raises(ValueError, match="still in the incubator"):
        finetune.kit(str(vi / "k1"), base_model="Qwen/Qwen3-4B")
    r = finetune.kit(str(vi / "kit"), base_model="Qwen/Qwen3-4B", force=True)
    d = vi / "kit"
    for f in ("dataset/qa.jsonl", "dataset/DATASHEET.md", "split/train.jsonl", "split/test.jsonl", "train.py", "evaluate.py", "convert.sh",
              "README-FINETUNE.md", "run.json", "eval_questions.json", "training_runtime.py", "requirements-train.txt"):
        assert (d / f).exists(), f
    train = {json.loads(x)["meta"]["id"] for x in (d / "split/train.jsonl").read_text().splitlines()}
    test = {json.loads(x)["meta"]["id"] for x in (d / "split/test.jsonl").read_text().splitlines()}
    assert train and test and not (train & test)  # split by object: no object on both sides
    assert r["split"]["train"] + r["split"]["test"] == 24 and r["eval_fragen"] == 1
    for f in ("train.py", "evaluate.py", "heldout_loss.py"):
        py_compile.compile(str(d / f), doraise=True)
        assert '"max_tokens"' not in (d / f).read_text() and "max_new_tokens" not in (d / f).read_text() and (d / f).stat().st_mode & 0o777 == 0o700
    assert (d / "run.json").stat().st_mode & 0o077 == 0 and json.loads((d / "run.json").read_text())["base_model"] == "Qwen/Qwen3-4B"


def test_documents_without_a_question_become_grounded_synthetic_pairs_never_echo_rows(vi, monkeypatch):
    """14.11.2026: 96 % of the first production run's SFT rows were "question = the document's beginning" — the model learned to echo.
    Now such knowledge goes to documents.jsonl; synth_qa.py keeps only grounded pairs, and `finetune split` puts them into the split."""
    import http.server
    import subprocess
    import sys
    import threading
    for i in range(6):
        g.new_object("erkenntnis", f"backup of db{i} runs nightly", ["thema:backup"],
                     {"quelle": f"https://wiki.example/db{i}", "inhalt": f"The backup of db{i}.example runs at 02:00 with borg to /srv/backup/db{i}."},
                     visibility="intern", instance="a")
    d = vi / "kit"
    finetune.kit(str(d), base_model="Qwen/Qwen3-4B", force=True)
    docs = [json.loads(x) for x in (d / "dataset/documents.jsonl").read_text().splitlines()]
    qa = [json.loads(x) for x in (d / "dataset/qa.jsonl").read_text().splitlines()]
    assert len(docs) == 6 and all(x["meta"]["question_source"] == "doc2query" for x in qa)
    assert not any(r["messages"][0]["content"].startswith("backup of db") for r in qa)  # never the text as its own question
    script = (d / "synth_qa.py").read_text()
    py_compile.compile(str(d / "synth_qa.py"), doraise=True)
    assert "max_tokens" not in script and '"enable_thinking": False' in script and (d / "synth_qa.py").stat().st_mode & 0o777 == 0o700
    calls = []

    class Fake(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            doc = body["messages"][1]["content"]
            n = doc.split("db", 1)[1][0]
            pairs = [{"question": f"When does the backup of db{n}.example run?", "answer": f"At 02:00 with borg to /srv/backup/db{n}."},
                     {"question": f"What does the document say about db{n}?", "answer": f"The backup of db{n}.example runs at 02:00."},
                     {"question": f"Who owns db{n}.example in the finance team?", "answer": "Alice Example from finance owns it since 2019."}]
            out = json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps({"pairs": pairs}) + "\n```"}, "finish_reason": "stop"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        run = lambda: subprocess.run([sys.executable, "synth_qa.py", "--url", url, "--model", "m", "--workers", "2"], cwd=d,
                                     capture_output=True, text=True, timeout=60)
        p = run()
        assert p.returncode == 0, p.stderr
        assert all("max_tokens" not in c and c["chat_template_kwargs"] == {"enable_thinking": False} for c in calls) and len(calls) == 6
        synth = [json.loads(x) for x in (d / "dataset/qa_synth.jsonl").read_text().splitlines()]
        assert len(synth) == 6 and all(r["meta"]["question_source"] == "synthetic" and r["meta"]["generator"] == "m" for r in synth)
        assert all(r["messages"][0]["content"].startswith("When does the backup") for r in synth)  # meta and ungrounded pairs dropped
        assert all(r["messages"][1]["content"].endswith(f"Source: https://wiki.example/{r['messages'][1]['content'].split('/srv/backup/')[1][:3]}") for r in synth)
        report = json.loads((d / "synth_report.json").read_text())
        assert report["pairs"] == 6 and report["rejected"] == {"refers to the document instead of its subject": 6, "not grounded in the document": 6}
        assert run().returncode == 0 and len(calls) == 6  # resumable: done documents are not asked again
    finally:
        srv.shutdown()
    split = finetune.resplit(str(d))
    assert split["train"] + split["test"] == 24 + 6
    assert sum(side.get("synthetic", 0) for side in split["qa_question_sources"].values()) == 6
    rj = json.loads((d / "run.json").read_text())
    assert rj["qa_synth_sha256"] and rj["split"] == split
    assert finetune.main(["split", str(d)]) == 0


def test_record_keeps_the_lineage_and_adopts_only_what_measured_better(vi):
    finetune.kit(str(vi / "kit"), base_model="Qwen/Qwen3-4B", force=True)
    d = vi / "kit"
    with pytest.raises(ValueError, match="no trained adapter"):
        finetune.record(str(d))
    (d / "out" / "sft").mkdir(parents=True)
    (d / "out" / "sft" / "adapter_model.safetensors").write_bytes(b"weights")
    o = finetune.record(str(d))
    assert o.mode == "test" and "not evaluated" in o.text and o.data["adapter_sha256"]["sft"]
    (d / "eval.json").write_text(json.dumps({"base": 0.31, "tuned": 0.52, "n": 25, "better": 14, "worse": 3}))
    o = finetune.record(str(d))
    assert o.mode == "test"  # old word-overlap reports never authorize adoption
    import hashlib
    ev = {"base": 0.31, "tuned": 0.52, "n": 25, "better": 14, "worse": 3, "metric": "verified_task_success", "lower_95": -0.1, "decision": True, "p_one_sided": 0.006,
          "evaluation_set_sha256": hashlib.sha256((d / "split/test.jsonl").read_bytes()).hexdigest()}
    (d / "eval.json").write_text(json.dumps(ev))
    o = finetune.record(str(d))
    assert o.mode == "aktiv" and "fine-tuned 0.52 vs base 0.31 on 25 held-out questions" in o.text
    assert o.data["dataset_manifest_sha256"] and o.data["incubator"]["zustand"] == "im_inkubator"
