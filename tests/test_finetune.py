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
    Knowledge without stored pairs goes to documents.jsonl; synth_qa.py keeps only grounded pairs (qa_pairs.py, the server's definition),
    holds one per document out as its probe, and `finetune split` puts pairs and probes of trained objects into the split. probe.py
    judges and unblinds; record names the probe."""
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
    assert len(docs) == 18 and all(x["meta"]["question_source"] == "doc2query" for x in qa)  # no stored pairs yet: every object
    assert not any(r["messages"][0]["content"].startswith("backup of db") for r in qa)  # never the text as its own question
    script = (d / "synth_qa.py").read_text()
    for f in ("synth_qa.py", "qa_pairs.py", "probe.py"):
        py_compile.compile(str(d / f), doraise=True)
    assert "max_tokens" not in script and '"enable_thinking": False' in script and (d / "synth_qa.py").stat().st_mode & 0o777 == 0o700
    assert (d / "qa_pairs.py").read_text() == (finetune.Path(finetune.__file__).with_name("qa_pairs.py")).read_text()  # one definition
    assert "max_tokens" not in (d / "probe.py").read_text()
    import re
    calls, support_calls = [], []

    class Fake(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            system, doc = body["messages"][0]["content"], body["messages"][1]["content"]
            if system.startswith("You check training data"):  # qa-v4 support check: everything supported here
                calls.pop()
                support_calls.append(body)
                n_pairs = len(re.findall(r"^\d+\. Q:", doc, re.M))
                content = json.dumps({"verdicts": [{"i": i, "supported": True, "reason": "stated"} for i in range(n_pairs)]})
            elif system.startswith("You grade answers"):
                content = json.dumps({"A": "correct", "B": "wrong", "reason": "A names the time"})
            elif "db" not in doc:
                content = json.dumps({"pairs": []})
            else:
                n = doc.split("db", 1)[1][0]
                more = [{"question": f"Which tool backs up db{n}.example at 02:00?", "answer": "borg."}] if "up to 7 questions" in system else []
                pairs = more + [{"question": f"At what time is db{n}.example backed up each night?", "answer": "At 02:00 with borg."},
                                {"answer": "a pair without a question is rejected, never a crash"},
                                {"question": f"When does the backup of db{n}.example run?", "answer": f"At 02:00 with borg to /srv/backup/db{n}."},
                                {"question": f"What does the document say about db{n}?", "answer": f"The backup of db{n}.example runs at 02:00."},
                                {"question": f"Who owns db{n}.example in the finance team?", "answer": "Alice Example from finance owns it since 2019."}]
                content = "```json\n" + json.dumps({"pairs": pairs}) + "\n```"
            out = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
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
        assert all("max_tokens" not in c and c["chat_template_kwargs"] == {"enable_thinking": False} for c in calls) and len(calls) == 18
        synth = [json.loads(x) for x in (d / "dataset/qa_synth.jsonl").read_text().splitlines()]
        probes = [json.loads(x) for x in (d / "dataset/probe_synth.jsonl").read_text().splitlines()]
        assert len(synth) == 6 and len(probes) == 6 and all(r["meta"]["question_source"] == "synthetic" and r["meta"]["generator"] == "m" for r in synth)
        good = ("At what time is db", "When does the backup")
        for r in synth:  # per document: one grounded pair trained, the other held out as its probe — never both on one side
            pr = next(x for x in probes if x["id"] == r["meta"]["id"])
            assert r["messages"][0]["content"].startswith(good) and pr["question"].startswith(good)
            assert r["messages"][0]["content"] != pr["question"] and pr["meta"]["question_source"] == "probe"
        assert all(r["messages"][1]["content"].split("Source: ")[1].startswith("https://wiki.example/db") for r in synth)
        report = json.loads((d / "synth_report.json").read_text())
        assert report["pairs"] == 6 and report["probes"] == 6 and len(support_calls) == 6  # every kept pair went through the check
        assert report["rejected"] == {"refers to the document instead of its subject": 6, "a number the document does not state": 6, "length": 6}
        assert run().returncode == 0 and len(calls) == 18  # resumable: done documents are not asked again
        # --more asks chosen documents again for further, varied pairs; a repeat of the probe is rejected, a repeat of a pair too
        ids = sorted(r["meta"]["id"] for r in synth)
        (d / "more.txt").write_text("\n".join(ids[:2]) + "\n")
        p = subprocess.run([sys.executable, "synth_qa.py", "--url", url, "--model", "m", "--pairs", "7", "--more", "more.txt"], cwd=d,
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        assert len(calls) == 20 and "up to 7 questions" in calls[-1]["messages"][0]["content"]
        synth = [json.loads(x) for x in (d / "dataset/qa_synth.jsonl").read_text().splitlines()]
        assert sorted(r["meta"]["id"] for r in synth if r["messages"][0]["content"].startswith("Which tool")) == ids[:2]
        report = json.loads((d / "synth_report.json").read_text())
        assert report["rejected"]["too close to a held-out evaluation question"] == 2 and report["rejected"]["duplicate question"] == 2
        assert report["pairs_asked"] == 7 and report["probes"] == 0  # a document keeps its one probe
        split = finetune.resplit(str(d))
        assert split["train"] + split["test"] == 24 + 8
        assert sum(side.get("synthetic", 0) for side in split["qa_question_sources"].values()) == 8
        assert split["probe"]["trained_objects"] + split["probe"]["objects_not_trained"] == 6
        trained = {json.loads(x)["meta"]["id"] for x in (d / "split/train.jsonl").read_text().splitlines()}
        kept = [json.loads(x) for x in (d / "split/probe.jsonl").read_text().splitlines()]
        assert len(kept) == split["probe"]["trained_objects"] and all(x["id"] in trained for x in kept)
        rj = json.loads((d / "run.json").read_text())
        assert rj["qa_synth_sha256"] and rj["probe_synth_sha256"] and rj["split"] == split
        assert finetune.main(["split", str(d)]) == 0
        # probe.py judge + summarize on recorded answers (answering needs the model; the files are what `probe.py answer` writes)
        assert kept, "the seeded split keeps at least one db object on the train side"
        (d / "probe").mkdir()
        with open(d / "probe/answers_blind.jsonl", "w") as f, open(d / "probe/ab_key.jsonl", "w") as k:
            for x in kept:
                f.write(json.dumps({"id": x["id"], "group": x["meta"]["group"], "question": x["question"], "reference": x["answer"],
                                    "A": "At 02:00.", "A_finish": "stop", "B": "I do not know.", "B_finish": "stop"}) + "\n")
                k.write(json.dumps({"id": x["id"], "A": "tuned", "B": "base"}) + "\n")
        import hashlib as _h
        p = subprocess.run([sys.executable, "probe.py", "summarize"], cwd=d, capture_output=True, text=True, timeout=60)
        assert p.returncode != 0  # nothing judged yet, and no answer run recorded: refused, never a decision
        (d / "probe/answer_run.json").write_text(json.dumps({"probe_sha256": _h.sha256((d / "split/probe.jsonl").read_bytes()).hexdigest()}))
        p = subprocess.run([sys.executable, "probe.py", "judge", "--url", url, "--model", "m"], cwd=d, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        assert all("max_tokens" not in c for c in calls)
        p = subprocess.run([sys.executable, "probe.py", "summarize"], cwd=d, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        ev = json.loads((d / "probe/probe_eval.json").read_text())
        assert ev["grades"]["tuned"]["correct"] == len(kept) and ev["grades"]["base"]["wrong"] == len(kept)
        assert ev["judges"] == ["m"] and "self-preference" in ev["warning"]  # the judge wrote the probes: named, not hidden
    finally:
        srv.shutdown()
    (d / "out" / "sft").mkdir(parents=True)
    (d / "out" / "sft" / "adapter_model.safetensors").write_bytes(b"weights")
    o = finetune.record(str(d))
    assert "knowledge probe" in o.text and o.data["wissensprobe"]["n"] == len(kept) and o.mode == "test"  # the probe is no adoption gate


def test_server_stores_training_pairs_with_a_probe_and_the_export_uses_them(vi, monkeypatch):
    """0.65.0 — the user: "die Daten müssen von Anfang an in der richtigen Art und Weise gespeichert sein". The server writes grounded
    pairs and one probe question onto each knowledge object (sleep step 8i / vi-mcp classify --pairs); the export trains the pairs,
    holds the probe out, and treats pairs of changed knowledge as stale."""
    from vi_mcp import classify, dataset, qa_pairs, reflection
    asked = []

    def fake(system, user):
        if system == qa_pairs.SUPPORT_SYSTEM:
            return json.dumps({"verdicts": [{"i": i, "supported": True, "reason": "stated"} for i in range(user.count(". Q:"))]})
        asked.append(system)
        if "fail" in user:
            raise ConnectionError("model down")
        i = user.split("knowledge ", 1)[1].split(":", 1)[0]
        return json.dumps({"pairs": [{"question": f"How do I deploy service {i}?", "answer": f"step one, step two for {i}"},
                                     {"question": f"What are the steps to deploy service {i}?", "answer": "step one, step two"},
                                     {"question": "What does the document say?", "answer": "step one"}]})
    monkeypatch.setattr(reflection, "ask_fn", fake)
    secret = g.new_object("erkenntnis", "knowledge 99: private", ["thema:deploy"], {"inhalt": "step one"}, visibility="privat", instance="a")
    failing = g.new_object("erkenntnis", "knowledge fail: model down", ["thema:deploy"], {"inhalt": "fail"}, visibility="intern", instance="a")
    assert secret.id not in classify.pairs_pending(100) and failing.id in classify.pairs_pending(100)
    r = classify.run_pairs(100, "test", workers=3)
    assert r["ok"] == 12 and r["error"] == 1 and r["pairs"] == 12 and r["probes"] == 12
    assert r["rejected"] == {"refers to the document instead of its subject": 12} and r["reasons"] == {"ConnectionError: model down": 1}
    assert asked[0] == qa_pairs.system_for() and "max_tokens" not in asked[0]
    assert classify.pairs_pending(100) == []  # a fresh failure waits a day instead of blocking every sleep
    o = g.load(failing.id)
    o.data["qa"]["time"] = "2020-01-01T00:00:00.000+00:00"
    g.save(o)
    assert classify.pairs_pending(100) == [failing.id]
    rep = classify.pairs_report()
    assert rep["ok"] == 12 and rep["error"] == 1 and rep["pairs"] == 12 and rep["probes"] == 12
    # changed knowledge: its pairs are stale — not exported, renewed by --stale
    ids = [x[0] for x in g.db().execute("SELECT id FROM objekt WHERE art = 'erkenntnis' AND json_extract(daten, '$.qa.state') = 'ok'")]
    changed = g.load(ids[0])
    changed.data["inhalt"] = "step one, step two, step three"
    g.save(changed)
    assert changed.id in classify.pairs_pending(100, stale=True) and changed.id not in classify.pairs_pending(100)
    data = dataset.build(agentic=False)
    synthetic = [x for x in data["qa"] if x["meta"]["question_source"] == "synthetic"]
    assert len(synthetic) == 11 and len(data["probe"]) == 11 and all(x["meta"]["generator"] for x in synthetic)
    trained_q = {x["messages"][0]["content"] for x in data["qa"]}
    assert not any(p["question"] in trained_q for p in data["probe"])  # a probe is never trained
    assert any(d["meta"]["id"] == changed.id for d in data["documents"])
    assert data["pair_states"] == {"stored pairs used": 11, "stored training pairs are stale (the knowledge changed since): exported to documents.jsonl": 1,
                                   "stored training pairs in state error: exported to documents.jsonl": 1}
    m = dataset.export(vi / "ds")
    assert m["counts"]["probe"] == 11 and (vi / "ds" / "probe.jsonl").stat().st_mode & 0o777 == 0o600
    assert "probe.jsonl | 11" in (vi / "ds" / "DATASHEET.md").read_text()
    import types
    from vi_mcp import config
    monkeypatch.setattr(config, "load", lambda: types.SimpleNamespace(llm=None, db=vi / "vi.db"))  # never the person's real database
    assert classify.main(["--pairs", "--dry-run"]) == 0
    g.open_db(vi / "vi.db")  # main closes the database it opened


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


def test_grounding_counts_facts_not_function_words():
    """qa-v2 (15.11.2026): with qa-v1, real Qwen3.6-35B answers to six short German findings kept 1 of 25 pairs — "die, wurde, auf"
    counted as ungrounded, and "host"/"dns01" hidden in host:dns01.example.org were not found. Invented facts must still fail."""
    from vi_mcp import qa_pairs
    doc = ('Bekannter Fehler: dns01: erstmals www.bad-shop.example (6×) sinkholed — neue Schad-Domain in Serie — Filter: '
           'host:dns01.example.org bedrohung:"generic:www.bad-shop.example(789736897)"')
    q = "Welche Domain wurde auf dem Host dns01 sinkholed?"
    assert qa_pairs.check({"question": q, "answer": "Die Domain www.bad-shop.example wurde auf dem Host dns01 erstmals sinkholed."}, doc) is None
    assert qa_pairs.check({"question": q, "answer": "Die Domain wurde auf dns02.example.org am 2025-03-01 von 10.1.2.3 gemeldet."}, doc) in (
        "a number the document does not state", "a name the document does not state")
    assert qa_pairs.check({"question": q, "answer": "Ja, das ist so."}, doc) == "not grounded in the document"  # no fact at all
    assert qa_pairs.SCHEMA_VERSION == "qa-v4" and qa_pairs.prompt_sha256() != qa_pairs.prompt_sha256(3)
    # qa-v3 (Codex's review): a question must name the subject; the content hash sees whitespace (code, tables)
    assert qa_pairs.check({"question": "What is the port number used?", "answer": "www.bad-shop.example"}, doc) == "names no subject of the document"
    assert qa_pairs.content_sha256("def f():\n    return 1") != qa_pairs.content_sha256("def f():\n  return 1")
    assert qa_pairs.check({"question": "Which port does the SSH service on web1.example use?", "answer": "The SSH service uses port 22."},
                          "SSH on web1.example: the SSH service uses port 80.") == "a number the document does not state"


def test_codex_review_findings_on_stored_pairs_are_fixed(vi, monkeypatch):
    """Codex's review of 0.65.0 (15.11.2026) reproduced: a credential coming back through the raw source line, a doc2query question
    equal to the probe being trained, pairs of an older schema exported, a lost update when the object changed during the model call,
    and changed knowledge never renewed by sleep."""
    from vi_mcp import classify, dataset, qa_pairs, reflection
    o = g.new_object("erkenntnis", "nginx reload on web1.example", ["thema:nginx"], {"quelle": "https://wiki.example/nginx",
                     "inhalt": "Run systemctl reload nginx on web1.example after changing /etc/nginx/sites-enabled.",
                     "klasse": {"fragen": ["How do I reload nginx on web1.example?"]}}, visibility="intern", instance="a")
    pairs = [{"question": "How do I reload nginx on web1.example?", "answer": "Run systemctl reload nginx on web1.example."},
             {"question": "Which directory holds the enabled sites on web1.example?", "answer": "/etc/nginx/sites-enabled"},
             {"question": "Which command reloads nginx on web1.example after a change?", "answer": "systemctl reload nginx"}]
    monkeypatch.setattr(reflection, "ask_fn", lambda system, user: json.dumps(
        {"verdicts": [{"i": i, "supported": True, "reason": "stated"} for i in range(user.count(". Q:"))]}
        if system == qa_pairs.SUPPORT_SYSTEM else {"pairs": pairs}))
    # the probe is chosen by id; make the doc2query question equal to whichever pair became the probe
    res = classify.make_pairs(o)
    assert res["state"] == "ok" and res["probe"]
    k = dict(o.data["klasse"])
    k["fragen"] = [res["probe"]["question"]]
    raw = {**o.data, "klasse": k, "quelle": "https://wiki.example/nginx?password=hunter2secret"}
    g.db().execute("UPDATE objekt SET daten = ? WHERE id = ?", (json.dumps(raw), o.id))  # as if stored before scrubbing existed
    g.db().commit()
    # lost update: the object changes while the model is asked — the result made from the old text is not stored
    cur = g.load(o.id)
    stale_res = classify.make_pairs(cur)
    cur.data["inhalt"] = cur.data["inhalt"] + " Check with nginx -t first."
    g.save(cur)
    assert classify.store_pairs(cur, stale_res, "test") is False and "qa" not in g.load(o.id).data
    assert classify.run_pairs(5, "test")["ok"] >= 1
    data = dataset.build(agentic=False)
    rows = [x for x in data["qa"] if x["meta"]["id"] == o.id]
    probe = next(p for p in data["probe"] if p["id"] == o.id)
    assert rows and not any(qa_pairs.jaccard(x["messages"][0]["content"], probe["question"]) >= 0.5 for x in rows)  # probe untrained
    assert data["skipped"].get("doc2query question too close to the object's probe (the probe stays untrained)") == 1
    assert not any("hunter2secret" in x["messages"][1]["content"] for x in rows) and "hunter2secret" not in probe["source"]
    assert not any("hunter2secret" in d["source"] for d in data["documents"])
    # an older schema is regenerated, never exported
    cur = g.load(o.id)
    cur.data["qa"]["version"]["schema"] = "qa-v1"
    g.save(cur)
    assert dataset.stored_pairs(cur, classify.document_of(cur))[0] is None
    # sleep finds changed knowledge by a random sample (the content hash is not in SQL)
    cur.data["qa"]["version"]["schema"] = qa_pairs.SCHEMA_VERSION
    cur.data["inhalt"] = cur.data["inhalt"] + " Rollback: systemctl restart nginx."
    g.save(cur)
    assert o.id not in classify.pairs_pending(10) and o.id in classify.pairs_pending(10, sample=50)


def test_qa_v4_english_pairs_are_grounded_by_names_and_a_model_check_that_may_only_reject(vi, monkeypatch):
    """qa-v4 (user, 15.11.2026: "Die Standardsprache sollte Englisch sein"): an English answer to a German page is grounded by its names
    (hosts, paths, numbers, quoted terms); the language model then checks each pair and may only reject; a check without a verdict for
    every pair stores nothing (retried later) — never a silent pass."""
    from vi_mcp import classify, qa_pairs, reflection
    doc = "Postfix auf mail01.example.org: inet_interfaces = loopback-only, relayhost = relay.example.org. Neustart mit systemctl restart postfix."
    q = "How is Postfix on mail01.example.org configured?"
    assert qa_pairs.check({"question": q, "answer": "Postfix listens on loopback-only and relays via relay.example.org."}, doc) is None
    assert qa_pairs.check({"question": q, "answer": "Postfix relays via smtp.other.example."}, doc) == "a name the document does not state"
    assert qa_pairs.check({"question": q, "answer": "It is configured well by the mail team."}, doc) == "not grounded in the document"
    assert "in English" in qa_pairs.SYSTEM and qa_pairs.SCHEMA_VERSION == "qa-v4"
    o = g.new_object("erkenntnis", "Postfix auf mail01.example.org", ["thema:mail"], {"inhalt": doc}, visibility="intern", instance="a")
    pairs = [{"question": q, "answer": "Postfix listens on loopback-only and relays via relay.example.org."},
             {"question": "How do I restart Postfix on mail01.example.org?", "answer": "Run systemctl restart postfix."},
             {"question": "Does Postfix on mail01.example.org accept mail from the internet?", "answer": "Yes, inet_interfaces allows it."}]
    verdicts = {"value": [True, True, False]}

    def fake(system, user):
        if system == qa_pairs.SUPPORT_SYSTEM:
            return json.dumps({"verdicts": [{"i": i, "supported": v, "reason": "x"} for i, v in enumerate(verdicts["value"])]})
        return json.dumps({"pairs": pairs})
    monkeypatch.setattr(reflection, "ask_fn", fake)
    r = classify.make_pairs(g.load(o.id))
    assert r["state"] == "ok" and len(r["pairs"]) + (1 if r["probe"] else 0) == 2  # the reversed fact was vetoed by the model
    assert r["rejected"]["not supported by the document (model check)"] == 1
    verdicts["value"] = [True]  # a verdict missing for two pairs: nothing is stored as checked
    r = classify.make_pairs(g.load(o.id))
    assert r["state"] == "error" and "incomplete" in r["reason"] and r["pairs"] == []
