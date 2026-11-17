"""0.69.3: credentials in documentation are found in more forms and checked at import, on access, in sleep and in the archive.
All values here are invented."""
from vi_mcp import anonymize, archive, classify, memory as g, qualify, secret_sweep, tools

SAMPLE = ("Alle Wiederherstellungs Kennwörter / Frage lauten: Frage: Kennwort Antwort: Xy7!kp2Q\n"
            "Benutzer | Passwort\nanschluss-a | Qw3rT!9z\n"
            "Zugang: feste-ip/ABC123@provider.example 99887766")


def test_the_forms_found_on_production_are_replaced_and_ordinary_text_stays():
    out, n = anonymize.strip_credentials(SAMPLE)
    assert n == 3 and "Xy7!kp2Q" not in out and "Qw3rT!9z" not in out and "99887766" not in out
    for keep in ("Initial-Kennwort muss mindestens 12 Zeichen haben", "Kennwortrichtlinie: Länge 12, Ablauf 90 Tage",
                 "Name | Position | Telefonnummer\nHerr Muster | Geschäftsführer | 030/1234567", "Upgrade to PHP 8.1 | DC-X 5.5.0"):
        assert anonymize.strip_credentials(keep) == (keep, 0), keep


def _doc(content, **kw):
    return g.new_object("insight", "Wiki: access lines of the branch offices", ["knowledge_kind:company-docs", "source_kind:wiki", "page:9"],
                        {"source": "wiki", "content": content, **kw}, instance="old-importer", visibility="internal")


def test_sleep_sweeps_documentation_once_and_records_only_the_count(gd):
    d = _doc(SAMPLE)
    r = secret_sweep.run()
    o = g.load(d.id)
    assert r["objects_changed"] == 1 and r["values_replaced"] == 3 and r["open"] == 0
    assert "99887766" not in o.data["content"] and o.data["secrets_checked"] == anonymize.CREDENTIALS_VERSION
    assert any("credentials replaced: 3" in c["what"] for c in o.changes) and not any("99887766" in str(c) for c in o.changes)
    assert secret_sweep.run()["checked"] == 0  # checked once per version


def test_a_read_document_is_checked_before_it_is_shown(gd):
    d = _doc(SAMPLE)
    out = tools.call_tool("vi_object", {"id": d.id}, "a", False)
    assert "Qw3rT!9z" not in out and "<geheimnis>" in out


def test_archived_documentation_is_swept_too(gd):
    d = _doc(SAMPLE)
    archive.store_in_archive(g.load(d.id), "superseded by a page object")
    assert "Xy7!kp2Q" in archive.content(d.id)["data"]["content"]
    r = secret_sweep.run_archive()
    assert r["entries_changed"] == 1 and "Xy7!kp2Q" not in archive.content(d.id)["data"]["content"]


def test_pairs_queue_puts_accessed_objects_first_and_the_budget_stops_between_rounds(gd, monkeypatch):
    old = [g.new_object("insight", f"older fact {i}", ["topic:x"], {"source": "t", "content": "text " * 30}, instance="a") for i in range(3)]
    newer = g.new_object("insight", "newest fact", ["topic:x"], {"source": "t", "content": "text " * 30}, instance="a")
    assert classify.pairs_pending(1)[0] == newer.id  # newest first without a queue
    assert classify.queue_pairs([old[0]]) == 1
    assert classify.pairs_pending(2)[:1] == [old[0].id]  # accessed first
    qualify.on_access([old[1]], "t", "read")
    assert old[1].id in classify.pairs_pending(3)[:2]
    monkeypatch.setattr(classify, "available", lambda: True)
    calls = []
    monkeypatch.setattr(classify, "make_pairs", lambda o: calls.append(o.id) or {"state": "error", "reason": "test", "pairs": [], "rejected": {}})
    monkeypatch.setattr(classify, "store_pairs", lambda o, res, who: True)
    r = classify.run_pairs(10, "test", budget_s=0.0)
    assert r.get("stopped", "").startswith("time budget") and len(calls) == 0
    r = classify.run_pairs(2, "test")
    assert len(calls) == 2 and r["error"] == 2


def test_the_model_name_comes_from_the_url_when_the_config_names_none(monkeypatch):
    monkeypatch.setattr(classify, "_config", lambda: {"url": "https://llm.example.org/qwen27b/v1/chat/completions"})
    assert classify.model_name() == "qwen27b"
    monkeypatch.setattr(classify, "_config", lambda: {"url": "http://127.0.0.1:8080/v1/chat/completions", "model": "m1"})
    assert classify.model_name() == "m1"
