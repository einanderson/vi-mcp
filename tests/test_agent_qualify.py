"""0.69.3: the AI that reads an object qualifies it — training pairs, a probe and labels through vi_qualify, checked by fixed rules."""
import json

import pytest

from vi_mcp import classify, memory as g, tools
from vi_mcp.tools import Rejected

DOC = ("Confluence IT: backup of web1.example.org — The nightly backup of web1.example.org runs at 02:30 with restic and keeps "
       "14 daily snapshots in the bucket backup-web. A restore is started with restic restore latest --target /srv/restore.")


def _doc():
    return g.new_object("insight", DOC[:120], ["knowledge_kind:company-docs", "source_kind:wiki", "host:web1.example.org"],
                        {"source": "wiki", "content": DOC}, instance="importer", visibility="internal")


PAIRS = [
    {"question": "When does the nightly backup of web1.example.org run?", "answer": "At 02:30, with restic."},
    {"question": "How many daily snapshots does the backup of web1.example.org keep?", "answer": "14 daily snapshots in the bucket backup-web."},
    {"question": "How is a restore of the web1.example.org backup started?", "answer": "restic restore latest --target /srv/restore"},
    {"question": "Which tool backs up web1.example.org?", "answer": "restic"},
    {"question": "Where does the backup of web1.example.org store its snapshots?", "answer": "In the bucket backup-web."},
]


def test_vi_object_asks_once_and_vi_qualify_stores_checked_pairs_a_probe_and_labels(gd):
    d = _doc()
    out = tools.call_tool("vi_object", {"id": d.id}, "agent-a", False)
    assert "QUALIFY" in out and "vi_qualify" in out
    assert "QUALIFY" not in tools.call_tool("vi_object", {"id": d.id}, "agent-a", False)  # once per caller and day
    bad = {"question": "Which host runs the backup of web1.example.org?", "answer": "db9.example.org at 03:15"}  # invented name and number
    r = json.loads(tools.call_tool("vi_qualify", {"id": d.id, "pairs": PAIRS + [bad], "answers": "how", "memory": "procedure"},
                                   "agent-a", False, family="claude"))
    assert r["pairs"]["kept"] >= 3 and r["pairs"]["probe"] and sum(r["pairs"]["rejected"].values()) >= 1
    o = g.load(d.id)
    assert o.data["qa"]["version"]["model"] == "agent:claude" and o.data["qa"]["state"] == "ok"
    assert "answers:how" in o.hashtags and "memory:procedure" in o.hashtags
    assert d.id not in classify.pairs_pending(50)  # sleep does not redo agent pairs
    assert "QUALIFY" not in tools.call_tool("vi_object", {"id": d.id}, "agent-b", False)  # nothing missing any more


def test_wrong_labels_private_objects_and_no_token_are_refused(gd):
    d = _doc()
    with pytest.raises(Rejected, match="answers must be"):
        tools.call_tool("vi_qualify", {"id": d.id, "answers": "maybe", "memory": "fact"}, "agent-a", False)
    p = g.new_object("insight", "my private note on web1", ["topic:x"], {"source": "t", "content": DOC}, instance="agent-a", visibility="private")
    with pytest.raises(Rejected, match="cannot be qualified"):
        tools.call_tool("vi_qualify", {"id": p.id, "pairs": PAIRS}, "agent-a", False)
    from vi_mcp import config
    with pytest.raises(Rejected, match="token"):
        tools.call_tool("vi_qualify", {"id": d.id, "pairs": PAIRS}, config.OPEN, False)


def test_tagging_and_qualifying_are_one_call(gd):
    d = g.new_object("insight", "restic backup of a web host", ["knowledge_kind:company-docs", "source_kind:wiki"],
                     {"source": "wiki", "content": DOC}, instance="importer", visibility="internal")
    out = tools.call_tool("vi_object", {"id": d.id}, "agent-a", False)
    assert "tags:" in out and "(it has none)" in out
    r = json.loads(tools.call_tool("vi_qualify", {"id": d.id, "pairs": PAIRS, "answers": "how", "memory": "procedure",
                                                  "tags": ["host:web1.example.org", "topic:backup", "instance:evil", "no-colon"]},
                                   "agent-a", False, family="claude"))
    assert r["tags"]["added"] == ["host:web1.example.org", "topic:backup"] and set(r["tags"]["refused"]) == {"instance:evil", "no-colon"}
    o = g.load(d.id)
    assert {"host:web1.example.org", "topic:backup", "answers:how"} <= set(o.hashtags) and "instance:evil" not in o.hashtags
    assert r["pairs"]["kept"] >= 3


def test_vi_learn_stores_tags_pairs_and_labels_in_one_call(gd):
    out = tools.call_tool("vi_learn", {"text": DOC[:120], "content": DOC, "hashtags": ["host:web1.example.org", "topic:backup"],
                                       "source": "restic snapshots on web1.example.org, 17.11.", "pairs": PAIRS,
                                       "answers": "how", "memory": "procedure"}, "agent-a", False, family="claude")
    assert "qualified:" in out and "pairs" in out and "answers:how" in out
    oid = out.split()[1]
    o = g.load(oid)
    assert o.data["qa"]["version"]["model"] == "agent:claude" and {"answers:how", "memory:procedure", "topic:backup"} <= set(o.hashtags)
    with pytest.raises(Rejected, match="both, or neither"):
        tools.call_tool("vi_learn", {"text": "web2 backs up at 03:00", "hashtags": ["topic:backup"], "source": "crontab -l on web2",
                                     "answers": "how"}, "agent-a", False)


def test_vi_learn_without_pairs_asks_for_them_while_the_text_is_in_context(gd):
    out = tools.call_tool("vi_learn", {"text": DOC[:120], "content": DOC, "hashtags": ["topic:backup"],
                                       "source": "restic snapshots on web1.example.org, 17.11."}, "agent-a", False)
    assert "QUALIFY" in out and "vi_qualify(id=" in out
