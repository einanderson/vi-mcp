"""0.68.0: the embedding fine-tuning kit — real positives (helped, adopted, judge used after success) are the evaluation set, negatives
only when confirmed (feedback wrong, judge misleading), never "shown but not used"; scripts written, never run."""
import json

from vi_mcp import finetune_embed as fe
from vi_mcp import memory as g


def test_kit_exports_confirmed_labels_only(gd, tmp_path):
    a = g.new_object("insight", "ssh needs the key file", ["topic:ssh"], {"source": "t"})
    b = g.new_object("insight", "dns is slow on monday", ["topic:dns"], {"source": "t"})
    c = g.new_object("insight", "unused but right", ["topic:x"], {"source": "t"})
    g.new_object("experience", "Bash(...) → ok", ["tool:Bash", "ok:yes"],
                 {"args": {"command": "ssh web1 uptime"}, "before": {"shown": [a.id, b.id, c.id], "triggers": ["tool:Bash", "topic:ssh", "host:web1"]},
                  "feedback": {a.id: 1, b.id: -1}})
    g.new_object("experience", "Bash(...) → ok", ["tool:Bash", "ok:yes"],
                 {"args": {"command": "dig web1"}, "before": {"shown": [b.id], "triggers": ["tool:Bash", "topic:dns"]}, "judge": {"hints": {b.id: "used"}}})
    r = fe.kit(str(tmp_path / "k"))
    assert r["real_pairs"] == 2 and r["negatives"] == 1 and r["hints"] == 3
    real = [json.loads(l) for l in open(tmp_path / "k/pairs_real.jsonl")]
    assert {x["label"] for x in real} == {"helped", "judge_used"}
    assert {x["situation"] for x in real} == {"tool:Bash topic:ssh", "tool:Bash topic:dns"}  # tags only: no command, no host
    assert "web1" not in (tmp_path / "k/pairs_real.jsonl").read_text()
    neg = [json.loads(l) for l in open(tmp_path / "k/negatives.jsonl")]
    assert [x["hint_id"] for x in neg] == [b.id]  # c was shown and not used: never a negative
    for f in ("README.md", "gen_synth.py", "train_embed.py", "eval_embed.py", "split.json", "hints.jsonl"):
        assert (tmp_path / "k" / f).exists()
    for f in ("gen_synth.py", "train_embed.py", "eval_embed.py"):
        compile((tmp_path / "k" / f).read_text(), f, "exec")
    assert "max_tokens" not in (tmp_path / "k/gen_synth.py").read_text()
