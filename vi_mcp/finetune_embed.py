"""Embedding fine-tuning kit (0.68.0): train the embedding model on this VI's own (situation, hint) pairs, so the router's similarity
features come from vectors that know the domain — the language model is not touched, the objects stay in the database.

The user (16.11.2026): instead of a LoRA layer in a chat model, improve the scikit-learn layer with what a model learns. Research (VI
insight:8f363e8e… and the embedding research of 16.11.2026) shapes the kit:
  - real labels are far too few for a gain on their own (production copy 16.11.2026: 206 helped + 96 adopted pairs over 128 hints; GPL
    2112.07577 and Promptagator 2209.11755 needed thousands to ~250k pairs) → they are the EVALUATION set; training volume comes from
    synthetic pairs: a language model writes situations (tool + arguments) in which a hint helps, kept only if the hint ranks first for
    its own situation among all hints (round-trip filter, Promptagator: +2.5 nDCG);
  - "shown but not used" is NOT a negative (RocketQA 2010.08191: ~70 % of such passages were relevant) → negatives only from feedback
    "wrong" or the hint judge's "misleading", otherwise in-batch negatives (MultipleNegativesRankingLoss);
  - a model tuned on a narrow domain forgets (LM-Cocktail 2311.13534: other tasks 49.81 → 47.49) → merge tuned and base weights (alpha
    0.5) and keep a general retrieval check;
  - privacy: an experience belongs to its instance (dataset.py never exports it); the kit takes only its trigger tags without hosts
    — the situation as the router already represents it — and only non-private hints;
  - evaluation by time: train on the older pairs, test Recall@4 of the used hint on the newest window; then the router's paired log loss
    with only the similarity feature swapped.
The kit is written, never run by vi-mcp (a person runs it on a GPU host, like the LoRA kit; skill finetune).
`vi-mcp finetune embed-kit DIR [--days N]`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import memory as g

TEST_SHARE = 0.2

README = """# Embedding fine-tuning kit (vi-mcp)

Trains the embedding model (default `BAAI/bge-m3`) on this VI's own (situation → hint) pairs. Nothing here touches a chat model.

Files:
- `pairs_real.jsonl` — {situation (trigger tags, no hosts, no command), hint, label: helped|adopted|judge_used, ts}: real positives. **Evaluation only** (too few to train).
- `negatives.jsonl` — {situation, hint, label: wrong|misleading}: confirmed negatives only (never "shown but not used").
- `hints.jsonl` — every shown hint {id, text}: the corpus for synthetic pairs.
- `split.json` — the time cut: pairs before it may be used for training checks, pairs after it are the test window.

Steps (each: run, then check the "OK if"):
1. Synthetic pairs: `python gen_synth.py --url http://LLM/v1 --model NAME --per-hint 4` — writes `pairs_synth.jsonl`; a pair is kept
   only if its hint ranks first for its own situation among all hints with the base model (round-trip filter).
   OK if: `synth_report.json` shows kept pairs for most hints.
2. Train: `python train_embed.py --base BAAI/bge-m3 --epochs 1` — MultipleNegativesRankingLoss on `pairs_synth.jsonl` (+ confirmed
   negatives as hard negatives), then merges tuned and base weights with alpha 0.5 into `out/merged`.
   OK if: `out/merged` exists and `train_report.json` shows the loss falling.
3. Evaluate: `python eval_embed.py --model out/merged` — Recall@4 of the real used hint on the test window, base vs merged, paired;
   plus a general retrieval check (a fixed set of the VI's own questions). Keep the merged model only if Recall@4 rises and the general
   check does not fall.
4. Only then serve it (llama.cpp GGUF of the merged model as the embed service) and let the router's offline gate decide with the
   similarity features recomputed — with the person's approval.

No token caps in any language-model call (thinking off instead). Write every result down with `vi_learn`.
"""


def _text(o: g.MemoryObject | None) -> str:
    return "" if o is None else (o.text + "\n" + str(o.data.get("content") or "")[:600]).strip()


def _situation(e: g.MemoryObject) -> str:
    """The situation as the router sees it: its trigger tags (tool, skill, topic, error), never the command or its arguments — an
    experience belongs to its instance and is never exported (dataset.py); the tags are the aggregate the router already uses."""
    from .learning import TRIGGER_KINDS
    trig = (e.data.get("before") or {}).get("triggers") or [t for t in e.hashtags if t.startswith(TRIGGER_KINDS)]
    return " ".join(sorted(t for t in trig if t.startswith(TRIGGER_KINDS) and not t.startswith("host:")))


def kit(directory: str, days: float = 60) -> dict:
    from datetime import datetime, timedelta, timezone
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with g.lock:
        rows = g.db().execute("SELECT id FROM object WHERE kind = 'experience' AND ts >= ? AND data LIKE ? ORDER BY ts",
                              (since, '%"shown": ["%')).fetchall()
    real, neg, hints = [], [], {}
    for (eid,) in rows:
        e = g.load(eid)
        if e is None:
            continue
        shown = (e.data.get("before") or {}).get("shown") or []
        fb = e.data.get("feedback") or {}
        judge = (e.data.get("judge") or {}).get("hints") or {}
        sit = _situation(e)
        for h in shown:
            ho = g.load(h)
            if ho is None or ho.visibility == "private":
                continue
            hints[h] = _text(ho)
            lab = ("helped" if fb.get(h, 0) > 0 else "adopted" if h in (e.data.get("used") or []) else
                   "judge_used" if judge.get(h) == "used" and "ok:yes" in e.hashtags else None)
            if lab:
                real.append({"situation": sit, "hint_id": h, "hint": hints[h], "label": lab, "ts": e.ts})
            elif fb.get(h, 0) < 0 or judge.get(h) == "misleading":
                neg.append({"situation": sit, "hint_id": h, "hint": hints[h], "label": "wrong" if fb.get(h, 0) < 0 else "misleading", "ts": e.ts})
    cut = real[int(len(real) * (1 - TEST_SHARE))]["ts"] if real else None
    for name, data in (("pairs_real.jsonl", real), ("negatives.jsonl", neg),
                       ("hints.jsonl", [{"id": k, "text": v} for k, v in hints.items()])):
        (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
    (d / "split.json").write_text(json.dumps({"test_from": cut, "test_share": TEST_SHARE}, indent=1))
    (d / "README.md").write_text(README)
    for name, body in SCRIPTS.items():
        (d / name).write_text(body)
    return {"directory": str(d), "real_pairs": len(real), "negatives": len(neg), "hints": len(hints), "test_from": cut,
            "by_label": {k: sum(1 for r in real if r["label"] == k) for k in ("helped", "adopted", "judge_used")}}


SCRIPTS = {
    "gen_synth.py": r'''"""Synthetic (situation, hint) pairs: the language model writes situations (tool + arguments) in which a hint helps; a pair is kept
only if the hint ranks first for its situation among all hints with the base embedding model (round-trip filter). No token cap.
usage: python gen_synth.py --url URL --model NAME [--per-hint 4] [--base BAAI/bge-m3]"""
import argparse, json, urllib.request
a = argparse.ArgumentParser(); a.add_argument("--url", required=True); a.add_argument("--model", required=True)
a.add_argument("--per-hint", type=int, default=4); a.add_argument("--base", default="BAAI/bge-m3"); o = a.parse_args()
hints = [json.loads(l) for l in open("hints.jsonl")]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ask(text):
    body = {"model": o.model, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content":
            f"Write {o.per_hint} different situations in which an AI agent about to act would be helped by this note. Each situation is one "
            "line of tags as the memory describes an action: tool:<name> plus topic:<word> or error:<key> tags (e.g. tool:Bash topic:ssh "
            "error:timeout). JSON only: "
            "{\"situations\": [...]}\nNote:\n" + text}]}
    r = json.load(op.open(urllib.request.Request(o.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    t = r["choices"][0]["message"]["content"]; return json.loads(t[t.index("{"): t.rindex("}") + 1])["situations"]
from sentence_transformers import SentenceTransformer
import numpy as np
m = SentenceTransformer(o.base); H = m.encode([h["text"] for h in hints], normalize_embeddings=True)
kept, tried = [], 0
for i, h in enumerate(hints):
    try: sits = ask(h["text"])
    except Exception as e: print("failed", h["id"], e); continue
    for s in sits:
        tried += 1
        if int(np.argmax(H @ m.encode([s], normalize_embeddings=True)[0])) == i:
            kept.append({"situation": s, "hint_id": h["id"], "hint": h["text"]})
open("pairs_synth.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
json.dump({"hints": len(hints), "tried": tried, "kept": len(kept)}, open("synth_report.json", "w"), indent=1)
print(len(kept), "of", tried, "kept")
''',
    "train_embed.py": r'''"""MultipleNegativesRankingLoss on the synthetic pairs (+ confirmed negatives as hard negatives), then merge tuned and base weights
(LM-Cocktail 2311.13534) against forgetting. usage: python train_embed.py [--base BAAI/bge-m3] [--epochs 1] [--alpha 0.5]"""
import argparse, json
a = argparse.ArgumentParser(); a.add_argument("--base", default="BAAI/bge-m3"); a.add_argument("--epochs", type=float, default=1)
a.add_argument("--alpha", type=float, default=0.5); o = a.parse_args()
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
pairs = [json.loads(l) for l in open("pairs_synth.jsonl")]
neg = {}
for l in open("negatives.jsonl"):
    r = json.loads(l); neg.setdefault(r["situation"], r["hint"])
ex = [InputExample(texts=[p["situation"], p["hint"]] + ([neg[p["situation"]]] if p["situation"] in neg else [])) for p in pairs]
m = SentenceTransformer(o.base)
dl = DataLoader(ex, shuffle=True, batch_size=16)
m.fit(train_objectives=[(dl, losses.MultipleNegativesRankingLoss(m))], epochs=int(max(1, o.epochs)), show_progress_bar=True)
m.save("out/tuned")
base = SentenceTransformer(o.base); sd_b = base.state_dict(); sd_t = m.state_dict()
merged = {k: (o.alpha * sd_t[k] + (1 - o.alpha) * sd_b[k]) if sd_t[k].dtype.is_floating_point else sd_t[k] for k in sd_t}
m.load_state_dict(merged); m.save("out/merged")
json.dump({"pairs": len(ex), "with_hard_negative": sum(len(e.texts) == 3 for e in ex), "alpha": o.alpha}, open("train_report.json", "w"), indent=1)
''',
    "eval_embed.py": r'''"""Recall@4 of the real used hint on the test window: base vs tuned model, paired per pair. usage: python eval_embed.py --model out/merged [--base BAAI/bge-m3]"""
import argparse, json
import numpy as np
a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("--base", default="BAAI/bge-m3"); o = a.parse_args()
from sentence_transformers import SentenceTransformer
cut = json.load(open("split.json"))["test_from"]
test = [json.loads(l) for l in open("pairs_real.jsonl") if cut and json.loads(l)["ts"] >= cut]
hints = [json.loads(l) for l in open("hints.jsonl")]; ids = [h["id"] for h in hints]
def recall(name):
    m = SentenceTransformer(name); H = m.encode([h["text"] for h in hints], normalize_embeddings=True)
    out = []
    for p in test:
        s = H @ m.encode([p["situation"]], normalize_embeddings=True)[0]
        out.append(ids.index(p["hint_id"]) in np.argsort(-s)[:4])
    return np.array(out)
b, t = recall(o.base), recall(o.model)
print(json.dumps({"test_pairs": len(test), "recall4_base": float(b.mean()) if len(b) else None, "recall4_tuned": float(t.mean()) if len(t) else None,
                  "only_tuned": int((t & ~b).sum()), "only_base": int((b & ~t).sum())}, indent=1))
''',
}
