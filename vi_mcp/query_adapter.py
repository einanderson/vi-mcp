"""Query adapter (0.68.0) — learned embedding weights by scikit-learn instead of fine-tuning the embedding model.

Why (the user, 16.11.2026: "ja dann machen wir das doch die embed gewichte per scikit learn"): a linear transform of the QUERY vector
alone leaves every stored vector and the embedding service unchanged — no re-embedding, no migration, nothing to convert for
llama.cpp. Query-only linear adapters reached up to +70 % NDCG from ~1,500 labelled pairs and were competitive with fine-tuning
(Chroma technical report "Embedding Adapters", 2024, https://research.trychroma.com/embedding-adapters).

The model: a per-dimension weighting of the similarity, fitted by logistic regression on the features
    [cos(q, h), D·q₁h₁, …, D·q_D h_D]        (q, h normalised; D = dimension, so a per-dimension weight is on the scale of the cosine)
score(q, h) = w_cos·cos + Σ wᵢ·D·qᵢhᵢ = ĥ · (q̂ ⊙ (w_cos + D·w)) — a scaling of the query vector, so the search keeps its one matrix product.
L2 pulls every wᵢ to 0, which is the plain cosine again: with little data the adapter IS the cosine.

Labels (only what exists, each named by its source):
  - `qa`     the grounded questions stored on knowledge objects (classify.run_pairs, data.qa.pairs) → their object;
  - `judged` the language model's relevance grades of real searches (relevance.py, UMBRELA): grade ≥ 2 positive, grade 0 a confirmed
             negative for that query;
  - `probe`  the one held-out question per object (data.qa.probe) — never trained, test only (it is held out for fine-tuning too);
             only the probes of HELD-OUT objects count (split by object, see split());
  - `sq`     short search-style queries the classify model wrote per object (search_questions.py, round-trip filtered).
The judged real searches are reported apart (judged_diff); an adapter that is better overall but worse on them stays off.
Negatives besides grade 0: objects at cosine ranks NEG_FROM…NEG_TO of the query, never the very top (an unjudged top hit is often
relevant — RocketQA 2010.08191: ~70 % of such "negatives" were relevant).

Gate — the same two steps as feature_writer and hint_prior: the regularisation is chosen inside the training part (fit on the older
part, best mean reciprocal rank on the newer part), then ONE test on the held-out queries (newest TEST_SHARE of the judged searches
plus every probe): per query the reciprocal rank of its best positive among the top CANDIDATES by cosine, adapter vs cosine, paired;
the adapter is ON only if it is better by more than 2 SE. State in table `query_adapter` (off · on), re-checked every sleep; the
weights belong to one embedding model and dimension — another model switches it off.
Migration: none — the table starts empty, the state starts off, and off is the plain cosine.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from . import memory as g

VERSION = "query-adapter-v1"
TEST_SHARE = 0.2
CANDIDATES = 50
NEG_FROM, NEG_TO, NEG_PER_QUERY = 5, 50, 8
MIN_TRAIN_QUERIES = 200
MIN_TEST_QUERIES = 30
MIN_JUDGED = 5
#: at most this many generated training queries (qa, sq) per gate, chosen by a stable hash; judged searches always count. Bounds the
#: table query_vector (4 KB per query at 1024 dimensions): every object with 3 search questions would be ~300 MB on a 25k-object memory
MAX_TRAIN_QUERIES = 3000  # with this many judged real searches in the test, the adapter must not make them worse
C_GRID = (0.01, 0.1, 1.0)
EMBED_BATCH = 32
SEED = 7

_cache: dict = {"at": 0.0, "db": None, "row": None, "model": ("", 0)}


def _table() -> None:
    with g.lock:
        g.db().execute("CREATE TABLE IF NOT EXISTS query_adapter (id INTEGER PRIMARY KEY, state TEXT NOT NULL, ts TEXT NOT NULL, model TEXT, "
                       "dim INTEGER, weights TEXT, measurement TEXT)")
        g.db().execute("CREATE TABLE IF NOT EXISTS query_vector (sha TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, raw BLOB NOT NULL)")
        g.db().commit()


def _row() -> tuple | None:
    if _cache["db"] != id(g.db()) or time.monotonic() - _cache["at"] > 300:
        _table()
        with g.lock:
            r = g.db().execute("SELECT state, ts, model, dim, weights, measurement FROM query_adapter WHERE id = 1").fetchone()
        _cache.update(at=time.monotonic(), db=id(g.db()), row=tuple(r) if r else None, model=_model() if r and r[0] == "on" else ("", 0))
    return _cache["row"]


def state() -> str:
    r = _row()
    return r[0] if r else "off"


def _model() -> tuple[str, int]:
    """(model, dimension) of the stored vectors: the most common pair."""
    from . import vector
    vector._table()
    with g.lock:
        r = g.db().execute("SELECT model, dim, COUNT(*) AS n FROM vector GROUP BY model, dim ORDER BY n DESC LIMIT 1").fetchone()
    return (r[0], int(r[1])) if r else ("", 0)


def scale(dim: int) -> np.ndarray | None:
    """The query scaling (w_cos + D·w) while the state is on and the stored vectors are still of the adapter's model and dimension;
    else None (= the plain cosine). The stored vectors' model is re-read with the state (at most every 300 s)."""
    r = _row()
    if not r or r[0] != "on" or (r[2], int(r[3] or 0)) != _cache["model"] or int(r[3] or 0) != dim:
        return None
    w = json.loads(r[4])
    return np.float32(w["cos"]) + np.float32(dim) * np.asarray(w["dims"], dtype=np.float32)


def transform(q: np.ndarray) -> np.ndarray:
    """The query vector as the search uses it: unchanged while off."""
    s = scale(len(q))
    return q if s is None else q * s


def _set(s: str, model: str, dim: int, weights: dict | None, measurement: dict) -> None:
    _table()
    with g.lock:
        g.db().execute("DELETE FROM query_adapter WHERE id = 1")
        g.db().execute("INSERT INTO query_adapter (id, state, ts, model, dim, weights, measurement) VALUES (1, ?, ?, ?, ?, ?, ?)",
                       (s, g.now(), model, dim, json.dumps(weights) if weights else None, json.dumps(measurement)))
        g.db().commit()
    _cache.update(at=0.0, db=None)


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def examples() -> list[dict]:
    """{query, positives: set, negatives: set, ts, source} per distinct query and source."""
    out: dict[tuple, dict] = {}

    def add(query, ts, source, pos=None, neg=None):
        query = " ".join(str(query or "").split())
        if not query:
            return
        e = out.setdefault((query, source), {"query": query, "positives": set(), "negatives": set(), "ts": ts, "source": source})
        e["ts"] = min(e["ts"], ts) if e["ts"] and ts else (e["ts"] or ts)
        if pos:
            e["positives"].add(pos)
        if neg:
            e["negatives"].add(neg)
    with g.lock:
        rows = g.db().execute("SELECT id, data FROM object WHERE mode = 'active' AND visibility != 'private' AND data LIKE ?",
                              ('%"qa": {%',)).fetchall()
    for oid, raw in rows:
        qa = (json.loads(raw) if isinstance(raw, str) else raw or {}).get("qa") or {}
        for p in qa.get("pairs") or []:
            add(p.get("question"), qa.get("time") or "", "qa", pos=oid)
        if (qa.get("probe") or {}).get("question"):
            add(qa["probe"]["question"], qa.get("time") or "", "probe", pos=oid)
    with g.lock:
        rows = g.db().execute("SELECT id, data FROM object WHERE mode = 'active' AND visibility != 'private' AND data LIKE ?",
                              ('%"search_q": {%',)).fetchall()
    for oid, raw in rows:  # search_questions.py: short search-style queries the classify model wrote, round-trip filtered
        sq = (json.loads(raw) if isinstance(raw, str) else raw or {}).get("search_q") or {}
        for q in sq.get("questions") or []:
            add(q, sq.get("time") or "", "sq", pos=oid)
    from . import relevance
    relevance._table()
    with g.lock:
        rows = g.db().execute("SELECT query, search_ts, object_id, grade FROM relevance_judgment WHERE grade IS NOT NULL").fetchall()
    for query, ts, oid, grade in rows:
        if grade >= 2:
            add(query, ts, "judged", pos=oid)
        elif grade == 0:
            add(query, ts, "judged", neg=oid)
    return [e for e in out.values() if e["positives"]]


def _sha(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def query_vectors(texts: list[str], model: str) -> dict[str, np.ndarray]:
    """Normalised query vectors, embedded once and kept (table query_vector) — a sleep re-uses them."""
    from . import vector
    _table()
    need, out = [], {}
    for t in dict.fromkeys(texts):
        with g.lock:
            r = g.db().execute("SELECT dim, raw FROM query_vector WHERE sha = ?", (_sha(t, model),)).fetchone()
        if r:
            out[t] = vector._unpack(r[1], r[0])
        else:
            need.append(t)
    for i in range(0, len(need), EMBED_BATCH):
        part = need[i:i + EMBED_BATCH]
        for t, v in zip(part, vector.embed_texts(part)):
            v = np.asarray(v, dtype=np.float32)
            out[t] = v
            with g.lock:
                g.db().execute("INSERT OR REPLACE INTO query_vector (sha, model, dim, raw) VALUES (?, ?, ?, ?)", (_sha(t, model), model, len(v), vector._pack(v)))
        with g.lock:
            g.db().commit()
    for t, v in out.items():
        out[t] = v / (float(np.linalg.norm(v)) or 1.0)
    return out


# ── fit and measurement ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _features(q: np.ndarray, H: np.ndarray) -> np.ndarray:
    return np.hstack([(H @ q)[:, None], H * q[None, :] * H.shape[1]])


def _rows(ex: list[dict], qv: dict, pos: dict, m: np.ndarray, rnd: np.random.Generator):
    X, y = [], []
    for e in ex:
        q = qv[e["query"]]
        order = np.argsort(-(m @ q))
        posr = [pos[i] for i in e["positives"] if i in pos]
        negr = [pos[i] for i in e["negatives"] if i in pos]
        band = [int(r) for r in order[NEG_FROM:NEG_TO] if int(r) not in posr and int(r) not in negr]
        negr += list(rnd.choice(band, size=min(NEG_PER_QUERY, len(band)), replace=False)) if band else []
        if not posr or not negr:
            continue
        rows = posr + negr
        X.append(_features(q, m[rows]))
        y += [1] * len(posr) + [0] * len(negr)
    return (np.vstack(X), np.array(y)) if X else (None, None)


def _fit(ex, qv, pos, m, c: float):
    from sklearn.linear_model import LogisticRegression
    X, y = _rows(ex, qv, pos, m, np.random.default_rng(SEED))
    if X is None or len(set(y)) < 2:
        return None
    lr = LogisticRegression(C=c, max_iter=10000)  # 2000 left lbfgs unconverged on the production copy (16.11.2026)
    lr.fit(X, y)
    return {"cos": float(lr.coef_[0][0]), "dims": [float(x) for x in lr.coef_[0][1:]]}


def _rr(ex: list[dict], qv: dict, pos: dict, m: np.ndarray, w: dict | None) -> np.ndarray:
    """Reciprocal rank of each query's best positive among the top CANDIDATES by cosine (0 if none is there)."""
    s = None if w is None else (np.float32(w["cos"]) + np.float32(m.shape[1]) * np.asarray(w["dims"], dtype=np.float32))
    out = []
    for e in ex:
        q = qv[e["query"]]
        cand = np.argsort(-(m @ q))[:CANDIDATES]
        score = m[cand] @ (q if s is None else q * s)
        ranked = cand[np.argsort(-score)]
        hits = [k for k, r in enumerate(ranked) if int(r) in {pos[i] for i in e["positives"] if i in pos}]
        out.append(1.0 / (hits[0] + 1) if hits else 0.0)
    return np.array(out)


def _held_out(oid: str) -> bool:
    """A fixed TEST_SHARE of the objects by hash: their questions are never trained."""
    return int(hashlib.sha256(oid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < TEST_SHARE


def split(ex: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split by OBJECT and by time, so the test asks for what the fit never saw:
    train: the qa and search questions of the objects not held out + the older judged searches;
    test:  the probes and search questions of the held-out objects + the newest TEST_SHARE of the judged searches.
    Measured why (production copy 16.11.2026): with probes of the TRAINED objects in the test, MRR rose 0.552 → 0.662 — the adapter may
    simply prefer the few objects that have questions at all; a per-dimension weighting can learn that cluster."""
    judged = sorted([e for e in ex if e["source"] == "judged"], key=lambda e: e["ts"] or "")
    cut = int(len(judged) * (1 - TEST_SHARE))
    train = [e for e in ex if e["source"] in ("qa", "sq") and not any(_held_out(i) for i in e["positives"])] + judged[:cut]
    test = judged[cut:] + [e for e in ex if e["source"] in ("probe", "sq") and all(_held_out(i) for i in e["positives"])]
    test_q = {e["query"] for e in test}
    train = [e for e in train if e["query"] not in test_q]
    gen = sorted([e for e in train if e["source"] != "judged"], key=lambda e: hashlib.sha256(e["query"].encode()).hexdigest())
    return gen[:MAX_TRAIN_QUERIES] + [e for e in train if e["source"] == "judged"], test


def prune_vectors(keep: list[str], model: str) -> int:
    """Remove cached query vectors no longer used by the gate — the table stays bounded (MAX_TRAIN_QUERIES + the test)."""
    _table()
    want = {_sha(t, model) for t in keep}
    with g.lock:
        have = [r[0] for r in g.db().execute("SELECT sha FROM query_vector").fetchall()]
        gone = [s for s in have if s not in want]
        for i in range(0, len(gone), 500):
            part = gone[i:i + 500]
            g.db().execute(f"DELETE FROM query_vector WHERE sha IN ({','.join('?' * len(part))})", part)
        g.db().commit()
    return len(gone)


def gate(ex: list[dict] | None = None) -> dict:
    from . import vector
    model, dim = _model()
    if not dim:
        return {"state": state(), "reason": "no stored vectors"}
    ex = examples() if ex is None else ex
    ids, pos, m = vector.matrix()
    ex = [e for e in ex if any(i in pos for i in e["positives"])]
    train, test = split(ex)
    if len(train) < MIN_TRAIN_QUERIES or len(test) < MIN_TEST_QUERIES:
        r = {"reason": f"too few queries with a positive that has a vector (train {len(train)} of {MIN_TRAIN_QUERIES}, "
                       f"test {len(test)} of {MIN_TEST_QUERIES})", "train": len(train), "test": len(test)}
        _set("off", model, dim, None, r)
        return {"state": "off", **r}
    qv = query_vectors([e["query"] for e in train + test], model)
    prune_vectors([e["query"] for e in train + test], model)
    if any(len(v) != m.shape[1] for v in qv.values()):
        r = {"reason": "the embedding service returns another dimension than the stored vectors"}
        _set("off", model, dim, None, r)
        return {"state": "off", **r}
    train = sorted(train, key=lambda e: e["ts"] or "")
    k = int(len(train) * 0.8)
    fit_part, val = train[:k], train[k:]
    base_val = _rr(val, qv, pos, m, None)
    best = None
    for c in C_GRID:  # step 1: choose inside the training part
        w = _fit(fit_part, qv, pos, m, c)
        if w is not None:
            mrr = float(_rr(val, qv, pos, m, w).mean())
            if best is None or mrr > best[1]:
                best = (c, mrr)
    if best is None:
        _set("off", model, dim, None, {"reason": "no fit (one class only)"})
        return {"state": "off", "reason": "no fit (one class only)"}
    w = _fit(train, qv, pos, m, best[0])  # step 2: refit on all training queries, ONE test on the held-out ones
    a, b = _rr(test, qv, pos, m, w), _rr(test, qv, pos, m, None)
    d = a - b
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    meas = {"version": VERSION, "train": len(train), "test": len(test), "c": best[0], "val_mrr_adapter": round(best[1], 4),
            "val_mrr_cosine": round(float(base_val.mean()), 4), "test_mrr_adapter": round(float(a.mean()), 4),
            "test_mrr_cosine": round(float(b.mean()), 4), "paired_diff": round(float(d.mean()), 4), "se": round(se, 4),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "test_by_source": {s: sum(1 for e in test if e["source"] == s) for s in ("judged", "probe", "sq")}}
    jd = np.array([x for x, e in zip(d, test) if e["source"] == "judged"])
    if len(jd):
        meas["judged_diff"] = round(float(jd.mean()), 4)
    s = "on" if float(d.mean()) > 2 * se else "off"
    if s == "on" and len(jd) >= MIN_JUDGED and float(jd.mean()) < 0:  # generated queries must not win against real searches
        s, meas["reason"] = "off", f"better overall but worse on the {len(jd)} judged real searches"
    _set(s, model, dim, w if s == "on" else None, meas)
    return {"state": s, **meas}


def run(who: str = "sleep") -> dict:
    from . import vector
    if not vector.available():
        return {"state": "skipped", "reason": "no embedding service configured (config embed)"}
    return gate()


def line() -> str:
    _table()
    with g.lock:
        r = g.db().execute("SELECT state, ts, measurement FROM query_adapter WHERE id = 1").fetchone()
    if not r:
        return "    Query adapter (learned embedding weights, 0.68.0): off, never gated"
    return f"    Query adapter (learned embedding weights, 0.68.0): {r[0]} since {r[1][:16]} · {r[2]}"


def main(argv: list[str]) -> int:
    """vi-mcp query-adapter run | report | off"""
    from . import config, vector
    c = config.load()
    g.open_db(c.db)
    vector.config = c.embed
    cmd = argv[0] if argv else "report"
    if cmd == "run":
        print(json.dumps(run("cli"), ensure_ascii=False))
    elif cmd == "report":
        print(line().strip())
    elif cmd == "off":  # the way back by hand: the plain cosine at once
        model, dim = _model()
        _set("off", model, dim, None, {"reason": "switched off by hand"})
        print(line().strip())
    else:
        print(main.__doc__)
        return 2
    return 0
