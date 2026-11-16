"""0.68.0: query adapter — learned embedding weights by scikit-learn on the query vector, switched by a two-step offline gate."""
import numpy as np

from vi_mcp import memory as g, query_adapter as qa, vector

D = 16


def _world(signal: bool, n=400, seed=1):
    """n objects with a vector; per object two training questions and a probe. With signal, a question's vector equals its object's
    vector on dims 0–7 and is strong noise elsewhere (the cosine is misled by the noise, a per-dimension weighting is not); without
    signal the questions are pure noise."""
    rnd = np.random.default_rng(seed)
    table = {}
    for i in range(n):
        h = rnd.normal(size=D).astype(np.float32)
        o = g.new_object("insight", f"object {i}", ["topic:x"], {"source": "t", "qa": {
            "pairs": [{"question": f"q{i}a"}, {"question": f"q{i}b"}], "probe": {"question": f"q{i}p"}, "time": f"2026-11-{1 + i % 28:02d}"}})
        vector.save(o.id, h, "m1")
        for s in "abp":
            noise = rnd.normal(size=D).astype(np.float32) * 2.0
            v = noise.copy()
            if signal:
                v[:8] = h[:8] * 3.0
            table[f"q{i}{s}"] = v
    vector.embed_fn = lambda texts: [table[t].tolist() for t in texts]
    return table


def test_off_and_plain_cosine_without_a_gate(gd):
    q = np.ones(D, dtype=np.float32)
    assert qa.state() == "off" and qa.transform(q) is q
    assert "never gated" in qa.line()


def test_too_few_queries_stays_off_with_the_reason(gd):
    _world(True, n=20)
    try:
        r = qa.gate()
    finally:
        vector.embed_fn = None
    assert r["state"] == "off" and "too few" in r["reason"]


def test_the_gate_switches_on_when_dimensions_carry_the_signal_and_the_search_uses_it(gd):
    _world(True)
    try:
        r = qa.gate()
        assert r["state"] == "on", r
        assert r["test_mrr_adapter"] > r["test_mrr_cosine"] and r["test_by_source"]["probe"] > 0  # probes are test only
        w = qa._row()
        assert w[2] == "m1" and w[3] == D
        s = qa.scale(D)
        assert s is not None and float(np.abs(s[:8]).mean()) > float(np.abs(s[8:]).mean())  # the signal dims weigh more
        assert qa.scale(D + 1) is None  # another dimension: plain cosine
        # the search orders by the adapter and still reports the cosine
        objs = [o for o in g.recall("object", limit=400)]
        cand = [o.id for _s, o, _w in objs]
        hits = vector.similar("q5p", cand, limit=5)
        assert len(hits) == 5 and all(-1.0 <= c <= 1.0 for _i, c in hits)
    finally:
        vector.embed_fn = None


def test_pure_noise_questions_leave_it_off(gd):
    _world(False)
    try:
        r = qa.gate()
    finally:
        vector.embed_fn = None
    assert r["state"] == "off" and r.get("paired_diff", 0) <= 2 * r.get("se", 0) + 1e-9


def test_another_embedding_model_switches_the_adapter_off(gd):
    _world(True)
    try:
        assert qa.gate()["state"] == "on"
        with g.lock:
            g.db().execute("UPDATE vector SET model = 'm2'")
            g.db().commit()
        qa._cache.update(at=0.0, db=None)
        assert qa.scale(D) is None
    finally:
        vector.embed_fn = None


def test_the_test_never_asks_for_a_trained_object(gd):
    """Split by object (production copy 16.11.2026: probes of trained objects inflated MRR 0.552 → 0.662, unseen objects 0.424 → 0.388)."""
    _world(True, n=120)
    vector.embed_fn = None
    train, test = qa.split(qa.examples())
    trained = {i for e in train for i in e["positives"]}
    assert test and all(not (e["positives"] & trained) for e in test if e["source"] == "probe")
    assert all(e["source"] != "probe" for e in train)


def test_search_questions_keep_only_short_round_trip_queries_and_feed_the_adapter(gd, monkeypatch):
    import json as _json
    from vi_mcp import classify, reflection, search_questions as sq
    table = _world(True, n=40)
    target = g.recall("object 7", limit=50)
    o = next(o for _s, o, _w in target if o.text == "object 7")
    good = table["q7a"]
    table["good query"] = good
    table["far query"] = -np.asarray(good)
    table["object 7"] = good
    monkeypatch.setattr(classify, "available", lambda: True)
    monkeypatch.setattr(reflection, "ask_fn", lambda s, u: _json.dumps({"queries": ["good query", "far query", "object 7",
                                                                                     " ".join(["w"] * 30)]}))
    monkeypatch.setattr(sq, "pending", lambda limit: [o])
    monkeypatch.setattr(sq, "PER_OBJECT", 4)
    try:
        r = sq.run(1, "test")
    finally:
        vector.embed_fn = None
    stored = g.load(o.id).data["search_q"]
    assert r["objects"] == 1 and stored["questions"] == ["good query"]
    assert stored["rejected"] == {"own object not in the top 20": 1, "copies the first sentence": 1, "too long or empty": 1}
    assert any(e["source"] == "sq" and e["query"] == "good query" for e in qa.examples())
    assert "Search questions per object" in sq.line()


def test_generated_training_queries_are_capped_and_the_vector_cache_pruned(gd, monkeypatch):
    monkeypatch.setattr(qa, "MAX_TRAIN_QUERIES", 100)
    monkeypatch.setattr(qa, "MIN_TRAIN_QUERIES", 50)
    _world(True)
    try:
        train, test = qa.split(qa.examples())
        assert sum(1 for e in train if e["source"] != "judged") == 100
        with g.lock:  # a vector cached by an earlier gate that the next gate no longer uses
            g.db().execute("CREATE TABLE IF NOT EXISTS query_vector (sha TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, raw BLOB NOT NULL)")
            g.db().execute("INSERT INTO query_vector VALUES ('stale', 'm1', 16, ?)", (np.zeros(16, dtype=np.float32).tobytes(),))
            g.db().commit()
        assert qa.gate()["train"] == 100
        with g.lock:
            n = g.db().execute("SELECT COUNT(*) FROM query_vector").fetchone()[0]
        assert n == len({e["query"] for e in train + test})  # only what the gate used stays cached
    finally:
        vector.embed_fn = None
