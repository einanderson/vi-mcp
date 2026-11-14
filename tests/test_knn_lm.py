"""kNN-LM probability semantics, mutable revisions and real neural integration when optional dependencies exist."""
import hashlib
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from vi_mcp.knn_lm import Datastore, KNNLanguageModel, TransformersCore, interpolate

IDENTITY = {"model": "fixture", "weights_sha256": "fixture", "tokenizer_sha256": "fixture",
            "key_function": "fixture", "dimension": 2, "vocab_size": 4}


def test_squared_l2_distribution_and_probability_interpolation(tmp_path):
    store = Datastore(tmp_path / "store", IDENTITY)
    store.replace("first", [[0, 0], [1, 0], [0, 2]], [1, 1, 2])
    p, info = store.distribution([0, 0], k=3)
    expected = np.exp(-np.array([0., 1., 4.]))
    expected /= expected.sum()
    np.testing.assert_allclose(p, [0, expected[:2].sum(), expected[2], 0])
    mixed, base = interpolate(np.log([.7, .1, .1, .1]), p, .75)
    np.testing.assert_allclose(mixed, .25 * base + .75 * p)
    assert mixed.argmax() == 1 and info["neighbors"] == 3 and info["nearest_squared_l2"] == 0
    p, info = store.distribution([0, 0], exclude_sources=["first"])
    mixed, base = interpolate([2, 1, 0, -1], p, 1)
    assert info["neighbors"] == 0
    np.testing.assert_array_equal(mixed, base)


def test_live_mutation_pinned_revision_restart_and_independent_snapshot(tmp_path):
    store = Datastore(tmp_path / "store", IDENTITY)
    store.replace("inventory", [[0, 0]], [1])
    pinned = store.view()
    clone = store.snapshot(tmp_path / "clone")
    second_writer = Datastore(tmp_path / "store", IDENTITY)
    second_writer.replace("inventory", [[0, 0]], [2])
    assert store.distribution([0, 0])[0].argmax() == 2
    assert store.distribution([0, 0], view=pinned)[0].argmax() == 1
    assert clone.distribution([0, 0])[0].argmax() == 1
    store.forget("inventory", "inventory removed")
    reopened = Datastore(tmp_path / "store", IDENTITY)
    assert not reopened.distribution([0, 0])[0].any() and len(reopened.view()["tombstones"]) == 2
    with pytest.raises(ValueError, match="mismatch"):
        Datastore(tmp_path / "store", {**IDENTITY, "weights_sha256": "other"})


def test_forget_erases_every_shard_of_the_source_and_snapshots_do_not_copy_it(tmp_path):
    """Review 14.11.2026 (Claude), item 7: forget only removed the manifest entry. The keys and the token IDs (the text itself)
    stayed on disk behind the tombstone, and a snapshot taken afterwards copied them into the new datastore."""
    import numpy as np
    store = Datastore(tmp_path / "store", IDENTITY)
    store.replace("note", [[7, 7]], [2])
    store.replace("note", [[7, 7]], [3])          # the older version becomes a "replaced" tombstone
    store.replace("public", [[0, 1]], [1])
    pinned = store.view()
    store.forget("note", "user asked to forget")
    holding = [p for p in (tmp_path / "store").glob("*.npy") if (np.load(p) == 7).all() or np.load(p).tolist() in ([2], [3])]
    assert not holding
    view = store.view()
    assert [t["erased"] for t in view["tombstones"] if t["source_hash"] == view["tombstones"][-1]["source_hash"]] == [True, True]
    snap = store.snapshot(tmp_path / "snap")
    assert len(list((tmp_path / "snap").glob("*.npy"))) == 2 and snap.distribution([0, 1])[0].argmax() == 1
    with pytest.raises(ValueError, match="forgotten"):
        store.distribution([7, 7], view=pinned)   # a generation pinned before the forget gets a named refusal


def test_invalid_keys_limits_and_corruption_fail_closed(tmp_path):
    store = Datastore(tmp_path / "store", IDENTITY, max_entries=2)
    for keys, tokens in (([[float("nan"), 0]], [1]), ([[0, 0]], [4]), ([[0, 0]], [1.5])):
        with pytest.raises(ValueError):
            store.replace("bad", keys, tokens)
    assert store.view()["revision"] == 0
    store.replace("valid", [[0, 0], [1, 1]], [1, 2])
    with pytest.raises(ValueError, match="limit"):
        store.replace("too much", [[0, 0]], [1])
    entry = next(iter(store.view()["documents"].values()))
    (store.directory / entry["keys"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.distribution([0, 0])


def test_frozen_checkpoint_holds_writers_until_the_symbolic_capture_finishes(tmp_path):
    first = Datastore(tmp_path / "live", IDENTITY)
    first.replace("inventory", [[0, 0]], [1])
    other = Datastore(tmp_path / "live", IDENTITY)
    started, finished = threading.Event(), threading.Event()
    def write():
        started.set()
        other.replace("inventory", [[0, 0]], [2])
        finished.set()
    with first.frozen_snapshot(tmp_path / "snapshot") as frozen:
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        assert started.wait(1) and not finished.wait(.05)
        assert frozen.distribution([0, 0])[0].argmax() == 1
    writer.join(timeout=2)
    assert finished.is_set() and first.distribution([0, 0])[0].argmax() == 2


def test_chat_endpoint_uses_inference_and_keeps_authentication(tmp_path):
    from http.server import ThreadingHTTPServer
    from vi_mcp.knn_cli import make_handler
    class Core:
        identity = IDENTITY
        def chat_prompt(self, messages):
            return messages[-1]["content"]
    class LM:
        core, store, weight, k = Core(), Datastore(tmp_path / "store", IDENTITY), .5, 2
        def generate(self, prompt, steps, stop):
            assert prompt == "question" and steps == 7
            return {"text": "answer", "revision": 0, "finish_reason": "stop"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(LM(), "test credential", 7))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        body = json.dumps({"messages": [{"role": "user", "content": "question"}], "temperature": 0}).encode()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(endpoint, body))
        assert e.value.code == 401
        req = urllib.request.Request(endpoint, body, {"Authorization": "Bearer test credential"})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
        assert result["choices"][0]["message"]["content"] == "answer" and result["knn"]["revision"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_transformer_changes_next_token_from_datastore_only(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    # Real causal transformer with deterministic untrained weights; no downloads or memorized fixture logits.
    torch.manual_seed(3)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "service": 1, "port": 2, "80": 3, "8080": 4, "[EOS]": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=32, n_embd=8, n_layer=1, n_head=2, eos_token_id=5))
    core = TransformersCore(model, tokenizer, "deterministic-test-transformer")
    store = Datastore(tmp_path / "store", core.identity)
    lm = KNNLanguageModel(core, store, weight=.99, k=1)
    original = core.weight_hash()
    baseline = lm.predict("service port")
    lm.remember("inventory", "service port", " 80")
    first = lm.predict("service port")
    lm.remember("inventory", "service port", " 8080")
    second = lm.predict("service port")
    assert first["token"] == 3 and second["token"] == 4 and second["probabilities"][4] > .99
    np.testing.assert_array_equal(baseline["base_probabilities"], second["base_probabilities"])
    assert core.weight_hash() == original and not any(p.requires_grad for p in core.model.parameters())
    score = lm.evaluate([{"source": "inventory", "prompt": "service port", "continuation": " 8080"}])
    assert score["tokens"] == 1 and score["source_exclusion"]
    assert score["base_nll"] == score["knn_nll"]  # the evaluator cannot retrieve its own target source
    store.forget("inventory", "test complete")
    np.testing.assert_array_equal(lm.predict("service port")["probabilities"], baseline["probabilities"])


def _tiny_lm(tmp_path, weight=.99):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(5)
    words = ["[UNK]", "[EOS]", "reload", "nginx", "check", "config", "with", "nginx-t", "then", "systemctl", "reload-nginx", "done",
             "backup", "database", "pg_dump", "into", "srv"]
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_positions=64, n_embd=16, n_layer=2, n_head=2, eos_token_id=1))
    core = TransformersCore(model, tok, "deterministic-draft-transformer")
    return KNNLanguageModel(core, Datastore(tmp_path / "store", core.identity), weight=weight, k=1)


def test_retrieval_drafts_keep_the_greedy_text_and_save_forward_passes(tmp_path):
    """Exact retrieval-based speculative decoding (REST/NEST without relaxed acceptance): same tokens as plain greedy decoding,
    fewer forward passes when the memory holds the continuation, and the accepted tokens name their source."""
    lm = _tiny_lm(tmp_path)
    runbook = " check config with nginx-t then systemctl reload-nginx done"
    lm.remember("runbook-nginx", "reload nginx", runbook)
    plain = lm.generate("reload nginx", steps=8)
    fast = lm.generate("reload nginx", steps=8, draft=6)
    assert fast["tokens"] == plain["tokens"] and fast["text"] == plain["text"]
    spec = fast["speculative"]
    assert spec["forward_passes"] < len(fast["tokens"]) and spec["accepted_from_memory"] > 0
    assert {s["source_hash"] for s in spec["spans"]} == {hashlib.sha256(b"runbook-nginx").hexdigest()}
    # Nothing relevant in memory: no usable drafts, still the identical text.
    other = _tiny_lm(tmp_path / "other", weight=.3)
    other.remember("backup", "backup database", " pg_dump into srv")
    assert other.generate("reload nginx", steps=6, draft=4)["tokens"] == other.generate("reload nginx", steps=6)["tokens"]
    with pytest.raises(ValueError, match="draft"):
        lm.generate("reload nginx", draft=65)


def test_compact_erases_replaced_shards_and_reports_what_it_freed(tmp_path):
    """Review item 7, open point: replaced shards stayed on disk forever. compact is the explicit storage management: it erases the
    shards of replaced versions (never live ones), marks their tombstones erased and reports the count and bytes."""
    store = Datastore(tmp_path / "store", IDENTITY)
    store.replace("inventory", [[0, 0]], [1])
    store.replace("inventory", [[0, 0]], [2])
    store.replace("inventory", [[0, 0]], [3])
    store.replace("other", [[1, 1]], [1])
    pinned = store.view()
    before = len(list((tmp_path / "store").glob("*.npy")))
    report = store.compact("disk cleanup after review")
    assert before == 8 and report["erased_versions"] == 2 and report["freed_bytes"] > 0
    assert len(list((tmp_path / "store").glob("*.npy"))) == 4          # only live shards remain
    assert store.distribution([0, 0])[0].argmax() == 3                    # live data untouched
    assert store.distribution([0, 0], view=pinned)[0].argmax() == 3      # a view pinned after the last replace still works
    assert store.compact("again")["erased_versions"] == 0                 # idempotent
    with pytest.raises(ValueError, match="reason"):
        store.compact("")


class _ChainCore:
    """Deterministic core: the state of a position is the one-hot of its token, logits are flat — with weight ~1 and k=1 the kNN
    memory decides alone, so greedy decoding follows the stored text exactly and drafts are accepted across any boundary."""
    WORDS = ["[EOS]"] + [f"w{i}" for i in range(1, 16)]

    def __init__(self, window):
        import threading
        self.window, self.lock = window, threading.RLock()
        self.identity = {"model": "chain", "weights_sha256": "chain", "tokenizer_sha256": "chain", "key_function": "onehot",
                         "dimension": len(self.WORDS), "vocab_size": len(self.WORDS)}
        words = self.WORDS
        class Tok:
            eos_token_id = 0
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(words[i] for i in ids if i)
        self.tokenizer = Tok()

    def encode(self, text):
        return [self.WORDS.index(w) for w in text.split()]

    def forward_all(self, ids):
        states = np.eye(len(self.WORDS))[ids]
        return states, np.zeros((len(ids), len(self.WORDS)))

    def forward(self, ids):
        states, logits = self.forward_all(ids)
        return states, logits[-1]


def test_drafts_match_plain_decoding_at_stop_strings_and_the_context_window(tmp_path):
    """Codex's review (14.11.2026): drafted generation appended a whole accepted span before checking stop strings and the window,
    so it could return more tokens than plain decoding. Both are now checked after every token, exactly like plain decoding."""
    core = _ChainCore(window=10)
    lm = KNNLanguageModel(core, Datastore(tmp_path / "store", core.identity), weight=1.0, k=1)
    lm.remember("chain", "w1", " w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12")
    for stop in (["w5"], ["w7"]):
        plain, fast = lm.generate("w1", steps=11, stop=stop), lm.generate("w1", steps=11, stop=stop, draft=6)
        assert fast["speculative"]["accepted_from_memory"] > 0
        assert (fast["tokens"], fast["text"], fast["finish_reason"]) == (plain["tokens"], plain["text"], plain["finish_reason"]), stop
    plain, fast = lm.generate("w1 w2 w3", steps=11), lm.generate("w1 w2 w3", steps=11, draft=6)
    assert plain["finish_reason"] == "context" and fast["speculative"]["accepted_from_memory"] > 0
    assert (fast["tokens"], fast["finish_reason"]) == (plain["tokens"], plain["finish_reason"])


def test_an_interrupted_erase_is_completed_by_compact(tmp_path, monkeypatch):
    """Codex's review (14.11.2026): forget marks the tombstone erased before the files are gone; a crash in between leaves the data
    on disk under an 'erased' tombstone. compact now removes every shard file no live document or kept tombstone references."""
    import pathlib
    store = Datastore(tmp_path / "store", IDENTITY)
    store.replace("note", [[7, 7]], [2])
    store.replace("public", [[0, 1]], [1])
    real_unlink = pathlib.Path.unlink
    monkeypatch.setattr(pathlib.Path, "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        store.forget("note", "user asked to forget")
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)
    assert len(list((tmp_path / "store").glob("*.npy"))) == 4      # the data survived the crash …
    report = store.compact("finish interrupted erase")
    assert report["orphans_removed"] == 2 and len(list((tmp_path / "store").glob("*.npy"))) == 2   # … until compact
    assert store.distribution([0, 1])[0].argmax() == 1
