"""kNN-LM probability semantics, mutable revisions and real neural integration when optional dependencies exist."""
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
