"""Demonstrate real pretrained-model token changes caused only by mutable kNN memory.

    python scripts/demo_knn_lm.py [--model sshleifer/tiny-gpt2] [--revision COMMIT] [--device cpu]

Downloads an explicitly selected pretrained model; no optimizer or model training. The datastore is temporary.
The tiny default model demonstrates the mechanism, not general task quality. Full weight hashes before/after must agree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vi_mcp.knn_lm import Datastore, KNNLanguageModel, TransformersCore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--revision", default="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    core = TransformersCore.load(a.model, a.revision, a.device)
    before_hash = core.weight_hash()
    prompt = "The access code is"
    with tempfile.TemporaryDirectory(prefix="vi-knn-demo-") as td:
        store = Datastore(Path(td) / "live", core.identity)
        lm = KNNLanguageModel(core, store, weight=.75, k=1)
        initial = lm.predict(prompt)
        def summary(result):
            token = result["token"]
            return {"revision": result["revision"], "next_token": core.tokenizer.decode([token]), "token_id": token,
                    "mixed_probability": float(result["probabilities"][token]),
                    "base_probability": float(result["base_probabilities"][token]), "neighbors": result["neighbors"]}
        lm.remember("checked-inventory", prompt, " blue")
        first = lm.predict(prompt)
        clone = store.snapshot(Path(td) / "snapshot")
        from http.server import ThreadingHTTPServer
        from vi_mcp.knn_cli import make_handler
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lm, steps=1))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def http_prediction():
            body = json.dumps({"model": a.model, "prompt": prompt, "temperature": 0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/completions", body,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                return json.load(response)
        try:
            http_first = http_prediction()
            lm.remember("checked-inventory", prompt, " green")
            second = lm.predict(prompt)
            http_second = http_prediction()
            assert http_first["choices"][0]["text"] == " blue" and http_second["choices"][0]["text"] == " green"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        historical = KNNLanguageModel(core, clone, weight=.75, k=1).predict(prompt)
        store.forget("checked-inventory", "inventory no longer applies")
        forgotten = lm.predict(prompt)
        assert core.tokenizer.decode([first["token"]]) == " blue"
        assert core.tokenizer.decode([second["token"]]) == " green"
        assert historical["token"] == first["token"]
        assert forgotten["token"] == initial["token"]
        assert before_hash == core.weight_hash()
        print(json.dumps({"model": a.model, "revision": a.revision, "prompt": prompt, "weight": .75,
                          "without_memory": summary(initial), "first_memory": summary(first), "replaced_memory": summary(second),
                          "snapshot": summary(historical), "forgotten": summary(forgotten), "weights_unchanged": True,
                          "http_predictions": [http_first["choices"][0]["text"], http_second["choices"][0]["text"]],
                          "weights_sha256": before_hash,
                          "scope": "pretrained neural token interpolation and memory lifecycle; no held-out general capability gain claimed"}, indent=2))


if __name__ == "__main__":
    main()
