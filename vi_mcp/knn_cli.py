"""vi-mcp knn add|predict|generate|forget|snapshot|serve — frozen causal model with mutable next-token memory.

Runs on the inference host. Install torch, transformers and numpy in its Python environment, not the memory daemon's.
JSONL experience rows: {source, prompt, continuation}; only independently approved experience should be ingested.
"""
from __future__ import annotations

import argparse
import hmac
import json
from pathlib import Path
import threading


def make_handler(lm, token=None, steps=256):
    """Minimal non-streaming OpenAI-compatible chat/completion endpoint for existing VI model clients."""
    from http.server import BaseHTTPRequestHandler
    from .knn_lm import FORMAT
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # prompts, completions and authentication stay out of access logs

        def send_json(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return token is None or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                return self.send_json(401, {"error": "authentication required"})
            if self.path == "/health":
                return self.send_json(200, {"status": "ok", "backend": FORMAT, "revision": lm.store.view()["revision"]})
            if self.path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [{"id": lm.core.identity["model"], "object": "model"}]})
            self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if not self.authorized():
                return self.send_json(401, {"error": "authentication required"})
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                return self.send_json(404, {"error": "unknown endpoint"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("request length outside bounds")
                body = json.loads(self.rfile.read(length))
                if body.get("stream") or body.get("temperature", 0) != 0 or body.get("tools"):
                    raise ValueError("this backend supports non-streaming greedy text generation; tools run through VIL")
                if body.get("model") not in (None, "default", lm.core.identity["model"]):
                    raise ValueError("requested model differs from this inference core")
                stop = body.get("stop") or []
                if isinstance(stop, str):
                    stop = [stop]
                if not isinstance(stop, list) or any(not isinstance(s, str) for s in stop):
                    raise ValueError("stop must be a string or list of strings")
                chat = self.path == "/v1/chat/completions"
                prompt = lm.core.chat_prompt(body.get("messages")) if chat else body.get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be text")
                with request_lock:
                    result = lm.generate(prompt, steps=steps, stop=stop)
                choice = {"index": 0, "finish_reason": "stop" if result["finish_reason"] == "stop" else "length"}
                choice.update({"message": {"role": "assistant", "content": result["text"]}} if chat else {"text": result["text"]})
                self.send_json(200, {"object": "chat.completion" if chat else "text_completion", "model": lm.core.identity["model"],
                                      "choices": [choice], "knn": {"revision": result["revision"], "weight": lm.weight,
                                          "neighbors": lm.k, "finish_reason": result["finish_reason"], "backend": FORMAT,
                                          "datastore_sha256": result.get("datastore_sha256"),
                                          "weights_sha256": lm.core.identity["weights_sha256"]}})
            except (ValueError, TypeError, KeyError) as e:
                self.send_json(400, {"error": str(e)})
    return Handler


def main(argv):
    """vi-mcp knn ACTION --store DIR [--model MODEL] [--revision COMMIT] [--data JSONL] [--prompt TEXT]."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["add", "predict", "generate", "evaluate", "forget", "snapshot", "serve", "status"])
    p.add_argument("--store", required=True)
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data")
    p.add_argument("--prompt")
    p.add_argument("--source")
    p.add_argument("--reason")
    p.add_argument("--destination")
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--neighbors", type=int, default=32)
    p.add_argument("--retrieval-temperature", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=256, help="explicit decoding resource budget; exhaustion is reported")
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8776)
    p.add_argument("--token-file")
    args = p.parse_args(argv)
    try:
        from .knn_lm import Datastore, KNNLanguageModel, TransformersCore
        if args.action in ("status", "forget", "snapshot"):
            store = Datastore(args.store)
            if args.action == "forget":
                if not args.source:
                    raise ValueError("forget requires --source and --reason")
                value = {"revision": store.forget(args.source, args.reason)}
            elif args.action == "snapshot":
                if not args.destination:
                    raise ValueError("snapshot requires --destination")
                value = {"revision": store.snapshot(args.destination).view()["revision"]}
            else:
                v = store.view()
                value = {"revision": v["revision"], "documents": len(v["documents"]), "entries": sum(d["entries"] for d in v["documents"].values()),
                         "tombstones": len(v["tombstones"]), "identity": v["identity"]}
        else:
            if not args.model:
                raise ValueError("inference and ingestion require --model")
            if args.action == "serve" and args.bind not in ("127.0.0.1", "::1", "localhost") and not args.token_file:
                raise ValueError("a non-loopback inference listener requires --token-file")
            core = TransformersCore.load(args.model, args.revision, args.device)
            store = Datastore(args.store, core.identity)
            lm = KNNLanguageModel(core, store, weight=args.weight, k=args.neighbors, temperature=args.retrieval_temperature)
            if args.action == "add":
                if not args.data:
                    raise ValueError("add requires --data JSONL")
                count = 0
                with Path(args.data).open(encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        lm.remember(row["source"], row["prompt"], row["continuation"])
                        count += 1
                value = {"sources_updated": count, "revision": store.view()["revision"]}
            elif args.action == "serve":
                from http.server import ThreadingHTTPServer
                token = Path(args.token_file).read_text().strip() if args.token_file else None
                if args.token_file and not token:
                    raise ValueError("empty token file")
                with ThreadingHTTPServer((args.bind, args.port), make_handler(lm, token, args.steps)) as server:
                    server.serve_forever()
                return 0
            elif args.action == "evaluate":
                if not args.data:
                    raise ValueError("evaluate requires held-out --data JSONL")
                with Path(args.data).open(encoding="utf-8") as f:
                    value = lm.evaluate(json.loads(line) for line in f)
            elif args.action == "generate":
                value = lm.generate(args.prompt, args.steps)
            else:
                result = lm.predict(args.prompt)
                token, base = result["token"], result["base_token"]
                value = {"revision": result["revision"], "token": token, "text": core.tokenizer.decode([token]),
                         "probability": float(result["probabilities"][token]), "base_token": base, "base_text": core.tokenizer.decode([base]),
                         "base_probability_of_chosen": float(result["base_probabilities"][token]), "neighbors": result["neighbors"]}
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError) as e:
        print(f"vi-mcp knn: {type(e).__name__}: {e}")
        return 1
