"""kNN-LM below MCP: mutable hidden-state/next-token memory interpolated with frozen-model probabilities.

Run on the inference host, with optional torch/transformers dependencies. No model training and no VI SQL writer.
The exact squared-L2 search scans immutable NumPy shards in bounded blocks; a revision pins a whole generation.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import uuid

import numpy as np

FORMAT = "vi-knn-lm-v1"


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Datastore:
    """Private local inference datastore. Atomic manifests reference immutable shards; replacements leave tombstones.
    Separate from the VI database and networks. Open with exactly the same model/tokenizer/key-function identity.
    """
    def __init__(self, directory, identity=None, max_entries=1_000_000):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.directory / "manifest.json"
        self._thread_lock = threading.RLock()
        self._verified = {}
        with self._writer():
            if not self.path.exists():
                if not identity:
                    raise ValueError("new datastore needs a model/tokenizer/key-function identity")
                if int(identity.get("dimension", 0)) <= 0 or int(identity.get("vocab_size", 0)) <= 0 or max_entries < 1:
                    raise ValueError("invalid datastore dimensions or entry limit")
                self._write(self.path, {"format": FORMAT, "identity": identity, "revision": 0, "documents": {},
                                       "tombstones": [], "max_entries": max_entries})
        view = self.view()
        if identity is not None and view["identity"] != identity:
            raise ValueError("datastore model/tokenizer/key function mismatch; rebuild for this inference core")
        self.identity = view["identity"]

    @contextlib.contextmanager
    def _writer(self):
        with self._thread_lock:
            fd = os.open(self.directory / ".writer.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if os.name == "nt":
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"0")
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _write(self, path, value):
        temporary = path.parent / (uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as f:
                os.chmod(temporary, 0o600)
                f.write(_json(value))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def view(self):
        view = json.loads(self.path.read_text(encoding="utf-8"))
        if view.get("format") != FORMAT:
            raise ValueError("unsupported kNN datastore format")
        return view

    def _array(self, descriptor, field):
        filename = descriptor[field]
        if Path(filename).name != filename or not filename.endswith(".npy"):
            raise ValueError("invalid datastore shard path")
        path = self.directory / filename
        try:
            stat = path.stat()
        except FileNotFoundError:
            raise ValueError("a source of this pinned revision was forgotten (its shards are erased); "
                             "restart on the current revision") from None
        stamp = (stat.st_size, stat.st_mtime_ns)
        if self._verified.get(filename) != stamp:
            if _digest(path) != descriptor[field + "_sha256"]:
                raise ValueError("datastore shard checksum mismatch")
            self._verified[filename] = stamp
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def replace(self, source, keys, tokens):
        """Atomically replace all entries of one source. Other readers see the old or the new revision, never a partial write."""
        if not isinstance(source, str) or not source.strip() or len(source) > 1000:
            raise ValueError("source must be a nonempty identifier of at most 1000 characters")
        keys = np.asarray(keys, dtype="<f4")
        raw_tokens = np.asarray(tokens)
        if raw_tokens.dtype.kind not in "iu" or raw_tokens.ndim != 1:
            raise ValueError("targets must be a vector of integer next-token IDs")
        tokens = raw_tokens.astype("<i8")
        if keys.shape != (len(tokens), self.identity["dimension"]) or not len(tokens) or not np.isfinite(keys).all():
            raise ValueError("keys need finite aligned context vectors of the core's dimension")
        if (tokens < 0).any() or (tokens >= self.identity["vocab_size"]).any():
            raise ValueError("next-token ID outside the bound tokenizer vocabulary")
        key = hashlib.sha256(source.encode()).hexdigest()
        with self._writer():
            view = self.view()
            total = sum(d["entries"] for s, d in view["documents"].items() if s != key) + len(tokens)
            if total > view["max_entries"]:
                raise ValueError("datastore entry limit reached; remove or replace sources explicitly")
            d = {"source": source, "entries": len(tokens)}
            for field, array in (("keys", keys), ("tokens", tokens)):
                filename = uuid.uuid4().hex + ".npy"
                path = self.directory / filename
                with path.open("xb") as f:
                    os.chmod(path, 0o600)
                    np.save(f, array, allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
                d[field], d[field + "_sha256"] = filename, _digest(path)
            old = view["documents"].get(key)
            if old:
                view["tombstones"].append({"source_hash": key, "revision": view["revision"], "reason": "replaced", "shard": old})
            view["documents"][key] = d
            view["revision"] += 1
            self._write(self.path, view)
            return view["revision"]

    def forget(self, source, reason):
        if not reason or not str(reason).strip():
            raise ValueError("forget needs a reason")
        key = hashlib.sha256(source.encode()).hexdigest()
        with self._writer():
            view = self.view()
            if key not in view["documents"]:
                raise ValueError("unknown datastore source")
            old = view["documents"].pop(key)
            view["tombstones"].append({"source_hash": key, "revision": view["revision"], "reason": str(reason), "shard": old})
            # Forgetting erases: the token IDs are the text itself and the keys can be inverted. Every shard of this source —
            # the live one and older replaced versions — is deleted after the manifest no longer references it; the tombstones
            # keep hash, reason and checksums (erased=True). Snapshots taken before stay independent copies (docs/KNN_LM.md).
            doomed = []
            for t in view["tombstones"]:
                if t["source_hash"] == key and not t.get("erased"):
                    doomed += [t["shard"][f] for f in ("keys", "tokens")]
                    t["erased"] = True
            view["revision"] += 1
            self._write(self.path, view)
            for filename in doomed:
                (self.directory / filename).unlink(missing_ok=True)
                self._verified.pop(filename, None)
            return view["revision"]

    def compact(self, reason):
        """Explicit storage management: erase the shards of replaced versions (never a live document), mark their tombstones erased
        and report what was freed. A request still pinned to a revision that used them gets the named refusal of _array."""
        if not reason or not str(reason).strip():
            raise ValueError("compact needs a reason")
        with self._writer():
            view = self.view()
            live = {d[f] for d in view["documents"].values() for f in ("keys", "tokens")}
            doomed, versions = [], 0
            for t in view["tombstones"]:
                if t.get("erased"):
                    continue
                files = [t["shard"][f] for f in ("keys", "tokens")]
                if set(files) & live:
                    continue  # defensive: a tombstone never shares files with a live document
                doomed += files
                t["erased"], t["erased_reason"] = True, str(reason)
                versions += 1
            # Orphans: shard files that no live document and no kept tombstone references — an erase interrupted after its
            # manifest write (Codex's review, 14.11.2026) or a crash between writing a shard and the manifest. Completed here.
            kept = live | {t["shard"][f] for t in view["tombstones"] if not t.get("erased") for f in ("keys", "tokens")}
            orphans = sorted(p.name for p in self.directory.glob("*.npy") if p.name not in kept and p.name not in doomed)
            freed = sum((self.directory / f).stat().st_size for f in doomed + orphans if (self.directory / f).exists())
            if versions:
                view["revision"] += 1
                self._write(self.path, view)
            for filename in doomed + orphans:
                (self.directory / filename).unlink(missing_ok=True)
                self._verified.pop(filename, None)
            return {"erased_versions": versions, "orphans_removed": len(orphans), "freed_bytes": freed, "revision": view["revision"],
                    "reason": str(reason)}

    @contextlib.contextmanager
    def frozen_snapshot(self, destination):
        """Copy one consistent revision into a new independent datastore, retaining its own tombstones and old shards."""
        destination = Path(destination)
        if destination.exists():
            raise ValueError("snapshot destination must not exist")
        with self._writer():
            view = self.view()
            destination.mkdir(mode=0o700, parents=True)
            descriptors = list(view["documents"].values()) + [x["shard"] for x in view["tombstones"] if not x.get("erased")]
            for d in descriptors:
                for field in ("keys", "tokens"):
                    self._array(d, field)
                    shutil.copyfile(self.directory / d[field], destination / d[field])
                    os.chmod(destination / d[field], 0o600)
            self._write(destination / "manifest.json", view)
            yield Datastore(destination, self.identity)

    def snapshot(self, destination):
        with self.frozen_snapshot(destination) as frozen:
            return frozen

    def distribution(self, query, *, k=32, temperature=1.0, view=None, exclude_sources=()):
        """Squared-L2 neighbors → softmax(-distance/temperature) → probability mass aggregated by next-token ID."""
        if not isinstance(k, int) or not 1 <= k <= 4096 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("k must be 1–4096 and retrieval temperature finite and positive")
        query = np.asarray(query, dtype=np.float64)
        if query.shape != (self.identity["dimension"],) or not np.isfinite(query).all():
            raise ValueError("invalid context query vector")
        view = self.view() if view is None else view
        if view["identity"] != self.identity:
            raise ValueError("incompatible datastore view")
        blocked = {hashlib.sha256(s.encode()).hexdigest() for s in exclude_sources}
        distances, targets = np.empty(0), np.empty(0, dtype=np.int64)
        origin, rows, sources = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), []
        for source, d in sorted(view["documents"].items()):
            if source in blocked:
                continue
            sources.append(source)
            keys, tokens = self._array(d, "keys"), self._array(d, "tokens")
            for start in range(0, len(tokens), 1024):
                delta = np.asarray(keys[start:start + 1024], dtype=np.float64) - query
                ds = np.einsum("ij,ij->i", delta, delta)
                distances = np.concatenate((distances, ds))
                targets = np.concatenate((targets, tokens[start:start + 1024]))
                origin = np.concatenate((origin, np.full(len(ds), len(sources) - 1, dtype=np.int64)))
                rows = np.concatenate((rows, np.arange(start, start + len(ds), dtype=np.int64)))
                # Stable ties: previous neighbors first, then source hash and row position.
                keep = np.argsort(distances, kind="stable")[:k]
                distances, targets, origin, rows = distances[keep], targets[keep], origin[keep], rows[keep]
        probability = np.zeros(self.identity["vocab_size"], dtype=np.float64)
        if len(targets):
            weights = np.exp(-(distances - distances[0]) / temperature)
            weights /= weights.sum()
            np.add.at(probability, targets, weights)
        return probability, {"revision": view["revision"], "neighbors": len(targets),
                             "nearest_squared_l2": float(distances[0]) if len(targets) else None,
                             "nearest": {"source_hash": sources[int(origin[0])], "row": int(rows[0])} if len(targets) else None}

    def continuation(self, view, source_hash, row, length):
        """The stored next tokens of one source from a row on, in order — a shard keeps a source's targets contiguous, so the
        tokens after a neighbor's target are the text that followed it (a draft for speculative decoding, never an answer by itself)."""
        d = view["documents"].get(source_hash)
        if d is None or length < 1:
            return []
        tokens = self._array(d, "tokens")
        return [int(t) for t in tokens[row:row + length]]


def interpolate(logits, memory_probability, weight):
    """Mix probabilities, not logits. An empty datastore returns the exact base distribution even at weight=1."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("kNN interpolation weight must be in [0,1]")
    logits = np.asarray(logits, dtype=np.float64)
    memory_probability = np.asarray(memory_probability, dtype=np.float64)
    if logits.ndim != 1 or logits.shape != memory_probability.shape or not np.isfinite(logits).all():
        raise ValueError("incompatible or nonfinite token distributions")
    if not np.isfinite(memory_probability).all() or (memory_probability < 0).any():
        raise ValueError("invalid kNN probabilities")
    base = np.exp(logits - logits.max())
    base /= base.sum()
    if not memory_probability.any() or weight == 0:
        return base, base
    if not np.isclose(memory_probability.sum(), 1):
        raise ValueError("kNN probabilities must sum to one")
    return (1 - weight) * base + weight * memory_probability, base


class TransformersCore:
    """Frozen Hugging Face causal LM. Real last-layer context states and next-token logits from the same forward pass."""
    def __init__(self, model, tokenizer, model_id="local", device="cpu"):
        import torch
        self.lock = threading.RLock()
        self.torch, self.model, self.tokenizer = torch, model.to(device).eval(), tokenizer
        self.model.requires_grad_(False)
        self.device = device
        self.window = min(int(getattr(model.config, "max_position_embeddings", 2048)), 131072)
        tokenizer_json = tokenizer.backend_tokenizer.to_str() if hasattr(tokenizer, "backend_tokenizer") else _json(tokenizer.get_vocab())
        self.identity = {"model": model_id, "weights_sha256": self.weight_hash(),
                         "config_sha256": hashlib.sha256(model.config.to_json_string(use_diff=False).encode()).hexdigest(),
                         "tokenizer_sha256": hashlib.sha256((tokenizer_json + _json(tokenizer.special_tokens_map)).encode()).hexdigest(),
                         "key_function": "transformers.hidden_states[-1].float32", "dimension": int(model.config.hidden_size),
                         "vocab_size": int(model.config.vocab_size), "context_window": self.window}

    @classmethod
    def load(cls, model_id, revision=None, device="cpu", dtype="float32"):
        """float32 by default (measured 14.11.2026, Qwen3-0.6B on CPU): in the checkpoint's bfloat16 the logits of the same position
        differed by up to 0.625 between sequence lengths (8 of 16 lengths), in float32 by 0.0 (0 of 16). kNN keys are written at
        one length and queried at another, and speculative drafts verify at a longer length — both need reproducible states."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=False, dtype=getattr(torch, dtype))
        return cls(model, tokenizer, model_id, device)

    def weight_hash(self):
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8).numpy()
            h.update(memoryview(raw).cast("B"))
        return h.hexdigest()

    def encode(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("context must contain at least one token")
        if len(ids) > self.window:
            raise ValueError("context exceeds model window; split the source explicitly")
        return ids

    def forward_all(self, token_ids):
        """Context states and next-token logits at every position of one forward pass (speculative verification)."""
        if not 0 < len(token_ids) <= self.window:
            raise ValueError("context exceeds model window or is empty")
        with self.lock, self.torch.inference_mode():
            out = self.model(input_ids=self.torch.tensor([token_ids], device=self.device),
                             output_hidden_states=True, use_cache=False, return_dict=True)
        return out.hidden_states[-1][0].float().cpu().numpy(), out.logits[0].float().cpu().numpy()

    def forward(self, token_ids):
        states, logits = self.forward_all(token_ids)
        return states, logits[-1]

    def chat_prompt(self, messages):
        if not isinstance(messages, list) or not messages or any(m.get("role") not in ("system", "user", "assistant", "tool")
                or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("messages must contain textual system/user/assistant/tool messages")
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nassistant:"


class KNNLanguageModel:
    """Shared frozen computation core + replaceable per-instance token datastore."""
    def __init__(self, core, store, *, weight=0.5, k=32, temperature=1.0):
        if core.identity != store.identity:
            raise ValueError("model and datastore identity mismatch")
        if not math.isfinite(weight) or not 0 <= weight <= 1 or not isinstance(k, int) or not 1 <= k <= 4096:
            raise ValueError("invalid interpolation weight or neighbor count")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("retrieval temperature must be finite and positive")
        self.core, self.store = core, store
        self.weight, self.k, self.temperature = weight, k, temperature
        self.lock = getattr(core, "lock", threading.RLock())

    def remember(self, source, prompt, continuation):
        """Store only continuation targets, with causal prefix states; atomically replace this source's previous experience."""
        prefix = self.core.encode(prompt)
        full = self.core.encode(prompt + continuation)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError("prompt must end at a tokenizer boundary and continuation must add tokens")
        with self.lock:
            states, _logits = self.core.forward(full)
        return self.store.replace(source, states[len(prefix) - 1:-1], full[len(prefix):])

    def predict(self, prompt, *, view=None, exclude_sources=()):
        return self._predict_ids(self.core.encode(prompt), view=view, exclude_sources=exclude_sources)

    def _predict_ids(self, ids, *, view=None, exclude_sources=()):
        with self.lock:
            states, logits = self.core.forward(ids)
        memory, info = self.store.distribution(states[-1], k=self.k, temperature=self.temperature, view=view,
                                                exclude_sources=exclude_sources)
        mixed, base = interpolate(logits, memory, self.weight)
        return {"probabilities": mixed, "base_probabilities": base, "memory_probabilities": memory, **info,
                "token": int(mixed.argmax()), "base_token": int(base.argmax())}

    def evaluate(self, rows):
        """Teacher-forced next-token log loss on explicitly held-out references; exclude each reference's source from retrieval.
        A fixed revision and interpolation settings are used throughout. Source exclusion cannot detect paraphrased duplicate sources.
        """
        import time
        started = time.monotonic()
        view, base_loss, mixed_loss, count, tasks = self.store.view(), 0.0, 0.0, 0, 0
        for row in rows:
            if not isinstance(row.get("source"), str) or not row["source"].strip():
                raise ValueError("evaluation references need a held-out source identifier")
            prefix = self.core.encode(row["prompt"])
            full = self.core.encode(row["prompt"] + row["continuation"])
            if full[:len(prefix)] != prefix or len(full) <= len(prefix):
                raise ValueError("evaluation prompt must end at a token boundary before a nonempty continuation")
            for pos in range(len(prefix), len(full)):
                prediction = self._predict_ids(full[:pos], view=view, exclude_sources=[row["source"]])
                target = full[pos]
                base_loss -= math.log(max(float(prediction["base_probabilities"][target]), np.finfo(float).tiny))
                mixed_loss -= math.log(max(float(prediction["probabilities"][target]), np.finfo(float).tiny))
                count += 1
            tasks += 1
        if not count:
            raise ValueError("evaluation needs nonempty reference continuations")
        return {"metric": "held_out_next_token_log_loss", "revision": view["revision"], "tasks": tasks, "tokens": count,
                "base_nll": base_loss / count, "knn_nll": mixed_loss / count, "difference_nll": (mixed_loss - base_loss) / count,
                "base_perplexity": math.exp(min(700, base_loss / count)), "knn_perplexity": math.exp(min(700, mixed_loss / count)),
                "weight": self.weight, "neighbors": self.k, "retrieval_temperature": self.temperature,
                "seconds": time.monotonic() - started, "source_exclusion": True,
                "datastore_sha256": hashlib.sha256(_json(view).encode()).hexdigest()}

    def generate(self, prompt, steps=64, stop=(), draft=0):
        """Greedy decoding with explicit resource budget. Report budget exhaustion, never pretend it was a complete answer.
        draft > 0: exact retrieval-based speculative decoding (REST arXiv 2311.08252, NEST arXiv 2405.19325, without NEST's relaxed
        acceptance): the nearest neighbor's stored continuation is a draft, one forward pass verifies it, and a drafted token is
        kept only if it IS the greedy choice of the mixed distribution — the text equals draft=0 up to floating-point differences
        between sequence shapes (PyTorch: batched vs. sliced results may differ), with fewer forward passes."""
        if not isinstance(draft, int) or not 0 <= draft <= 64:
            raise ValueError("draft length must be 0–64 tokens")
        if draft:
            return self._generate_drafted(prompt, steps, stop, draft)
        if not isinstance(steps, int) or not 1 <= steps <= 4096:
            raise ValueError("decoding step budget must be 1–4096")
        ids, generated, view = self.core.encode(prompt), [], self.store.view()
        eos = self.core.tokenizer.eos_token_id
        finish = "budget"
        for _ in range(steps):
            with self.lock:
                states, logits = self.core.forward(ids)
            memory, _info = self.store.distribution(states[-1], k=self.k, temperature=self.temperature, view=view)
            mixed, _base = interpolate(logits, memory, self.weight)
            token = int(mixed.argmax())
            if token == eos:
                finish = "stop"
                break
            generated.append(token)
            text = self.core.tokenizer.decode(generated, skip_special_tokens=True)
            if any(s and s in text for s in stop):
                text = text[:min(text.index(s) for s in stop if s and s in text)]
                return {"text": text, "finish_reason": "stop", "revision": view["revision"], "tokens": generated,
                        "datastore_sha256": hashlib.sha256(_json(view).encode()).hexdigest()}
            ids.append(token)
            if len(ids) >= self.core.window:
                finish = "context"
                break
        return {"text": self.core.tokenizer.decode(generated, skip_special_tokens=True), "finish_reason": finish,
                "revision": view["revision"], "tokens": generated, "datastore_sha256": hashlib.sha256(_json(view).encode()).hexdigest()}

    def _generate_drafted(self, prompt, steps, stop, draft):
        if not isinstance(steps, int) or not 1 <= steps <= 4096:
            raise ValueError("decoding step budget must be 1–4096")
        ids, generated, view = self.core.encode(prompt), [], self.store.view()
        eos = self.core.tokenizer.eos_token_id
        proposal, finish, passes = [], "budget", 0
        drafted = accepted_from_memory = 0
        spans = []  # attribution: which stored source supplied which generated tokens
        digest = hashlib.sha256(_json(view).encode()).hexdigest()
        dtype = str(getattr(getattr(self.core, "model", None), "dtype", "float32")).replace("torch.", "")
        exact = dtype == "float32"
        while len(generated) < steps:
            proposal = proposal[:max(0, min(steps - len(generated) - 1, self.core.window - len(ids)))]
            with self.lock:
                states, logits = self.core.forward_all(ids + [t for t, _src in proposal])
            passes += 1
            drafted += len(proposal)
            base = len(ids) - 1
            new, nearest = [], None
            for j in range(len(proposal) + 1):
                memory, info = self.store.distribution(states[base + j], k=self.k, temperature=self.temperature, view=view)
                mixed, _base = interpolate(logits[base + j], memory, self.weight)
                token, nearest = int(mixed.argmax()), info.get("nearest")
                if token == eos:
                    finish = "stop"
                    break
                from_memory = j < len(proposal) and token == proposal[j][0]
                new.append((token, proposal[j][1] if from_memory else None))
                if not from_memory or len(generated) + len(new) >= steps:
                    break
            # Token by token, in the order plain decoding checks: append, stop string, context window. An accepted span must not
            # run past a stop or the window (Codex's review, 14.11.2026: it returned 8 tokens where plain decoding returned 4).
            text, done = self.core.tokenizer.decode(generated, skip_special_tokens=True), False
            for token, src in new:
                generated.append(token)
                if src:
                    accepted_from_memory += 1
                    if spans and spans[-1]["source_hash"] == src and spans[-1]["end"] == len(generated) - 1:
                        spans[-1]["end"] += 1
                    else:
                        spans.append({"source_hash": src, "start": len(generated) - 1, "end": len(generated)})
                text = self.core.tokenizer.decode(generated, skip_special_tokens=True)
                if any(s_ and s_ in text for s_ in stop):
                    text = text[:min(text.index(s_) for s_ in stop if s_ and s_ in text)]
                    finish, done = "stop", True
                    break
                ids.append(token)
                if len(ids) >= self.core.window:
                    finish, done = "context", True
                    break
            if done or finish == "stop":
                break
            # Next draft: the nearest neighbor of the last verified state predicted the token just emitted; what followed it in
            # its source is the proposal. If that neighbor predicted something else, no draft (one token per pass, as without).
            proposal = []
            if nearest and new and finish != "stop":
                cont = self.store.continuation(view, nearest["source_hash"], nearest["row"], draft + 1)
                if cont and cont[0] == new[-1][0]:
                    proposal = [(t, nearest["source_hash"]) for t in cont[1:]]
        else:
            text = self.core.tokenizer.decode(generated, skip_special_tokens=True)
        return {"text": text, "finish_reason": finish, "revision": view["revision"], "tokens": generated, "datastore_sha256": digest,
                "speculative": {"forward_passes": passes, "drafted": drafted, "accepted_from_memory": accepted_from_memory,
                                "tokens_per_pass": round(len(generated) / passes, 3) if passes else None, "spans": spans,
                                "acceptance": ("exact greedy match: equals draft=0 up to floating-point differences between sequence "
                                               "shapes (measured 72/72 on Qwen3-0.6B, float32, CPU; not a universal guarantee)" if exact else
                                               f"greedy match in {dtype}: near-ties may differ from draft=0 (length-dependent rounding)")}}
