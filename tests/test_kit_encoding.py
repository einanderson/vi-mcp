"""0.68.1: every text file a fine-tuning kit reads or writes names its encoding. Without it Python on Windows uses the ANSI code page
(cp1252) instead of UTF-8 — found by a run on a Windows 11 GPU host (17.11.2026), which needed PYTHONUTF8=1 to read the kit."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["vi_mcp/training_runtime.py", "vi_mcp/finetune.py", "vi_mcp/finetune_embed.py", "vi_mcp/training_eval.py",
         "clingo_object/runtime.py"]
NOT_FILES = {"op", "opener", "os", "gzip", "tarfile", "zipfile", "io"}  # urllib openers and binary-only modules


def _missing(src: str) -> list[int]:
    out = []
    for n in ast.walk(ast.parse(src)):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in NOT_FILES:
            continue
        if not ((isinstance(f, ast.Name) and f.id == "open") or (isinstance(f, ast.Attribute) and f.attr in ("open", "read_text", "write_text"))):
            continue
        if any(k.arg == "encoding" for k in n.keywords):
            continue
        mode = [a.value for a in (n.args[1:2] if isinstance(f, ast.Name) else n.args[0:1] if f.attr == "open" else [])
                if isinstance(a, ast.Constant)] + [k.value.value for k in n.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
        if any("b" in str(m) for m in mode):
            continue
        out.append(n.lineno)
    return out


def _sources():
    for rel in FILES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        yield rel, src
        for n in ast.walk(ast.parse(src)):  # the kit scripts finetune.py writes are string constants
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                body = n.value.value
                if "import " in body and len(body) > 300:
                    try:
                        ast.parse(body)
                    except SyntaxError:
                        continue
                    yield f"{rel}:{getattr(n.targets[0], 'id', '?')}", body


def test_every_kit_file_access_names_its_encoding():
    bad = {name: lines for name, src in _sources() if (lines := _missing(src))}
    assert not bad, bad
