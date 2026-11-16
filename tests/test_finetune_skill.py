"""0.68.0: the fine-tuning skill — one source (finetune_skill.SKILL), reachable as vi_finetune(action="skill") and `vi-mcp finetune skill`,
written for small instruction-following models: every step names its check, decisions are tables, the rules come first."""
import re
from pathlib import Path

from vi_mcp import finetune, finetune_skill, tools

ROOT = Path(__file__).resolve().parents[1]


def test_the_tool_and_the_cli_give_the_one_source(gd, capsys):
    assert tools.call_tool("vi_finetune", {"action": "skill"}, "a", False) == finetune_skill.SKILL
    out = tools.call_tool("vi_finetune", {"action": "skill", "base_model": "Qwen/Qwen3-1.7B"}, "a", False)
    assert "Qwen/Qwen3-1.7B" in out and "m='Qwen/Qwen3-1.7B'" in out and "MODEL" not in out
    assert finetune.main(["skill", "--file"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("---\nname: finetune\n") and printed.endswith(finetune_skill.SKILL)


def test_docs_copy_is_current():
    assert (ROOT / "docs/FINETUNING_SKILL.md").read_text() == finetune_skill.SKILL, "regenerate: vi-mcp finetune skill > docs/FINETUNING_SKILL.md"


def test_written_for_small_models():
    s = finetune_skill.SKILL
    steps = re.split(r"\n## Step \d+ — ", s)[1:]
    assert len(steps) == 13  # steps 0..12
    for body in steps:
        body = body.split("\n## ")[0]
        assert "OK if" in body, body.splitlines()[0]  # every step says how to know it worked
    assert s.index("## Rules you must never break") < s.index("## Step 0")
    for rule in ("without a clear \"yes\"", "max_tokens", "Never train on the evaluation data", "vi_learn"):
        assert rule in s
    assert "| model size |" in s and "| what you see | cause | what to do |" in s  # decisions as tables


def test_commands_match_the_kit():
    s = finetune_skill.SKILL
    for flag in ("--tokenizer-only", "--preflight", "--lora-rank", "--qlora", "--skip-oversize", "--fp32"):
        assert flag in s
    runtime = (ROOT / "vi_mcp/training_runtime.py").read_text()
    for flag in ("--tokenizer-only", "--preflight", "--lora-rank", "--qlora", "--skip-oversize", "--fp32", "--from-base"):
        assert f'"{flag}"' in runtime
    assert "phil_answers.py --adapter out/philosophy" in s and "--adapter" in finetune.PHIL_ANSWERS_PY
    assert "python phil_score.py\n" in s and "python phil_score.py   ->" in finetune.PHIL_SCORE_PY
