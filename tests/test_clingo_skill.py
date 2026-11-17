"""0.69.4: skills live in the server as knowledge_kind:skill — extensible with vi_learn, inherited by every AI (vi_capabilities). The
initial skill clingo teaches only what the engine does: every example runs through the real vi_logic tool."""
import pytest

pytest.importorskip("clingo")

from vi_mcp import capabilities, clingo_skill, knowledge_skills, tools  # noqa: E402
from vi_mcp import memory as g  # noqa: E402


@pytest.mark.parametrize("ex", clingo_skill.EXAMPLES, ids=[e["name"] for e in clingo_skill.EXAMPLES])
def test_every_example_of_the_skill_is_what_the_engine_answers(gd, ex):
    out = tools.call_tool("vi_logic", dict(ex["args"]), "a", False)
    for line in ex["expect"]:
        assert line in out, (line, out)


def test_the_measured_mistakes_fail_or_stay_silent_as_the_skill_says(gd):
    for wrong, happens, _fix in clingo_skill.PITFALLS:
        src = wrong.split(" and later ")[0]
        if happens.startswith("SILENT"):
            out = tools.call_tool("vi_logic", {"action": "check", "source": src}, "a", False)
            assert "state: satisfied" in out, (wrong, out)
            continue
        with pytest.raises(tools.Rejected) as e:
            tools.call_tool("vi_logic", {"action": "check", "source": src}, "a", False)
        quoted = happens.split('"')[1]
        assert quoted in str(e.value), (wrong, str(e.value))


def test_the_skill_text_names_every_example_and_shows_english_states():
    text = clingo_skill.skill()
    assert text.startswith("# Skill: clingo") and all(e["name"] in text for e in clingo_skill.EXAMPLES)
    assert "state: contradiction" in text and "widerspruch" not in text


def test_initial_skills_are_written_at_start_listed_first_and_refreshed(gd):
    assert set(knowledge_skills.ensure().values()) == {"created"}
    assert set(knowledge_skills.ensure().values()) == {"current"}
    o = g.load(knowledge_skills.object_id("clingo"))
    assert o.instance == g.SERVER_INSTANCE and {"knowledge_kind:skill", "docs:clingo", "origin:builtin"} <= set(o.hashtags)
    assert o.data["content"] == clingo_skill.skill()
    capabilities._cache.clear()
    rows = {r["skill"]: r for r in capabilities.table("a")}
    assert {"clingo", "finetune", "philosophy"} <= set(rows) and rows["clingo"]["tools"] == ["vi_logic"]
    assert rows["clingo"]["description"].startswith("Logic programs with vi_logic")
    o.data["content"] = "old text"; o.data["content_sha256"] = "x"; o.mode = "refuted"; g.save(o)
    assert knowledge_skills.ensure()["clingo"] == "updated"
    o = g.load(knowledge_skills.object_id("clingo"))
    assert o.mode == "active" and o.data["content"] == clingo_skill.skill() and "skill updated" in o.changes[-1]["what"]
    capabilities._cache.clear()


MANUAL = "## Steps\n" + "\n".join(f"{i}. step {i}: run `tool_x --step {i}` and check its output for the word ok" for i in range(1, 12))


def _learn(instance, name="backup-check", content=MANUAL, key=None, extra=()):
    return tools.call_tool("vi_learn", {"text": f"Skill, docs:{name}. Check the nightly backups.", "content": content,
                                        "hashtags": ["knowledge_kind:skill", f"docs:{name}", *extra], "key": key or f"skill:{name}",
                                        "source": "written by hand, 17.11."}, instance, False)


def test_an_ai_adds_and_updates_a_skill_with_vi_learn_and_every_ai_sees_it(gd):
    knowledge_skills.ensure()
    assert _learn("alice").startswith("learned")
    assert _learn("alice", content=MANUAL + "\n12. done").startswith("updated")  # same key, same instance: an update with history
    capabilities._cache.clear()
    table = tools.call_tool("vi_capabilities", {}, "bob", False)
    assert "- backup-check: Check the nightly backups" in table and "- clingo:" in table
    with pytest.raises(tools.Rejected, match="initial skill of the package"):
        _learn("alice", name="clingo")
    with pytest.raises(tools.Rejected, match='key="skill:backup-check"'):
        _learn("alice", key="backup")
    with pytest.raises(tools.Rejected, match="exactly one hashtag docs:"):
        _learn("alice", extra=("docs:other",))
    capabilities._cache.clear()


def test_a_skills_folder_is_imported_through_vi_learn(gd, tmp_path):
    d = tmp_path / "skills" / "stoerung"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: stoerung\ndescription: Handle an outage step by step. Use when a service is down.\n---\n"
                                + MANUAL, encoding="utf-8")
    (d / "references" / "checks.md").write_text("Run `checkmk_get_alerts` first.", encoding="utf-8")
    (tmp_path / "skills" / "empty").mkdir()
    r = knowledge_skills.import_dir(tmp_path / "skills", "skills-import")
    assert r["learned"] == ["stoerung"] and not r["refused"]
    capabilities._cache.clear()
    row = next(x for x in capabilities.table("bob") if x["skill"] == "stoerung")
    o = g.load(row["id"])
    assert o.instance == "skills-import" and "Run `checkmk_get_alerts` first." in o.data["content"] and row["description"] == "Handle an outage step by step."
    assert knowledge_skills.import_dir(tmp_path / "skills", "skills-import")["updated"] == ["stoerung"]
    capabilities._cache.clear()


def test_an_old_german_skill_header_shows_its_real_description():
    assert capabilities._description("Vollständige Doku zu awx-ng: vi_erinnere doku:awx-ng. AWX NextGen admin tools.", "awx-ng") == \
        "AWX NextGen admin tools."
