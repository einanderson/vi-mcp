"""Questions between conversations of ONE instance (Unreleased): every Claude Code session of a machine uses the instance claude-code,
so vi_ask to the own instance reaches its other conversations; never the asking one, answers go back to it. A Claude Code session is
two conversations (the agent's MCP client and its hook); the hook's announcement of the agent's call pairs them (session.py)."""
import pytest

from vi_mcp import intuition, questions, session, tools
from vi_mcp import memory as g

I = "claude-code"


@pytest.fixture(autouse=True)
def _fresh_pairing():
    for d in (session._announced, session._agent_of, session._hook_of):
        d.clear()
    session._announcers.clear()
    session.pairing_counts.update(gekoppelt=0, mehrdeutig=0)
    yield


def call(s, name, args=None):
    with session.on_behalf(s):
        return tools.call_tool(name, args or {}, s.instance, True)


def hints(s):
    with session.on_behalf(s):
        return intuition.before("Bash", {"command": "true"}, s.instance)["hinweise"]


def asked(s):
    return [h for h in hints(s) if h.startswith("Question from")]


def answered(s):
    return [h for h in hints(s) if h.startswith("Answer from")]


def _qid(reply: str) -> str:
    return reply.split()[1]


ASK = {"to": I, "text": "Which session is changing vi_mcp/questions.py right now?", "source": "two sessions in one repository"}


def test_two_conversations_ask_deliver_answer_and_back(gd):
    a, b = session.begin(I, "claude-code 1"), session.begin(I, "claude-code 1")
    r = call(a, "vi_ask", ASK)
    qid = _qid(r)
    assert f"from conversation {a.serial}" in r and "other conversations of this instance" in r
    d = g.load(qid).data
    assert d["from_conversation"] == a.serial and d["to"] == I and d["zugestellt_gespraeche"] == []
    assert not asked(a)  # never to the asker
    got = asked(b)
    assert len(got) == 1 and f"Question from {I}#{a.serial}" in got[0] and qid in got[0]
    assert not asked(b)  # once per conversation
    with pytest.raises(tools.Rejected, match="your own question"):
        call(a, "vi_answer", {"id": qid, "text": "I answer myself here", "evidence": "nothing at all"})
    call(b, "vi_answer", {"id": qid, "text": "This one, on branch ask-same-instance", "evidence": "git worktree list"})
    assert not answered(b)  # the answer goes back to the asker only
    back = answered(a)
    assert len(back) == 1 and f"Answer from {I}#{b.serial}" in back[0] and "ask-same-instance" in back[0]
    assert not answered(a)


def test_the_hook_of_the_asker_never_gets_its_own_question(gd):
    a, a_hook = session.begin(I, "claude-code 1"), session.begin(I, "claude-code-hook 1")
    b, b_hook = session.begin(I, "claude-code 1"), session.begin(I, "claude-code-hook 1")
    call(a_hook, "vi_before", {"tool": "vi_ask", "args": ASK})  # the hook announces the agent's call …
    qid = _qid(call(a, "vi_ask", ASK))  # … and the agent's own call pairs both
    assert session.canonical(a_hook.serial) == a.serial and g.load_keyed("conversation", a.serial).data["partner"] == a_hook.serial
    assert not asked(a_hook)  # the asker's hook stands for the asker
    assert qid in asked(b_hook)[0]
    ans = {"id": qid, "text": "Session b holds the file open in its worktree", "evidence": "lsof on the file"}
    call(b_hook, "vi_before", {"tool": "vi_answer", "args": ans})
    call(b, "vi_answer", ans)
    assert g.load(qid).data["answers"][0]["conversation"] == b.serial
    assert not asked(b)  # b was shown it through its hook before the pairing: shown
    assert any("lsof" in h for h in answered(a_hook))  # the answer reaches the asker through its hook


def test_one_conversation_can_be_targeted(gd):
    a, b, c = (session.begin(I, "claude-code 1") for _ in range(3))
    qid = _qid(call(a, "vi_ask", {**ASK, "to": f"{I}#{c.serial}"}))
    assert g.load(qid).data["to_conversation"] == c.serial
    assert not asked(b) and not asked(a)
    assert len(asked(c)) == 1
    with pytest.raises(tools.Rejected, match="not to this one"):
        call(b, "vi_answer", {"id": qid, "text": "b answers although c was asked", "evidence": "none given here"})
    call(c, "vi_answer", {"id": qid, "text": "c answers as it was asked to", "evidence": "the question named c"})
    assert answered(a)
    with pytest.raises(tools.Rejected, match="own conversation"):
        call(a, "vi_ask", {**ASK, "to": f"{I}#{a.serial}"})
    with pytest.raises(tools.Rejected, match="serial number"):
        call(a, "vi_ask", {**ASK, "to": f"{I}#abc"})
    with pytest.raises(tools.Rejected, match="serial number"):
        call(a, "vi_ask", {**ASK, "to": f"alle#{c.serial}"})


def test_a_different_instance_is_unchanged(gd):
    a = session.begin("a", "claude-code 1")
    b1, b2 = session.begin("b", "claude-code 1"), session.begin("b", "claude-code 1")
    qid = _qid(call(a, "vi_ask", {**ASK, "to": "b"}))
    assert "zugestellt_gespraeche" not in g.load(qid).data
    assert len(asked(b1)) == 1 and not asked(b2)  # once per instance, as before
    call(b2, "vi_answer", {"id": qid, "text": "answered from the other conversation of b", "evidence": "same instance b"})
    assert answered(a)


def test_an_answer_to_an_ended_conversation_is_not_lost(gd):
    a, b, a2 = (session.begin(I, "claude-code 1") for _ in range(3))
    qid = _qid(call(a, "vi_ask", ASK))
    session.finish(a.id)
    call(b, "vi_answer", {"id": qid, "text": "answer after the asker has gone", "evidence": "session a finished"})
    got = answered(a2)
    assert got and f"asking conversation {a.serial} has ended" in got[0]


def test_vi_questions_shows_both_sides_and_the_own_conversation(gd):
    a, b = session.begin(I, "claude-code 1"), session.begin(I, "claude-code 1")
    qid = _qid(call(a, "vi_ask", ASK))
    la, lb = call(a, "vi_questions"), call(b, "vi_questions")
    assert f"This conversation: {I}#{a.serial}" in la and f'vi_ask(to="{I}#{a.serial}")' in la
    assert "Questions to" in la and f"Own questions: 1" in la and qid in la.split("Own questions")[1]
    assert qid in lb.split("Own questions")[0] and "Own questions: 0" in lb  # to b, not b's own
    assert f"from {I}#{a.serial}" in call(b, "vi_questions", {"id": qid})


def test_a_question_without_a_conversation_to_the_own_instance_is_refused(gd):
    with pytest.raises(ValueError, match="needs a conversation"):
        questions.ask(I, I, "a question from no conversation at all", [], "")


def test_two_hooks_announcing_the_same_call_pair_nothing(gd):
    a, h1, h2 = session.begin(I, "claude-code 1"), session.begin(I, "claude-code-hook 1"), session.begin(I, "claude-code-hook 1")
    for h in (h1, h2):
        call(h, "vi_before", {"tool": "vi_recall", "args": {"filter": "host:web1"}})
    call(a, "vi_recall", {"filter": "host:web1"})
    assert session.pairing_of(a.serial) is None and session.pairing_counts["mehrdeutig"] == 1
    assert session.fingerprint("vi_ask", {"text": "x" * 3000}) == session.fingerprint("vi_ask", {"text": "x" * 2000})  # the hook cuts
