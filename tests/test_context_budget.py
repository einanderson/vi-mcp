"""Context budget (vi_mcp/context_budget.py, ported from IT-AIKB): the reranker picks which chunks reach the reader. The cases of the
original suite for the selection itself; the wiring into vi_recall is tested with the collect mode."""
from vi_mcp import context_budget as cb

SECTION_A = "Print system: extend tablespace > Check current usage"
SECTION_B = "Print system: extend tablespace > Enlarge tablespace"


def _text(section: str, body: str) -> str:
    return f"{section}\n\n{body}"


def test_section_of_reads_the_heading_path_of_a_chunk():
    assert cb.section_of(_text(SECTION_A, "db@host> stats")) == SECTION_A
    assert cb.section_of("just one paragraph without heading") == ""


def test_score_gap_cutoff_cuts_at_the_largest_drop():
    assert cb.score_gap_cutoff([2.0, 1.8, 1.7, -6.0, -6.5, -7.0, -7.1, -8.0, -9.0, -9.5]) == 1.7
    assert cb.score_gap_cutoff([1.0, 0.5]) is None  # too few to tell a drop from noise


def test_neighbours_come_along_only_inside_the_same_section():
    pieces = [cb.Piece(0, 0, _text(SECTION_A, "a" * 350), SECTION_A), cb.Piece(0, 1, _text(SECTION_A, "b" * 350), SECTION_A),
              cb.Piece(0, 2, _text(SECTION_B, "c" * 350), SECTION_B)]
    selection = cb.select(pieces, [0.0, 2.0, -9.0], {0: 10}, min_tokens=10_000, max_tokens=10_000, per_entry_core=1)
    assert selection.keep == {0: {0, 1}}  # piece 2 is the next section, not a continuation
    assert selection.dropped["page_cap"] == 1


def test_a_page_cannot_eat_the_budget_and_forced_host_chunks_go_first():
    pieces = [cb.Piece(0, k, f"page zero chunk {k} " * 20) for k in range(6)] + [cb.Piece(1, 0, "host03 storage -n " * 10)]
    selection = cb.select(pieces, [5.0, 4.9, 4.8, 4.7, 4.6, 4.5, -3.0], {0: 10, 1: 10}, min_tokens=10_000, max_tokens=10_000,
                          per_entry_core=2, forced={6})
    assert selection.keep[0] == {0, 1}
    assert selection.keep[1] == {0}  # low score, but it names the host of the question


def test_the_budget_is_a_hard_ceiling_including_headers():
    pieces = [cb.Piece(e, 0, "x" * 3500) for e in range(5)]
    selection = cb.select(pieces, [5.0, 4.0, 3.0, 2.0, 1.0], {e: 100 for e in range(5)}, min_tokens=3000, max_tokens=3000, per_entry_core=4)
    assert selection.tokens <= 3000 and set(selection.keep) == {0, 1}
    assert selection.dropped["over_budget"] == 3


def test_below_the_score_drop_nothing_is_added_once_the_minimum_is_reached():
    pieces = [cb.Piece(e, 0, "y" * 350) for e in range(6)]
    selection = cb.select(pieces, [3.0, 2.9, 2.8, -8.0, -8.1, -8.2], {}, min_tokens=150, max_tokens=10_000, per_entry_core=4)
    assert set(selection.keep) == {0, 1, 2}
    assert selection.dropped["below_cutoff"] == 3


def test_render_keeps_document_order_and_marks_gaps():
    texts = ["t0", "t1", "t2", "t3", "t4"]
    assert cb.render(texts, {3, 1, 2}) == "[…]\n\nt1\n\nt2\n\nt3\n\n[…]"
    assert cb.render(texts, {0, 4}) == "t0\n\n[…]\n\nt4"
