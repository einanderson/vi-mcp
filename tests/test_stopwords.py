"""fts-stopwords (15.11.2026): function words leave a free-text query, never the index; a query of stop words only stays as typed."""
import hashlib

from vi_mcp import memory as g
from vi_mcp import stopwords as sw


def test_the_lists_are_the_published_snowball_lists_without_additions():
    """German https://snowballstem.org/algorithms/german/stop.txt, English .../english/stop.txt (fetched 15.11.2026), BSD-3-Clause.
    A change to either list is a new decision with a new measurement, not an edit."""
    assert len(sw.GERMAN) == 231 and len(sw.ENGLISH) == 174
    assert hashlib.sha256(" ".join(sorted(sw.GERMAN)).encode()).hexdigest() == "60f2df7346d68de44d83973718a38794a900d92ca4909aeff0b0f3e0fe4dc8e5"
    assert hashlib.sha256(" ".join(sorted(sw.ENGLISH)).encode()).hexdigest() == "760af36a43a49c495117ee88e2666999bf4c114fb751f46ba4ee13b40a3edf5d"


def test_content_words_drop_function_words_and_never_empty_a_query():
    assert sw.content_words(["Wie", "tausche", "ich", "das", "Zertifikat", "aus"]) == ["tausche", "Zertifikat"]
    assert sw.content_words(["How", "do", "I", "renew", "the", "certificate"]) == ["renew", "certificate"]
    assert sw.content_words(["wie", "ist", "das"]) == ["wie", "ist", "das"]  # only stop words: unchanged
    assert sw.content_words([]) == []


def test_a_natural_question_finds_the_runbook_and_a_stop_word_query_still_searches(gd):
    runbook = g.new_object("insight", "Zertifikat tauschen auf dem Webserver: neues Zertifikat einspielen, Dienst neu laden", ["topic:tls"])
    chatter = g.new_object("insight", "Wie ist das bei dir, was machst du und wie geht es dir", ["topic:smalltalk"])
    question = "Wie tausche ich das Zertifikat auf dem Webserver"
    hits = [o.id for _s, o, _w in g.recall(question, limit=5, or_fallback=True, viewer=None)]
    assert hits[0] == runbook.id and chatter.id not in hits
    token = g.stop_words_off.set(True)  # the old behaviour: every word of the question in the AND set, then the OR fallback
    try:
        old = [o.id for _s, o, _w in g.recall(question, limit=5, or_fallback=True, viewer=None)]
    finally:
        g.stop_words_off.reset(token)
    assert chatter.id in old  # it matched only on "wie", "ist", "das" — what the removal ends
    only_stop = [o.id for _s, o, _w in g.recall("wie ist das", limit=5, or_fallback=True, viewer=None)]
    assert only_stop and only_stop[0] == chatter.id  # never emptied: the words as typed are searched
