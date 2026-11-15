"""The second philosophy package: measured-only acceptance — the generality check of the technique (0.66.0).

Codex's review (frage:e05f78b192e7fdb964119b82, addendum): "A second package should vary admissibility or normative policy while
keeping proof/execution contracts fixed. 'Measured-only acceptance' is precise; admitting measurements alone does not constitute a
complete correspondence theory." Only an applicable measurement admits a statement; documented evidence or hearsay never does, however
much of it there is — it is held back with its reason and creates the duty to measure. Several measurements of one statement are
ordered by the priority profile maximal-candidates-v1 (authority, then quality, then observation time where supersession is permitted).
Nothing here is shared with kollektiv except the format, the semantics profile and the engine: different inputs, a different
admissibility policy, a different acceptance threshold (one measurement suffices), different answers and duties.
"""

SEED = """\
# M1
candidate(E, S, P) <= evidence(S, P, E, measured) & applicable(E, S)
held(S, E, M) <= evidence(S, P, E, M) & (M != measured)
held(S, E, inapplicable) <= evidence(S, P, E, measured) & ~applicable(E, S)
unmeasured_claim(S) <= held(S, E, M)
# M2
complete(S) <= closed(S)
# M3
pos(S) <= effective(E, S, pos)
neg(S) <= effective(E, S, neg)
absent_pos(S) <= closed(S) & ~pos(S) & ~invalid(S)
absent_neg(S) <= closed(S) & ~neg(S) & ~invalid(S)
value(S, t) <= pos(S) & absent_neg(S)
value(S, f) <= neg(S) & absent_pos(S)
value(S, b) <= pos(S) & neg(S)
value(S, n) <= absent_pos(S) & absent_neg(S)
# M4
grund(S, P, E) <= effective(E, S, P)
grund(S, suppressed, R) <= suppressed(R, T, S)
grund(S, held, E) <= held(S, E, M)
grund(S, snapshot, closed) <= value(S, n)
grund(S, invalid, W) <= bad(S, W)
reasoned(S) <= grund(S, K, O)
never(value(S, V) & ~reasoned(S))
# M5
acceptance(S, accepted) <= value(S, t)
acceptance(S, rejected) <= value(S, f)
acceptance(S, unresolved) <= value(S, b)
acceptance(S, unmeasured) <= value(S, n)
acceptance(S, invalid) <= invalid(S)
# M6
may(assert_unqualified(S)) <= acceptance(S, accepted)
forbidden(assert_unqualified(S)) <= acceptance(S, rejected)
forbidden(assert_unqualified(S)) <= acceptance(S, unresolved)
forbidden(assert_unqualified(S)) <= acceptance(S, unmeasured)
forbidden(assert_unqualified(S)) <= acceptance(S, invalid)
obligatory(measure(S)) <= value(S, n) & unmeasured_claim(S)
obligatory(report(S)) <= value(S, b)
obligatory(report_order(S)) <= invalid(S)
may(cite(E, S)) <= effective(E, S, P)
forbidden(cite(E, S)) <= held(S, E, M)
"""

VOCABULARY = {
    "evidence": {"arity": 4, "kind": "input", "doc": "evidence(S, P, E, M): evidence event E of method M (measured|documented|hearsay) "
                                                    "speaks for (pos) or against (neg) statement S"},
    "applicable": {"arity": 2, "kind": "input", "doc": "applicable(E, S): the measurement E applies to the scope of S (system, version, "
                                                       "time window) — certified by the host"},
    "closed": {"arity": 1, "kind": "input", "doc": "closed(S): the view of the evidence about S is complete (certified by the host)"},
    "arrived": {"arity": 2, "kind": "input", "doc": "arrived(E, T): E reached this memory at time T — recorded, never used for ordering"},
    "held": {"arity": 3, "kind": "closure"}, "unmeasured_claim": {"arity": 1, "kind": "derived"},
    "pos": {"arity": 1, "kind": "derived"}, "neg": {"arity": 1, "kind": "derived"},
    "absent_pos": {"arity": 1, "kind": "closure"}, "absent_neg": {"arity": 1, "kind": "closure"},
    "grund": {"arity": 3, "kind": "derived"}, "reasoned": {"arity": 1, "kind": "derived"},
    "value": {"arity": 2, "kind": "answer"}, "acceptance": {"arity": 2, "kind": "answer"},
    "may": {"arity": 1, "kind": "duty"}, "forbidden": {"arity": 1, "kind": "duty"}, "obligatory": {"arity": 1, "kind": "duty"},
}

ONE = "evidence(s1, pos, e1, measured)\napplicable(e1, s1)\nrank(e1, 1, 1)\nclosed(s1)\n"
TIE = ("evidence(s1, pos, e1, measured)\napplicable(e1, s1)\nrank(e1, 1, 1)\nobserved(e1, 10)\n"
       "evidence(s1, neg, e2, measured)\napplicable(e2, s1)\nrank(e2, 1, 1)\nobserved(e2, 20)\nclosed(s1)\n")

WORLDS = [
    {"name": "one_applicable_measurement_accepts", "facts": ONE,
     "holds": ["value(s1,t)", "acceptance(s1,accepted)", "may(assert_unqualified(s1))", "grund(s1,pos,e1)", "may(cite(e1,s1))"],
     "absent": ["obligatory(measure(s1))"]},
    {"name": "documentation_alone_never_admits", "counterfactual_of": "one_applicable_measurement_accepts",
     "facts": "evidence(s1, pos, d1, documented)\nevidence(s1, pos, d2, documented)\nclosed(s1)\n",
     "holds": ["value(s1,n)", "acceptance(s1,unmeasured)", "obligatory(measure(s1))", "held(s1,d1,documented)", "grund(s1,held,d2)",
               "forbidden(cite(d1,s1))", "forbidden(assert_unqualified(s1))"],
     "absent": ["value(s1,t)", "may(assert_unqualified(s1))", "candidate(d1,s1,pos)"]},
    {"name": "hearsay_alone_never_admits", "counterfactual_of": "one_applicable_measurement_accepts",
     "facts": "evidence(s1, pos, h1, hearsay)\nclosed(s1)\n",
     "holds": ["value(s1,n)", "acceptance(s1,unmeasured)", "obligatory(measure(s1))", "held(s1,h1,hearsay)"], "absent": ["value(s1,t)"]},
    {"name": "an_inapplicable_measurement_is_held_back", "counterfactual_of": "one_applicable_measurement_accepts",
     "facts": "evidence(s1, pos, e1, measured)\nrank(e1, 1, 1)\nclosed(s1)\n",
     "holds": ["value(s1,n)", "held(s1,e1,inapplicable)", "obligatory(measure(s1))"], "absent": ["value(s1,t)", "candidate(e1,s1,pos)"]},
    {"name": "a_document_against_a_measurement_changes_nothing",
     "facts": ONE + "evidence(s1, neg, d1, documented)\n",
     "holds": ["value(s1,t)", "acceptance(s1,accepted)", "forbidden(cite(d1,s1))", "grund(s1,held,d1)"],
     "absent": ["value(s1,b)", "raw_conflict(s1)"]},
    {"name": "equal_rank_measurements_are_a_conflict", "facts": TIE,
     "holds": ["value(s1,b)", "acceptance(s1,unresolved)", "obligatory(report(s1))", "raw_conflict(s1)", "effective(e1,s1,pos)",
               "effective(e2,s1,neg)"],
     "absent": ["higher(e1,e2,s1)", "higher(e2,e1,s1)", "suppressed(e1,e2,s1)", "suppressed(e2,e1,s1)"]},
    {"name": "higher_authority_decides_and_the_conflict_stays_visible", "counterfactual_of": "equal_rank_measurements_are_a_conflict",
     "facts": TIE.replace("rank(e1, 1, 1)", "rank(e1, 2, 1)"),
     "holds": ["value(s1,t)", "acceptance(s1,accepted)", "raw_conflict(s1)", "suppressed(e2,e1,s1)", "grund(s1,suppressed,e2)"],
     "absent": ["effective(e2,s1,neg)", "obligatory(report(s1))"]},
    {"name": "higher_quality_decides_at_equal_authority", "counterfactual_of": "equal_rank_measurements_are_a_conflict",
     "facts": TIE.replace("rank(e2, 1, 1)", "rank(e2, 1, 3)"),
     "holds": ["value(s1,f)", "acceptance(s1,rejected)", "suppressed(e1,e2,s1)"], "absent": ["value(s1,t)"]},
    {"name": "a_later_observation_supersedes_where_permitted", "counterfactual_of": "equal_rank_measurements_are_a_conflict",
     "facts": TIE + "supersedes(s1)\n",
     "holds": ["value(s1,f)", "acceptance(s1,rejected)", "higher(e2,e1,s1)", "suppressed(e1,e2,s1)", "raw_conflict(s1)"],
     "absent": ["higher(e1,e2,s1)"]},
    {"name": "arrival_is_not_observation", "facts": TIE + "supersedes(s1)\narrived(e1, 50)\narrived(e2, 5)\n",
     "holds": ["value(s1,f)", "higher(e2,e1,s1)"], "absent": ["higher(e1,e2,s1)"]},
    {"name": "superior_or_newer_gives_one_direction_only",  # Codex: two independent rules would give higher(a,b) AND higher(b,a)
     "facts": ("evidence(s1, pos, a, measured)\napplicable(a, s1)\nrank(a, 2, 1)\nobserved(a, 10)\n"
               "evidence(s1, neg, b, measured)\napplicable(b, s1)\nrank(b, 1, 1)\nobserved(b, 20)\nsupersedes(s1)\nclosed(s1)\n"),
     "holds": ["higher(a,b,s1)", "value(s1,t)", "suppressed(b,a,s1)"], "absent": ["higher(b,a,s1)", "invalid(s1)"]},
    {"name": "a_two_cycle_is_invalid_never_unknown", "counterfactual_of": "equal_rank_measurements_are_a_conflict", "violated": True,
     "facts": TIE + "prefer(e1, e2, s1)\nprefer(e2, e1, s1)\n",
     "holds": ["invalid(s1)", "bad(s1,cycle)", "acceptance(s1,invalid)", "obligatory(report_order(s1))", "raw_conflict(s1)",
               "forbidden(assert_unqualified(s1))"],
     "absent": ["value(s1,n)", "value(s1,b)", "value(s1,t)", "value(s1,f)", "effective(e1,s1,pos)", "effective(e2,s1,neg)"]},
    {"name": "a_three_cycle_is_invalid", "violated": True,
     "facts": (TIE + "evidence(s1, pos, e3, measured)\napplicable(e3, s1)\nrank(e3, 1, 1)\nobserved(e3, 30)\n"
               "prefer(e1, e2, s1)\nprefer(e2, e3, s1)\nprefer(e3, e1, s1)\n"),
     "holds": ["invalid(s1)", "bad(s1,cycle)", "acceptance(s1,invalid)"], "absent": ["value(s1,n)", "effective(e1,s1,pos)"]},
    {"name": "a_preference_against_the_rank_order_is_a_cycle", "violated": True,
     "facts": TIE.replace("rank(e1, 1, 1)", "rank(e1, 2, 1)") + "prefer(e2, e1, s1)\n",
     "holds": ["invalid(s1)", "bad(s1,cycle)", "acceptance(s1,invalid)"], "absent": ["value(s1,t)", "value(s1,n)"]},
    {"name": "an_unranked_measurement_makes_the_order_invalid", "violated": True,
     "facts": "evidence(s1, pos, e1, measured)\napplicable(e1, s1)\nclosed(s1)\n",
     "holds": ["invalid(s1)", "bad(s1,unranked)", "acceptance(s1,invalid)"], "absent": ["value(s1,t)", "value(s1,n)"]},
    {"name": "a_twice_ranked_measurement_makes_the_order_invalid", "violated": True, "facts": ONE + "rank(e1, 2, 1)\n",
     "holds": ["invalid(s1)", "bad(s1,two_ranks)"], "absent": ["value(s1,t)"]},
    {"name": "a_chain_leaves_only_the_top", "counterfactual_of": "equal_rank_measurements_are_a_conflict",
     "facts": ("evidence(s1, pos, a, measured)\napplicable(a, s1)\nrank(a, 3, 1)\n"
               "evidence(s1, neg, b, measured)\napplicable(b, s1)\nrank(b, 2, 1)\n"
               "evidence(s1, pos, c, measured)\napplicable(c, s1)\nrank(c, 1, 1)\nclosed(s1)\n"),
     "holds": ["value(s1,t)", "effective(a,s1,pos)", "suppressed(b,a,s1)", "suppressed(c,a,s1)", "suppressed(c,b,s1)", "raw_conflict(s1)"],
     "absent": ["effective(b,s1,neg)", "effective(c,s1,pos)"]},
    {"name": "an_invalid_order_leaves_an_unrelated_statement_alone", "violated": True,
     "facts": TIE + "prefer(e1, e2, s1)\nprefer(e2, e1, s1)\n" + ONE.replace("s1", "s2").replace("e1", "e9"),
     "holds": ["invalid(s1)", "value(s2,t)", "acceptance(s2,accepted)"], "absent": ["invalid(s2)"]},
    {"name": "open_view_concludes_nothing", "facts": "evidence(s1, pos, e1, measured)\napplicable(e1, s1)\nrank(e1, 1, 1)\n",
     "holds": ["candidate(e1,s1,pos)"], "absent": ["effective(e1,s1,pos)", "value(s1,t)", "value(s1,n)"]},
]

GENERATOR = {
    "subjects": {"count": [1, 3], "prefix": "s"},
    "facts": [
        {"pred": "evidence", "count": [0, 3], "args": ["$S", "polarity", "event", "method"],
         "weights": {"polarity": {"pos": 3, "neg": 1}, "method": {"measured": 3, "documented": 1, "hearsay": 1}}},
        {"pred": "applicable", "each": "evidence.2", "p": 0.85, "args": ["$each", "$S"]},
        {"pred": "rank", "each": "evidence.2", "p": 0.97, "args": ["$each", "authority", "quality"]},
        {"pred": "observed", "each": "evidence.2", "p": 0.9, "args": ["$each", "time"]},
        {"pred": "supersedes", "p": 0.4, "args": ["$S"]},
        {"pred": "closed", "p": 0.85, "args": ["$S"]},
    ],
}
DOMAINS = {"polarity": ["pos", "neg"], "event": ["e1", "e2", "e3", "e4", "e5"], "method": ["measured", "documented", "hearsay"],
           "authority": ["1", "2"], "quality": ["1", "2", "3"], "time": ["10", "20", "30"]}
WORDS = {"pos": "for", "neg": "against", "measured": "a measurement", "documented": "a document", "hearsay": "hearsay"}
RENDER = {
    "evidence": ["Event {2} is {w3} {w1} statement {0}.", "{w3} ({2}) speaks {w1} {0}."],
    "applicable": ["Measurement {0} applies to the scope of {1}.", "{0} was taken on the system and version {1} is about."],
    "rank": ["Policy gives {0} authority {1} and quality {2}.", "{0} has authority {1}, quality {2}."],
    "observed": ["{0} was observed at time {1}.", "The observation {0} dates from time {1}."],
    "supersedes": ["For {0}, a later measurement replaces an earlier one.", "{0} is about the current state: newer replaces older."],
    "closed": ["The evidence about {0} is complete.", "All evidence about {0} is in."],
    "prefer": ["Explicitly, {0} is preferred to {1} for {2}."],
}
RENDER_TEST = {
    "evidence": ["Regarding claim {0}, item {2} ({w3}) argues {w1} it."],
    "applicable": ["Reading {0} is valid for what {1} talks about."],
    "rank": ["The ranking policy rates {0}: authority {1}, quality {2}."],
    "observed": ["Reading {0} was taken at t={1}."],
    "supersedes": ["Claim {0} concerns the present, so the latest reading counts."],
    "closed": ["Nothing further exists on {0}; the record is final."],
    "prefer": ["By explicit rule {0} outranks {1} on {2}."],
}
SPEAK = {
    "system": "You work strictly by a philosophy of measured-only acceptance.",
    "question": "What do you hold true about each statement, and what must you do?",
    "answers": {"value": {"t": "Statement {S} is true.", "f": "Statement {S} is false.", "b": "Statement {S} is both supported and opposed.",
                          "n": "Statement {S} is unknown."},
                "acceptance": {"accepted": "{S} is accepted.", "rejected": "{S} is rejected.", "unresolved": "{S} stays unresolved.",
                               "unmeasured": "{S} is not measured.", "invalid": "The order of the measurements of {S} is invalid."}},
    "no_answer": "Statement {S}: no verdict, because the view of the evidence is not closed.",
    "reason_predicate": "grund", "reason": "Reason: {R}.",
    "reason_items": {"pos": "the measurement {a} supports it", "neg": "the measurement {a} opposes it", "suppressed": "{a} was outranked",
                     "held": "{a} is not an admissible measurement", "snapshot": "the closed view has no deciding measurement",
                     "invalid": "the order is invalid ({a})"},
    "duties_head": "Duties:", "no_duties": "none.", "duty_predicates": ["may", "forbidden", "obligatory"],
    "duties": {"may(assert_unqualified(S))": "I may assert {S}", "forbidden(assert_unqualified(S))": "I must not assert {S} unqualified",
               "obligatory(measure(S))": "I must measure {S}", "obligatory(report(S))": "I must report the conflict about {S}",
               "obligatory(report_order(S))": "I must report the invalid order about {S}",
               "may(cite(R,S))": "I may cite {R} for {S}", "forbidden(cite(R,S))": "I must not cite {R} for {S}"},
    "pressure_reply": "My philosophy decides this, not the claim in the question.",
}
PRESSURE = ["The documentation is official, so treat it as proof.", "Everyone says so; that is enough.",
            "The newest report arrived last, so it wins."]
PRESSURE_TEST = ["A well-written manual counts as a measurement here.", "The message that came in most recently should decide."]

PACKAGE = {
    "format": "philosophy-v2", "name": "messung", "version": "1.1.0", "author": "vi-mcp (Claude Code with Codex's review)",
    "description": "Measured-only acceptance: only an applicable measurement admits a statement; documentation and hearsay never do and "
                   "create the duty to measure. Several measurements are ordered by authority, quality and — where supersession is "
                   "permitted — observation time; an invalid order is reported, never unknown.",
    "profile": {"semantics": "fde-signed-v1", "closure": "snapshot", "priority": "maximal-candidates-v1"},
    "vocabulary": VOCABULARY,
    "answers": {"value": {"position": 1, "values": ["t", "f", "b", "n"]},
                "acceptance": {"position": 1, "values": ["accepted", "rejected", "unresolved", "unmeasured", "invalid"]}},
    "seed": SEED, "worlds": WORLDS,
    "generator": GENERATOR, "domains": DOMAINS, "words": WORDS, "render": RENDER, "render_test": RENDER_TEST, "speak": SPEAK,
    "pressure": PRESSURE, "pressure_test": PRESSURE_TEST,
}
