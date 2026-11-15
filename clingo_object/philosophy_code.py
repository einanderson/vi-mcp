"""The philosophy package `code`: architecture rules over the structure facts of a repository (Unreleased).

The user (15.11.2026): learn source code — code structure as clingo-object facts plus an architecture-rules philosophy package. The
input predicates are what vi_mcp.code_facts derives in the importer's ast pass (calls, imports, table writes, raises, tests, history,
literal constants); `vi_mcp.code_facts.bridge(repo)` reads them from the stored fact objects, `direct_facts` straight from a scan.
The seed is generic; what is particular to one project — which tables need the lock, which constant is an alias table of which
registry, which finding is excused and why — is a PROFILE of input facts (PROFILE_VI_MCP for this repository), never a rule.

Default negation only in closure predicates, each over `snapshot(R)`: a synced commit's structure is complete, so "no locked
caller", "not registered", "no test" are known only over it. An excused finding stays visible with its reason (excused/3); the
commits that last touched a function are shown as where to look (last_touched_by/3), never as the cause (post hoc).
"""

SEED = """\
# P1
rule(no_import_cycle)
rule(locked_writes)
rule(english_tools)
rule(no_duplicates)
test_module(M) <= in_module(F, M) & test(F)
production(F) <= in_module(F, M) & snapshot(R) & ~test_module(M)
excused(Q, X, W) <= candidate(Q, X) & exempt(Q, X, W)
unexcused(Q, X) <= candidate(Q, X) & snapshot(R) & ~is_excused(Q, X)
is_excused(Q, X) <= exempt(Q, X, W)
finding(Q, X) <= unexcused(Q, X)
has_finding(Q) <= finding(Q, X)
clean(Q) <= rule(Q) & snapshot(R) & ~has_finding(Q)
status(Q, violated) <= has_finding(Q)
status(Q, ok) <= clean(Q)
# P2
reaches(A, B) <= imports(A, B) & module(B)
reaches(A, C) <= reaches(A, B) & imports(B, C) & module(C)
candidate(no_import_cycle, A) <= reaches(A, A)
grund(no_import_cycle, A, B) <= candidate(no_import_cycle, A) & imports(A, B) & reaches(B, A)
# P3
open_write(F, T) <= writes_unlocked(F, T) & guarded(T) & production(F)
unlocked_caller(F) <= calls(C, F) & snapshot(R) & ~calls_locked(C, F)
covered(F) <= calls_locked(C, F) & snapshot(R) & ~unlocked_caller(F)
uncovered(F) <= open_write(F, T) & snapshot(R) & ~covered(F)
candidate(locked_writes, F) <= uncovered(F)
grund(locked_writes, F, T) <= open_write(F, T) & uncovered(F)
# P4
registered(G, N) <= alias_of(A, G) & const_value(G, N)
dangling(A, E) <= alias_of(A, G) & const_value(A, E) & snapshot(R) & ~registered(G, E)
candidate(english_tools, E) <= dangling(A, E)
grund(english_tools, E, A) <= dangling(A, E)
candidate(english_tools, N) <= alias_of(A, G) & const_key(A, N) & const_value(G, N)
grund(english_tools, N, G) <= alias_of(A, G) & const_key(A, N) & const_value(G, N)
# P5
same_body(F, G) <= body(F, H) & body(G, H) & in_module(F, M) & in_module(G, N) & (M != N) & production(F) & production(G)
candidate(no_duplicates, F) <= same_body(F, G)
grund(no_duplicates, F, G) <= same_body(F, G)
# P6
reasoned(Q, X) <= grund(Q, X, W)
never(finding(Q, X) & ~reasoned(Q, X))
obligatory(fix(Q, X)) <= finding(Q, X)
last_touched_by(Q, X, C) <= finding(Q, X) & changed_in(X, C)
# P7
tested_by(F, T) <= calls(T, F) & test(T)
tested(F) <= tested_by(F, T)
untested(F) <= writes_table(F, T) & guarded(T) & production(F) & snapshot(R) & ~tested(F)
obligatory(add_test(F)) <= untested(F)
"""

_IN = "input"
VOCABULARY = {
    "snapshot": {"arity": 1, "kind": _IN, "doc": "snapshot(R): the facts are the complete structure of repository R at one commit"},
    "module": {"arity": 1, "kind": _IN, "doc": "module(M): M is a Python module of the repository"},
    "imports": {"arity": 2, "kind": _IN, "doc": "imports(A, B): module A imports B when it is loaded"},
    "imports_late": {"arity": 2, "kind": _IN, "doc": "imports_late(A, B): a function of module A imports B when it runs"},
    "defines": {"arity": 2, "kind": _IN, "doc": "defines(D, F): file D defines function or method F"},
    "in_module": {"arity": 2, "kind": _IN, "doc": "in_module(F, M): function F belongs to module M"},
    "calls": {"arity": 2, "kind": _IN, "doc": "calls(F, G): F calls G (resolved through imports, module definitions, self/cls)"},
    "calls_locked": {"arity": 2, "kind": _IN, "doc": "calls_locked(F, G): F calls G inside `with <x>.lock`"},
    "writes_table": {"arity": 2, "kind": _IN, "doc": "writes_table(F, T): F holds SQL that inserts, updates or deletes rows of table T"},
    "writes_unlocked": {"arity": 2, "kind": _IN, "doc": "writes_unlocked(F, T): such SQL of F outside `with <x>.lock`"},
    "raises": {"arity": 2, "kind": _IN, "doc": "raises(F, E): F raises exception E"},
    "test": {"arity": 1, "kind": _IN, "doc": "test(F): F is a test function"},
    "body": {"arity": 2, "kind": _IN, "doc": "body(F, H): H is the hash of F's exact body (without the docstring)"},
    "changed_in": {"arity": 2, "kind": _IN, "doc": "changed_in(F, C): commit C last touched a line of F (git blame)"},
    "const_key": {"arity": 2, "kind": _IN, "doc": "const_key(K, V): literal module constant K has the string key V"},
    "const_value": {"arity": 2, "kind": _IN, "doc": "const_value(K, V): literal module constant K holds the string V (for a list of "
                                                    "records: their name values)"},
    "guarded": {"arity": 1, "kind": _IN, "doc": "guarded(T): table T may be written only under the lock (profile)"},
    "alias_of": {"arity": 2, "kind": _IN, "doc": "alias_of(A, G): constant A maps aliases to names registered in constant G (profile)"},
    "exempt": {"arity": 3, "kind": _IN, "doc": "exempt(Q, X, W): rule Q excuses X for the written reason W (profile)"},
    "rule": {"arity": 1, "kind": "derived"}, "test_module": {"arity": 1, "kind": "derived"}, "production": {"arity": 1, "kind": "closure"},
    "candidate": {"arity": 2, "kind": "derived"}, "excused": {"arity": 3, "kind": "derived"}, "is_excused": {"arity": 2, "kind": "derived"},
    "unexcused": {"arity": 2, "kind": "closure"}, "finding": {"arity": 2, "kind": "derived"}, "has_finding": {"arity": 1, "kind": "derived"},
    "clean": {"arity": 1, "kind": "closure"}, "status": {"arity": 2, "kind": "answer"},
    "reaches": {"arity": 2, "kind": "derived"}, "grund": {"arity": 3, "kind": "derived"},
    "open_write": {"arity": 2, "kind": "derived"}, "unlocked_caller": {"arity": 1, "kind": "closure"}, "covered": {"arity": 1, "kind": "closure"},
    "uncovered": {"arity": 1, "kind": "closure"}, "registered": {"arity": 2, "kind": "derived"}, "dangling": {"arity": 2, "kind": "closure"},
    "same_body": {"arity": 2, "kind": "derived"}, "reasoned": {"arity": 2, "kind": "derived"},
    "last_touched_by": {"arity": 3, "kind": "derived"}, "tested_by": {"arity": 2, "kind": "derived"}, "tested": {"arity": 1, "kind": "derived"},
    "untested": {"arity": 1, "kind": "closure"},
    "obligatory": {"arity": 1, "kind": "duty"},
}

SNAP = "snapshot(r)\n"
CYCLE = SNAP + "module(a)\nmodule(b)\nimports(a, b)\nimports(b, a)\n"
LOCK = SNAP + "module(m)\nin_module(w, m)\nwrites_table(w, objekt)\nwrites_unlocked(w, objekt)\nguarded(objekt)\n"
ALIAS = SNAP + "alias_of(aliases, tools)\nconst_value(tools, vi_before)\nconst_key(aliases, vi_vorher)\nconst_value(aliases, vi_before)\n"
DUP = SNAP + "in_module(f, m)\nin_module(g, n)\nbody(f, h1)\nbody(g, h1)\n"

WORLDS = [
    {"name": "an_import_cycle_is_a_finding", "facts": CYCLE,
     "holds": ["status(no_import_cycle,violated)", "finding(no_import_cycle,a)", "grund(no_import_cycle,a,b)",
               "obligatory(fix(no_import_cycle,a))"], "absent": ["status(no_import_cycle,ok)"]},
    {"name": "a_late_import_is_no_cycle", "counterfactual_of": "an_import_cycle_is_a_finding",
     "facts": SNAP + "module(a)\nmodule(b)\nimports(a, b)\nimports_late(b, a)\n",
     "holds": ["status(no_import_cycle,ok)"], "absent": ["finding(no_import_cycle,a)"]},
    {"name": "a_longer_cycle_is_found_too", "facts": CYCLE.replace("imports(b, a)", "imports(b, c)\nmodule(c)\nimports(c, a)"),
     "holds": ["finding(no_import_cycle,a)", "finding(no_import_cycle,b)", "finding(no_import_cycle,c)"]},
    {"name": "an_unlocked_write_is_a_finding", "facts": LOCK,
     "holds": ["status(locked_writes,violated)", "finding(locked_writes,w)", "grund(locked_writes,w,objekt)", "obligatory(add_test(w))"]},
    {"name": "a_write_under_the_lock_is_fine", "counterfactual_of": "an_unlocked_write_is_a_finding",
     "facts": LOCK.replace("writes_unlocked(w, objekt)\n", ""), "holds": ["status(locked_writes,ok)"], "absent": ["finding(locked_writes,w)"]},
    {"name": "a_helper_called_only_under_the_lock_is_covered", "counterfactual_of": "an_unlocked_write_is_a_finding",
     "facts": LOCK + "in_module(c, m)\ncalls(c, w)\ncalls_locked(c, w)\ntest(t)\nin_module(t, tm)\ncalls(t, w)\ncalls_locked(t, w)\n",
     "holds": ["covered(w)", "status(locked_writes,ok)", "tested_by(w,t)"], "absent": ["finding(locked_writes,w)", "obligatory(add_test(w))"]},
    {"name": "one_unlocked_caller_breaks_the_cover",
     "facts": LOCK + "in_module(c, m)\ncalls(c, w)\ncalls_locked(c, w)\nin_module(d, m)\ncalls(d, w)\n",
     "holds": ["unlocked_caller(w)", "finding(locked_writes,w)"], "absent": ["covered(w)"]},
    {"name": "an_unguarded_table_needs_no_lock", "facts": LOCK.replace("guarded(objekt)\n", ""),
     "holds": ["status(locked_writes,ok)"], "absent": ["finding(locked_writes,w)", "obligatory(add_test(w))"]},
    {"name": "test_code_is_out_of_scope", "facts": LOCK + "in_module(t, m)\ntest(t)\n",
     "holds": ["test_module(m)", "status(locked_writes,ok)"], "absent": ["finding(locked_writes,w)", "production(w)"]},
    {"name": "an_exemption_excuses_with_its_reason",
     "facts": LOCK + 'exempt(locked_writes, w, "runs before the connection is shared")\n',
     "holds": ["status(locked_writes,ok)", 'excused(locked_writes,w,"runs before the connection is shared")'],
     "absent": ["finding(locked_writes,w)", "obligatory(fix(locked_writes,w))"]},
    {"name": "every_alias_resolves", "facts": ALIAS, "holds": ["status(english_tools,ok)", "registered(tools,vi_before)"]},
    {"name": "a_dangling_alias_is_a_finding", "counterfactual_of": "every_alias_resolves",
     "facts": ALIAS.replace("const_value(tools, vi_before)\n", "const_value(tools, vi_after)\n"),
     "holds": ["status(english_tools,violated)", "finding(english_tools,vi_before)", "grund(english_tools,vi_before,aliases)"]},
    {"name": "a_tool_registered_under_an_alias_is_a_finding", "counterfactual_of": "every_alias_resolves",
     "facts": ALIAS + "const_value(tools, vi_vorher)\n",
     "holds": ["finding(english_tools,vi_vorher)", "grund(english_tools,vi_vorher,tools)"]},
    {"name": "a_duplicate_body_is_a_finding", "facts": DUP + "changed_in(f, c1)\n",
     "holds": ["status(no_duplicates,violated)", "finding(no_duplicates,f)", "finding(no_duplicates,g)", "grund(no_duplicates,f,g)",
               "last_touched_by(no_duplicates,f,c1)"]},
    {"name": "the_same_body_in_one_module_is_no_duplicate", "counterfactual_of": "a_duplicate_body_is_a_finding",
     "facts": DUP.replace("in_module(g, n)", "in_module(g, m)"), "holds": ["status(no_duplicates,ok)"], "absent": ["finding(no_duplicates,f)"]},
    {"name": "an_open_view_concludes_nothing_from_absence", "facts": LOCK.replace(SNAP, "") + CYCLE.replace(SNAP, ""),
     "holds": ["reaches(a,a)", "candidate(no_import_cycle,a)"],
     "absent": ["status(locked_writes,ok)", "status(no_import_cycle,violated)", "finding(locked_writes,w)", "production(w)"]},
    {"name": "a_clean_snapshot_holds_every_rule", "facts": SNAP + "module(a)\nmodule(b)\nimports(a, b)\n",
     "holds": ["status(no_import_cycle,ok)", "status(locked_writes,ok)", "status(english_tools,ok)", "status(no_duplicates,ok)"]},
]

PACKAGE = {
    "format": "philosophy-v2", "name": "code", "version": "1.1.0", "author": "vi-mcp",
    "profile": {"semantics": "fde-signed-v1", "closure": "snapshot", "priority": "none"},
    "vocabulary": VOCABULARY,
    "answers": {"status": {"position": 1, "values": ["ok", "violated"]}},
    "seed": SEED, "worlds": WORLDS,
}

#: The profile of this repository: which table needs the lock, which constant is an alias table, and the excused findings — each
#: with its reason. Input facts, not rules (intension and extension kept apart).
PROFILE_VI_MCP = """\
guarded("objekt")
alias_of("vi_mcp.api_en.TOOL_ALIASES", "vi_mcp.tools.TOOLS")
exempt(locked_writes, "vi_mcp.memory._migrate", "runs on the new connection inside open_db before the connection is shared")
exempt(locked_writes, "scripts.measure_code_facts.questions", "a question catalogue: 'INSERT INTO objekt' is the text of a search query, nothing is written")
exempt(no_duplicates, "clingo_object.filters.is_condition", "clingo_object never imports vi_mcp (AGENTS.md), so it keeps its own copy")
exempt(no_duplicates, "vi_mcp.memory._is_condition", "the server side of the copy in clingo_object.filters (which never imports vi_mcp)")
"""
