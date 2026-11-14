"""Independent task judgments and executable program evaluation; no word-overlap adoption gate."""
from __future__ import annotations

import math
import ast
import hashlib


def program_family(source: str) -> str:
    """Group structural variants together; changing constants or variable names does not create an independent task family."""
    tree = ast.parse(source)
    names = {}
    class Shape(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value="VALUE" if isinstance(node.value, str) else 0), node)
        def visit_Name(self, node):
            if node.id[:1].isupper():
                names.setdefault(node.id, "V" + str(len(names)))
                node.id = names[node.id]
            return node
    return hashlib.sha256(ast.dump(Shape().visit(tree), include_attributes=False).encode()).hexdigest()


SIGN_FLIP_EXACT = 20      # up to 2^20 sign patterns over the non-zero groups are enumerated exactly
SIGN_FLIP_SAMPLES = 200_000


def sign_flip_p(sums: list[int], seed: int = 0) -> float | None:
    """One-sided p-value that the tuned model is better: under the null, tuned and base are exchangeable within each independent
    group, so every group sum keeps its size with a random sign. Groups with sum 0 carry no evidence either way. Exact up to
    SIGN_FLIP_EXACT non-zero groups, otherwise a seeded Monte Carlo with the (hits + 1) / (samples + 1) correction, which stays valid.
    Review 14.11.2026 (Claude): the Hoeffding bound alone needed about 600 independent tasks to show +10 points, so with today's
    data no tuned model could ever be recorded as better."""
    xs = [abs(s) for s in sums if s]
    if not xs:
        return None
    observed = sum(s for s in sums if s)
    if len(xs) <= SIGN_FLIP_EXACT:
        dist = {0: 1}
        for x in xs:  # convolution of the sign distribution; integer sums, exact counts
            nxt: dict[int, int] = {}
            for t, c in dist.items():
                nxt[t + x] = nxt.get(t + x, 0) + c
                nxt[t - x] = nxt.get(t - x, 0) + c
            dist = nxt
        return sum(c for t, c in dist.items() if t >= observed) / 2 ** len(xs)
    import random
    rnd = random.Random(seed)
    hits = sum(sum(x if rnd.random() < .5 else -x for x in xs) >= observed for _ in range(SIGN_FLIP_SAMPLES))
    return (hits + 1) / (SIGN_FLIP_SAMPLES + 1)


def summarize_judgments(rows: list[dict]) -> dict:
    """Paired correctness, with uncertainty clustered by task family/source group."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if type(row.get("base_correct")) is not bool or type(row.get("tuned_correct")) is not bool or not row.get("reference"):
            raise ValueError("every judgment needs boolean base_correct/tuned_correct and an independent reference")
        groups.setdefault(str(row.get("group") or row.get("id") or i), []).append(int(row["tuned_correct"]) - int(row["base_correct"]))
    n = len(rows)
    delta = sum(sum(v) for v in groups.values()) / n if n else 0.0
    # Cluster-robust SE of the paired mean; a single task family gives no uncertainty estimate.
    k = len(groups)
    se = math.sqrt(k / (k - 1) * sum((sum(v) - len(v) * delta) ** 2 for v in groups.values())) / n if k > 1 else None
    p = sign_flip_p([sum(v) for v in groups.values()])
    return {"metric": "verified_task_success", "base": sum(r["base_correct"] for r in rows) / n if n else None,
            "p_one_sided": p, "decision": bool(n and delta > 0 and p is not None and p < 0.05),
            "decision_method": "exact one-sided sign-flip test over independent group sums (clustered McNemar); "
                               "the Hoeffding lower_95 is the worst-case size bound, not the gate",
            "tuned": sum(r["tuned_correct"] for r in rows) / n if n else None, "n": n, "groups": k,
            "better": sum(not r["base_correct"] and r["tuned_correct"] for r in rows),
            "worse": sum(r["base_correct"] and not r["tuned_correct"] for r in rows),
            "difference": delta, "standard_error": se,
            "lower_95": delta - math.sqrt(2 * math.log(20) * sum((len(v) / n) ** 2 for v in groups.values())) if n else None,
            "confidence_method": "one-sided Hoeffding bound over independent groups; outcome differences in [-1,1]"}


def evaluate_program(source: str, case: dict) -> dict:
    """Execute against frozen objects and compare the complete answer, including extra and missing rows.
    case = {objects: object_data JSON, expected: atoms, predicate: 'ans', expected_state: 'erfuellt'}.
    """
    from clingo_object import notation as n
    try:
        program, stmts, filters = n.translate_full(source)
        data = case.get("objects") or {"objects": [], "members": [], "edges": [], "filters": []}
        if "input_objects" in case:
            from clingo_object.filters import object_data
            data = object_data(case["input_objects"], filters)
        if filters and list(filters) != data.get("filters"):
            raise ValueError("program filters differ from the frozen evaluation world")
        if (data.get("counts") or {}).get("cut"):
            raise ValueError("evaluation world was cut")
        raw = n.run_program("\n".join(n.facts(data)) + "\n" + program, stmts)
        result = n.check_result(raw)
        if not result.get("vollstaendig") or raw.get("timeout"):
            raise ValueError("incomplete execution")
        predicate = case.get("predicate", "ans")
        projections = [{a for a in m if a == predicate or a.startswith(predicate + "(")} for m in raw["map"]]
        if not projections or any(s != projections[0] for s in projections):
            raise ValueError("answer is undetermined")
        got = projections[0]
        want = set(n.reference_answers(case["expected"], predicate))
        tp = len(got & want)
        return {"correct": got == want and result["zustand"] == case.get("expected_state", "erfuellt"),
                "precision": tp / len(got) if got else float(not want), "recall": tp / len(want) if want else float(not got),
                "missing": len(want - got), "unexpected": len(got - want), "state": result["zustand"]}
    except (n.LogicError, ValueError, KeyError, TypeError) as e:
        return {"correct": False, "error": type(e).__name__, "state": "refused"}
