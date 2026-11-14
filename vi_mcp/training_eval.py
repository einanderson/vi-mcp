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
    return {"metric": "verified_task_success", "base": sum(r["base_correct"] for r in rows) / n if n else None,
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
