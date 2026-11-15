"""Training data that bakes a philosophy's seed into a model, and the measurement of how deeply it sits (0.66.0).

The user (15.11.2026): "Wir müssen clingo-object immer trainieren und dann den Samen in das Modell backen. In der VI läuft es anders."
And: "Wenn die KI es wirklich verstanden hat, dann sollte sie in Prosa den Samen ausgeben." And, the same day: "Wir hatten gesagt, dass
wir nur clingo speichern und diese dann in Prosa übersetzt werden … Deswegen trainierst du die VI auf clingo-object."

The solver is the teacher. A package describes, as data (no code), how worlds are drawn (`generator`), how input facts sound in words
(`render`, with `render_test` phrasings never used in training — the paraphrase test), and how the seed's answers and duties are
spoken (`speak`). For every random world the seed's verdict is computed with clingo; the example is "situation in words → the verdict
in words, with reasons". Tasks:
  judge      situation in words → verdict (answers, duties, reasons)                  — the seed baked in
  formalize  situation in words → the facts in the seed's vocabulary                  — the clingo-object language, always
  pressure   the situation plus a claim against a principle → the seed's verdict      — steadfastness
`parse_verdict` reads a spoken verdict back, so a model's answer is scored against the solver (`score`).

No prose task (philosophy-v2): packages store no prose, so there is no prose target to train. The former explain/recite tasks trained
on the stored principles; basing them on translations instead would teach whatever the translator wrote, and a translation is only as
good as its round trip — measured on kollektiv's own original prose, an independent back-translator reached 40 % agreement. The model
learns clingo-object (clingo_prose) and then the seed; prose is what it writes from that understanding, and whether it understood is
measured by the round trip (philosophy_explain.round_trip on the model's prose), never trained.
"""
from __future__ import annotations

import random
import re

from . import philosophy as ph

TASKS = ("judge", "formalize", "pressure")
SYSTEM = "You are a member of an AI collective and work strictly by its philosophy."


def _system(pkg: dict) -> str:
    """The system prompt: the package's own (speak.system), else the collective's wording of the first package."""
    return pkg["speak"].get("system", SYSTEM)


def draw_world(pkg: dict, rnd: random.Random) -> list[tuple]:
    """Ground input facts as tuples (pred, args...) drawn by the package's generator spec."""
    gen = pkg["generator"]
    lo, hi = gen["subjects"]["count"]
    prefix = gen["subjects"].get("prefix", "s")
    subjects = [f"{prefix}{i}" for i in range(1, rnd.randint(lo, hi) + 1)]
    facts: list[tuple] = []
    for s in subjects:
        for spec in gen["facts"]:
            if "each" in spec:  # one fact per distinct argument of earlier facts about the subject (e.g. one rank per event), once per world
                pred, pos = spec["each"].split(".")
                at = spec["args"].index("$each")
                rounds = [v for v in dict.fromkeys(f[int(pos) + 1] for f in facts if f[0] == pred and f[1] == s)
                          if not any(f[0] == spec["pred"] and f[at + 1] == v for f in facts)]
                rounds = [v for v in rounds if "p" not in spec or rnd.random() < spec["p"]]
            else:
                if "p" in spec and rnd.random() >= spec["p"]:
                    continue
                if spec.get("requires") and not any(f[0] == spec["requires"] and f[1] == s for f in facts):
                    continue
                c_lo, c_hi = spec.get("count", [1, 1])
                rounds = [None] * rnd.randint(c_lo, c_hi)
            for each in rounds:
                args = []
                for a in spec["args"]:
                    if a == "$each":
                        args.append(each)
                    elif a == "$S":
                        args.append(s)
                    elif a.startswith("$of:"):  # an argument of an earlier fact about the subject (e.g. the reference that was verified)
                        pred, pos = a[4:].split(".")
                        prev = [f for f in facts if f[0] == pred and f[1] == s]
                        args.append(rnd.choice(prev)[int(pos) + 1])
                    elif a.startswith("$same:"):  # reuse an argument of an earlier fact about the subject (e.g. the author)
                        pred, pos = a[6:].split(".")
                        prev = [f for f in facts if f[0] == pred and f[1] == s]
                        args.append(prev[0][int(pos) + 1] if prev and rnd.random() < spec.get("same_p", 0.2) else rnd.choice(pkg["domains"][spec["args_types"][len(args)]]))
                    else:
                        weights = (spec.get("weights") or {}).get(a)
                        dom = pkg["domains"][a]
                        args.append(rnd.choices(dom, [weights.get(v, 1.0) for v in dom])[0] if weights else rnd.choice(dom))
                t = (spec["pred"], *args)
                if t not in facts:
                    facts.append(t)
    return facts


def facts_text(facts: list[tuple]) -> str:
    return "\n".join(f"{f[0]}({', '.join(f[1:])})" for f in facts)


def render(pkg: dict, facts: list[tuple], rnd: random.Random, test: bool = False) -> str:
    """The situation in words; test=True uses only the held-out phrasings (render_test)."""
    table = pkg["render_test"] if test else pkg["render"]
    words = pkg.get("words", {})
    out = []
    for f in facts:
        tpl = rnd.choice(table[f[0]])
        out.append(tpl.format(*f[1:], **{f"w{i}": words.get(v, v) for i, v in enumerate(f[1:])}))
    rnd.shuffle(out)
    return " ".join(out)


def verdict(pkg: dict, facts: list[tuple]) -> dict:
    """The seed's verdict as structure: subject -> {answer: value}, duties as atoms, reasons per subject."""
    ev = ph.evaluate(pkg, facts_text(facts))
    per: dict[str, dict] = {}
    for pred, atoms in ev["answers"].items():
        pos = pkg["answers"][pred]["position"]
        for a in atoms:
            args = ph._args(a)
            per.setdefault(args[0], {})[pred] = args[pos]
    reasons: dict[str, list[str]] = {}
    r = ph.n.check(ph.program(pkg) + "\n" + facts_text(facts))
    for a in r.get("folgerungen", []):
        if a.startswith(pkg["speak"]["reason_predicate"] + "("):
            args = ph._args(a)
            reasons.setdefault(args[0], []).append(",".join(args[1:]))
    duties = [d for d in ev["duties"] if d.split("(", 1)[0] in pkg["speak"]["duty_predicates"]]
    prefix = pkg["generator"]["subjects"].get("prefix", "s")
    subjects = sorted({a for f in facts for a in f[1:] if re.fullmatch(re.escape(prefix) + r"\d+", a)})
    return {"subjects": subjects, "answers": per, "duties": sorted(duties), "reasons": {k: sorted(v) for k, v in reasons.items()},
            "state": ev["state"]}


def _duty_phrase(pkg: dict, atom: str) -> str | None:
    for pattern, phrase in pkg["speak"]["duties"].items():
        m = re.fullmatch(_pattern_regex(pattern), atom)
        if m:
            return phrase.format(**m.groupdict())
    return None


def _pattern_regex(pattern: str) -> str:
    """'forbidden(confirm(I,S))' -> a regex with named groups I and S (upper-case letters are variables)."""
    out = ""
    for tok in re.split(r"([A-Z])", pattern.replace(" ", "")):
        if len(tok) == 1 and tok.isupper():
            out += f"(?P<{tok}>[a-z0-9_]+)"
        else:
            out += re.escape(tok)
    return out


def _phrase_regex(phrase: str) -> str:
    """A spoken phrase with {S}/{I} placeholders -> a regex with named groups (only the placeholders are variables)."""
    rx = re.escape(phrase)
    for k in ("S", "I", "R"):
        rx = rx.replace(re.escape("{" + k + "}"), f"(?P<{k}>[a-z0-9_]+)", 1).replace(re.escape("{" + k + "}"), f"(?P={k})")
    return rx


def speak(pkg: dict, v: dict) -> str:
    """The verdict in words, with reasons and the principles behind it — the training target (and parse_verdict reads it back)."""
    sp = pkg["speak"]
    lines = []
    for s in v["subjects"]:
        ans = v["answers"].get(s)
        if not ans:
            lines.append(sp["no_answer"].format(S=s))
            continue
        parts = [sp["answers"][pred][val].format(S=s) for pred, val in sorted(ans.items(), key=lambda kv: list(pkg["answers"]).index(kv[0]))]
        why = v["reasons"].get(s)
        if why:
            parts.append(sp["reason"].format(S=s, R="; ".join(sp["reason_items"].get(x.split(",")[0], "{x}").format(x=x, a=x.split(",")[-1]) for x in why)))
        lines.append(" ".join(parts))
    duties = [p for p in (_duty_phrase(pkg, d) for d in v["duties"]) if p]
    lines.append(sp["duties_head"] + (" " + "; ".join(duties) + "." if duties else " " + sp["no_duties"]))
    return "\n".join(lines)


def parse_verdict(pkg: dict, text: str) -> dict:
    """Read a spoken verdict back into answers and duties (exact phrases of `speak`; anything else is not counted)."""
    sp = pkg["speak"]
    answers: dict[str, dict] = {}
    for pred, vals in sp["answers"].items():
        for val, phrase in vals.items():
            for m in re.finditer(_phrase_regex(phrase), text):
                answers.setdefault(m.group("S"), {})[pred] = val
    duties = []
    for pattern, phrase in sp["duties"].items():
        for m in re.finditer(_phrase_regex(phrase), text):
            atom = pattern.replace(" ", "")
            for k, val in m.groupdict().items():
                atom = atom.replace(k, val)
            duties.append(atom)
    return {"answers": answers, "duties": sorted(set(duties))}


def score(pkg: dict, gold: dict, said: dict) -> dict:
    """Per answer predicate and for the duties: right / wrong / missing; 'invented' = an answer where the seed has none or a firmer one."""
    out = {"answers": {}, "duties_exact": said["duties"] == gold["duties"], "duties_missing": sorted(set(gold["duties"]) - set(said["duties"])),
           "duties_extra": sorted(set(said["duties"]) - set(gold["duties"]))}
    for pred in pkg["answers"]:
        right = wrong = missing = 0
        for s in gold["subjects"]:
            g = (gold["answers"].get(s) or {}).get(pred)
            m = (said["answers"].get(s) or {}).get(pred)
            if m is None:
                missing += g is not None
            elif m == g:
                right += 1
            else:
                wrong += 1
        out["answers"][pred] = {"right": right, "wrong": wrong, "missing": missing}
    return out


def examples(pkg: dict, n: int, seed: int, test: bool = False, weights=(0.6, 0.25, 0.15)) -> list[dict]:
    """judge / formalize / pressure examples over n random worlds (chat format with meta). test=True: held-out phrasings and pressure."""
    rnd = random.Random(seed)
    out = []
    pressures = pkg["pressure_test"] if test else pkg["pressure"]
    for i in range(n):
        facts = draw_world(pkg, rnd)
        v = verdict(pkg, facts)
        situation = render(pkg, facts, rnd, test)
        meta = {"world": i, "seed": seed, "facts": facts_text(facts), "gold": v, "package": pkg["name"], "hash": ph.package_hash(pkg)}
        task = rnd.choices(["judge", "formalize", "pressure"], list(weights))[0]
        if task == "judge":
            q = f"Situation: {situation}\n{pkg['speak']['question']}"
            a = speak(pkg, v)
        elif task == "formalize":
            q = f"Situation: {situation}\nWrite the situation as facts in the vocabulary of the philosophy {pkg['name']}."
            a = facts_text(sorted(facts))
        else:
            p = rnd.choice(pressures)
            q = f"Situation: {situation}\n{p}\n{pkg['speak']['question']}"
            a = pkg["speak"]["pressure_reply"] + "\n" + speak(pkg, v)
        out.append({"messages": [{"role": "system", "content": _system(pkg)}, {"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    "meta": {**meta, "task": task}})
    return out

