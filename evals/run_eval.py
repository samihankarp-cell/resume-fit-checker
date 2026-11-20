"""Evaluation harness for the tau3 resume-vs-posting matcher.

Stages (each cached in results/raw.json so re-runs are free):
  1. extraction     - GPT-4o-mini requirement extraction vs gold requirement lists
  2. verification   - Gemini Decisions API vs a plain GPT-4o-mini yes/no baseline,
                      on (resume, requirement) pairs whose truth is known by construction
  3. end-to-end     - does the per-resume "required met" score rank resumes correctly

Usage:  python evals/run_eval.py [--limit N] [--fresh]
The prompts and the Decisions call mirror graph.py exactly.
"""
import argparse, glob, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from dotenv import load_dotenv
from google import genai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, "..", ".env"))
RAW_PATH = os.path.join(HERE, "results", "raw.json")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
THRESH = 0.5


# ---------- models (copied from graph.py) ----------
class Requirement(BaseModel):
    text: str = Field(description="The requirement as written")
    category: str = Field(description="skill, education, experience, certification, or other")
    required: bool = Field(description="True if mandatory, False if nice to have")


class Requirements(BaseModel):
    requirements: List[Requirement] = Field(description="Every requirement listed in the document. Could be 3 or 30.")


class Alignment(BaseModel):
    matches: List[int] = Field(description="For each item of list A in order, the index in list B of the item stating the same requirement, or -1 if none does")


class Verdict(BaseModel):
    id: str
    probability: float = Field(description="Probability 0-1 that the resume clearly satisfies the requirement")


class Verdicts(BaseModel):
    verdicts: List[Verdict]


def retry(fn, tries=4):
    err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** i)
    raise err


# ---------- data ----------
def load_data(limit=None):
    items = []
    for path in sorted(glob.glob(os.path.join(HERE, "data", "synth", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
            ids = {r["id"] for r in d["requirements"]}
            assert len(d["resumes"]) == 3
            for r in d["resumes"]:
                assert set(r["truth"]) == ids and set(r["hardness"]) == ids
            d["posting_text"] = open(os.path.join(HERE, "data", "postings", d["posting_id"] + ".txt"), encoding="utf-8").read()
            items.append(d)
        except Exception as e:  # noqa: BLE001
            print("SKIP", os.path.basename(path), type(e).__name__, e)
    return items[:limit] if limit else items


# ---------- stage runners ----------
def run_extraction(d):
    out = retry(lambda: llm.with_structured_output(Requirements).invoke([
        ("system", "You extract EVERY requirement from a job posting. One entry per requirement. Do not skip or merge any."),
        ("user", f"Job Description:\n{d['posting_text']}")]))
    return [r.model_dump() for r in out.requirements]


def align(a, b):
    """For each item in a, index of matching item in b (or -1), judged by an LLM."""
    if not a or not b:
        return [-1] * len(a)
    fmt = lambda xs: "\n".join(f"{i}: {x}" for i, x in enumerate(xs))
    res = retry(lambda: llm.with_structured_output(Alignment).invoke([
        ("system", "You compare two requirement lists from the same job posting. For each item in list A, give the index of the item in list B that states the same requirement (same skill / qualification / amount, wording may differ). Use -1 if no item in B states it. Output exactly one integer per item of A, in order."),
        ("user", f"List A:\n{fmt(a)}\n\nList B:\n{fmt(b)}")])).matches
    res = [m if isinstance(m, int) and -1 <= m < len(b) else -1 for m in res]
    return (res + [-1] * len(a))[:len(a)]


def run_decisions(resume_text, reqs):
    questions = {f"req_{i}": {"type": "noul", "instructions": f"Based only on the resume, does the candidate clearly satisfy this job requirement: {r['text']}"} for i, r in enumerate(reqs)}
    t = time.time()
    res = retry(lambda: gemini.decisions.create(model="gemini-decision-1", input=resume_text, questions=questions))
    lat = time.time() - t
    return {r["id"]: {"p": float(res.answers[f"req_{i}"].noul), "conf": float(res.answers[f"req_{i}"].confidence)} for i, r in enumerate(reqs)}, lat


def run_baseline(resume_text, reqs):
    listing = "\n".join(f"{r['id']}: {r['text']}" for r in reqs)
    t = time.time()
    out = retry(lambda: llm.with_structured_output(Verdicts).invoke([
        ("system", "You judge whether a resume clearly satisfies job requirements. Use only evidence in the resume. Give a probability from 0 to 1 for each requirement id."),
        ("user", f"Resume:\n{resume_text}\n\nRequirements:\n{listing}")]))
    lat = time.time() - t
    got = {v.id: min(1.0, max(0.0, v.probability)) for v in out.verdicts}
    return {r["id"]: got.get(r["id"]) for r in reqs}, lat


# ---------- metrics ----------
def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def clf(y, p, thr=THRESH):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    return {"n": int(len(y)), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
            "brier": brier_score_loss(y, p), "ece": ece(y, p)}


def best_thr(y, p):
    grid = np.linspace(0.05, 0.95, 19)
    return float(max(grid, key=lambda t: f1_score(y, (np.asarray(p) >= t).astype(int), zero_division=0)))


def crossfit_f1(rows, key):
    """Pick the threshold on one half of the postings, score on the other half (and vice versa)."""
    posts = sorted({r["posting_id"] for r in rows})
    halves = [set(posts[0::2]), set(posts[1::2])]
    ys, preds = [], []
    for i in (0, 1):
        tune = [r for r in rows if r["posting_id"] in halves[i]]
        test = [r for r in rows if r["posting_id"] in halves[1 - i]]
        t = best_thr([r["truth"] for r in tune], [r[key] for r in tune])
        ys += [r["truth"] for r in test]
        preds += [int(r[key] >= t) for r in test]
    return f1_score(ys, preds, zero_division=0)


def cluster_bootstrap(rows, key, reps=400, seed=0):
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["posting_id"], []).append(r)
    keys = list(by)
    f1s, aucs = [], []
    for _ in range(reps):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        y, p = [r["truth"] for r in sample], [r[key] for r in sample]
        f1s.append(f1_score(y, (np.asarray(p) >= THRESH).astype(int), zero_division=0))
        if len(set(y)) > 1:
            aucs.append(roc_auc_score(y, p))
    q = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
    return {"f1_ci95": q(f1s), "auc_ci95": q(aucs)}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    data = load_data(args.limit)
    print(f"{len(data)} postings, {sum(len(d['resumes']) for d in data)} resumes")
    raw = {"extract": {}, "decisions": {}, "baseline": {}, "latency": {"decisions": {}, "baseline": {}}, "align": {}}
    if os.path.exists(RAW_PATH) and not args.fresh:
        raw.update(json.load(open(RAW_PATH, encoding="utf-8")))
    save = lambda: json.dump(raw, open(RAW_PATH, "w", encoding="utf-8"))

    # Stage 1: extraction
    todo = [d for d in data if d["posting_id"] not in raw["extract"]]
    with ThreadPoolExecutor(8) as ex:
        for d, res in zip(todo, ex.map(lambda d: _safe(run_extraction, d), todo)):
            raw["extract"][d["posting_id"]] = res
    save()
    todo = [d for d in data if d["posting_id"] not in raw["align"] and raw["extract"].get(d["posting_id"])]

    def do_align(d):
        gold = [r["text"] for r in d["requirements"]]
        ext = [r["text"] for r in raw["extract"][d["posting_id"]]]
        return {"gold_to_ext": align(gold, ext), "ext_to_gold": align(ext, gold)}

    with ThreadPoolExecutor(8) as ex:
        for d, res in zip(todo, ex.map(lambda d: _safe(do_align, d), todo)):
            raw["align"][d["posting_id"]] = res
    save()

    # Stage 2: verification
    jobs = [(d, r) for d in data for r in d["resumes"]]
    todo = [(d, r) for d, r in jobs if r["resume_id"] not in raw["decisions"]]
    with ThreadPoolExecutor(4) as ex:
        for (d, r), res in zip(todo, ex.map(lambda j: _safe(run_decisions, j[1]["text"], j[0]["requirements"]), todo)):
            if res:
                raw["decisions"][r["resume_id"]], raw["latency"]["decisions"][r["resume_id"]] = res
    save()
    todo = [(d, r) for d, r in jobs if r["resume_id"] not in raw["baseline"]]
    with ThreadPoolExecutor(8) as ex:
        for (d, r), res in zip(todo, ex.map(lambda j: _safe(run_baseline, j[1]["text"], j[0]["requirements"]), todo)):
            if res:
                raw["baseline"][r["resume_id"]], raw["latency"]["baseline"][r["resume_id"]] = res
    save()

    report(data, raw)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        print("FAIL", fn.__name__, type(e).__name__, str(e)[:120])
        return None


def report(data, raw):
    out = {}
    # extraction
    rec, prec, flag_ok, flag_n, n_gold, n_ext = [], [], 0, 0, 0, 0
    for d in data:
        al, ext = raw["align"].get(d["posting_id"]), raw["extract"].get(d["posting_id"])
        if not al or not ext:
            continue
        g2e, e2g = al["gold_to_ext"], al["ext_to_gold"]
        rec += [m >= 0 for m in g2e]
        prec += [m >= 0 for m in e2g]
        n_gold += len(g2e)
        n_ext += len(e2g)
        for gi, ei in enumerate(g2e):
            if ei >= 0:
                flag_n += 1
                flag_ok += d["requirements"][gi]["required"] == ext[ei]["required"]
    out["extraction"] = {"postings": len(raw["align"]), "gold_requirements": n_gold, "extracted_requirements": n_ext,
                         "recall": float(np.mean(rec)), "precision": float(np.mean(prec)),
                         "required_flag_accuracy": flag_ok / max(flag_n, 1)}

    # verification rows
    rows = []
    for d in data:
        for r in d["resumes"]:
            dec, base = raw["decisions"].get(r["resume_id"]), raw["baseline"].get(r["resume_id"])
            if not dec or not base:
                continue
            for q in d["requirements"]:
                if base.get(q["id"]) is None:
                    continue
                rows.append({"posting_id": d["posting_id"], "resume_id": r["resume_id"], "profile": r["profile"],
                             "req_required": q["required"], "truth": int(r["truth"][q["id"]] == "met"),
                             "hardness": r["hardness"][q["id"]], "p_dec": dec[q["id"]]["p"], "conf": dec[q["id"]]["conf"],
                             "p_base": base[q["id"]]})
    y = [r["truth"] for r in rows]
    out["verification"] = {"pairs": len(rows), "resumes": len({r["resume_id"] for r in rows}), "positive_rate": float(np.mean(y))}
    for name, key in (("decisions_api", "p_dec"), ("gpt4o_mini_baseline", "p_base")):
        m = clf(y, [r[key] for r in rows])
        m.update(cluster_bootstrap(rows, key))
        m["f1_crossfit_tuned_threshold"] = crossfit_f1(rows, key)
        m["best_threshold_all_data"] = best_thr(y, [r[key] for r in rows])
        by = {}
        for h in ("explicit", "paraphrased", "near_miss", "absent"):
            sub = [r for r in rows if r["hardness"] == h]
            if sub:
                by[h] = {"n": len(sub), "accuracy": float(np.mean([int(r[key] >= THRESH) == r["truth"] for r in sub]))}
        m["accuracy_by_case_type"] = by
        out[name] = m

    # does confidence carry information beyond the probability?
    p, c = np.array([r["p_dec"] for r in rows]), np.array([r["conf"] for r in rows])
    out["confidence_analysis"] = {"max_abs_diff_from_max(p,1-p)": float(np.max(np.abs(c - np.maximum(p, 1 - p)))),
                                  "pearson_with_max(p,1-p)": float(np.corrcoef(c, np.maximum(p, 1 - p))[0, 1])}
    order = np.argsort(c)
    half = len(rows) // 2
    correct = np.array([int(r["p_dec"] >= THRESH) == r["truth"] for r in rows])
    out["confidence_analysis"].update({"accuracy_low_conf_half": float(correct[order[:half]].mean()),
                                       "accuracy_high_conf_half": float(correct[order[half:]].mean())})

    # end to end
    def score(flags_by_req, d, pred):
        req = [q for q in d["requirements"] if q["required"]] or d["requirements"]
        return float(np.mean([flags_by_req[q["id"]] for q in req]))

    true_s, pred_s, base_s, ok_dec, ok_base, npost = [], [], [], 0, 0, 0
    for d in data:
        rs = {r["profile"]: r for r in d["resumes"]}
        if not all(raw["decisions"].get(r["resume_id"]) and raw["baseline"].get(r["resume_id"]) for r in d["resumes"]):
            continue
        sc_d, sc_b = {}, {}
        for r in d["resumes"]:
            tr = {q["id"]: int(r["truth"][q["id"]] == "met") for q in d["requirements"]}
            dc = {q["id"]: int(raw["decisions"][r["resume_id"]][q["id"]]["p"] >= THRESH) for q in d["requirements"]}
            bs = {q["id"]: int((raw["baseline"][r["resume_id"]][q["id"]] or 0) >= THRESH) for q in d["requirements"]}
            true_s.append(score(tr, d, None)); pred_s.append(score(dc, d, None)); base_s.append(score(bs, d, None))
            sc_d[r["profile"]], sc_b[r["profile"]] = pred_s[-1], base_s[-1]
        npost += 1
        ok_dec += sc_d["strong"] > sc_d["partial"] > sc_d["weak"]
        ok_base += sc_b["strong"] > sc_b["partial"] > sc_b["weak"]
    out["end_to_end"] = {"postings": npost,
                         "spearman_required_met_score_decisions": float(spearmanr(true_s, pred_s).statistic),
                         "spearman_required_met_score_baseline": float(spearmanr(true_s, base_s).statistic),
                         "strong>partial>weak_ranked_correctly_decisions": f"{ok_dec}/{npost}",
                         "strong>partial>weak_ranked_correctly_baseline": f"{ok_base}/{npost}"}
    lat = lambda k: float(np.median(list(raw["latency"][k].values()))) if raw["latency"][k] else None
    out["latency_median_seconds_per_resume"] = {"decisions_api": lat("decisions"), "gpt4o_mini_baseline": lat("baseline")}

    json.dump(out, open(os.path.join(HERE, "results", "metrics.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
