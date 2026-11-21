"""Complete-workflow comparison, each system doing its OWN extraction, one run at a time.
  A = GPT-4o-mini does everything (extracts requirements, then verifies them)
  B = GPT-4o-mini extracts, Google Decisions API verifies   (graph.py in its default mode)
Inputs: the 18 held-out (odd-indexed) postings x 3 resumes = 54 resumes. Resumable: results cached in results/e2e_systems_raw.json."""
import os
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
import json, random, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from langchain_core.callbacks import get_usage_metadata_callback
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
import graph, run_eval as v1

assert graph.VERIFIER == "decisions", graph.VERIFIER
PRICE = {"gpt_in": 0.15, "gpt_out": 0.60, "dec_in": 0.10, "dec_out": 0.0}  # USD per 1M tokens, as supplied by the user
RAW = os.path.join(HERE, "results", "e2e_systems_raw.json")

# spy on the Decisions client so token counts can be read without changing graph.py
dec_log = []
_real = graph.gemini
class _Dec:
    def create(self, **kw):
        r = _real.decisions.create(**kw); u = r.usage
        dec_log.append((int(u.total_input_tokens or 0), int(u.total_output_tokens or 0)) if u else (0, 0))
        return r
class _Client:
    decisions = _Dec()
graph.gemini = _Client()


def run_A(posting_text, resume_text):
    clean_p, _ = graph.sanitize_posting(posting_text)
    ext = graph.extract_from_text(clean_p)
    clean_r, _ = graph.sanitize_untrusted(resume_text)
    ps = graph.gpt_verify(clean_r, ext) if ext else []
    return [{"text": r.text, "required": r.required, "p": p, "met": p >= 0.5} for r, p in zip(ext, ps)]


def run_B(posting_text, resume_text, tag, tmp):
    pp, rp = os.path.join(tmp, tag + "_p.txt"), os.path.join(tmp, tag + "_r.txt")
    open(pp, "w", encoding="utf-8").write(posting_text); open(rp, "w", encoding="utf-8").write(resume_text)
    out = graph.app.invoke({"posting_path": pp, "resume_path": rp})
    return [{"text": m.requirement.text, "required": m.requirement.required, "p": m.probability, "met": m.met} for m in out["matches"]]


def retry(fn, n=3):
    for i in range(n):
        try: return fn()
        except Exception as e: err = e; time.sleep(2 ** i)
    raise err


def main():
    data = v1.load_data(); posts = sorted(d["posting_id"] for d in data)
    test = [d for d in data if d["posting_id"] in set(posts[1::2])]
    pairs = [(d, r) for d in test for r in d["resumes"]]
    raw = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {"runs": {}, "align": {}, "fails": []}
    save = lambda: json.dump(raw, open(RAW, "w", encoding="utf-8"))
    tmp = tempfile.mkdtemp()

    for idx, (d, r) in enumerate(pairs):
        for s in (("A", "B") if idx % 2 == 0 else ("B", "A")):  # alternate order so neither system is always first
            key = f"{s}|{r['resume_id']}"
            if key in raw["runs"]: continue
            n = len(dec_log)
            try:
                with get_usage_metadata_callback() as cb:
                    t = time.perf_counter()
                    reqs = run_A(d["posting_text"], r["text"]) if s == "A" else run_B(d["posting_text"], r["text"], r["resume_id"], tmp)
                    secs = time.perf_counter() - t
            except Exception as e:  # noqa: BLE001
                print("FAIL", key, type(e).__name__, str(e)[:100], flush=True); raw["fails"].append(key); continue
            raw["runs"][key] = {"reqs": reqs, "secs": secs,
                                "gpt_in": sum(v.get("input_tokens", 0) for v in cb.usage_metadata.values()),
                                "gpt_out": sum(v.get("output_tokens", 0) for v in cb.usage_metadata.values()),
                                "dec_in": sum(x[0] for x in dec_log[n:]), "dec_out": sum(x[1] for x in dec_log[n:])}
        save()
        print(f"[{idx + 1}/{len(pairs)}] {r['resume_id']}  A {raw['runs'].get('A|' + r['resume_id'], {}).get('secs', 0):.1f}s  B {raw['runs'].get('B|' + r['resume_id'], {}).get('secs', 0):.1f}s", flush=True)

    # match each extracted requirement to the answer key (GPT-4o-mini judge, same method as the earlier evals)
    todo = [(d, r, s) for d, r in pairs for s in "AB" if f"{s}|{r['resume_id']}" in raw["runs"] and f"{s}|{r['resume_id']}" not in raw["align"]]
    def do_align(j):
        d, r, s = j
        ext = raw["runs"][f"{s}|{r['resume_id']}"]["reqs"]
        return retry(lambda: v1.align([q["text"] for q in d["requirements"]], [e["text"] for e in ext]))
    with ThreadPoolExecutor(8) as ex:
        for (d, r, s), res in zip(todo, ex.map(do_align, todo)): raw["align"][f"{s}|{r['resume_id']}"] = res
    save()
    report(pairs, raw)


def report(pairs, raw):
    ok = [(d, r) for d, r in pairs if all(f"{s}|{r['resume_id']}" in raw["align"] for s in "AB")]  # both systems finished
    out = {"resumes_compared": len(ok), "postings": len({d["posting_id"] for d, _ in ok}), "failed_runs": raw["fails"], "systems": {}}
    per = {}
    for s in "AB":
        rows, scores, gold_scores, cost, secs, tok = [], {}, [], [], [], {"gpt_in": [], "gpt_out": [], "dec_in": [], "dec_out": []}
        n_items = []
        for d, r in ok:
            run, al = raw["runs"][f"{s}|{r['resume_id']}"], raw["align"][f"{s}|{r['resume_id']}"]
            ext = run["reqs"]; n_items.append(len(ext))
            for gi, q in enumerate(d["requirements"]):
                ei = al[gi]; rows.append((d["posting_id"], int(r["truth"][q["id"]] == "met"), int(ext[ei]["met"]) if ei >= 0 else 0, ei >= 0))
            req = [e for e in ext if e["required"]] or ext
            scores.setdefault(d["posting_id"], {})[r["profile"]] = float(np.mean([e["met"] for e in req])) if req else 0.0
            greq = [q for q in d["requirements"] if q["required"]] or d["requirements"]
            gold_scores.append((d["posting_id"], r["profile"], float(np.mean([r["truth"][q["id"]] == "met" for q in greq]))))
            secs.append(run["secs"])
            for k in tok: tok[k].append(run[k])
            cost.append((run["gpt_in"] * PRICE["gpt_in"] + run["gpt_out"] * PRICE["gpt_out"] + run["dec_in"] * PRICE["dec_in"] + run["dec_out"] * PRICE["dec_out"]) / 1e6)
        y = [x[1] for x in rows]; p = [x[2] for x in rows]; matched = [x for x in rows if x[3]]
        ranked = sum(1 for v in scores.values() if len(v) == 3 and v["strong"] > v["partial"] > v["weak"])
        pred_flat = [scores[pid][prof] for pid, prof, _ in gold_scores]
        per[s] = {"rows": rows}
        out["systems"][s] = {
            "accuracy_all_gold_pairs": {"pairs": len(rows), "f1": f1_score(y, p), "precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
                                        "accuracy": float(np.mean(np.array(y) == np.array(p)))},
            "verification_only_f1_on_matched_pairs": f1_score([x[1] for x in matched], [x[2] for x in matched]),
            "requirements_found_of_gold": float(np.mean([x[3] for x in rows])),
            "extracted_items_per_run": float(np.mean(n_items)),
            "ranking_strong>partial>weak": f"{ranked}/{len(scores)}",
            "spearman_vs_gold_score": float(spearmanr([g[2] for g in gold_scores], pred_flat).statistic),
            "time_per_resume_s": {"median": float(np.median(secs)), "mean": float(np.mean(secs)), "p90": float(np.percentile(secs, 90))},
            "tokens_per_resume_mean": {k: float(np.mean(v)) for k, v in tok.items()},
            "cost_per_resume_usd": float(np.mean(cost)), "cost_per_1000_resumes_usd": 1000 * float(np.mean(cost)), "total_cost_this_test_usd": float(np.sum(cost))}
    # paired bootstrap over postings: B minus A on the all-gold-pairs F1
    byp = {}
    for i, x in enumerate(per["A"]["rows"]): byp.setdefault(x[0], []).append(i)
    keys = list(byp); rng = random.Random(0); diffs = []
    for _ in range(1000):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in byp[k]]
        ya = [per["A"]["rows"][i][1] for i in idx]
        diffs.append(f1_score(ya, [per["B"]["rows"][i][2] for i in idx]) - f1_score(ya, [per["A"]["rows"][i][2] for i in idx]))
    out["F1_B_minus_A_95ci_over_postings"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    if out["systems"]["B"]["tokens_per_resume_mean"]["dec_in"] == 0 or out["systems"]["B"]["tokens_per_resume_mean"]["gpt_in"] < 1500:
        out["WARNING"] = "token capture looks incomplete for system B"
    json.dump(out, open(os.path.join(HERE, "results", "e2e_systems.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
