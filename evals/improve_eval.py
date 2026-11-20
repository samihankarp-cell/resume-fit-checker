"""Does the hardened/improved pipeline score better than the original?  Dev/test split to avoid tuning on the test set.

Selection rule (fixed before looking at results): among candidate verifiers, pick the highest DEV-half AUC, set its
threshold by maximising DEV-half F1, then report everything on the held-out TEST half (odd-indexed postings).
"""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
import graph
import run_eval as v1

RAW2 = os.path.join(HERE, "results", "improve_raw.json")
raw = json.load(open(os.path.join(HERE, "results", "raw.json"), encoding="utf-8"))
raw2 = json.load(open(RAW2, encoding="utf-8")) if os.path.exists(RAW2) else {"dec2": {}, "gpt2": {}, "extract2": {}, "align2": {}, "judge": {}}
save2 = lambda: json.dump(raw2, open(RAW2, "w", encoding="utf-8"))
data = v1.load_data()
posts = sorted(d["posting_id"] for d in data)
DEV, TEST = set(posts[0::2]), set(posts[1::2])


def reqs_of(d):
    return [graph.Requirement(text=q["text"], category=q["category"], required=q["required"]) for q in d["requirements"]]


def safe(fn, *a):
    for i in range(3):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            err = e
    print("FAIL", fn.__name__, type(err).__name__, str(err)[:100])


# ---------------------------------------------------------------- verification
def run_v2_verifiers():
    jobs = [(d, r) for d in data for r in d["resumes"]]
    def one(j):
        d, r = j
        text = graph.sanitize_untrusted(r["text"])[0]
        rq = reqs_of(d)
        key = r["resume_id"]
        if key not in raw2["dec2"]:
            res = safe(graph.decisions_verify, text, rq, "v2")
            if res: raw2["dec2"][key] = {q["id"]: p for q, (p, _) in zip(d["requirements"], res)}
        if key not in raw2["gpt2"]:
            res = safe(graph.gpt_verify, text, rq)
            if res: raw2["gpt2"][key] = {q["id"]: p for q, p in zip(d["requirements"], res)}
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    save2()


def build_rows():
    rows = []
    for d in data:
        for r in d["resumes"]:
            rid = r["resume_id"]
            if not all(rid in x for x in (raw["decisions"], raw["baseline"], raw2["dec2"], raw2["gpt2"])):
                continue
            for q in d["requirements"]:
                i = q["id"]
                p = {"dec_v1": raw["decisions"][rid][i]["p"], "gpt_v1": raw["baseline"][rid][i] or 0.0,
                     "dec_v2": raw2["dec2"][rid][i], "gpt_v2": raw2["gpt2"][rid][i]}
                p["ens_v1"] = (p["dec_v1"] + p["gpt_v1"]) / 2
                p["ens_v2"] = (p["dec_v2"] + p["gpt_v2"]) / 2
                rows.append({"posting_id": d["posting_id"], "resume_id": rid, "profile": r["profile"], "req_id": i,
                             "required": q["required"], "truth": int(r["truth"][i] == "met"), "hardness": r["hardness"][i], **p})
    return rows


def best_thr(rows, key):
    y = [r["truth"] for r in rows]; p = np.array([r[key] for r in rows])
    return float(max(np.linspace(0.05, 0.95, 19), key=lambda t: f1_score(y, (p >= t).astype(int), zero_division=0)))


def metrics(rows, key, thr):
    y = np.array([r["truth"] for r in rows]); p = np.array([r[key] for r in rows]); pred = (p >= thr).astype(int)
    by = {}
    for h in ("explicit", "paraphrased", "near_miss", "absent"):
        idx = [i for i, r in enumerate(rows) if r["hardness"] == h]
        by[h] = round(float(np.mean([pred[i] == y[i] for i in idx])), 3)
    return {"n": len(y), "thr": round(thr, 2), "f1": round(f1_score(y, pred, zero_division=0), 3), "auc": round(roc_auc_score(y, p), 3),
            "precision": round(float((y[pred == 1] == 1).mean()), 3) if pred.sum() else 0.0,
            "recall": round(float((pred[y == 1] == 1).mean()), 3),
            "brier": round(float(np.mean((p - y) ** 2)), 3), "accuracy_by_case_type": by}


def paired_bootstrap(rows, key_a, thr_a, key_b, thr_b, reps=1000, seed=1):
    by = {}
    for r in rows: by.setdefault(r["posting_id"], []).append(r)
    keys = list(by); rng = random.Random(seed); diffs_f1, diffs_auc = [], []
    for _ in range(reps):
        s = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        y = [r["truth"] for r in s]
        if len(set(y)) < 2: continue
        fa = f1_score(y, [int(r[key_a] >= thr_a) for r in s], zero_division=0); fb = f1_score(y, [int(r[key_b] >= thr_b) for r in s], zero_division=0)
        diffs_f1.append(fb - fa); diffs_auc.append(roc_auc_score(y, [r[key_b] for r in s]) - roc_auc_score(y, [r[key_a] for r in s]))
    q = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return {"f1_diff_ci95": q(diffs_f1), "auc_diff_ci95": q(diffs_auc)}


def end_to_end(rows, key, thr):
    sc = {}
    for r in rows:
        sc.setdefault(r["resume_id"], {"p": r["profile"], "post": r["posting_id"], "t": [], "y": []})
        if r["required"]:
            sc[r["resume_id"]]["t"].append(r["truth"]); sc[r["resume_id"]]["y"].append(int(r[key] >= thr))
    true_s = [np.mean(v["t"]) for v in sc.values() if v["t"]]; pred_s = [np.mean(v["y"]) for v in sc.values() if v["t"]]
    by = {}
    for v in sc.values():
        if v["t"]: by.setdefault(v["post"], {})[v["p"]] = np.mean(v["y"])
    ok = sum(1 for s in by.values() if len(s) == 3 and s["strong"] > s["partial"] > s["weak"])
    return {"spearman": round(float(spearmanr(true_s, pred_s).statistic), 3), "ranked_correctly": f"{ok}/{len(by)}"}


# ------------------------------------------------------------------ extraction
class _Genuine(BaseModel):
    genuine: list[bool]


def run_extraction_v2():
    def one(d):
        pid = d["posting_id"]
        if pid in raw2["extract2"]: return
        text = graph.sanitize_untrusted(d["posting_text"])[0]
        ext = safe(graph.extract_from_text, text)
        if ext is None: return
        raw2["extract2"][pid] = [r.model_dump() for r in ext]
        gold = [r["text"] for r in d["requirements"]]; et = [r.text for r in ext]
        raw2["align2"][pid] = {"gold_to_ext": v1.align(gold, et), "ext_to_gold": v1.align(et, gold)}
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, data))
    save2()

    def judge(job):
        cfg, d = job
        pid = d["posting_id"]; key = f"{cfg}|{pid}"
        if key in raw2["judge"]: return
        ext = raw["extract"][pid] if cfg == "v1" else raw2["extract2"][pid]
        al = raw["align"][pid] if cfg == "v1" else raw2["align2"][pid]
        unmatched = [i for i, m in enumerate(al["ext_to_gold"]) if m < 0]
        if not unmatched: raw2["judge"][key] = {}; return
        listing = "\n".join(f"{n}: {ext[i]['text']}" for n, i in enumerate(unmatched))
        res = safe(lambda: v1.llm.with_structured_output(_Genuine).invoke([
            ("system", "For each numbered item, decide whether it is a genuine candidate qualification or requirement (skill, experience, education, certification, language, or an ability the role truly needs) that is actually stated in the job posting. Answer false for benefits, company description, hiring process, application instructions, or anything not in the posting. One boolean per item, in order."),
            ("user", f"Posting:\n{d['posting_text']}\n\nItems:\n{listing}")]).genuine)
        if res is None: return
        res = (list(res) + [False] * len(unmatched))[:len(unmatched)]
        raw2["judge"][key] = {str(i): bool(g) for i, g in zip(unmatched, res)}
    jobs = [(c, d) for c in ("v1", "v2") for d in data if (d["posting_id"] in (raw["align"] if c == "v1" else raw2["align2"]))]
    with ThreadPoolExecutor(6) as ex: list(ex.map(judge, jobs))
    save2()


def extraction_report():
    out = {}
    for cfg in ("v1", "v2"):
        al_all = raw["align"] if cfg == "v1" else raw2["align2"]; ex_all = raw["extract"] if cfg == "v1" else raw2["extract2"]
        rec, matched, total, genuine, flag_ok, flag_n = [], 0, 0, 0, 0, 0
        for d in data:
            pid = d["posting_id"]
            if pid not in al_all or not ex_all.get(pid): continue
            al, ext = al_all[pid], ex_all[pid]
            rec += [m >= 0 for m in al["gold_to_ext"]]
            matched += sum(m >= 0 for m in al["ext_to_gold"]); total += len(ext)
            genuine += sum(raw2["judge"].get(f"{cfg}|{pid}", {}).values())
            for gi, ei in enumerate(al["gold_to_ext"]):
                if ei >= 0:
                    flag_n += 1; flag_ok += d["requirements"][gi]["required"] == ext[ei]["required"]
        out[cfg] = {"extracted_items": total, "recall": round(float(np.mean(rec)), 3), "precision_vs_gold_list": round(matched / total, 3),
                    "precision_counting_judged_genuine_extras": round((matched + genuine) / total, 3),
                    "required_flag_accuracy": round(flag_ok / flag_n, 3)}
    return out


def main():
    run_v2_verifiers()
    rows = build_rows()
    dev = [r for r in rows if r["posting_id"] in DEV]; test = [r for r in rows if r["posting_id"] in TEST]
    cands = ["dec_v1", "gpt_v1", "dec_v2", "gpt_v2", "ens_v1", "ens_v2"]
    res = {"split": {"dev_postings": len(DEV), "test_postings": len(TEST), "dev_pairs": len(dev), "test_pairs": len(test)}}
    res["dev_selection_auc"] = {k: round(roc_auc_score([r["truth"] for r in dev], [r[k] for r in dev]), 4) for k in cands}
    chosen = max(cands, key=lambda k: res["dev_selection_auc"][k])
    thr = best_thr(dev, chosen)
    res["chosen"] = {"system": chosen, "threshold_tuned_on_dev": round(thr, 2)}
    res["test_all_candidates_at_0.5"] = {k: metrics(test, k, 0.5) for k in cands}
    res["test_original_vs_chosen"] = {"original(dec_v1@0.5)": metrics(test, "dec_v1", 0.5), f"chosen({chosen}@{thr:.2f})": metrics(test, chosen, thr),
                                      "paired_bootstrap_chosen_minus_original": paired_bootstrap(test, "dec_v1", 0.5, chosen, thr)}
    res["test_end_to_end"] = {"original": end_to_end(test, "dec_v1", 0.5), "chosen": end_to_end(test, chosen, thr)}
    res["all_data_end_to_end_sanity"] = {"original": end_to_end(rows, "dec_v1", 0.5), "chosen": end_to_end(rows, chosen, thr)}
    run_extraction_v2()
    res["extraction_all_36_postings"] = extraction_report()
    json.dump(res, open(os.path.join(HERE, "results", "improve.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
