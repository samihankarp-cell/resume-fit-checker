"""Cost and speed of the TWO-STAGE pipeline (extract once per posting, then loop over resumes), systems A and B.
  A = GPT extracts once, GPT verifies each resume        B = GPT extracts once, Decisions verifies each resume (graph.py)
10 postings x 10 resumes each, one call at a time. Stage 1 (per posting) and stage 2 (per resume) are measured separately."""
import os
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
import json, shutil, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from langchain_core.callbacks import get_usage_metadata_callback
import graph, run_eval as v1

assert graph.VERIFIER == "decisions"
PRICE = {"gpt_in": 0.15, "gpt_out": 0.60, "dec_in": 0.10, "dec_out": 0.0}  # USD per 1M tokens, as supplied by the user
shutil.rmtree(graph.REQ_CACHE_DIR, ignore_errors=True)

dec_log = []; _real = graph.gemini
class _Dec:
    def create(self, **kw):
        r = _real.decisions.create(**kw); u = r.usage
        dec_log.append((int(u.total_input_tokens or 0), int(u.total_output_tokens or 0)) if u else (0, 0)); return r
class _Client: decisions = _Dec()
graph.gemini = _Client()

def measure(fn):
    n = len(dec_log)
    with get_usage_metadata_callback() as cb:
        t = time.perf_counter(); out = fn(); secs = time.perf_counter() - t
    gi = sum(v.get("input_tokens", 0) for v in cb.usage_metadata.values()); go = sum(v.get("output_tokens", 0) for v in cb.usage_metadata.values())
    di = sum(x[0] for x in dec_log[n:]); do = sum(x[1] for x in dec_log[n:])
    cost = (gi * PRICE["gpt_in"] + go * PRICE["gpt_out"] + di * PRICE["dec_in"] + do * PRICE["dec_out"]) / 1e6
    return out, {"secs": secs, "gpt_in": gi, "gpt_out": go, "dec_in": di, "cost": cost}

data = v1.load_data(); posts = sorted(d["posting_id"] for d in data)
test = [d for d in data if d["posting_id"] in set(posts[1::2])][:10]
pool = [(r["resume_id"], r["text"], d["posting_id"]) for d in data if d["posting_id"] in set(posts[1::2]) for r in d["resumes"]]
tmp = tempfile.mkdtemp()
S1 = {"A": [], "B": []}; S2 = {"A": [], "B": []}; reqs_n = []; fails = []
for i, d in enumerate(test):
    pp = os.path.join(tmp, d["posting_id"] + ".txt"); open(pp, "w", encoding="utf-8").write(d["posting_text"])
    own = [(r["resume_id"], r["text"]) for r in d["resumes"]]
    others = [(rid, t) for rid, t, pid in pool if pid != d["posting_id"]]
    pick = own + [others[(i * 7 + k) % len(others)] for k in range(7)]
    paths = []
    for rid, t in pick:
        rp = os.path.join(tmp, f"{i}_{rid}.txt"); open(rp, "w", encoding="utf-8").write(t); paths.append((rid, rp, t))
    # ---- stage 1, once per posting, per system
    (rA, mA) = measure(lambda: graph.extract_from_text(graph.sanitize_posting(d["posting_text"])[0]))
    (rB, mB) = measure(lambda: graph.extract_once(pp)[0])
    S1["A"].append(mA); S1["B"].append(mB); reqs_n.append(len(rB))
    # ---- stage 2, per resume
    for j, (rid, rp, t) in enumerate(paths):
        for s in (("A", "B") if j % 2 == 0 else ("B", "A")):
            try:
                if s == "A":
                    _, m = measure(lambda: graph.gpt_verify(graph.sanitize_untrusted(t)[0], rA))
                else:
                    _, m = measure(lambda: graph.app.invoke({"posting_path": pp, "resume_path": rp}))
                S2[s].append(m)
            except Exception as e:  # noqa: BLE001
                fails.append(f"{s}|{rid}|{type(e).__name__}")
    print(f"posting {i + 1}/10 done ({len(rB)} requirements)  stage1 A {mA['secs']:.1f}s B {mB['secs']:.1f}s", flush=True)

M = lambda L, k: float(np.mean([x[k] for x in L])); P = lambda L, k, q: float(np.percentile([x[k] for x in L], q))
out = {"postings": len(test), "resumes_per_posting": 10, "requirements_per_posting_mean": float(np.mean(reqs_n)), "failed_calls": fails, "systems": {}}
for s in "AB":
    ext_cost, chk_cost = M(S1[s], "cost"), M(S2[s], "cost")
    out["systems"][s] = {
        "stage1_per_posting": {"secs_median": float(np.median([x["secs"] for x in S1[s]])), "cost_usd": ext_cost, "gpt_in": M(S1[s], "gpt_in"), "gpt_out": M(S1[s], "gpt_out")},
        "stage2_per_resume": {"n": len(S2[s]), "secs_median": float(np.median([x["secs"] for x in S2[s]])), "secs_mean": M(S2[s], "secs"), "secs_p90": P(S2[s], "secs", 90),
                              "cost_usd": chk_cost, "gpt_in": M(S2[s], "gpt_in"), "gpt_out": M(S2[s], "gpt_out"), "dec_in": M(S2[s], "dec_in")},
        "per_resume_all_in_by_resumes_per_posting": {str(N): {"cost_usd": (ext_cost + N * chk_cost) / N, "cost_per_1000_usd": 1000 * (ext_cost + N * chk_cost) / N,
                                                             "secs": (float(np.mean([x["secs"] for x in S1[s]])) + N * M(S2[s], "secs")) / N} for N in (1, 10, 100, 1000)},
        "measured_batch_total": {"resumes": len(S2[s]), "cost_usd": sum(x["cost"] for x in S1[s]) + sum(x["cost"] for x in S2[s]),
                                 "secs": sum(x["secs"] for x in S1[s]) + sum(x["secs"] for x in S2[s])}}
json.dump(out, open(os.path.join(HERE, "results", "cost_speed_two_stage.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
