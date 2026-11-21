"""Cost and speed of the verifier: GPT-4o-mini only vs Decisions only vs both (the shipped node).
Runs the real functions from graph.py, one call at a time (clean latency), tokens read from each API response."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from langchain_core.callbacks import get_usage_metadata_callback
import graph, run_eval as v1

PRICE = {"gpt_in": 0.15, "gpt_out": 0.60, "dec_in": 0.10, "dec_out": 0.0}  # USD per 1M tokens (as supplied by the user)

# spy on the Decisions client to read response.usage without changing graph.py
dec_log = []
_real = graph.gemini
class _Dec:
    def create(self, **kw):
        r = _real.decisions.create(**kw)
        u = r.usage
        dec_log.append((int(u.total_input_tokens or 0), int(u.total_output_tokens or 0)) if u else (0, 0))
        return r
class _Client:
    decisions = _Dec()
graph.gemini = _Client()

def gpt_tokens(cb):
    i = sum(v.get("input_tokens", 0) for v in cb.usage_metadata.values()); o = sum(v.get("output_tokens", 0) for v in cb.usage_metadata.values())
    return i, o

def timed(fn):
    t = time.perf_counter(); out = fn(); return out, time.perf_counter() - t

data = v1.load_data(); posts = sorted(d["posting_id"] for d in data); TEST = set(posts[1::2])
sample = [d for d in data if d["posting_id"] in TEST][:10]
reqs_of = lambda d: [graph.Requirement(text=q["text"], category=q["category"], required=q["required"]) for q in d["requirements"]]

# warm-up (not counted): open connections
w = sample[0]; wt = graph.sanitize_untrusted(w["resumes"][0]["text"])[0]
graph.gpt_verify(wt, reqs_of(w)); graph.decisions_verify(wt, reqs_of(w)); dec_log.clear()

rows = []
for d in sample:
    for r in d["resumes"]:
        text = graph.sanitize_untrusted(r["text"])[0]; rq = reqs_of(d)
        rec = {"resume_id": r["resume_id"], "n_reqs": len(rq)}
        with get_usage_metadata_callback() as cb:
            _, rec["gpt_s"] = timed(lambda: graph.gpt_verify(text, rq))
        rec["gpt_tok"] = gpt_tokens(cb)
        n = len(dec_log); _, rec["dec_s"] = timed(lambda: graph.decisions_verify(text, rq)); rec["dec_tok"] = dec_log[n]
        st = graph.State(posting_path="", resume_path="", resume_text=text, requirements=graph.Requirements(requirements=rq))
        n = len(dec_log)
        with get_usage_metadata_callback() as cb:
            _, rec["both_s"] = timed(lambda: graph.verify_requirements(st))
        rec["both_gpt_tok"] = gpt_tokens(cb); rec["both_dec_tok"] = dec_log[n]
        rows.append(rec); print("done", r["resume_id"], f'gpt {rec["gpt_s"]:.2f}s dec {rec["dec_s"]:.2f}s both {rec["both_s"]:.2f}s', flush=True)

ext = []
for d in sample:
    with get_usage_metadata_callback() as cb:
        (clean, _), t1 = timed(lambda: graph.sanitize_posting(d["posting_text"]))
        _, t2 = timed(lambda: graph.extract_from_text(clean))
    ext.append({"posting_id": d["posting_id"], "s": t1 + t2, "tok": gpt_tokens(cb)})

gcost = lambda i, o: (i * PRICE["gpt_in"] + o * PRICE["gpt_out"]) / 1e6
dcost = lambda i, o: (i * PRICE["dec_in"] + o * PRICE["dec_out"]) / 1e6
M = lambda xs: float(np.mean(xs)); P = lambda xs, q: float(np.percentile(xs, q))
cfg = {
    "gpt_only":      {"s": [r["gpt_s"] for r in rows],  "in": [r["gpt_tok"][0] for r in rows],  "out": [r["gpt_tok"][1] for r in rows], "cost": [gcost(*r["gpt_tok"]) for r in rows]},
    "decisions_only": {"s": [r["dec_s"] for r in rows], "in": [r["dec_tok"][0] for r in rows],  "out": [r["dec_tok"][1] for r in rows], "cost": [dcost(*r["dec_tok"]) for r in rows]},
    "both_shipped":  {"s": [r["both_s"] for r in rows], "in": [r["both_gpt_tok"][0] + r["both_dec_tok"][0] for r in rows],
                      "out": [r["both_gpt_tok"][1] + r["both_dec_tok"][1] for r in rows],
                      "cost": [gcost(*r["both_gpt_tok"]) + dcost(*r["both_dec_tok"]) for r in rows]},
}
out = {"resumes_timed": len(rows), "requirements_per_resume_mean": M([r["n_reqs"] for r in rows]), "prices_usd_per_1M_tokens": PRICE, "verification": {}}
for k, v in cfg.items():
    out["verification"][k] = {"latency_median_s": float(np.median(v["s"])), "latency_mean_s": M(v["s"]), "latency_p90_s": P(v["s"], 90),
                              "input_tokens_mean": M(v["in"]), "output_tokens_mean": M(v["out"]),
                              "cost_per_resume_usd": M(v["cost"]), "cost_per_1000_resumes_usd": 1000 * M(v["cost"])}
out["adding_decisions_to_gpt"] = {"extra_latency_median_s": float(np.median(np.array(cfg["both_shipped"]["s"]) - np.array(cfg["gpt_only"]["s"]))),
                                  "extra_cost_per_resume_usd": M(cfg["both_shipped"]["cost"]) - M(cfg["gpt_only"]["cost"]),
                                  "extra_cost_per_1000_resumes_usd": 1000 * (M(cfg["both_shipped"]["cost"]) - M(cfg["gpt_only"]["cost"]))}
ec = [gcost(*e["tok"]) for e in ext]
out["extraction_per_posting_shared_by_all_setups"] = {"latency_median_s": float(np.median([e["s"] for e in ext])), "input_tokens_mean": M([e["tok"][0] for e in ext]),
                                                      "output_tokens_mean": M([e["tok"][1] for e in ext]), "cost_per_posting_usd": M(ec), "postings": len(ext)}
json.dump(out, open(os.path.join(HERE, "results", "cost_speed.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
