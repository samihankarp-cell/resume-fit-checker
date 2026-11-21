"""Does extract-once + loop work? 3 postings x 3 resumes. Counts AI extraction calls, time and cost vs the earlier no-cache run."""
import os
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
import json, shutil, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from langchain_core.callbacks import get_usage_metadata_callback
import graph, run_eval as v1

PRICE = {"gpt_in": 0.15, "gpt_out": 0.60, "dec_in": 0.10}
shutil.rmtree(graph.REQ_CACHE_DIR, ignore_errors=True)  # start with nothing saved

calls = {"extract": 0}; _orig = graph.extract_from_text
def counted(t): calls["extract"] += 1; return _orig(t)
graph.extract_from_text = counted
dec_log = []; _real = graph.gemini
class _Dec:
    def create(self, **kw):
        r = _real.decisions.create(**kw); u = r.usage; dec_log.append(int(u.total_input_tokens or 0)); return r
class _Client: decisions = _Dec()
graph.gemini = _Client()

raw = json.load(open(os.path.join(HERE, "results", "e2e_systems_raw.json"), encoding="utf-8"))
data = v1.load_data(); posts = sorted(d["posting_id"] for d in data)
test = [d for d in data if d["posting_id"] in set(posts[1::2])]
chosen = [d for d in test if all(f"B|{r['resume_id']}" in raw["runs"] for r in d["resumes"])][:3]
tmp = tempfile.mkdtemp()

def cost(gi, go, di): return (gi * PRICE["gpt_in"] + go * PRICE["gpt_out"] + di * PRICE["dec_in"]) / 1e6
new = {"secs": 0.0, "cost": 0.0, "resumes": 0}; old = {"secs": 0.0, "cost": 0.0}
for d in chosen:
    pp = os.path.join(tmp, d["posting_id"] + ".txt"); open(pp, "w", encoding="utf-8").write(d["posting_text"])
    rps = []
    for r in d["resumes"]:
        rp = os.path.join(tmp, r["resume_id"] + ".txt"); open(rp, "w", encoding="utf-8").write(r["text"]); rps.append(rp)
        b = raw["runs"][f"B|{r['resume_id']}"]
        old["secs"] += b["secs"]; old["cost"] += cost(b["gpt_in"], b["gpt_out"], b["dec_in"])
    n0, d0 = calls["extract"], len(dec_log)
    with get_usage_metadata_callback() as cb:
        t = time.perf_counter(); results = graph.check_resumes(pp, rps); secs = time.perf_counter() - t
    gi = sum(v.get("input_tokens", 0) for v in cb.usage_metadata.values()); go = sum(v.get("output_tokens", 0) for v in cb.usage_metadata.values())
    lists = [tuple(m.requirement.text for m in out["matches"]) for out in results.values()]
    print(f"{d['posting_id']}: extraction AI calls = {calls['extract'] - n0} (3 resumes) | all 3 resumes scored against the SAME list: {len(set(lists)) == 1} "
          f"({len(lists[0])} requirements) | {secs:.1f}s total, {secs / 3:.1f}s per resume", flush=True)
    new["secs"] += secs; new["cost"] += cost(gi, go, sum(dec_log[d0:])); new["resumes"] += 3

# second run of the same posting: should load from disk with ZERO extraction calls (persistence)
d = chosen[0]; pp = os.path.join(tmp, d["posting_id"] + ".txt"); n0 = calls["extract"]
reqs, _, cached = graph.extract_once(pp)
print(f"\nRe-opening {d['posting_id']} later: loaded from saved file = {cached}, extraction AI calls = {calls['extract'] - n0}")
print("saved files:", sorted(os.listdir(graph.REQ_CACHE_DIR)))
n = new["resumes"]
print(f"\nSAME 9 RESUMES   earlier (re-extract every time): {old['secs'] / n:.1f}s and ${old['cost'] / n:.5f} per resume  -> ${1000 * old['cost'] / n:.2f} per 1,000")
print(f"                 now (extract once per posting) : {new['secs'] / n:.1f}s and ${new['cost'] / n:.5f} per resume  -> ${1000 * new['cost'] / n:.2f} per 1,000")
print(f"                 extraction AI calls: {calls['extract'] - 0} now vs {n} before")
