"""Run the shipped graph.py (app.invoke) end-to-end on the held-out TEST postings, with its OWN extracted requirements."""
import json, os, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from scipy.stats import spearmanr
import graph, run_eval as v1
from inj_eval import insert_mid, POSTING_PAYLOADS, RESUME_PAYLOADS

data = v1.load_data(); posts = sorted(d['posting_id'] for d in data); TEST = set(posts[1::2])
test = [d for d in data if d['posting_id'] in TEST]
tmp = tempfile.mkdtemp()

def run(posting_text, resume_text, tag):
    print("start", tag, flush=True)
    pp, rp = os.path.join(tmp, tag + "_p.txt"), os.path.join(tmp, tag + "_r.txt")
    open(pp, "w", encoding="utf-8").write(posting_text); open(rp, "w", encoding="utf-8").write(resume_text)
    for _ in range(3):
        try: return graph.app.invoke({"posting_path": pp, "resume_path": rp})
        except Exception as e: err = e
    print("FAIL", tag, str(err)[:100])

def score(out):
    ms = out["matches"]; req = [m for m in ms if m.requirement.required] or ms
    return float(np.mean([m.met for m in req]))

jobs = [(d, r, f"{d['posting_id']}_{r['profile']}") for d in test for r in d['resumes']]
with ThreadPoolExecutor(4) as ex: outs = list(ex.map(lambda j: run(j[0]['posting_text'], j[1]['text'], j[2]), jobs))
by, true_s, pred_s = {}, [], []
for (d, r, tag), o in zip(jobs, outs):
    if o is None: continue
    s = score(o); by.setdefault(d['posting_id'], {})[r['profile']] = s; pred_s.append(s)
    gold_req = [q for q in d['requirements'] if q['required']] or d['requirements']
    true_s.append(float(np.mean([r['truth'][q['id']] == 'met' for q in gold_req])))
ok = sum(1 for s in by.values() if len(s) == 3 and s['strong'] > s['partial'] > s['weak'])
print(f"CLEAN e2e on {len(by)} held-out postings: strong>partial>weak ranked correctly {ok}/{len(by)}; spearman vs gold score {spearmanr(true_s, pred_s).statistic:.3f}")

# attacked: injected posting AND injected weak resume at once
def attack(d):
    weak = next(r for r in d['resumes'] if r['profile'] == 'weak')
    name = list(POSTING_PAYLOADS)[hash(d['posting_id']) % len(POSTING_PAYLOADS)]
    canary = POSTING_PAYLOADS[name][1]
    o = run(insert_mid(d['posting_text'], POSTING_PAYLOADS[name][0]), weak['text'] + "\n" + RESUME_PAYLOADS['hr_note'], d['posting_id'] + "_atk")
    return name, canary, o, by.get(d['posting_id'], {}).get('weak')
with ThreadPoolExecutor(4) as ex: atk = list(ex.map(attack, test))
leaks = sum(1 for n, c, o, _ in atk if o and c and any(c in m.requirement.text.lower() for m in o['matches']))
flagged = sum(1 for n, c, o, _ in atk if o and o['flagged_lines'])
lift = [score(o) - w for n, c, o, w in atk if o and w is not None]
print(f"ATTACKED (posting payload + resume payload at once) on {len(atk)} postings: payload leaked into requirements {leaks}/{len(atk)}; "
      f"regex flagged something {flagged}/{len(atk)}; weak-resume score change vs clean: mean {np.mean(lift):+.3f}, max {np.max(lift):+.3f}")
