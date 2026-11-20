"""Extraction v1 (original) vs v2 (first hardened) vs v3 (current graph.py prompt), by dev/test half."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import graph, run_eval as v1
from improve_eval import _Genuine, safe

raw = json.load(open('results/raw.json', encoding='utf-8')); raw2 = json.load(open('results/improve_raw.json', encoding='utf-8'))
raw2.setdefault("extract4", {}); raw2.setdefault("align4", {})
data = v1.load_data(); posts = sorted(d['posting_id'] for d in data); DEV, TEST = set(posts[0::2]), set(posts[1::2])

def one(d):
    pid = d['posting_id']
    if pid in raw2['extract4']: return
    ext = safe(graph.extract_from_text, graph.sanitize_posting(d["posting_text"])[0])
    if ext is None: return
    raw2['extract4'][pid] = [r.model_dump() for r in ext]
    raw2['align4'][pid] = {"gold_to_ext": v1.align([r['text'] for r in d['requirements']], [r.text for r in ext]),
                           "ext_to_gold": v1.align([r.text for r in ext], [r['text'] for r in d['requirements']])}
with ThreadPoolExecutor(6) as ex: list(ex.map(one, data))

def judge(d):
    pid = d['posting_id']; key = f"v4|{pid}"
    if key in raw2['judge'] or pid not in raw2['align4']: return
    ext, al = raw2['extract4'][pid], raw2['align4'][pid]
    un = [i for i, m in enumerate(al['ext_to_gold']) if m < 0]
    if not un: raw2['judge'][key] = {}; return
    res = safe(lambda: v1.llm.with_structured_output(_Genuine).invoke([
        ("system", "For each numbered item, decide whether it is a genuine candidate qualification or requirement (skill, experience, education, certification, language, or an ability the role truly needs) that is actually stated in the job posting. Answer false for benefits, company description, hiring process, application instructions, or anything not in the posting. One boolean per item, in order."),
        ("user", f"Posting:\n{d['posting_text']}\n\nItems:\n" + "\n".join(f"{n}: {ext[i]['text']}" for n, i in enumerate(un)))]).genuine)
    if res is not None:
        res = (list(res) + [False] * len(un))[:len(un)]; raw2['judge'][key] = {str(i): bool(g) for i, g in zip(un, res)}
with ThreadPoolExecutor(6) as ex: list(ex.map(judge, data))
json.dump(raw2, open('results/improve_raw.json', 'w', encoding='utf-8'))

src = {"v1_original": ("v1", raw['align'], raw['extract']), "v2_first_hardening": ("v2", raw2['align2'], raw2['extract2']), "v4_final": ("v4", raw2['align4'], raw2['extract4'])}
out = {}
for name, (cfg, AL, EX) in src.items():
    out[name] = {}
    for split, S in (("dev", DEV), ("test", TEST), ("all", DEV | TEST)):
        rec, matched, total, gen, fo, fn = [], 0, 0, 0, 0, 0
        for d in data:
            pid = d['posting_id']
            if pid not in S or pid not in AL or not EX.get(pid): continue
            al, ext = AL[pid], EX[pid]
            rec += [m >= 0 for m in al['gold_to_ext']]; matched += sum(m >= 0 for m in al['ext_to_gold']); total += len(ext)
            gen += sum(raw2['judge'].get(f"{cfg}|{pid}", {}).values())
            for gi, ei in enumerate(al['gold_to_ext']):
                if ei >= 0: fn += 1; fo += d['requirements'][gi]['required'] == ext[ei]['required']
        out[name][split] = {"items": total, "recall": round(float(np.mean(rec)), 3), "precision_vs_gold": round(matched / total, 3),
                            "precision_with_judged_genuine_extras": round((matched + gen) / total, 3), "required_flag_accuracy": round(fo / fn, 3)}
json.dump(out, open('results/extraction_compare4.json', 'w'), indent=1)
for n, v in out.items():
    print(n)
    for s, m in v.items(): print("  ", s, m)
