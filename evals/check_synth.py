"""Validate a generated synth file: python check_synth.py <posting_id>"""
import json, os, sys

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synth", sys.argv[1] + ".json")
try:
    d = json.load(open(path, encoding="utf-8"))
    ids = {r["id"] for r in d["requirements"]}
    assert 8 <= len(ids) <= 14, f"{len(ids)} requirements, need 8-14"
    assert [r["profile"] for r in d["resumes"]] == ["strong", "partial", "weak"], "need strong, partial, weak in order"
    for r in d["resumes"]:
        assert set(r["truth"]) == ids, f"{r['resume_id']}: truth keys mismatch"
        assert set(r["hardness"]) == ids, f"{r['resume_id']}: hardness keys mismatch"
        assert set(r["truth"].values()) <= {"met", "not_met"}, "truth values must be met/not_met"
        assert set(r["hardness"].values()) <= {"explicit", "paraphrased", "absent", "near_miss"}, "bad hardness value"
        assert len(r["text"].split()) >= 120, f"{r['resume_id']}: resume too short"
    print("ok", len(ids), "requirements")
except Exception as e:
    print("ERROR:", type(e).__name__, e)
