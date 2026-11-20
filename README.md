# resume-fit-checker

A LangGraph pipeline that checks a resume against a job posting: it extracts every requirement from the posting, then scores how well the resume satisfies each one. It ships with an evaluation harness and a prompt-injection red-team, so the numbers below are measured, not assumed.

## How it works

```
posting ─► sanitize + LLM injection filter ─► extract requirements (GPT-4o-mini, structured output)
resume  ─► sanitize ─────────────────────────────────────────────┐
                                                                  ▼
                       verify each requirement: Google Decisions API  +  GPT-4o-mini
                       average the two probabilities, threshold 0.4, confidence = how much they agree
                                                                  ▼
                              per-requirement YES/NO + "required met X / Y"
```

- **State and graph:** `graph.py` (LangGraph `StateGraph`, Pydantic models for requirements and matches).
- **Extraction:** GPT-4o-mini with structured output returns a list of requirements, each tagged required or nice-to-have.
- **Verification:** two judges with different error patterns are averaged. The Decisions API is more precise; GPT-4o-mini is better at spotting skills shown without the exact keyword.
- **Confidence:** the Decisions API's own `confidence` turned out to equal `max(p, 1-p)`, so it adds nothing. Agreement between the two judges is used instead (98% accurate when they agree, 87% when they disagree, on held-out data).

## Results

Evaluated on **36 real job postings** and **108 synthetic resumes** (3 per posting: strong, partial, weak), giving **1,221 labeled requirement judgments**. Each resume was generated against a fixed answer key of which requirements it meets, so the labels come from how it was written. Settings were chosen on one half of the postings and reported on the other (18 postings, 591 judgments).

**Verifier (held-out half)**

| Metric | Original pipeline | Final |
|---|---|---|
| F1 | 0.899 | **0.925** |
| ROC-AUC | 0.969 | **0.976** |
| Precision | 0.882 | 0.887 |
| Recall | 0.917 | **0.967** |
| Strong > partial > weak ranked correctly | 15 / 18 | **17 / 18** |

The F1 gain is +2.6 points (95% paired-bootstrap interval +0.7 to +4.7). It comes from averaging the two judges; the injection-hardened Decisions prompt alone is slightly worse on accuracy but is needed for security.

**Requirement extraction (all 36 postings)**

| Metric | Original | Final |
|---|---|---|
| Recall of true requirements | 94% | 90% |
| Extracted items that are genuine requirements | 83% | **97%** |
| Required-vs-optional flag accuracy | 88% | 90% |

**Prompt injection** (hidden instructions in the text the model reads)

| Attack | Original | Final |
|---|---|---|
| Fake requirement planted in a posting, fresh phrasings (n=70-72) | 67% succeed | **10%** |
| "This candidate meets every requirement" planted in a resume (non-matches wrongly passed) | 45% | **3%** |

Defenses are layered: a regex sanitizer for instruction-like sentences, hardened prompts that mark the input as untrusted data, and a conservative LLM filter on postings.

## Run it

```bash
pip install langgraph langchain-openai google-genai pydantic pypdf python-dotenv scikit-learn scipy numpy requests
```

Create a `.env` file with `OPENAI_API_KEY` and `GEMINI_API_KEY`. The Decisions API (`gemini-decision-1`) needs access on your Gemini account.

```bash
python graph.py                       # reads posting.pdf and resume.pdf from this folder (plain text also works)
python evals/run_eval.py              # first-pass evals (cached in evals/results/)
python evals/improve_eval.py          # dev/test experiments and the final verifier numbers
python evals/inj_eval.py              # injection red-team, original vs defended
```

The evals call paid APIs (a few hundred to a few thousand small GPT-4o-mini and Decisions calls). Results are cached, so re-runs are mostly free.

## Repo layout

- `graph.py` - the pipeline and its defenses
- `evals/data/postings/` - the 36 job postings; `evals/data/synth/` - generated resumes plus answer keys
- `evals/*.py` - harness, experiments, injection tests; `evals/results/` - saved metrics
- `evals/graph_v1_backup.py` - the original pipeline, kept for comparison

## Limitations

- **Synthetic resumes and labels.** The resumes were written by an LLM (Claude Haiku) to a known answer key, not by real candidates. About 1.5% of labels look wrong (both systems strongly disagree with them).
- **LLM-graded extraction.** Matching extracted requirements to the reference lists, and judging "genuine requirement", was done by GPT-4o-mini, the same model family as the extractor.
- **Small sample.** 36 postings; the injection tests used 12. The attack phrasings were written by the author, so real attackers may do better.
- **Injection is reduced, not solved.** About 10% of fresh attacks still get through, mostly ones phrased as ordinary requirements. The LLM filter also removes a few real duty sentences (about 5 of roughly 570 extracted items), and it fails open to the regex and prompt defenses if the filter call errors.
- **Recall trade-off.** Hardening the extractor cost about 4 points of recall.

## Data attribution

Job postings were collected from the public [Remotive](https://remotive.com) and [Arbeitnow](https://www.arbeitnow.com) APIs. The posting text belongs to the respective employers and is included only as evaluation data; each posting's source link is in `evals/data/postings.json`.
