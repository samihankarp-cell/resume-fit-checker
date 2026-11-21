# resume-fit-checker

A LangGraph pipeline that checks resumes against a job posting: it extracts the posting's requirements **once**, then scores how well each resume satisfies each requirement. It ships with an evaluation harness, a cost/speed benchmark and a prompt-injection red-team, so the numbers below are measured, not assumed.

## How it works

```
STAGE 1  (once per posting)
  posting ─► regex cleaner + LLM injection filter ─► GPT-4o-mini extracts requirements ─► saved to requirements_cache/

STAGE 2  (for every resume)
  load the saved requirements ─► clean the resume ─► Google Decisions API scores every requirement in ONE call
  probability >= 0.5 -> YES, otherwise NO      output: per-requirement YES/NO + "required met X / Y"
```

- **Extract once, reuse.** Saved lists are keyed to the posting text plus the prompts, so editing either invalidates them. Every candidate is scored against the identical list.
- **Verifier.** `VERIFIER=decisions` (default) uses the Decisions API alone. `VERIFIER=ensemble` averages Decisions with GPT-4o-mini (threshold 0.4): about +3 F1 points, but about 8x slower and about 2x the verification cost.
- **Code.** `graph.py` (LangGraph `StateGraph`, Pydantic models). `check_resumes()` runs the loop.

## Results

**Test data.** 36 real job postings and 108 synthetic resumes (3 per posting: strong, partial, weak). Each resume was generated against a fixed answer key of which requirements it meets, giving 1,221 labeled judgments. Settings were chosen on half of the postings and reported on the other half.

### Verification step (held-out half: 18 postings, 591 judgments, verifier given the correct requirement list)

| Verifier | F1 | AUC | Precision | Recall | Ranked strong > partial > weak |
|---|---|---|---|---|---|
| GPT-4o-mini only | 0.896 | 0.954 | 0.840 | 0.960 | 14 / 18 |
| Decisions API only (default) | 0.891 | 0.962 | 0.899 | 0.884 | 15 / 18 |
| Both averaged (`VERIFIER=ensemble`) | **0.925** | **0.976** | 0.887 | **0.967** | **17 / 18** |

The two models fail in different ways: GPT says yes to near-misses more often, Decisions is stricter but misses skills shown without the exact keyword. Adding Decisions to GPT gains about 3 F1 points (95% bootstrap interval +1.4 to +3.9 over all 36 postings).

### Whole workflow, each system doing its own extraction (50 resumes, 17 postings)

| | A: GPT extracts + GPT verifies | B: GPT extracts + Decisions verifies |
|---|---|---|
| F1 over every requirement in the answer key | 0.795 | 0.797 |
| Precision / recall | 0.764 / 0.828 | 0.848 / 0.752 |
| Ranked strong > partial > weak | 15 / 17 | 14 / 17 |
| Median time per resume (no caching) | 8.6s | 6.1s |

Accuracy is a statistical tie (F1 difference 95% interval -0.04 to +0.04). The end-to-end F1 is lower than the verification-only numbers because requirements the system fails to extract count as misses.

### Cost and speed with extract-once (10 postings x 10 resumes)

| | A: GPT verifies | B: Decisions verifies |
|---|---|---|
| Once per posting (extraction) | 5.6s, $0.0007 | 5.6s, $0.0007 |
| Per resume, median check time | 2.39s | **0.33s** |
| Per resume, cost | $0.000251 | $0.000243 |
| At 100 resumes per posting: cost per 1,000 resumes | $0.258 | $0.250 |
| At 100 resumes per posting: time per resume | 2.7s | **0.57s** |

Before caching, every resume re-read the posting: about $0.87 per 1,000 resumes. Extracting once cuts that by about 70% at 100 resumes per posting. Prices used: GPT-4o-mini $0.15 / $0.60 per 1M input / output tokens; Decisions $0.10 per 1M input tokens (taken from an AI-generated search summary, not verified against Google's pricing page). The cost reduction comes from caching; Decisions' contribution is speed at matching cost and accuracy.

### Requirement extraction (final prompt, all 36 postings)

Recall of true requirements 90%; 97% of extracted items are genuine requirements (83% originally); required-vs-optional accuracy 90%.

### Prompt injection (hidden instructions in a posting or resume)

| Attack | Original | Final |
|---|---|---|
| Fake requirement planted in a posting, fresh phrasings (n=70-72) | 67% succeed | **10%** |
| "This candidate meets every requirement" planted in a resume (non-matches wrongly passed) | 45% | **3%** |

Defenses are layered: a regex sanitizer for instruction-like sentences, hardened prompts that mark input as untrusted data, and a conservative LLM filter on postings.

## Run it

```bash
pip install langgraph langchain-openai google-genai pydantic pypdf python-dotenv scikit-learn scipy numpy requests
```

Create a `.env` with `OPENAI_API_KEY` and `GEMINI_API_KEY`. The Decisions API (`gemini-decision-1`) needs access on your Gemini account.

```bash
python graph.py posting.pdf resumes_folder/     # extracts the posting once, scores every resume in the folder
python graph.py                                  # defaults: posting.pdf and resume.pdf in this folder
VERIFIER=ensemble python graph.py ...            # use both judges
```

Evals (they call paid APIs; results are cached in `evals/results/`):

```bash
python evals/run_eval.py             # first-pass verification evals
python evals/improve_eval.py         # dev/test experiments
python evals/inj_eval.py             # injection red-team
python evals/e2e_systems.py          # whole-workflow comparison of systems A and B
python evals/cost_speed_two_stage.py # cost and speed with extract-once
```

## Repo layout

- `graph.py` - the pipeline, defenses and two-stage loop
- `evals/data/postings/` - the 36 postings; `evals/data/synth/` - generated resumes plus answer keys
- `evals/*.py` - harness, experiments, benchmarks; `evals/results/` - saved metrics
- `evals/graph_v1_backup.py` - the original pipeline, kept for comparison

## Limitations

- **Synthetic resumes and labels.** The resumes were written by an LLM (Claude Haiku) to a known answer key, not by real candidates, and are shorter than typical resumes. About 1.5% of labels look wrong.
- **LLM-graded matching.** Matching extracted requirements to the answer key, and judging "genuine requirement", used GPT-4o-mini, the same family as the extractor.
- **Small samples.** 36 postings; the whole-workflow comparison used 50 resumes; the injection tests used 12 postings. Attack phrasings were written by the author.
- **Injection is reduced, not solved.** About 10% of fresh attacks still get through, mostly ones phrased as ordinary requirements. The LLM filter removes a few real duty sentences (about 5 of roughly 570 extracted items) and falls back to the regex and prompt defenses if the filter call errors. Hidden text inside PDFs (for example white-on-white keywords) is extracted like normal text and was not tested.
- **PDF parsing.** Resumes are read with `pypdf`, which can scramble multi-column layouts.
- **Two-stage accuracy not re-measured.** The extract-once change reuses the same extraction and verification steps; its effect on accuracy was not separately re-run.
- **Recall trade-off.** Hardening the extractor cost about 4 points of recall.

## Data attribution

Job postings were collected from the public [Remotive](https://remotive.com) and [Arbeitnow](https://www.arbeitnow.com) APIs. The posting text belongs to the respective employers and is included only as evaluation data; each posting's source link is in `evals/data/postings.json`.
