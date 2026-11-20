import os, time
from dotenv import load_dotenv
from google import genai
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
c = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
t = time.time()
try:
    r = c.decisions.create(model="gemini-decision-1",
        input="Jane Doe. 5 years Python backend engineer. BSc Computer Science. Built REST APIs with Django.",
        questions={"req_0": {"type": "noul", "instructions": "Based only on the resume, does the candidate clearly satisfy this job requirement: 3+ years of Python experience"},
                   "req_1": {"type": "noul", "instructions": "Based only on the resume, does the candidate clearly satisfy this job requirement: Experience with Kubernetes"}})
    for k, a in r.answers.items():
        print(k, a.noul, a.confidence)
    print("latency", round(time.time() - t, 2))
except Exception as e:
    print(type(e).__name__, str(e)[:600])
