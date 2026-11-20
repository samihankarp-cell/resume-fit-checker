import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader
from google import genai
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"},
)

YES_THRESHOLD = 0.5


class Requirement(BaseModel):
    text: str = Field(description="The requirement as written")
    category: str = Field(description="skill, education, experience, certification, or other")
    required: bool = Field(description="True if mandatory, False if nice to have")


class Requirements(BaseModel):
    requirements: List[Requirement] = Field(
        description="Every requirement listed in the document. Could be 3 or 30."
    )


class Match(BaseModel):
    requirement: Requirement
    met: bool
    probability: float
    confidence: float


class State(BaseModel):
    posting_path: str
    resume_path: str
    posting_text: str = ""
    resume_text: str = ""
    requirements: Optional[Requirements] = None
    matches: List[Match] = []


def read_document(path: str) -> str:
    with open(path, "rb") as f:
        header = f.read(5)
    if header == b"%PDF-":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_posting(state: State):
    return {"posting_text": read_document(state.posting_path)}


def extract_requirements(state: State):
    structured_llm = llm.with_structured_output(Requirements)
    result = structured_llm.invoke([
        ("system", "You extract EVERY requirement from a job posting. One entry per requirement. Do not skip or merge any."),
        ("user", f"Job Description:\n{state.posting_text}"),
    ])
    return {"requirements": result}


def load_resume(state: State):
    return {"resume_text": read_document(state.resume_path)}


def verify_requirements(state: State):
    reqs = state.requirements.requirements

    questions = {
        f"req_{i}": {
            "type": "noul",
            "instructions": (
                "Based only on the resume, does the candidate clearly satisfy this "
                f"job requirement: {r.text}"
            ),
        }
        for i, r in enumerate(reqs)
    }

    result = gemini.decisions.create(
        model="gemini-decision-1",
        input=state.resume_text,
        questions=questions,
    )

    matches = []
    for i, r in enumerate(reqs):
        answer = result.answers[f"req_{i}"]
        matches.append(Match(
            requirement=r,
            met=answer.noul >= YES_THRESHOLD,
            probability=answer.noul,
            confidence=answer.confidence,
        ))
    return {"matches": matches}


builder = StateGraph(State)
builder.add_node("load_posting", load_posting)
builder.add_node("extract_requirements", extract_requirements)
builder.add_node("load_resume", load_resume)
builder.add_node("verify_requirements", verify_requirements)

builder.add_edge(START, "load_posting")
builder.add_edge("load_posting", "extract_requirements")
builder.add_edge("extract_requirements", "load_resume")
builder.add_edge("load_resume", "verify_requirements")
builder.add_edge("verify_requirements", END)

app = builder.compile()

out = app.invoke({"posting_path": "posting.pdf", "resume_path": "resume.pdf"})

for m in out["matches"]:
    tag = "YES" if m.met else "NO "
    kind = "REQUIRED" if m.requirement.required else "nice to have"
    print(f"{tag} | p={m.probability:.2f} | {kind} | {m.requirement.text}")

required = [m for m in out["matches"] if m.requirement.required]
print(f"\nRequired met: {sum(m.met for m in required)} / {len(required)}")