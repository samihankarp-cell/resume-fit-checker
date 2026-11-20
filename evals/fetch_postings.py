"""Collect real, public job postings (Remotive + Arbeitnow public APIs) into data/postings.json."""
import html, json, re, requests

TARGET = 36
MIN_CHARS, MAX_CHARS = 1500, 9000


def clean(raw: str) -> str:
    text = re.sub(r"<(br|/p|/li|/h\d|/div)[^>]*>", "\n", raw, flags=re.I)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def looks_english(t: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", t.lower())
    if not words:
        return False
    common = sum(w in {"the", "and", "to", "of", "with", "you", "for", "a", "in"} for w in words)
    german = sum(w in {"und", "der", "die", "das", "mit", "für", "wir", "sie", "ein", "eine"} for w in words)
    return common / len(words) > 0.12 and german / len(words) < 0.02


postings, seen = [], set()


def add(src, jid, title, company, url, raw):
    body = clean(raw)
    if url in seen or not (MIN_CHARS <= len(body) <= MAX_CHARS) or not looks_english(body):
        return
    seen.add(url)
    postings.append({"id": f"{src}-{jid}", "source": src, "title": title, "company": company, "url": url, "text": body})


for j in requests.get("https://remotive.com/api/remote-jobs", timeout=30).json()["jobs"]:
    add("remotive", j["id"], j["title"], j["company_name"], j["url"], j["description"])

for page in (1, 2, 3):
    data = requests.get("https://www.arbeitnow.com/api/job-board-api", params={"page": page}, timeout=30).json()["data"]
    for j in data:
        if len(postings) >= TARGET:
            break
        add("arbeitnow", j["slug"][-12:], j["title"], j["company_name"], j["url"], j["description"])

postings = postings[:TARGET]
for p in postings:
    with open(f"data/postings/{p['id']}.txt", "w", encoding="utf-8") as f:
        f.write(f"{p['title']} at {p['company']}\n\n{p['text']}")
with open("data/postings.json", "w", encoding="utf-8") as f:
    json.dump(postings, f, indent=1)
print(len(postings), "postings;", len({p['url'] for p in postings}), "unique urls")
for p in postings:
    print(p["id"], "|", p["title"][:60])
