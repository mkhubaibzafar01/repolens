import os, json, re, httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY", "")
GH_TOKEN = os.getenv("GITHUB_TOKEN", "")

app = FastAPI(title="RepoLens")
CACHE = {}

SKIP = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".lock", ".min.js", ".map",
        ".woff", ".woff2", ".ttf", ".pdf", ".zip", ".mp4", ".mp3")
SKIP_DIRS = ("node_modules/", "dist/", "build/", ".git/", "vendor/", "__pycache__/", "test", "docs/")
KEY_NAMES = ("readme", "main", "app", "index", "server", "package.json", "requirements.txt",
             "pyproject.toml", "go.mod", "cargo.toml", "dockerfile", "manage.py", "setup.py")


class Repo(BaseModel):
    repo_url: str

class Question(Repo):
    question: str


def parse_repo(url: str):
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        raise HTTPException(400, "Enter a link like https://github.com/owner/repo")
    return m.group(1), m.group(2).removesuffix(".git")


def gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    return h


def load_repo(url: str):
    owner, name = parse_repo(url)
    key = f"{owner}/{name}"
    if key in CACHE:
        return key, CACHE[key]
    with httpx.Client(timeout=30, headers=gh_headers()) as c:
        info = c.get(f"https://api.github.com/repos/{key}")
        if info.status_code == 404:
            raise HTTPException(404, "Repo not found or it is private.")
        info.raise_for_status()
        branch = info.json()["default_branch"]
        tree = c.get(f"https://api.github.com/repos/{key}/git/trees/{branch}?recursive=1").json().get("tree", [])
        paths = [t for t in tree if t["type"] == "blob" and not t["path"].lower().endswith(SKIP)
                 and not any(d in t["path"].lower() for d in SKIP_DIRS) and t.get("size", 0) < 60000]

        def score(t):
            p = t["path"].lower()
            base = p.split("/")[-1]
            s = 0
            if any(base.startswith(k) or base == k for k in KEY_NAMES): s += 10
            s -= p.count("/") * 2
            s += min(t.get("size", 0), 20000) / 20000 * 3
            return s

        chosen = sorted(paths, key=score, reverse=True)[:25]
        files = {}
        for t in chosen:
            r = c.get(f"https://raw.githubusercontent.com/{key}/{branch}/{t['path']}")
            if r.status_code == 200:
                files[t["path"]] = r.text[:6000]
    data = {"tree": [t["path"] for t in paths][:400], "files": files, "desc": info.json().get("description") or ""}
    CACHE[key] = data
    return key, data


def context(data):
    out = "FILE TREE:\n" + "\n".join(data["tree"]) + "\n\nKEY FILES:\n"
    for p, txt in data["files"].items():
        out += f"\n===== {p} =====\n{txt}\n"
    return out[:90000]


def ask_claude(system, user, max_tokens=3000):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": 8192, "thinkingConfig": {"thinkingBudget": 0}},
    }
    r = httpx.post(url, json=payload, headers={"x-goog-api-key": API_KEY}, timeout=90)
    if r.status_code == 429:
        raise HTTPException(429, "Free limit reached. Wait a minute and try again.")
    if r.status_code != 200:
        raise HTTPException(502, f"AI error: {r.text[:200]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


@app.post("/api/analyze")
def analyze(body: Repo):
    key, data = load_repo(body.repo_url)
    system = ("You are a senior engineer onboarding a new teammate. Respond with ONLY valid JSON, no markdown fences. "
              'Schema: {"overview": str (3-4 plain-English sentences), "tech_stack": [str], '
              '"modules": [{"name": str, "path": str, "summary": str}], '
              '"reading_path": [{"file": str, "why": str}] (6-8 items, beginner order, only files that exist in the tree), '
              '"mermaid": str}. '
              'The mermaid value is a "graph TD" diagram of 5-9 nodes. Use simple ids (A, B, C), '
              'labels in double quotes like A["API Server"], and no special characters or parentheses in labels.')
    raw = ask_claude(system, f"Repo: {key}\nDescription: {data['desc']}\n\n{context(data)}")
    raw = re.sub(r"^`{3}(?:json)?|`{3}$", "", raw.strip(), flags=re.M).strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, "AI returned an unreadable answer. Try again.")
    result["repo"] = key
    return result


@app.post("/api/ask")
def ask(body: Question):
    key, data = load_repo(body.repo_url)
    system = ("You answer questions about a GitHub repository using ONLY the provided code. "
              "Be concise, cite file paths in backticks, and say so if the answer is not in the provided files.")
    answer = ask_claude(system, f"{context(data)}\n\nQUESTION: {body.question}", 1200)
    return {"answer": answer}


@app.get("/")
def home():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")