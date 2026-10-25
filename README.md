# RepoLens
Paste a GitHub repo, get an architecture diagram, module summaries, a reading path, and a Q&A chat.

## Run locally
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    export GITHUB_TOKEN=ghp_...
    uvicorn main:app --reload

## Deploy (Render)
Build: pip install -r requirements.txt
Start: uvicorn main:app --host 0.0.0.0 --port $PORT
Env vars: ANTHROPIC_API_KEY, GITHUB_TOKEN
