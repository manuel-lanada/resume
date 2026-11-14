# Resume Digital Twin

A small RAG chatbot that answers questions about your professional background **as you**, in the first person, using only what is in your resume. Recruiters and hiring managers open the page, ask a question, and get an answer grounded in `resume.pdf`.

## How it works

```
resume.pdf ──pypdf──▶ text ──▶ 500-char chunks (100 overlap)
                                   │
                 sentence-transformers (all-MiniLM-L6-v2, local CPU)
                                   ▼
                          FAISS IndexFlatL2 (in memory)
                                   ▲
visitor question ──embed──▶ top 4 chunks ──▶ Gemini ──▶ first-person answer
```

- The resume is parsed, embedded and indexed **once** when the app boots (`@st.cache_resource`). Each chat turn only runs a vector search and one Gemini call.
- The system prompt keeps the twin in the first person, limits it to resume facts, and makes it decline politely when the resume does not cover a question.
- Embeddings are computed locally, so the only paid or rate-limited service is Gemini.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | The whole app: PDF ingestion, FAISS index, Gemini call, Streamlit chat UI |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image for Render or any Docker host |
| `.gitignore` | Keeps `.env`, caches and virtualenvs out of Git |
| `resume.pdf` | **You add this.** Your resume as a text-based PDF |

## Quickstart (local)

You need Python 3.11 or newer (3.12 recommended) and a Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your resume next to app.py
cp /path/to/your-resume.pdf resume.pdf

# 4. Add your API key (the .env file is git-ignored)
echo "GEMINI_API_KEY=your-key-here" > .env

# 5. Run
streamlit run app.py
```

Open http://localhost:8501. The first start downloads the embedding model (about 90 MB) from Hugging Face; later starts reuse the cached copy.

No `resume.pdf` yet? Start the app anyway and upload a PDF in the sidebar. If the key or the resume is missing, the page shows a setup checklist instead of an error.

## Configuration

All settings are environment variables. Only the first is required.

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | none | Your Gemini API key. Also read from Streamlit secrets. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for answers. |
| `CANDIDATE_NAME` | empty | If set to `Maria`, the title reads "Chat with Maria's Digital Twin". |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model name or local path. |

> **About `gemini-2.5-flash`:** Google's [deprecations page](https://ai.google.dev/gemini-api/docs/deprecations) (checked November 2026) lists no shutdown date for it, but says access to 2.5 models is limited to accounts that have used them before. If the chat replies that the model is not available for your key, set `GEMINI_MODEL` to a current model such as `gemini-3.8-flash`. No code change is needed.

Chunk size, overlap, number of retrieved chunks, the starter questions and the system prompt are constants at the top of `app.py`.

## Run with Docker

```bash
docker build -t digital-twin .
docker run --rm -p 8501:8501 -e GEMINI_API_KEY=your-key-here digital-twin
```

The image installs CPU-only PyTorch and bakes in the embedding model, so the container starts without downloading anything. `resume.pdf` is copied into the image at build time; rebuild after you change it.

## Deploy

### Step 0: push the project to GitHub (both options)

```bash
git init
git add .
git commit -m "Resume digital twin chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Check two things before you push:

- `resume.pdf` **is** committed. Both hosts build from the repo, so the resume has to be in it. Your resume holds personal details; use a private repo, or remove your phone number and home address from the PDF first.
- `.env` is **not** committed (`git status` should not list it). The API key is added in the host's dashboard instead.

### Option A: Streamlit Community Cloud (free, recommended)

This is the best free fit. Streamlit lists its per-app memory limits as approximately 690 MB minimum and 2.7 GB maximum (figures dated February 2024, subject to change), which leaves room for the roughly 1 GB this stack uses (see "Memory" under Option B).

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **Create app** (upper right), then **Yup, I have an app**.
3. Fill in your **repository**, the branch `main`, and the file path `app.py`.
4. Optional: choose a subdomain in the **App URL** field.
5. Click **Advanced settings**.
   - **Python version:** 3.12.
   - **Secrets:** paste this line with your real key:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
     Optional extra lines: `GEMINI_MODEL = "..."`, `CANDIDATE_NAME = "..."`.
6. Click **Save**, then deploy.

**Port settings:** none. Community Cloud runs `streamlit run app.py` itself and ignores the `Dockerfile`.

The first build takes several minutes because the default PyTorch package is large. To change the key later, open the app, click **Manage app** (lower right), then the menu, **Settings**, **Secrets**. Every `git push` redeploys the app. Apps with no visitors for 12 hours go to sleep and wake on the next visit.

### Option B: Render (Docker)

1. Sign in at [dashboard.render.com](https://dashboard.render.com) and click **New > Web Service**.
2. Choose **Git Provider**, connect GitHub if asked, and select your repo.
3. Fill in the form:
   - **Name:** for example `digital-twin` (becomes `digital-twin.onrender.com`).
   - **Region:** the one closest to your visitors.
   - **Branch:** `main`.
   - **Language:** **Docker**. Render builds the `Dockerfile` in the repo root. Leave any build or start command empty.
4. Choose a compute plan (see "Memory" below).
5. Under **Advanced**, add an environment variable:
   - Key `GEMINI_API_KEY`, value: your key.
   - Optional: `GEMINI_MODEL`, `CANDIDATE_NAME`.
   - Optional: set the health check path to `/_stcore/health`.
6. Click **Create Web Service** and follow the build on the **Deploys** page. Your app is live at `https://<name>.onrender.com` when the deploy finishes.

**Port settings:** nothing to configure. Render injects a `PORT` environment variable (default `10000`) and expects the app to listen on `0.0.0.0`. The `Dockerfile` starts Streamlit with `--server.port=${PORT:-8501} --server.address=0.0.0.0`, so it follows whatever Render assigns. Do not hard-code a port.

**Memory (read before choosing a plan):** PyTorch plus sentence-transformers is the heavy part of this stack. With the default PyPI build of PyTorch the app process measured about 1 GB of RAM in testing. The CPU-only build used by the `Dockerfile` is lighter, but you should still expect it to need more than the 512 MB that Render's Free and Starter plans provide; when a service runs out of memory Render restarts it. For a dependable Render deployment pick a plan with 2 GB (Standard). You can try Free first, and if the logs show out-of-memory restarts, upgrade or use Option A. Free Render services also spin down after 15 minutes without traffic and take about a minute to wake.

Every push to `main` triggers a new deploy.

## Notes for a public URL

- **Quota:** anyone with the link uses your Gemini key. The app caps each browser session at 40 questions of 500 characters (constants in `app.py`). Also set a quota or budget alert on the key in Google AI Studio.
- **Data flow:** resume excerpts and visitor questions are sent to the Gemini API. Embedding and search happen on your server.
- **Uploads:** a PDF uploaded in the sidebar is processed in memory for that visitor's session only. It does not replace your resume for other visitors.
- **Accuracy:** the twin is instructed to answer only from the resume and to say so when something is not there, but language models can still make mistakes. Use the **Show resume excerpts used** toggle in the sidebar to see what each answer was based on.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Setup checklist instead of the chat | Add `resume.pdf` next to `app.py` and set `GEMINI_API_KEY`, then restart. |
| "No selectable text was found in this PDF" | The PDF is a scan. Export the resume from Word or Google Docs as a PDF instead. |
| "The Gemini model ... is not available for this API key" | Set `GEMINI_MODEL` to a model your key can use. |
| "My Gemini API key was rejected" | Re-copy the key; check it is enabled in AI Studio. |
| Render deploy keeps restarting | Out of memory. Use a 2 GB plan or deploy on Streamlit Community Cloud. |
| Answers miss details that are in the resume | Raise `TOP_K` or `CHUNK_SIZE` in `app.py`. |
