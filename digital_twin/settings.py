"""Settings for the Digital Twin.

Anything that differs between deployments is read from an environment
variable (or a local .env file). Everything else is a constant you can edit
here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# The folder that holds app.py, resume.pdf and (optionally) .env.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reads a local .env file if one exists; does nothing in the cloud.
load_dotenv(PROJECT_ROOT / ".env")

RESUME_PATH = PROJECT_ROOT / "resume.pdf"
MAX_UPLOAD_MB = 10    # sidebar upload limit; a resume is far smaller than this

# --- Retrieval ------------------------------------------------------------- #

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks
TOP_K = 4             # chunks retrieved per question

# --- Models ---------------------------------------------------------------- #

# Both models can be swapped with environment variables, no code change needed.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_API_VERSION = "v1beta"
TEMPERATURE = 0.3     # low temperature keeps answers close to the resume

# --- Chat ------------------------------------------------------------------ #

# Optional: shown in the page title, e.g. "Chat with Maria's Digital Twin".
CANDIDATE_NAME = os.getenv("CANDIDATE_NAME", "").strip()

HISTORY_EXCHANGES = 3            # previous question/answer pairs sent to the LLM
MAX_QUESTION_CHARS = 500         # cap on a single visitor message
MAX_QUESTIONS_PER_SESSION = 40   # protects your API quota on a public URL

STARTER_QUESTIONS = (
    "Tell me about yourself",
    "What is your strongest technical skill?",
    "Walk me through your most recent role",
    "What project are you most proud of?",
)
