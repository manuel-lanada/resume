# Digital Twin: Streamlit + sentence-transformers + FAISS + Gemini
FROM python:3.12-slim

# HF_HOME keeps the embedding model cache inside the image.
# The thread and malloc settings lower RAM use on small instances.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

# 1) CPU-only PyTorch. Installing it first stops pip from downloading the
#    default build with ~2 GB of GPU libraries this app never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else (cached by Docker until requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

USER appuser

# 3) Download the embedding model at build time so the container starts fast
#    and never needs to reach Hugging Face at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 4) App code and the resume. The [f] pattern lets the build succeed even if
#    resume.pdf is not there (the app then shows its setup message).
#    Files are copied by name on purpose, so a local .env can never end up
#    inside the image (.dockerignore guards the package folder as well).
COPY --chown=appuser:appuser app.py resume.pd[f] ./
COPY --chown=appuser:appuser digital_twin ./digital_twin

# Render (and most hosts) tell the container which port to use via $PORT.
# Locally it falls back to Streamlit's default, 8501.
EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
