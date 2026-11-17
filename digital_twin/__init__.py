"""Digital Twin: a resume-grounded RAG chatbot.

Pipeline
--------
1. Read resume.pdf (or a PDF uploaded in the sidebar) with pypdf.
2. Split the text into overlapping ~500-character chunks.
3. Embed the chunks locally with sentence-transformers (all-MiniLM-L6-v2, CPU).
4. Store the vectors in an in-memory FAISS IndexFlatL2.
5. For every visitor question: embed it, fetch the 4 nearest chunks, and ask
   Gemini to answer in the first person using only those chunks.

Layout
------
The core modules know nothing about Streamlit, so they can be tested and
reused on their own:

    settings      environment-driven settings and tunable constants
    prompts       the system prompt and the per-question prompt
    conversation  chat messages and which of them the LLM gets to see
    ingestion     steps 1-2: PDF -> clean text -> chunks
    retrieval     steps 3-5: embeddings, FAISS index, nearest-chunk search
    llm           step 5: Gemini client, request assembly, streaming, errors

The `ui` subpackage is the Streamlit app built on top of them.
"""
