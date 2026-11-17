"""Digital Twin: a resume-grounded RAG chatbot.

This file is only the Streamlit entry point. The application lives in the
`digital_twin` package; start with `digital_twin/__init__.py` for the layout
and `digital_twin/ui/page.py` for the flow of a page run.

Run locally:  streamlit run app.py
"""

from digital_twin.ui.page import main

if __name__ == "__main__":
    main()
