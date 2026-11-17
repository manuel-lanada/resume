"""The sidebar: optional resume upload and a few chat controls."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from digital_twin import settings
from digital_twin.ui import state


@dataclass(frozen=True)
class SidebarControls:
    """What the visitor chose in the sidebar on this run."""

    uploaded_pdf: bytes | None   # None means "use the default resume"
    show_sources: bool


def render_sidebar() -> SidebarControls:
    with st.sidebar:
        st.header("Resume")
        uploaded = st.file_uploader(
            "Try a different resume (PDF)",
            type=["pdf"],
            max_upload_size=settings.MAX_UPLOAD_MB,
            help=(
                "Only affects your own session. "
                "Remove the file to go back to the default resume."
            ),
        )
        show_sources = st.toggle("Show resume excerpts used", value=False)
        if st.button("Clear chat", width="stretch"):
            state.clear_messages()
            st.rerun()

    return SidebarControls(
        uploaded_pdf=uploaded.getvalue() if uploaded is not None else None,
        show_sources=show_sources,
    )


def render_index_status(*, uploaded: bool, chunk_count: int) -> None:
    """Tell the visitor which resume and model are answering."""
    source = "uploaded PDF" if uploaded else "resume.pdf"
    with st.sidebar:
        st.caption(
            f"Using {source}: {chunk_count} chunks indexed. Model: {settings.GEMINI_MODEL}."
        )
