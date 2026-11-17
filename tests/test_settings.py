"""Settings that other files (README, Dockerfile, hosts) rely on."""

from __future__ import annotations

from digital_twin import settings


def test_project_root_is_the_folder_that_holds_app_py():
    assert (settings.PROJECT_ROOT / "app.py").is_file()
    assert (settings.PROJECT_ROOT / "digital_twin" / "settings.py").is_file()


def test_default_resume_is_expected_next_to_app_py():
    assert settings.RESUME_PATH == settings.PROJECT_ROOT / "resume.pdf"


def test_shipped_chunking_settings_are_valid():
    assert settings.CHUNK_SIZE > 0
    assert 0 <= settings.CHUNK_OVERLAP <= settings.CHUNK_SIZE // 2
    assert settings.TOP_K > 0
