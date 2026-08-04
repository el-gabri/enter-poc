"""Regression tests for the documented Streamlit entrypoint."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_from_frontend_script_path(monkeypatch) -> None:
    """The documented command must resolve sibling frontend modules."""
    monkeypatch.setenv("LITIGATION_API_URL", "http://127.0.0.1:1")
    app_path = Path(__file__).parents[1] / "frontend" / "streamlit_app.py"

    app = AppTest.from_file(app_path).run(timeout=10)

    assert not app.exception
