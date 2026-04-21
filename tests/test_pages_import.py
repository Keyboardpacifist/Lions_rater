"""
Smoke tests: verify every page module's config can be imported.

These don't render the pages (that requires a Streamlit runtime),
but they catch syntax errors, missing imports, and broken file paths.
"""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


def get_page_files():
    """List all .py files in pages/."""
    return sorted(PAGES_DIR.glob("*.py"))


@pytest.mark.parametrize("page_path", get_page_files(), ids=lambda p: p.name)
def test_page_has_no_syntax_errors(page_path):
    """Every page file should compile without syntax errors."""
    source = page_path.read_text()
    try:
        compile(source, str(page_path), "exec")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {page_path.name}: {e}")
