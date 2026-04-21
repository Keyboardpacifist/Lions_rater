"""
Shared test fixtures for Lions Rater.

These fixtures provide test data and mock Streamlit's secrets
so tests can run without a live Supabase connection.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLEGE_DIR = DATA_DIR / "college"


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """Mock st.secrets so lib_shared imports without real credentials."""
    mock_secrets = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
    }
    mock_st = MagicMock()
    mock_st.secrets = mock_secrets
    mock_st.cache_data = lambda func=None, **kwargs: func if func else (lambda f: f)
    mock_st.cache_resource = lambda func=None, **kwargs: func if func else (lambda f: f)
    monkeypatch.setitem(sys.modules, "streamlit", mock_st)


@pytest.fixture
def sample_player_data():
    """Minimal DataFrame for testing scoring logic."""
    import pandas as pd

    return pd.DataFrame({
        "player_display_name": ["Player A", "Player B", "Player C"],
        "stat_a_z": [1.0, 0.0, -1.0],
        "stat_b_z": [0.5, 1.5, -0.5],
        "stat_c_z": [0.0, 0.0, 2.0],
    })
