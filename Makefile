.PHONY: install run test lint data-refresh game-logs game-logs-nfl game-logs-college clean

PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

# ── Setup ────────────────────────────────────────────────────

install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	touch $(VENV)/bin/activate

# ── Development ──────────────────────────────────────────────

run: install
	$(STREAMLIT) run app.py --server.port 8501 --server.headless true

# ── Quality ──────────────────────────────────────────────────

test: install
	$(PYTEST) tests/ -v

lint: install
	$(RUFF) check .

lint-fix: install
	$(RUFF) check --fix .

# ── Data ─────────────────────────────────────────────────────

data-refresh: install
	@echo "Usage: $(VENV)/bin/python tools/data_pull.py --position POSITION --seasons SEASONS"
	@echo ""
	@echo "Examples:"
	@echo "  $(VENV)/bin/python tools/data_pull.py --position wr --seasons 2024"
	@echo "  $(VENV)/bin/python tools/data_pull.py --position wr --seasons 2016-2025"
	@echo "  $(VENV)/bin/python tools/data_pull.py --position wr --seasons 2024 --dry-run"

data-refresh-wr: install
	$(VENV)/bin/python tools/data_pull.py --position wr --seasons 2016-2025

data-refresh-rb: install
	@echo "RB config is stubbed but not yet registered. See tools/pipeline/positions/rb.py"

# ── Game logs (per-game player stats) ────────────────────────

game-logs: install game-logs-nfl game-logs-college

game-logs-nfl: install
	$(VENV)/bin/python tools/game_logs/pull_nfl_weekly.py

game-logs-college: install
	$(VENV)/bin/python tools/game_logs/pull_college_games.py

# ── Cleanup ──────────────────────────────────────────────────

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
