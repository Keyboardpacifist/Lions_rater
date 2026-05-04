.PHONY: install run test lint data-refresh game-logs game-logs-nfl game-logs-college game-logs-pbp game-logs-participation defense-scheme data-upload clean api-install api-run api-test

PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn

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

# Heavy NFL feeds (play-by-play + participation) — needed for the
# defensive-scheme summary. Local-only (gitignored) since they're large.

game-logs-pbp: install
	$(VENV)/bin/python tools/game_logs/pull_nfl_pbp.py

game-logs-participation: install
	$(VENV)/bin/python tools/game_logs/pull_nfl_participation.py

# Per-(defense, season) and per-(defense, season, week) scheme profile.
# Depends on pbp + participation parquets — run those first if missing.
defense-scheme: install
	$(VENV)/bin/python tools/game_logs/build_defense_scheme.py

# Upload runtime parquets to Supabase Storage so production reads them.
# Run after `make game-logs` / `make defense-scheme` to refresh live.
data-upload: install
	$(VENV)/bin/python tools/game_logs/upload_to_supabase.py

# ── EdgeAcademy API ──────────────────────────────────────────

api-install: install
	$(PIP) install -r api/requirements.txt

api-run: api-install
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

api-test: api-install
	$(PYTEST) api/tests/ -v

# ── Cleanup ──────────────────────────────────────────────────

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
