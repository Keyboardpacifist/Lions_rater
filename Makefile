.PHONY: install run test lint data-refresh clean

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
	@echo "Available positions: wr, rb"
	@echo "Usage: make data-refresh-wr  or  make data-refresh-rb"
	@echo ""
	@echo "NOTE: Most positions do not have data pull scripts yet."
	@echo "See CLAUDE.md Phase 3 for the plan to add them."

data-refresh-wr: install
	$(PYTHON) tools/wr_data_pull.py

data-refresh-rb: install
	$(PYTHON) tools/rb_data_pull.py

# ── Cleanup ──────────────────────────────────────────────────

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
