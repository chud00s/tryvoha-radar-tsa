PY = ./.venv/bin/python
STREAMLIT = ./.venv/bin/streamlit

.PHONY: setup data forecast events test app all clean

setup:                ## create venv + install deps
	python3 -m venv .venv && ./.venv/bin/pip install -U pip && ./.venv/bin/pip install -r requirements.txt

data:                 ## (re)download alerts + rebuild hourly series
	$(PY) -m src.transform --rebuild

forecast:             ## backtest + write near-term risk
	$(PY) -m src.forecast

events:               ## extract OSINT events (LLM if key, else rules)
	$(PY) -m src.ai_extractor

test:                 ## run unit tests
	$(PY) -m pytest

app:                  ## launch the dashboard
	$(STREAMLIT) run app/dashboard.py

all: data forecast events test   ## full pipeline + tests

clean:
	rm -f data/processed/*.parquet data/processed/*.json
