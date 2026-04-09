# StrategyPilot Elite

StrategyPilot Elite is an upgraded competitive intelligence product that feels closer to a real consulting / strategy copilot.

## What changed vs MVP
- richer real-company knowledge base
- stronger capability extraction with weighted signals
- semantic competitor matching using TF-IDF similarity
- rationale generation for *why* competitors were matched
- strategy recommendations with clearer business logic
- downloadable executive report
- deployment-safe design (no external model downloads)

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
