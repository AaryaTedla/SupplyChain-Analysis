# DataCo Supply Chain Dashboard

An interactive Streamlit dashboard for delivery-risk classification, demand
forecasting, and operational recommendations.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run supply_chain_dashboard.py
```

## Live deployment

This repository is ready for Streamlit Community Cloud. Set the entrypoint to
`supply_chain_dashboard.py`; dependencies are installed from
`requirements.txt` automatically.

Upload a DataCo-compatible CSV. The engineered
`data/SupplyChain_FullMerged_20260314.csv` format provides the most reliable
model inputs; raw DataCo files are accepted but missing engineered features use
neutral defaults and trigger a warning.
