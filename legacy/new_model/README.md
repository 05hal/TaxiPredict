# STID Taxi Demand Forecasting

This folder contains the STID-based regional taxi demand training workflow.

## Files

- `stid_taxi_demand.py`: builds hourly region demand from raw Uber orders, joins hourly weather, trains STID, and writes predictions.
- `sweep_stid_params.py`: runs a grid search over `grid_size` and `top_regions`.
- `requirements_stid.txt`: Python dependencies.
- `sweep_summary_with_normalized.csv`: first-round parameter sweep summary.
- `stid_refine_summary.csv`: refined candidate comparison.
- `stid_best_0.02_top30/`: best balanced run in the current experiment.

## Current Best Balanced Configuration

- `grid_size=0.02`
- `top_regions=30`
- `input_len=24`
- `horizon=1`

Refined test metrics:

- MAE: `5.840814`
- RMSE: `11.306272`
- R2: `0.933728`
- WAPE: `21.538729%`

## Run

```powershell
pip install -r new_model/requirements_stid.txt
python new_model/stid_taxi_demand.py --grid-size 0.02 --top-regions 30 --epochs 40 --device cpu
```

## Sweep

```powershell
python new_model/sweep_stid_params.py --grid-sizes 0.005,0.01,0.02 --top-regions 30,60,100 --epochs 12 --device cpu
```
