# Time-series forecasting inference test

`forecast_inference.ipynb` runs three zero-shot forecasting models on 10 real series, scores them on
held-out data and plots everything. The full notebook runs in under a minute on CPU once the weights are downloaded.

| Model | Source | Runs on CPU |
|---|---|---|
| Toto 2.0 | `Datadog/Toto-2.0-{22m,313m,1B,2.5B}` (`toto-models`) | 22m / 313m yes; 1B with ~8 GB+ free RAM; **2.5B needs ~12 GB+ free RAM or a GPU** |
| TiRex-2 | `NX-AI/TiRex-2` (`tirex-2`) | yes (with `TORCHDYNAMO_DISABLE=1` if there is no C++ compiler) |
| Chronos-2 | `amazon/chronos-2` (`chronos-forecasting>=2.0`) | yes |

MOMENT-1-base is not included: its forecasting head has to be fine-tuned first.

## Environment (already created on this machine as `forecast`)

```bash
conda create -y -n forecast python=3.12          # toto-models needs Python >= 3.12
conda activate forecast
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build
pip install toto-models tirex-2 "chronos-forecasting>=2.0" "pandas[pyarrow]" matplotlib jinja2 ipykernel
python -m ipykernel install --user --name forecast --display-name "Python (forecast)"
```

## Run

```bash
cd forecast_test
python prepare_data.py          # only needed if data/series/ is missing (reads data/raw/)
jupyter lab forecast_inference.ipynb             # pick the "Python (forecast)" kernel, Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace forecast_inference.ipynb
```

Change `TOTO_SIZE` in the Settings cell to try other Toto sizes.

## Outputs

- `outputs/plots/<series>.png`: history, actual values and each model's median plus 80% range
- `outputs/forecasts.csv`: every forecast step: model, series, timestamp, actual, quantiles q0.1 … q0.9
- `outputs/summary.csv`: mean MASE, wins, 80% coverage and timing per model

## Using your own series

Put a CSV with `timestamp,target` columns (plus optional covariate columns) in `data/series/`, and add an entry
to `data/series/meta.json` with `freq`, `horizon`, `season` (the seasonal period, e.g. 12 for monthly, 24 for hourly,
7 for daily) and `unit`.

## Data sources (in `data/raw/`)

- `airline-passengers`, `monthly-sunspots`, `daily-min-temperatures`, `daily-total-female-births`,
  `monthly-car-sales`: github.com/jbrownlee/Datasets
- `co2_mm_mlo.csv`: NOAA Global Monitoring Laboratory, Mauna Loa monthly CO2
- `ETTh1.csv`: github.com/zhouhaoyi/ETDataset (electricity transformer data)
- `bike/`: UCI Bike Sharing Dataset (Capital Bikeshare, Washington DC, 2011–2012)
