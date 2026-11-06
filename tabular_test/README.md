# Tabular foundation model inference test

`tabular_inference.ipynb` runs three tabular foundation models (classifier and regressor of each) plus a
gradient-boosting baseline on 8 small datasets, scores them on a held-out 20% split and saves every prediction.

| Model | Package / weights | Params | Runs on CPU |
|---|---|---|---|
| TabICLv2 | `tabicl` (`tabicl-{classifier,regressor}-v2-20260212.ckpt`) | ~28M | yes, seconds per dataset |
| TabDPT v1.2 | `tabdpt==1.2.0` (`Layer6/TabDPT`, `tabdpt1_2.safetensors`) | ~64M | yes, with `compile=False` |
| Mitra v1 | `autogluon.tabular[mitra]` (`autogluon/mitra-{classifier,regressor}`) | ~76M | yes but slow: ~2 min for 1,000 rows, ~12 min for 1,460 rows × 79 features |
| HistGradientBoosting | scikit-learn (baseline) | — | yes |

By default Mitra runs only on the 4 smallest datasets. Set `MITRA_DATASETS = "all"` in the Settings cell
to run it everywhere (best on a GPU).

Note: `tabdpt` 1.3.x downloads the newer **v1.3** weights and cannot load v1.2 weights. Pin `tabdpt==1.2.0` for v1.2.

## Environment (already created on this machine as `tabular`)

```bash
conda create -y -n tabular python=3.12
conda activate tabular
pip install "autogluon.tabular[mitra]" tabicl "tabdpt==1.2.0" openml matplotlib jinja2 ipykernel
# AutoGluon 1.6 needs torch < 2.14 and pip pulls the CUDA build; on a CPU-only machine swap it for the CPU build:
pip install --force-reinstall --no-deps torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip uninstall -y $(pip list | awk '/^(nvidia|cuda|triton)/{print $1}')
python -m ipykernel install --user --name tabular --display-name "Python (tabular)"
```

On a GPU machine (e.g. the HPC), skip the swap and keep the CUDA build.

## Run

```bash
cd tabular_test
python prepare_data.py              # only if data/ is missing (downloads from scikit-learn / OpenML)
jupyter lab tabular_inference.ipynb # pick the "Python (tabular)" kernel, Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 tabular_inference.ipynb
```

## Outputs

- `outputs/scores.csv` - every model × dataset: accuracy / ROC AUC / log loss, or R² / RMSE / MAE (+ 80% coverage), seconds
- `outputs/predictions.csv` - every test row: actual, predicted, class probabilities or TabICLv2's 10%/90% range
- `outputs/regression_scatter.png` - predicted vs actual for each regression dataset and model

## Using your own data

Save a CSV with your feature columns plus a `target` column into `data/`, and add an entry to `data/meta.json`
(`task`: `"classification"` or `"regression"`, plus `rows`, `features`, `categorical`, `missing_cells`, `description`).
Text categories and missing values are fine.

## Data sources

scikit-learn built-ins (breast_cancer, wine, diabetes, california_housing) and OpenML (adult, credit-g, abalone, house_prices).
Datasets over 2,000 rows are randomly subsampled to 2,000.
