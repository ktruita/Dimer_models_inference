"""Download 8 small tabular datasets (4 classification, 4 regression) into data/ as CSV.

Each CSV keeps the raw columns (text categories and missing values included) with the target in a column named `target`.
The large ones are subsampled so every model runs in reasonable time on CPU.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_breast_cancer, load_diabetes, load_wine

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
MAX_ROWS = 2000
SEED = 0


def save(name, task, X, y, desc):
    df = X.copy()
    df["target"] = y.values if hasattr(y, "values") else y
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    df.to_csv(OUT / f"{name}.csv", index=False)
    n_cat = int((df.drop(columns="target").dtypes == "object").sum() + (df.drop(columns="target").dtypes == "category").sum())
    meta[name] = {"task": task, "rows": len(df), "features": df.shape[1] - 1, "categorical": n_cat,
                  "missing_cells": int(df.drop(columns="target").isna().sum().sum()), "description": desc}
    print(f"{name:18s} {task:14s} rows={len(df):5d} features={df.shape[1] - 1:3d} categorical={n_cat}")


meta = {}

d = load_breast_cancer(as_frame=True)
save("breast_cancer", "classification", d.data, d.target.map({0: "malignant", 1: "benign"}),
     "tumour measurements -> malignant / benign")

d = load_wine(as_frame=True)
save("wine", "classification", d.data, d.target.map(lambda i: f"cultivar_{i}"), "chemical analysis -> 3 grape cultivars")

d = fetch_openml("adult", version=2, as_frame=True)
save("adult_income", "classification", d.data, d.target, "census record -> income >50K or <=50K")

d = fetch_openml("credit-g", version=1, as_frame=True)
save("credit_risk", "classification", d.data, d.target, "loan applicant -> good / bad credit risk")

d = load_diabetes(as_frame=True, scaled=False)
save("diabetes", "regression", d.data, d.target, "patient measurements -> disease progression after 1 year")

d = fetch_california_housing(as_frame=True)
save("california_housing", "regression", d.data, d.target * 100_000, "census block -> median house value (USD)")

d = fetch_openml("abalone", version=1, as_frame=True)
y = pd.to_numeric(d.data.pop("Class_number_of_rings") if "Class_number_of_rings" in d.data else d.target)
save("abalone", "regression", d.data, y, "shell measurements -> age in rings")

d = fetch_openml("house_prices", version=1, as_frame=True)
X = d.data.drop(columns=["Id"], errors="ignore")
save("house_prices", "regression", X, d.target, "Ames house features -> sale price (USD)")

(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
