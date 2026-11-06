"""Turn the raw downloads in data/raw/ into 10 clean series in data/series/.

Each output CSV has columns: timestamp, target (+ covariate columns for the bike series).
The test set is always the LAST `horizon` rows; the notebook hides them from the models.
"""
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "series"
OUT.mkdir(parents=True, exist_ok=True)


def monthly(df, date_col, value_col):
    return pd.DataFrame({"timestamp": pd.to_datetime(df[date_col]), "target": df[value_col].astype(float)})


series = {}

df = pd.read_csv(RAW / "airline-passengers.csv")
series["01_airline_passengers"] = (monthly(df, "Month", "Passengers"), "MS", 24, 12, "passengers (thousands)")

df = pd.read_csv(RAW / "co2_mm_mlo.csv", comment="#")
df = df[df["average"] > 0]
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
series["02_co2_mauna_loa"] = (monthly(df, "date", "average"), "MS", 36, 12, "CO2 (ppm)")

df = pd.read_csv(RAW / "monthly-sunspots.csv").tail(2048)
series["03_sunspots"] = (monthly(df, "Month", "Sunspots"), "MS", 60, 132, "sunspot number")

df = pd.read_csv(RAW / "daily-min-temperatures.csv")
s = monthly(df, "Date", "Temp").set_index("timestamp").asfreq("D").interpolate().reset_index()
series["04_melbourne_min_temp"] = (s, "D", 60, 365, "min temperature (C)")

df = pd.read_csv(RAW / "daily-total-female-births.csv")
series["05_female_births"] = (monthly(df, "Date", "Births"), "D", 30, 7, "births per day")

df = pd.read_csv(RAW / "monthly-car-sales.csv")
series["06_car_sales"] = (monthly(df, "Month", "Sales"), "MS", 12, 12, "cars sold")

df = pd.read_csv(RAW / "ETTh1.csv").tail(3000)
series["07_ett_oil_temp"] = (monthly(df, "date", "OT"), "h", 96, 24, "oil temperature (C)")
series["08_ett_load"] = (monthly(df, "date", "HUFL"), "h", 96, 24, "high useful load")

df = pd.read_csv(RAW / "bike" / "hour.csv")
df["timestamp"] = pd.to_datetime(df.dteday) + pd.to_timedelta(df.hr, unit="h")
s = df.set_index("timestamp")[["cnt"]].asfreq("h", fill_value=0).tail(3000)  # missing hours = no rentals
series["09_bike_hourly"] = (s.reset_index().rename(columns={"cnt": "target"}).astype({"target": float}),
                            "h", 168, 168, "rentals per hour")

df = pd.read_csv(RAW / "bike" / "day.csv")
s = pd.DataFrame({"timestamp": pd.to_datetime(df.dteday), "target": df.cnt.astype(float),
                  "workingday": df.workingday.astype(float), "temp": df.temp * 41.0})
series["10_bike_daily"] = (s, "D", 30, 7, "rentals per day")

meta = {}
for name, (s, freq, horizon, season, unit) in series.items():
    s.to_csv(OUT / f"{name}.csv", index=False)
    meta[name] = {"freq": freq, "horizon": horizon, "season": season, "unit": unit, "length": len(s)}
    print(f"{name:24s} {len(s):5d} rows  freq={freq:2s} horizon={horizon}")
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
