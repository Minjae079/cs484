"""TSLA next-session Ridge prediction, implemented in one Python file.

Run: python ridge_project.py
Offline snapshot: python ridge_project.py --train-only
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np



DATA_URL = "https://query1.finance.yahoo.com/v8/finance/chart/TSLA?range=5y&interval=1d"
SOURCE_PAGE = "https://finance.yahoo.com/quote/TSLA/history/"
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "AdjustedClose"]
CALENDAR_SOURCE = "https://www.nasdaq.com/market-activity/stock-market-holiday-schedule"
ORIGINAL_FEATURES = ["Close", "SMA5", "SMA20", "Return1D", "Volatility20D"]
RELATIVE_FEATURES = ["Return1D", "LagReturn1D", "Return5D", "Return20D",
                     "PriceToSMA5", "PriceToSMA20", "Volatility20D"]
EXTRA_FEATURES = ["VolumeRatio20", "RangeRatio", "GapReturn", "IntradayReturn"]
FEATURE_SETS = {"original": ORIGINAL_FEATURES, "relative": RELATIVE_FEATURES,
                "relative_volume": [*RELATIVE_FEATURES, *EXTRA_FEATURES]}
FEATURE_NAMES = list(dict.fromkeys([*ORIGINAL_FEATURES, *RELATIVE_FEATURES, *EXTRA_FEATURES]))
FEATURE_COLUMNS = ["Date", *FEATURE_NAMES, "TargetDate", "TargetClose"]
REFIT_EVERY = 20


def csv_bytes(rows, columns=COLUMNS):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_outputs(directory, outputs):
    """Atomically replace each generated file; keep CSV line endings intact."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        temporary = directory / (name + ".tmp")
        temporary.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        temporary.replace(directory / name)


def parse_prices(payload, now=None):
    """Validate Yahoo observations and keep only completed trading sessions."""
    now = now or datetime.now(timezone.utc)
    document = json.loads(payload.decode("utf-8-sig"))
    chart = document.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance returned an error: {chart['error']}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise ValueError("Expected exactly one Yahoo Finance price series.")
    result = results[0]
    meta = result.get("meta", {})
    if meta.get("symbol") != "TSLA" or meta.get("currency") != "USD":
        raise ValueError("Expected TSLA prices denominated in USD.")
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quotes = indicators.get("quote") or []
    adjustments = indicators.get("adjclose") or []
    if not timestamps or not quotes or not adjustments:
        raise ValueError("Price timestamps, OHLCV or adjusted close data are missing.")
    quote = quotes[0]
    series = {name: quote.get(name.lower(), []) for name in COLUMNS[1:-1]}
    series["AdjustedClose"] = adjustments[0].get("adjclose", [])
    if any(len(values) != len(timestamps) for values in series.values()):
        raise ValueError("Mismatched timestamp and price-array lengths.")

    # For US daily bars, the opening timestamp maps to the trading date.
    # gmtoffset is sufficient for the date (historical DST changes do not
    # move a US regular-session opening across midnight).
    market_tz = timezone(timedelta(seconds=int(meta.get("gmtoffset", -18000))))
    market_today = now.astimezone(market_tz).date()
    regular = meta.get("currentTradingPeriod", {}).get("regular", {})
    regular_end = regular.get("end")
    raw_rows, rows = [], []
    seen = set()
    skipped_missing = skipped_incomplete = 0
    for index, timestamp in enumerate(timestamps):
        day = datetime.fromtimestamp(timestamp, market_tz).date()
        values = {key: data[index] for key, data in series.items()}
        raw_rows.append({"Date": day.isoformat(), **values})
        if day in seen:
            raise ValueError(f"Duplicate trading date: {day}")
        seen.add(day)
        if day > market_today:
            raise ValueError(f"Unexpected future trading date: {day}")
        if day == market_today:
            # Allow a 15-minute delay after the provider's regular close.
            end_day = (datetime.fromtimestamp(regular_end, market_tz).date()
                       if regular_end is not None else None)
            if end_day != day or now.timestamp() < regular_end + 900:
                skipped_incomplete += 1
                continue
        if any(value is None for value in values.values()):
            skipped_missing += 1
            continue
        try:
            values = {key: float(value) for key, value in values.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric data on {day}") from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite data on {day}")
        if any(values[key] <= 0 for key in ("Open", "High", "Low", "Close", "AdjustedClose")):
            raise ValueError(f"Non-positive price on {day}")
        if values["Volume"] < 0 or not values["Volume"].is_integer():
            raise ValueError(f"Invalid volume on {day}")
        if not (values["Low"] <= min(values["Open"], values["Close"])
                <= max(values["Open"], values["Close"]) <= values["High"]):
            raise ValueError(f"Inconsistent OHLC prices on {day}")
        values["Volume"] = int(values["Volume"])
        rows.append({"Date": day.isoformat(), **values})
    raw_rows.sort(key=lambda row: row["Date"])
    rows.sort(key=lambda row: row["Date"])
    if len(rows) < 100:
        raise ValueError("Fewer than 100 usable daily prices returned for the 5-year request.")
    last_day = datetime.fromisoformat(rows[-1]["Date"]).date()
    if (market_today - last_day).days > 7:
        raise ValueError(f"Downloaded series is stale: last trading date is {last_day}.")
    info = {"currency": meta["currency"],
            "exchange_timezone": meta.get("exchangeTimezoneName"),
            "downloaded_observations": len(timestamps),
            "skipped_missing": skipped_missing,
            "skipped_incomplete": skipped_incomplete}
    return rows, raw_rows, info


def download_prices():
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read(5_000_001)
            if len(payload) > 5_000_000:
                raise ValueError("Unexpectedly large price response.")
            return payload
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(1)


def collect_prices(output_dir):
    payload = download_prices()
    rows, raw_rows, info = parse_prices(payload)
    metadata = {
        "symbol": "TSLA", "provider": "Yahoo Finance", "requested_range": "5y",
        "source_url": DATA_URL, "source_page": SOURCE_PAGE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_json_sha256": hashlib.sha256(payload).hexdigest(),
        "observations": len(rows), "first_date": rows[0]["Date"],
        "last_date": rows[-1]["Date"], "columns": COLUMNS, **info,
        "notes": [
            "Rolling 5-year daily history; the date range changes with retrieval date.",
            "Regular-session daily data; not a real-time trading feed.",
            "Provider Close and AdjustedClose are kept separately; do not mix adjustment bases.",
            "Current-day bars excluded until 15 minutes after the provider regular-session close.",
            "Missing OHLCV/adjusted-close observations skipped and counted; no imputation.",
            "A last observation older than seven calendar days fails freshness validation.",
            "This collection stage downloads data; model fitting follows in later stages.",
        ],
    }
    # Prepare all content before replacing generated files. Each replacement
    # is atomic so readers do not encounter a partially written CSV.
    contents = {"tsla_raw.json": payload,
                "tsla_daily.csv": csv_bytes(rows),
                "data_metadata.json": (json.dumps(metadata, indent=2) + "\n").encode("utf-8")}
    write_outputs(output_dir, contents)
    print(f"TSLA data: {len(rows)} rows | {rows[0]['Date']} to {rows[-1]['Date']}")
    return rows, raw_rows


def load_price_snapshot(output_dir):
    payload = (output_dir / "tsla_raw.json").read_bytes()
    metadata = json.loads((output_dir / "data_metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != metadata.get("raw_json_sha256"):
        raise ValueError("Raw price snapshot hash mismatch.")
    source = json.loads(payload)["chart"]["result"][0]
    if source["meta"]["symbol"] != "TSLA":
        raise ValueError("Snapshot must contain TSLA data.")
    market_tz = timezone(timedelta(seconds=int(source["meta"].get("gmtoffset", -18000))))
    raw_rows = [{"Date": datetime.fromtimestamp(value, market_tz).date().isoformat()}
                for value in source["timestamp"]]
    with (output_dir / "tsla_daily.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows, raw_rows


def close_features(closes, index):
    """Shared calculation for historical rows and the unlabeled final row."""
    closes = np.asarray(closes, dtype=float)
    if index < 20 or index >= len(closes):
        raise ValueError("Twenty prior closing prices are required for this feature date.")
    window = closes[index - 20:index + 1]
    if not np.isfinite(window).all() or np.any(window <= 0):
        raise ValueError("Latest feature window must contain positive finite closing prices.")
    returns = window[1:] / window[:-1] - 1.0
    return {
        "Close": float(closes[index]),
        "SMA5": float(np.mean(closes[index - 4:index + 1])),
        "SMA20": float(np.mean(closes[index - 19:index + 1])),
        "Return1D": float(returns[-1]),
        "Volatility20D": float(np.std(returns, ddof=1)),
    }


def enriched_features(rows, index):
    """All information is available at the feature-date regular close."""
    closes = np.asarray([float(row["Close"]) for row in rows[index - 20:index + 1]])
    values = close_features(closes, 20)
    today = rows[index]
    volume = np.asarray([float(row["Volume"]) for row in rows[index - 19:index + 1]])
    mean_volume = float(np.mean(volume))
    values.update({
        "LagReturn1D": float(closes[-2] / closes[-3] - 1),
        "Return5D": float(closes[-1] / closes[-6] - 1),
        "Return20D": float(closes[-1] / closes[0] - 1),
        "PriceToSMA5": float(closes[-1] / values["SMA5"] - 1),
        "PriceToSMA20": float(closes[-1] / values["SMA20"] - 1),
        "VolumeRatio20": float(volume[-1] / mean_volume - 1) if mean_volume > 0 else 0.0,
        "RangeRatio": (float(today["High"]) - float(today["Low"])) / float(closes[-1]),
        "GapReturn": float(today["Open"]) / float(closes[-2]) - 1,
        "IntradayReturn": float(closes[-1]) / float(today["Open"]) - 1,
    })
    return values


def build_features(rows, source_dates=None):
    """Use information through t only; target the next observed session t+1."""
    if len(rows) < 22:
        raise ValueError("At least 22 daily prices are needed for features and a target.")
    dates = [row["Date"] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Feature input dates must be unique and increasing.")
    closes = np.asarray([float(row["Close"]) for row in rows])
    if not np.isfinite(closes).all() or np.any(closes <= 0):
        raise ValueError("Feature input closing prices must be finite and positive.")
    source_dates = dates if source_dates is None else source_dates
    if source_dates != sorted(set(source_dates)):
        raise ValueError("Source dates must be unique and increasing.")
    positions = {day: index for index, day in enumerate(source_dates)}
    if not set(dates).issubset(positions):
        raise ValueError("Some daily prices are absent from the source-date sequence.")
    features = []
    skipped_gaps = 0
    # Twenty returns need 21 prices. The last price has no known next target.
    for index in range(20, len(rows) - 1):
        # Do not bridge a source observation removed for missing values.
        if positions[dates[index + 1]] - positions[dates[index - 20]] != 21:
            skipped_gaps += 1
            continue
        features.append({
            "Date": dates[index],
            **enriched_features(rows, index),
            "TargetDate": dates[index + 1],
            "TargetClose": float(closes[index + 1]),
        })
    if not features:
        raise ValueError("No complete feature windows with a next-session target.")
    return features, skipped_gaps


def save_features(rows, raw_rows, output_dir):
    features, skipped_gaps = build_features(rows, [row["Date"] for row in raw_rows])
    metadata = {
        "symbol": "TSLA",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": "tsla_daily.csv",
        "source_sha256": hashlib.sha256((output_dir / "tsla_daily.csv").read_bytes()).hexdigest(),
        "input_observations": len(rows),
        "feature_observations": len(features),
        "excluded_initial_warmup": 20,
        "excluded_final_unlabeled": 1,
        "excluded_windows_with_source_gaps": skipped_gaps,
        "first_feature_date": features[0]["Date"],
        "last_feature_date": features[-1]["Date"],
        "first_target_date": features[0]["TargetDate"],
        "last_target_date": features[-1]["TargetDate"],
        "feature_columns": FEATURE_NAMES,
        "target_column": "TargetClose",
        "definitions": {
            "Date": "Feature date t; prediction made after the regular-session close.",
            "Close": "Provider Close at t in USD (not AdjustedClose).",
            "LagReturn1D": "Close[t-1] / Close[t-2] - 1.",
            "Return5D": "Close[t] / Close[t-5] - 1.",
            "Return20D": "Close[t] / Close[t-20] - 1.",
            "PriceToSMA5": "Close[t] / SMA5[t] - 1.",
            "PriceToSMA20": "Close[t] / SMA20[t] - 1.",
            "VolumeRatio20": "Volume[t] / mean(Volume[t-19:t+1]) - 1; 0 if mean volume is zero.",
            "RangeRatio": "(High[t] - Low[t]) / Close[t].",
            "GapReturn": "Open[t] / Close[t-1] - 1.",
            "IntradayReturn": "Close[t] / Open[t] - 1.",
            "SMA5": "Mean Close from t-4 through t inclusive (five observations).",
            "SMA20": "Mean Close from t-19 through t inclusive (twenty observations).",
            "Return1D": "Close[t] / Close[t-1] - 1; fraction, not percentage points.",
            "Volatility20D": "Sample standard deviation (ddof=1) of the last 20 one-session returns through t; not annualized.",
            "TargetDate": "Next available source trading observation after t.",
            "TargetClose": "Close at TargetDate in USD; never an input feature.",
        },
        "notes": [
            "All five inputs use only observations dated on or before Date.",
            "No scaling, fitting, splitting or evaluation is performed at this stage.",
            "TargetClose and TargetDate must not be included as model input columns.",
            "Exchange holidays are not filled; no official exchange-calendar check.",
            "Windows crossing removed source observations are excluded, not bridged.",
        ],
    }
    outputs = {
        "tsla_features.csv": csv_bytes(features, FEATURE_COLUMNS),
        "feature_metadata.json": json.dumps(metadata, indent=2) + "\n",
    }
    write_outputs(output_dir, outputs)
    print(f"Prepared {len(features)} labeled rows.")


def load_feature_rows(output_dir):
    path = output_dir / "tsla_features.csv"
    metadata = json.loads((output_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("symbol") != "TSLA" or metadata.get("feature_columns") != FEATURE_NAMES:
        raise ValueError("Feature metadata must describe the expected TSLA input columns.")
    daily_hash = hashlib.sha256((output_dir / "tsla_daily.csv").read_bytes()).hexdigest()
    if daily_hash != metadata.get("source_sha256"):
        raise ValueError("Daily prices changed since features were generated; rerun preparation.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FEATURE_COLUMNS:
            raise ValueError("Unexpected feature CSV columns.")
        features = []
        for raw in reader:
            row = {key: float(raw[key]) for key in [*FEATURE_NAMES, "TargetClose"]}
            row.update({key: raw[key] for key in ("Date", "TargetDate")})
            if not all(math.isfinite(row[key]) for key in [*FEATURE_NAMES, "TargetClose"]):
                raise ValueError("Features and target must be finite.")
            if datetime.fromisoformat(row["Date"]) >= datetime.fromisoformat(row["TargetDate"]):
                raise ValueError("A target must follow its prediction date.")
            features.append(row)
    dates = [row["Date"] for row in features]
    targets = [row["TargetDate"] for row in features]
    if dates != sorted(set(dates)) or targets != sorted(set(targets)):
        raise ValueError("Feature and target dates must be unique and increasing.")
    if any(features[i]["TargetDate"] > features[i + 1]["Date"] for i in range(len(features) - 1)):
        raise ValueError("Overlapping future labels would leak across chronological boundaries.")
    if len(features) != metadata.get("feature_observations"):
        raise ValueError("Feature CSV row count differs from its metadata.")
    return features, hashlib.sha256(path.read_bytes()).hexdigest()


def fit_ridge(X, y, alpha):
    """Fit single-target Ridge using training statistics only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be (samples, features), and y must be (samples,).")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Training data must not be empty.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training data must contain finite values.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    X_scaled = (X - mean) / scale
    # Center y to leave the intercept unpenalized.
    y_mean = y.mean()
    y_centered = y - y_mean
    # Solve the Ridge normal equations without forming an inverse.
    A = X_scaled.T @ X_scaled + alpha * np.eye(X.shape[1])
    rhs = X_scaled.T @ y_centered
    weights = np.linalg.solve(A, rhs)
    return {"mean": mean, "scale": scale,
            "y_mean": y_mean, "weights": weights}


def predict_ridge(model, X):
    """Apply the training transformation and predict."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(model["weights"]):
        raise ValueError("Prediction data must have the training feature count.")
    if not np.isfinite(X).all():
        raise ValueError("Prediction data must contain finite values.")
    X_scaled = (X - model["mean"]) / model["scale"]
    return model["y_mean"] + X_scaled @ model["weights"]


def regression_metrics(actual, predicted):
    """Compute errors in USD (MAE/RMSE) and USD squared (MSE)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require matching, nonempty one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values.")
    errors = predicted - actual
    mse = float(np.mean(errors ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    mape = float(np.mean(np.abs(errors / actual)) * 100) if np.all(actual != 0) else None
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAPE": mape,
            "CustomAccuracyPercent": 100.0 - mape if mape is not None else None,
            "MAE": float(np.mean(np.abs(errors))),
            "R2": float(1 - np.sum(errors ** 2) / total) if total > 0 else None}


def fit_configuration(features, end, config):
    """Fit only labels already observed by the next prediction date."""
    start = max(0, end - config["window"]) if config["window"] else 0
    history = features[start:end]
    columns = FEATURE_SETS[config["feature_set"]]
    X = np.asarray([[row[key] for key in columns] for row in history])
    y = np.asarray([row["TargetClose"] for row in history])
    if config["target"] == "return":
        y = y / np.asarray([row["Close"] for row in history]) - 1
    model = fit_ridge(X, y, config["alpha"])
    return model, start


def configuration_prediction(model, rows, config):
    columns = FEATURE_SETS[config["feature_set"]]
    prediction = predict_ridge(model, np.asarray([[row[key] for key in columns] for row in rows]))
    if config["target"] == "return":
        prediction = np.asarray([row["Close"] for row in rows]) * (1 + prediction)
    return prediction


def walk_forward_predict(features, start, end, config):
    """Refit every 20 prediction dates, with past observed labels only."""
    predictions = []
    audit = []
    for block_start in range(start, end, REFIT_EVERY):
        block_end = min(block_start + REFIT_EVERY, end)
        if features[block_start - 1]["TargetDate"] > features[block_start]["Date"]:
            raise ValueError("Model fitting would use an unavailable future label.")
        model, fit_start = fit_configuration(features, block_start, config)
        predictions.extend(configuration_prediction(model, features[block_start:block_end], config))
        audit.append({"first_prediction_date": features[block_start]["Date"],
                      "last_prediction_date": features[block_end - 1]["Date"],
                      "fit_rows": block_start - fit_start,
                      "first_fit_target_date": features[fit_start]["TargetDate"],
                      "last_fit_target_date": features[block_start - 1]["TargetDate"]})
    return np.asarray(predictions), audit


def select_configuration(features, development_end):
    """Three expanding validation folds entirely before the final test region."""
    development = features[:development_end]
    boundaries = [int(development_end / 2), int(development_end * 2 / 3),
                  int(development_end * 5 / 6), development_end]
    folds = list(zip(boundaries[:-1], boundaries[1:]))
    if folds[0][0] < 252:
        raise ValueError("At least 630 feature rows are needed for this validation design.")
    candidates = []
    for feature_set in FEATURE_SETS:
        for target in ("price", "return"):
            for window in (252, 504, None):
                for alpha in np.logspace(-4, 6, 11):
                    config = {"feature_set": feature_set, "target": target,
                              "window": window, "alpha": float(alpha)}
                    fold_metrics = []
                    for start, end in folds:
                        prediction, _ = walk_forward_predict(development, start, end, config)
                        actual = [row["TargetClose"] for row in development[start:end]]
                        fold_metrics.append(regression_metrics(actual, prediction))
                    candidates.append({
                        **config,
                        "ValidationMAPE": float(np.mean([item["MAPE"] for item in fold_metrics])),
                        "ValidationMAE": float(np.mean([item["MAE"] for item in fold_metrics])),
                        "ValidationMSE": float(np.mean([item["MSE"] for item in fold_metrics])),
                        **{f"Fold{i + 1}MAPE": item["MAPE"] for i, item in enumerate(fold_metrics)},
                    })
    best = min(candidates, key=lambda row: (row["ValidationMAPE"], row["ValidationMAE"]))
    config = {key: best[key] for key in ("feature_set", "target", "window", "alpha")}
    fold_records = [{"first_validation_feature_date": development[start]["Date"],
                     "last_validation_feature_date": development[end - 1]["Date"],
                     "first_validation_target_date": development[start]["TargetDate"],
                     "last_validation_target_date": development[end - 1]["TargetDate"],
                     "rows": end - start}
                    for start, end in folds]
    return config, best, candidates, fold_records


def legacy_reference(features, development_end):
    """Reproduce the previous five-input model without selecting on test data."""
    train_end = int(len(features) * .6)
    X = np.asarray([[row[key] for key in ORIGINAL_FEATURES] for row in features])
    y = np.asarray([row["TargetClose"] for row in features])
    best_alpha, best_error = None, float("inf")
    for alpha in np.logspace(-6, 6, 25):
        model = fit_ridge(X[:train_end], y[:train_end], float(alpha))
        prediction = predict_ridge(model, X[train_end:development_end])
        error = regression_metrics(y[train_end:development_end], prediction)["MSE"]
        if error < best_error:
            best_alpha, best_error = float(alpha), error
    model = fit_ridge(X[:development_end], y[:development_end], best_alpha)
    return predict_ridge(model, X[development_end:]), best_alpha


def serialize_configured_model(model, config, fit_rows, last_target_date, purpose):
    return {
        "symbol": "TSLA", "purpose": purpose, "configuration": config,
        "alpha": config["alpha"], "features": FEATURE_SETS[config["feature_set"]],
        "target_mode": config["target"], "refit_rows": fit_rows,
        "last_fit_target_date": last_target_date,
        "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
        "y_mean": float(model["y_mean"]), "weights": model["weights"].tolist(),
        "raw_prediction_formula": "y_mean + ((X - mean) / scale) @ weights",
        "price_conversion": "today_close * (1 + raw_prediction)" if config["target"] == "return" else "raw_prediction",
    }


def train_and_evaluate(features, results_dir, source_hash):
    """Freeze settings on three past folds, then walk forward over the final 20%."""
    count = len(features)
    development_end = int(count * .8)
    print("Selecting settings with three past-only validation folds...")
    config, best, search, folds = select_configuration(features, development_end)
    # Do not inspect final-region errors until configuration selection is over.
    prediction, audit = walk_forward_predict(features, development_end, count, config)
    previous_prediction, previous_alpha = legacy_reference(features, development_end)
    actual = np.asarray([row["TargetClose"] for row in features[development_end:]])
    baseline = np.asarray([row["Close"] for row in features[development_end:]])
    metrics = regression_metrics(actual, prediction)
    previous_metrics = regression_metrics(actual, previous_prediction)
    baseline_metrics = regression_metrics(actual, baseline)
    comparisons = [{"Model": name, **value} for name, value in (
        ("UpdatedRidge", metrics), ("PreviousRidge", previous_metrics), ("PersistenceBaseline", baseline_metrics))]
    predictions = []
    for row, predicted in zip(features[development_end:], prediction):
        difference = float(abs(predicted - row["TargetClose"]))
        rate = difference / abs(row["TargetClose"]) * 100
        predictions.append({"Date": row["TargetDate"], "ActualClose": row["TargetClose"],
                            "PredictedClose": float(predicted), "AbsoluteErrorUSD": difference,
                            "ErrorRatePercent": rate, "AccuracyPercent": 100 - rate})
    improvement = {key: 100 * (previous_metrics[key] - metrics[key]) / previous_metrics[key]
                   for key in ("MAE", "MAPE", "MSE") if previous_metrics[key] > 0}
    initial_model, fit_start = fit_configuration(features, development_end, config)
    record = serialize_configured_model(initial_model, config, development_end - fit_start,
                                         features[development_end - 1]["TargetDate"],
                                         "First test block; subsequent test blocks refit per walk_forward_audit.csv")
    report = {
        "symbol": "TSLA", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_csv_sha256": source_hash, "selected_config": config,
        "features": FEATURE_SETS[config["feature_set"]], "selected_alpha": config["alpha"],
        "selection_metric": "Mean validation MAPE across three sequential folds; MAE breaks ties",
        "selected_validation_metrics": best, "validation_candidates": len(search),
        "validation_folds": folds, "development_rows": development_end,
        "test_rows": count - development_end, "refit_every": REFIT_EVERY,
        "test_target_dates": {"first": features[development_end]["TargetDate"],
                              "last": features[-1]["TargetDate"]},
        "test_metrics": {"Ridge": metrics, "PreviousRidge": previous_metrics,
                         "PersistenceBaseline": baseline_metrics},
        "improvement_over_previous_percent": improvement,
        "previous_alpha": previous_alpha,
        "protocol": [
            "Final 20% region is excluded from configuration selection.",
            "Each validation fold and test period refits every 20 observations using only then-known labels.",
            "All scaling is fitted on each training window only; there is no shuffling.",
            "Chosen feature set, target, training window and alpha are frozen before final-region comparison.",
            "During test, earlier realized test outcomes may be used for later scheduled refits; future labels cannot.",
            "PreviousRidge exactly reproduces the prior fixed five-feature, single-validation MSE model.",
            "This test period has been seen in earlier project iterations: comparisons are descriptive, not a fresh independent confirmation.",
            "The accuracy display is 100 - MAPE, a custom similarity score, not a success probability.",
        ],
    }
    best_curve = [row for row in search if all(row[key] == config[key]
                  for key in ("feature_set", "target", "window"))]
    outputs = {
        "evaluation.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "ridge_model.json": json.dumps(record, indent=2, allow_nan=False) + "\n",
        "prediction_summary.csv": csv_bytes(predictions, list(predictions[0])),
        "model_comparison.csv": csv_bytes(comparisons, list(comparisons[0])),
        "validation_search.csv": csv_bytes(search, list(search[0])),
        "alpha_validation.csv": csv_bytes(best_curve, list(best_curve[0])),
        "walk_forward_audit.csv": csv_bytes(audit, list(audit[0])),
    }
    write_outputs(results_dir, outputs)
    print(f"Selected: {config['feature_set']} | target={config['target']} | "
          f"window={config['window'] or 'all'} | alpha={config['alpha']:.6g}")
    print(f"Previous vs updated average dollar error: ${previous_metrics['MAE']:.2f} -> ${metrics['MAE']:.2f}")
    print(f"Previous vs updated average error rate: {previous_metrics['MAPE']:.4f}% -> {metrics['MAPE']:.4f}%")
    print_prediction_table(predictions, metrics)
    plot_comparisons(predictions, best_curve, config["alpha"], results_dir)
    print(f"\nSaved table and charts to: {results_dir.resolve()}")
    return report


def print_prediction_table(rows, metrics):
    print("\nLast 10 test predictions")
    print(f"{'Date':<12} {'Actual $':>10} {'Predicted $':>12} {'Difference $':>13} {'Error %':>10} {'Accuracy %':>12}")
    for row in rows[-10:]:
        error = f"{row['ErrorRatePercent']:.2f}" if row['ErrorRatePercent'] is not None else "n/a"
        accuracy = f"{row['AccuracyPercent']:.2f}" if row['AccuracyPercent'] is not None else "n/a"
        print(f"{row['Date']:<12} {row['ActualClose']:10.2f} {row['PredictedClose']:12.2f} "
              f"{row['AbsoluteErrorUSD']:13.2f} {error:>10} {accuracy:>12}")
    print(f"\nAverage absolute difference: ${metrics['MAE']:.2f}")
    if metrics['MAPE'] is not None:
        print(f"Average error rate: {metrics['MAPE']:.2f}%")
        print(f"Accuracy (100 - average error rate): {metrics['CustomAccuracyPercent']:.2f}%")
    else:
        print("Error rate and accuracy: n/a (actual price contains zero)")


def plot_comparisons(predictions, validation_results, best_alpha, results_dir):
    """Save a simple actual-versus-Ridge comparison and optional diagnostics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as exc:
        raise RuntimeError("Charts require Matplotlib: python -m pip install matplotlib") from exc
    palette = {"actual": "#18324b", "ridge": "#008c95", "baseline": "#d48032"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    for filename, selected, title in (
        ("test_comparison.png", predictions, "TSLA | Actual vs predicted closing price"),
        ("test_recent_30.png", predictions[-30:], "TSLA | Actual vs predicted: last 30 test dates"),
    ):
        dates = [datetime.fromisoformat(row["Date"]) for row in selected]
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        ax.plot(dates, [row["ActualClose"] for row in selected],
                color=palette["actual"], label="Actual", linewidth=1.8)
        ax.plot(dates, [row["PredictedClose"] for row in selected],
                color=palette["ridge"], label="Predicted", linewidth=1.5)
        ax.set(title=title, xlabel="Target date", ylabel="Closing price (USD)")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(results_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.semilogx([row["alpha"] for row in validation_results],
                [row["ValidationMAPE"] for row in validation_results], "o-",
                label="Mean of three validation folds", color=palette["ridge"], markersize=4)
    ax.axvline(best_alpha, color=palette["baseline"], linestyle="--", label=f"Selected alpha = {best_alpha:.6g}")
    ax.set(title="Chosen configuration: validation error by Ridge alpha",
           xlabel="Alpha (log scale)", ylabel="Mean absolute percentage error (%)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(results_dir / "alpha_validation.png", dpi=160)
    plt.close(fig)


def scheduled_market_holidays(year):
    """Regular US equity full-day holidays; exceptional closures are not predicted."""
    if year < 2022 or year > 2099:
        raise ValueError("Scheduled market calendar supports years 2022 through 2099.")

    def nth_weekday(month, weekday, occurrence):
        first = datetime(year, month, 1).date()
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))

    def observed(month, day):
        holiday = datetime(year, month, day).date()
        if holiday.weekday() == 5:
            return holiday - timedelta(days=1)
        if holiday.weekday() == 6:
            return holiday + timedelta(days=1)
        return holiday

    # Gregorian Easter (Meeus/Jones/Butcher), to obtain Good Friday.
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, remainder = divmod(h + l - 7 * m + 114, 31)
    easter = datetime(year, month, remainder + 1).date()
    new_year = datetime(year, 1, 1).date()
    # US equity markets do not observe a Saturday Jan 1 on the preceding Friday.
    if new_year.weekday() == 6:
        new_year += timedelta(days=1)
    last_may = datetime(year, 6, 1).date() - timedelta(days=1)
    memorial = last_may - timedelta(days=last_may.weekday())
    return {
        new_year, nth_weekday(1, 0, 3), nth_weekday(2, 0, 3),
        easter - timedelta(days=2), memorial, observed(6, 19),
        observed(7, 4), nth_weekday(9, 0, 1), nth_weekday(11, 3, 4),
        observed(12, 25),
    }


def next_market_date(as_of):
    """Next scheduled session; early-close days are still trading days."""
    candidate = datetime.fromisoformat(as_of).date() + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in scheduled_market_holidays(candidate.year):
        candidate += timedelta(days=1)
    return candidate.isoformat()


def forecast_next_close(features, config, output_dir, source_hash):
    """After evaluation, fit a separate model on every known labeled row."""
    with (output_dir / "tsla_daily.csv").open(encoding="utf-8-sig", newline="") as handle:
        daily = list(csv.DictReader(handle))
    if len(daily) < 21:
        raise ValueError("Not enough daily prices for the latest forecast input.")
    dates = [row["Date"] for row in daily]
    if dates != sorted(set(dates)):
        raise ValueError("Forecast prices must have unique increasing dates.")
    # Avoid constructing the final return window across a removed source row.
    payload = (output_dir / "tsla_raw.json").read_bytes()
    metadata = json.loads((output_dir / "data_metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != metadata.get("raw_json_sha256"):
        raise ValueError("Raw price JSON does not match the data metadata.")
    source = json.loads(payload)["chart"]["result"][0]
    if source["meta"]["symbol"] != "TSLA":
        raise ValueError("Forecast source must contain TSLA observations.")
    market_tz = timezone(timedelta(seconds=int(source["meta"].get("gmtoffset", -18000))))
    source_dates = [datetime.fromtimestamp(value, market_tz).date().isoformat()
                    for value in source["timestamp"]]
    positions = {day: index for index, day in enumerate(source_dates)}
    if positions[dates[-1]] - positions[dates[-21]] != 20:
        raise ValueError("Latest forecast window crosses missing source observations.")
    closes = np.asarray([float(row["Close"]) for row in daily])
    latest = enriched_features(daily, len(daily) - 1)
    as_of = dates[-1]
    target_date = next_market_date(as_of)
    if any(row["TargetDate"] > as_of for row in features):
        raise ValueError("A forecast model cannot train on labels after its as-of date.")
    model, fit_start = fit_configuration(features, len(features), config)
    predicted = float(configuration_prediction(model, [latest], config)[0])
    if not math.isfinite(predicted) or predicted <= 0:
        raise ValueError("The model produced a non-positive or non-finite price forecast.")
    forecast = {
        "symbol": "TSLA", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of, "target_date": target_date, "currency": "USD",
        "last_actual_close": float(closes[-1]), "predicted_close": predicted,
        "actual_close": None, "absolute_error_usd": None,
        "error_rate_percent": None, "accuracy_percent": None,
        "selected_alpha": config["alpha"], "configuration": config, "training_rows": len(features) - fit_start,
        "last_training_target_date": features[-1]["TargetDate"],
        "features": {key: latest[key] for key in FEATURE_SETS[config["feature_set"]]}, "feature_csv_sha256": source_hash,
        "daily_csv_sha256": hashlib.sha256((output_dir / "tsla_daily.csv").read_bytes()).hexdigest(),
        "calendar_source": CALENDAR_SOURCE,
        "notes": [
            "Configuration is taken from historical validation; no new tuning on test results.",
            "This separate forecast model uses the validation-selected historical training window.",
            "It does not overwrite ridge_model.json or change historical test metrics.",
            "The latest unlabeled input is used for inference, not as a fabricated training target.",
            "Target is the next scheduled market date after the data snapshot, not after the computer date.",
            "Regular market holidays are excluded; unexpected exchange closures cannot be predicted.",
            "Future actual close, error and accuracy remain unknown until the target close is observed.",
        ],
    }
    model_record = serialize_configured_model(model, config, len(features) - fit_start,
                                               features[-1]["TargetDate"], "Next-session forecast")
    summary = {"AsOfDate": as_of, "TargetDate": target_date, "LastActualClose": float(closes[-1]),
               "PredictedClose": predicted, "ActualClose": None, "AbsoluteErrorUSD": None,
               "ErrorRatePercent": None, "AccuracyPercent": None}
    outputs = {"next_forecast.json": json.dumps(forecast, indent=2, allow_nan=False) + "\n",
               "next_forecast.csv": csv_bytes([summary], list(summary)),
               "forecast_model.json": json.dumps(model_record, indent=2, allow_nan=False) + "\n"}
    results_dir = output_dir / "results"
    write_outputs(results_dir, outputs)
    print("\nNext trading-session close forecast")
    print(f"Data through: {as_of} | Last actual close: ${closes[-1]:.2f}")
    print(f"Target date: {target_date} | Predicted close: ${predicted:.2f}")
    print("Actual close / error / accuracy: pending (not available in this snapshot)")
    print("Saved: next_forecast.csv, next_forecast.json, forecast_model.json")
    return forecast


def main():
    parser = argparse.ArgumentParser(description="Ridge project: prepare TSLA data, train and evaluate next-session predictions.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="Destination for downloaded daily data.")
    parser.add_argument("--train-only", action="store_true", help="Reuse the existing feature snapshot without downloading.")
    args = parser.parse_args()
    try:
        if not args.train_only:
            rows, raw_rows = collect_prices(args.output_dir)
        else:
            rows, raw_rows = load_price_snapshot(args.output_dir)
        save_features(rows, raw_rows, args.output_dir)
        features, source_hash = load_feature_rows(args.output_dir)
        report = train_and_evaluate(features, args.output_dir / "results", source_hash)
        forecast_next_close(features, report["selected_config"], args.output_dir, source_hash)
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
