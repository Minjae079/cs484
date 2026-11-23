# TSLA next-trading-session Ridge prediction

## Install and run

Python 3.9+, NumPy and Matplotlib are required. Downloading, CSV handling
and scheduled-date calculations use Python's standard library.

```powershell
python -m pip install numpy matplotlib
python ridge_project.py
```

The default command downloads the latest available five years of completed
daily TSLA prices, builds features, evaluates Ridge, saves charts and predicts
the next scheduled trading-session close. No API key is required.

To reuse the saved snapshot without downloading:

```powershell
python ridge_project.py --train-only
```

The offline command requires an existing complete `data/` snapshot. Its forecast
date is relative to that snapshot, which may be old. Use the default command
for an updated forecast. A separate snapshot can be selected with
`--output-dir data_snapshot`.

## Data and features

Source: https://finance.yahoo.com/quote/TSLA/history/

Endpoint: https://query1.finance.yahoo.com/v8/finance/chart/TSLA?range=5y&interval=1d

The program uses completed daily observations, not real-time intraday prices.
Provider delays may affect availability. Weekends and closed market days have
no new daily closing price. The rolling dataset changes over time; preserve
the raw response and metadata for reproducible experiments.

The feature date is after that day's regular-session close. All candidates use
past-only data from the current and preceding observations. Stored feature pools:

| Group | Inputs |
| --- | --- |
| Original | Close, SMA5, SMA20, Return1D, Volatility20D |
| Relative | Return1D, previous one-day return, 5-day/20-day returns, Close/SMA5-1, Close/SMA20-1, Volatility20D |
| Relative + volume | Relative inputs plus Volume/20-day-average-volume-1, (High-Low)/Close, Open/previous-Close-1, Close/Open-1 |

Returns and relative differences are fractions. Volatility is the sample standard
deviation (ddof=1) of the last 20 daily returns, not annualized. Price inputs and
targets consistently use provider Close, not AdjustedClose. The same input builder
is used for historical and new forecasts. Each group is a separate candidate;
extra features are not assumed to improve prediction automatically.

The target observation is the next available source trading date. TargetDate and
TargetClose never enter the input columns. Twenty returns need 21 prices: the
first 20 rows are warm-up observations. The final daily price has no known next
target and is used only as the newest forecast input.

Null observations are counted and skipped, not filled. Windows crossing removed
source observations are excluded. Dates, price ranges, numeric values and source
hashes are checked. Current-day bars are excluded until 15 minutes after the
provider's regular close. A feed older than seven calendar days fails collection.
Network requests have a 15-second timeout and at most three attempts. Download
failures do not silently substitute synthetic data or old prices.

## Model and evaluation

Ridge minimizes `||Xw-y||^2 + alpha*||w||^2`, implemented with NumPy normal
equations. Each fitting window supplies its own scaler and target mean.
The intercept is unpenalized. No prebuilt ML estimator is used.

The first 80% of supervised rows are development data. The last 20% are excluded
from configuration selection. Three sequential validation folds begin at 1/2,
2/3 and 5/6 of development data and end at the next boundary.

The fixed search contains 198 configurations:

- Three input groups listed above.
- Two training targets: next closing price, or next simple return.
- Three maximum training windows: 252 rows, 504 rows, or all available past rows.
- Eleven alpha values, logarithmically spaced from 1e-4 to 1e6.

Every candidate refits every 20 prediction dates using only then-known labels.
Settings are selected by the mean validation MAPE across the three folds;
mean validation MAE breaks ties. The final region is evaluated only AFTER this
selection. Its earlier realized outcomes can be used for later scheduled fits,
but a future outcome is never used for an earlier prediction.

For a return target, `r_next = next_close / current_close - 1`, and the predicted
closing price is `current_close * (1 + predicted_return)`.
The target transformation and feature group are used consistently in evaluation,
model serialization and the next-session forecast.

The next-session model is fitted with the selected configuration and window on
all currently available labeled history. The newest unlabeled input is used for
inference. The saved `ridge_model.json` is the first historical test-block model;
`forecast_model.json` is the separate model for the new forecast.

Comparison records include an exact reproduction of the previous five-input
model (single validation MSE selection, fixed test model) and persistence.
The earlier test period has already been observed during this project's previous
iterations: improvement on this reused period is descriptive, not independent
confirmation. The new selection code never reads its errors to choose settings.
Confirm prospective improvement on data collected after freezing the design.

The terminal and `prediction_summary.csv` show actual price, Ridge prediction,
absolute dollar difference, absolute percentage error, and requested accuracy:

```text
Dollar difference = abs(predicted - actual)
Error rate (%) = 100 * abs(predicted - actual) / abs(actual)
Accuracy (%) = 100 - error rate (%)
Overall accuracy (%) = 100 - mean absolute percentage error (%)
```

This accuracy is a user-defined price similarity score, not classification or
up/down accuracy. Scores are not clipped; errors over 100% yield negative scores.
Percentage metrics are undefined if an actual price is zero.
MSE/RMSE/MAE and baseline results remain available in the evaluation JSON.

## Files

```text
ridge_project.py                 Single-file program
README.md                       This guide
data/
  tsla_raw.json                 Exact downloaded response
  tsla_daily.csv                Validated daily prices
  data_metadata.json            Source, retrieval time, counts and raw hash
  tsla_features.csv             Historical inputs and next-session targets
  feature_metadata.json         Definitions, exclusions and daily-data hash
  results/
    prediction_summary.csv      All held-out price comparisons and error scores
    evaluation.json             Splits, validation selection and test metrics
    alpha_validation.csv        Alpha curve for the selected group/target/window
    validation_search.csv       All 198 configurations and validation scores
    model_comparison.csv        Updated vs previous Ridge and persistence
    walk_forward_audit.csv      Prediction blocks and last available fitting labels
    ridge_model.json            Model at the start of historical test evaluation
    test_comparison.png         Actual vs predicted prices, full test period
    test_recent_30.png          Actual vs predicted prices, last 30 test dates
    alpha_validation.png        Mean validation MAPE by alpha
    next_forecast.csv           New forecast, simple table
    next_forecast.json          New forecast, inputs and provenance
    forecast_model.json         Separate forecast model with the selected window
```

Generated files are refreshed on each successful run. Each file is replaced
atomically after its content has been prepared; preserve a snapshot if needed.
Future actual price, error and accuracy remain blank until the target close is
observed. Past test accuracy is not a guarantee for the new forecast.

The next-session helper skips weekends and regular US equity holidays for
2022-2099. The 2026 dates were checked against Nasdaq's published schedule:
https://www.nasdaq.com/market-activity/stock-market-holiday-schedule
Early-close days remain valid sessions. Exceptional closures or future rule
changes cannot be predicted by these scheduled-calendar rules.

For submission, the Python file downloads its own data; no private local CSV
is required. A written PDF report remains a separate project deliverable.

## Verification snapshot

With 1,255 daily observations (2021-11-22 through 2026-11-20) and 1,234 labeled
rows, development-only validation selected relative+volume inputs, a return
target, all past rows and alpha=1000. On the existing final 247 rows:

| Metric | Previous Ridge | Updated Ridge |
| --- | ---: | ---: |
| MAE (USD) | 8.6149 | 8.5090 |
| MAPE (%) | 2.2046 | 2.1843 |
| RMSE (USD) | 11.2995 | 11.2421 |

The change is modest and does not improve every date. The search is rerun when
data changes, so selected settings and results may differ on a later snapshot.
