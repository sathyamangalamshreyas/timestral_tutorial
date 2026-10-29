import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from pathlib import Path

    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd

    from timestral import TimestralForecaster


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Data Preprocessing
    """)
    return


@app.cell
def _():
    # Import data

    data_path = Path("data/raw/hour.csv")

    df = pd.read_csv(data_path, parse_dates=["dteday"])

    df["datetime"] = (
        df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    # Reorder columns

    col_names = list(df.columns)

    col_names.insert(2, col_names[-1])

    col_names.pop()

    print(f"Column Names: {col_names}")

    df = df[col_names]

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Inspect data quality and sampling frequency
    """)
    return


@app.cell
def _(df):
    print(f"Rows: {len(df):,}")
    print(f"Start: {df['datetime'].min()}")
    print(f"End: {df['datetime'].max()}")

    print(f"\nMissing Values:")
    print(df.isna().sum())

    print(f"\nTotal Missing Values:")
    print(df.isna().sum().sum())

    print(f"\nDuplicate Timestamps: {df['datetime'].duplicated().sum()}")

    print(f"\nTime-step frequencies:")
    print(df["datetime"].diff().value_counts().head(10))

    print("\nTarget Summary:")
    print(df["cnt"].describe())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the data

    Plot the first two weeks and overall demand pattern
    """)
    return


@app.cell
def _(df):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    df_first_two_weeks = df.iloc[:14 * 24]

    axes[0].plot(
        df_first_two_weeks["datetime"],
        df_first_two_weeks["cnt"],
    )

    axes[0].set_title("Hourly bike rentals - first two weeks")
    axes[0].set_ylabel("Rented Bikes")

    axes[1].plot(
        df["datetime"],
        df["cnt"],
    )

    axes[1].set_title("Hourly Bike Rentals - Complete Dataset")
    axes[1].set_ylabel("Rented Bikes")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Create a continuous hourly timeline
    """)
    return


@app.cell
def _(df):
    # Create every hourly timestamp between the first and last records
    hourly_index = pd.date_range(
        start=df["datetime"].min(),
        end=df["datetime"].max(),
        freq="h",
    )

    # Insert rows for timestamps that are absent from the original data
    df_hourly = (
        df.set_index("datetime")
        .reindex(hourly_index)
    )

    df_hourly.index_name = "datetime"

    print(f"Original Rows: {len(df):,}")
    print(f"\nExpected hourly timestamps: {len(hourly_index):,}")
    print(f"\nRows after reindexing: {len(df_hourly):,}")

    print(f"\nMissing target observations: {df_hourly['cnt'].isna().sum()}")

    print(f"\nDuplicate timestamps: {df_hourly.index.duplicated().sum()}")

    print(f"\nTime-step frequencies:")
    print(
        df_hourly.index.to_series()
        .diff()
        .dropna()
        .value_counts()
    )
    return (df_hourly,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Check for missing oservations at the end of the time-frame
    This is because we want to use the latest 24 hour period as the test horizon
    """)
    return


@app.cell
def _(df_hourly):
    print("Last 48 hourly positions:")
    print(f"{df_hourly[["cnt"]].tail(48)}")

    print(f"\nMissing target values in the last 24 hours:")
    print(f"{df_hourly['cnt'].tail(24).isna().sum()}")

    print(f"\nMissing target values in the last 7 days:")
    print(f"{df_hourly['cnt'].tail(7 * 24).isna().sum()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Prepare the forecast context
    """)
    return


@app.cell
def _(df_hourly):
    HORIZON = 24

    target = df_hourly["cnt"].to_numpy(dtype=np.float32)

    # All observations before the final day
    context = target[:-HORIZON]

    # The final 24 acutal observations are held out for evaluation
    actual_future = target[-HORIZON:]

    future_index = df_hourly.index[-HORIZON:]

    print(f"Context Length: {len(context):,}")
    print(f"\nMissing Values in context: {np.isnan(context).sum()}")
    print(f"\nFuture Horizon: {len(actual_future)}")
    print(f"\nMissing Values in Future: {np.isnan(actual_future).sum()}")
    print(f"Forecast Start: {future_index[0]}")
    print(f"Forecast End: {future_index[-1]}")
    return HORIZON, actual_future, context, future_index, target


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generate the 24-hour forecast
    """)
    return


@app.cell
def _(HORIZON, context):
    forecaster = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M",
        device="cpu",
    )

    forecast = forecaster.forecast(
        context=context,
        prediction_length=HORIZON,
    )

    median = forecast.median[0, 0].detach().cpu().numpy()
    lower = forecast.quantile(0.1)[0, 0].detach().cpu().numpy()
    upper = forecast.quantile(0.9)[0, 0].detach().cpu().numpy()

    print(f"\nMedian Forecast Shape: {median.shape}")
    print(f"\nQuantile Forecast Shape: {tuple(forecast.quantiles.shape)}")
    return forecast, forecaster, lower, median, upper


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Evaluate Timestral against a daily seasonal-naive baseline
    Our baseline will predict December 31 using the rental counts from the same hours on December 30. Since the immediately preceding day has complete observations, this comparison is straightforward.
    """)
    return


@app.cell
def _(actual_future, context, forecast, lower, median, upper):
    errors = actual_future - median

    absolute_errors = np.abs(errors)

    mae = absolute_errors.mean()
    rmse = np.sqrt(np.mean(errors**2))

    mape = (
        np.mean(absolute_errors / np.abs(actual_future)) * 100
        if np.all(np.abs(actual_future) > 1e-8)
        else np.nan
    )

    smape_denominator = np.abs(actual_future) + np.abs(median)

    smape_values = np.divide(
        200 * absolute_errors,
        smape_denominator,
        out=np.zeros_like(absolute_errors),
        where=smape_denominator > 0,
    )

    smape = smape_values.mean()

    # Repeat the previous day's 24 hourly observations
    seasonal_naive = context[-24].copy()

    baseline_mae = np.mean(np.abs(actual_future - seasonal_naive))

    # Calculate the MASE scale from valid historical
    # pairs separated by 24 hours
    seasonal_differences = np.abs(context[24:] - context[:-24])
    valid_differences = np.isfinite(seasonal_differences)

    seasonal_scale = seasonal_differences[valid_differences].mean()

    mase = mae / seasonal_scale if seasonal_scale > 0 else np.nan

    mae_improvement = (
        100 * (baseline_mae - mae) / baseline_mae
        if baseline_mae > 0
        else np.nan
    )

    # Probabilistic forecast metrics
    coverage_80 = np.mean(
        (actual_future >= lower) & (actual_future <= upper)
    ) * 100

    interval_width = np.mean(upper - lower)

    quantiles = forecast.quantiles[0, 0].detach().cpu().numpy()
    levels = np.asarray(forecast.quantile_levels)

    quantile_errors = actual_future[:, None] - quantiles

    pinball_loss = np.maximum(
        levels[None, :] * quantile_errors,
        (levels[None, :] - 1) * quantile_errors,
    ).mean()

    metrics = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "MAPE (%)",
                "sMAPE (%)",
                "MASE (daily period)",
                "80% interval coverage",
                "Mean 80% interval width",
                "Mean pinball loss",
            ],
            "Timestral": [
                mae,
                rmse,
                mape,
                smape,
                mase,
                coverage_80,
                interval_width,
                pinball_loss,
            ],
            "Seasonal naive": [
                baseline_mae,
                np.sqrt(np.mean((actual_future - seasonal_naive) ** 2)),
                np.mean(
                    np.abs(actual_future - seasonal_naive)
                    / np.abs(actual_future)
                ) * 100,
                np.mean(
                    np.divide(
                        200 * np.abs(actual_future - seasonal_naive),
                        np.abs(actual_future) + np.abs(seasonal_naive),
                        out=np.zeros_like(actual_future),
                        where=(
                            np.abs(actual_future) + np.abs(seasonal_naive)
                        ) > 0,
                    )
                ),
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    print(f"MAE improvement over daily naive baseline: {mae_improvement:.2f}%")

    metrics.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the forecast
    """)
    return


@app.cell
def _(HORIZON, actual_future, df_hourly, future_index, lower, median, upper):
    def _():
        history_start = -(7 * 24 + HORIZON)

        history_index = df_hourly.index[history_start:-HORIZON]
        history_values = df_hourly["cnt"].iloc[history_start:-HORIZON]

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(
            history_index,
            history_values,
            label="Historical Rentals",
        )

        ax.plot(
            future_index,
            actual_future,
            label="Acutal Rentals",
        )

        ax.plot(
            future_index,
            median,
            label="Timestral Median Forecast",
        )

        ax.fill_between(
            future_index,
            lower,
            upper,
            alpha=0.2,
            label="80% predictive interval",
        )

        ax.axvline(
            future_index[0],
            linestyle="--",
            color='r',
            label="Forecast Origin",
        )

        ax.set(
            title="UCI Bike Sharing: 24-hour Forecast",
            xlabel="Date and Time",
            ylabel="Bikes Rented per Hour",
        )

        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Next experiment — Forecast a full seven days
    We'll use the last 168 hourly positions as the test window. One of those hours has a missing rental count, so we'll generate a forecast for all 168 hours but evaluate only the 167 hours with known actual values.

    Timestral supports missing observations in the historical context, and 168 forecast steps fit within a single forecasting block.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Prepare the seven-day test window
    """)
    return


@app.cell
def _(df_hourly, target):
    HORIZON_7D = 7 * 24

    context_7d = target[:-HORIZON_7D]
    actual_7d = target[-HORIZON_7D:]
    future_index_7d = df_hourly.index[-HORIZON_7D:]

    print(f"Context Length: {len(context_7d):,}")
    print(f"\nMissing Values in context: {np.isnan(context_7d).sum()}")
    print(f"\nForecast Horizon: {len(actual_7d)}")
    print(f"\nMissing Future Observations: {np.isnan(actual_7d).sum()}")
    print(f"\nValid Future Observations: {np.isfinite(actual_7d).sum()}")
    print(f"\nForecast Start: {future_index_7d[0]}")
    print(f"\nForecast End: {future_index_7d[-1]}")
    return HORIZON_7D, actual_7d, context_7d, future_index_7d


@app.cell
def _(HORIZON_7D, context_7d, forecaster):
    forecast_7d = forecaster.forecast(
        context_7d,
        prediction_length=HORIZON_7D,
    )

    median_7d = (
        forecast_7d.median[0, 0].detach().cpu().numpy()
    )

    lower_7d = (
        forecast_7d.quantile(0.1)[0, 0].detach().cpu().numpy()
    )

    upper_7d = (
        forecast_7d.quantile(0.9)[0, 0].detach().cpu().numpy()
    )

    quantiles_7d = (
        forecast_7d.quantiles[0, 0].detach().cpu().numpy()
    )

    print(f"Median Forecast Shape: {median_7d.shape}")

    print(f"\nQuantile Forecast Shape: {quantiles_7d.shape}")
    return forecast_7d, lower_7d, median_7d, quantiles_7d, upper_7d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare with a daily seasonal-naive baseline
    """)
    return


@app.cell
def _(context_7d):
    baseline_day = None

    for end_idx in range(len(context_7d), 23, -24):
        candidate = context_7d[end_idx - 24:end_idx]

        if np.isfinite(candidate).all():
            baseline_day = candidate.copy()
            break

    if baseline_day is None:
            raise ValueError("No complete historical 24-hour day found")

    baseline_7d = np.tile(baseline_day, 7)

    print(f"Baseline Length: {len(baseline_7d)}")
    return (baseline_7d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Calculate the Metrics
    """)
    return


@app.cell
def _(
    actual_7d,
    baseline_7d,
    context_7d,
    forecast_7d,
    lower_7d,
    median_7d,
    quantiles_7d,
    upper_7d,
):
    def _():
        valid = np.isfinite(actual_7d)

        y_true = actual_7d[valid]
        y_pred = median_7d[valid]
        y_baseline = baseline_7d[valid]

        lower_valid = lower_7d[valid]
        upper_valid = upper_7d[valid]
        quantiles_valid = quantiles_7d[valid]

        errors = y_true - y_pred
        baseline_errors = y_true - y_baseline

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        baseline_mae = np.mean(np.abs(baseline_errors))
        baseline_rmse = np.sqrt(np.mean(baseline_errors**2))

        mape = np.mean(np.abs(errors) / np.abs(y_true)) * 100
        baseline_mape = (
            np.mean(np.abs(baseline_errors) / np.abs(y_true)) * 100
        )

        smape_denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean(
            np.divide(
                200 * np.abs(errors),
                smape_denominator,
                out=np.zeros_like(errors),
                where=smape_denominator > 0,
            )
        )

        baseline_denominator = np.abs(y_true) + np.abs(y_baseline)
        baseline_smape = np.mean(
            np.divide(
                200 * np.abs(baseline_errors),
                baseline_denominator,
                out=np.zeros_like(baseline_errors),
                where=baseline_denominator > 0,
            )
        )

        seasonal_differences = np.abs(
            context_7d[24:] - context_7d[:-24]
        )
        valid_differences = np.isfinite(seasonal_differences)
        seasonal_scale = seasonal_differences[valid_differences].mean()

        mase = mae / seasonal_scale if seasonal_scale > 0 else np.nan

        coverage_80 = np.mean(
            (y_true >= lower_valid) & (y_true <= upper_valid)
        ) * 100

        interval_width = np.mean(upper_valid - lower_valid)

        levels = np.asarray(forecast_7d.quantile_levels)
        quantile_errors = y_true[:, None] - quantiles_valid

        pinball_loss = np.maximum(
            levels[None, :] * quantile_errors,
            (levels[None, :] - 1) * quantile_errors,
        ).mean()

        improvement = (
            100 * (baseline_mae - mae) / baseline_mae
            if baseline_mae > 0
            else np.nan
        )

        results_7d = pd.DataFrame({
            "Metric": [
                "MAE",
                "RMSE",
                "MAPE (%)",
                "sMAPE (%)",
                "MASE (daily period)",
                "80% interval coverage (%)",
                "Mean 80% interval width",
                "Mean pinball loss",
            ],
            "Timestral": [
                mae, rmse, mape, smape, mase,
                coverage_80, interval_width, pinball_loss,
            ],
            "Daily seasonal naive": [
                baseline_mae, baseline_rmse, baseline_mape,
                baseline_smape, np.nan, np.nan, np.nan, np.nan,
            ],
        })

        print("Valid evaluation points:", valid.sum())
        print(f"MAE improvement: {improvement:.2f}%")
        return results_7d.round(4)


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Plot the forecast
    """)
    return


@app.cell
def _(
    HORIZON_7D,
    actual_7d,
    baseline_7d,
    df_hourly,
    future_index_7d,
    lower_7d,
    median_7d,
    upper_7d,
):
    def _():
        fig, ax = plt.subplots(figsize=(14, 6))

        history_index_7d = df_hourly.index[-2 * HORIZON_7D:-HORIZON_7D]
        history_values_7d = df_hourly["cnt"].iloc[-2 * HORIZON_7D:-HORIZON_7D]

        ax.plot(
            history_index_7d,
            history_values_7d,
            label="Previous week",
        )

        ax.plot(
            future_index_7d,
            actual_7d,
            label="Actual future rentals",
        )

        ax.plot(
            future_index_7d,
            median_7d,
            label="Timestral median forecast",
        )

        ax.plot(
            future_index_7d,
            baseline_7d,
            label="Daily seasonal-naive baseline",
        )

        ax.fill_between(
            future_index_7d,
            lower_7d,
            upper_7d,
            alpha=0.2,
            label="Timestral 80% interval",
        )

        ax.axvline(
            future_index_7d[0],
            linestyle="--",
            label="Forecast origin",
        )

        ax.set(
            title="UCI Bike Sharing: Seven-day forecast",
            xlabel="Date",
            ylabel="Bikes rented per hour",
        )

        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Add known-future calendar covariates
    We'll compare four approaches on exactly the same seven-day holdout:
    1. Timestral using the target history only.
    2. Timestral using the target history plus hour-of-day and day-of-week categories.
    3. The daily seasonal-naive baseline you already evaluated.
    4. A weekly seasonal-naive baseline, which repeats the previous complete week.
    The last comparison is worthwhile because hourly bike rentals have both daily and weekly patterns.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Construct the calendar features
    """)
    return


@app.cell
def _(HORIZON_7D, context_7d, df_hourly):
    # Derive calendar features from the complete hourly index.
    calendar_index = df_hourly.index

    hour_codes = calendar_index.hour.to_numpy(dtype=np.float32)
    weekday_codes = calendar_index.dayofweek.to_numpy(dtype=np.float32)

    # Two categorical feature rows:
    # row 0 = hour of day (0–23)
    # row 1 = day of week (Monday=0, Sunday=6)
    calendar_categories = np.stack(
        [hour_codes, weekday_codes],
        axis=0,
    )[None, :, :]

    assert calendar_categories.shape == (
        1,
        2,
        len(context_7d) + HORIZON_7D,
    )

    print("Calendar covariate shape:", calendar_categories.shape)
    print("Hour categories:", np.unique(hour_codes))
    print("Weekday categories:", np.unique(weekday_codes))
    return (calendar_categories,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Generate the calendar-aware forecast
    """)
    return


@app.cell
def _(HORIZON_7D, calendar_categories, context_7d, forecaster):
    forecast_calendar_7d = forecaster.forecast(
        context_7d,
        prediction_length=HORIZON_7D,
        categorical_known_covariates=calendar_categories,
    )

    median_calendar_7d = (
        forecast_calendar_7d.median[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    lower_calendar_7d = (
        forecast_calendar_7d.quantile(0.1)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    upper_calendar_7d = (
        forecast_calendar_7d.quantile(0.9)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    quantiles_calendar_7d = (
        forecast_calendar_7d.quantiles[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    print("Median shape:", median_calendar_7d.shape)
    print("Quantile shape:", quantiles_calendar_7d.shape)
    return (
        forecast_calendar_7d,
        lower_calendar_7d,
        median_calendar_7d,
        quantiles_calendar_7d,
        upper_calendar_7d,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Build a weekly seasonal-naive baseline
    """)
    return


@app.cell
def _(context_7d, df_hourly):
    def compute_weekly_baseline():
        weekly_baseline_7d = None
        weekly_baseline_end = None

        for end_idx in range(len(context_7d), 167, -24):
            candidate = context_7d[end_idx - 168 : end_idx]

            if len(candidate) == 168 and np.isfinite(candidate).all():
                weekly_baseline_7d = candidate.copy()
                weekly_baseline_end = df_hourly.index[end_idx - 1]
                break

        if weekly_baseline_7d is None:
            raise ValueError("No complete historical week found.")

        print("Weekly baseline shape:", weekly_baseline_7d.shape)
        print("Baseline week ends:", weekly_baseline_end)

        return weekly_baseline_7d, weekly_baseline_end


    # Unpack variables at top-level scope
    weekly_baseline_7d_new, weekly_baseline_end = compute_weekly_baseline()
    return (weekly_baseline_7d_new,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare all four approaches
    """)
    return


@app.function
def score_bike_forecast(
    name,
    actual,
    predicted,
    context,
    lower=None,
    upper=None,
    quantiles=None,
    quantile_levels=None,
):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Use the same observed future timestamps for every approach.
    valid = np.isfinite(actual)

    y = actual[valid]
    pred = predicted[valid]

    errors = y - pred
    absolute_errors = np.abs(errors)

    mae = absolute_errors.mean()
    rmse = np.sqrt(np.mean(errors**2))

    mape = (
        np.mean(absolute_errors / np.abs(y)) * 100
        if np.all(np.abs(y) > 1e-8)
        else np.nan
    )

    denominator = np.abs(y) + np.abs(pred)
    smape = np.mean(
        np.divide(
            200 * absolute_errors,
            denominator,
            out=np.zeros_like(absolute_errors),
            where=denominator > 0,
        )
    )

    context = np.asarray(context, dtype=float)
    differences = np.abs(context[24:] - context[:-24])
    differences = differences[np.isfinite(differences)]

    scale = differences.mean() if len(differences) else np.nan
    mase = mae / scale if scale > 0 else np.nan

    row = {
        "Approach": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "sMAPE (%)": smape,
        "MASE (daily period)": mase,
        "80% coverage (%)": np.nan,
        "80% interval width": np.nan,
        "Mean pinball loss": np.nan,
    }

    if lower is not None and upper is not None:
        lo = np.asarray(lower)[valid]
        hi = np.asarray(upper)[valid]

        row["80% coverage (%)"] = (
            np.mean((y >= lo) & (y <= hi)) * 100
        )
        row["80% interval width"] = np.mean(hi - lo)

    if quantiles is not None and quantile_levels is not None:
        q = np.asarray(quantiles)[valid]
        levels = np.asarray(quantile_levels)

        q_errors = y[:, None] - q
        pinball = np.maximum(
            levels[None, :] * q_errors,
            (levels[None, :] - 1) * q_errors,
        )

        row["Mean pinball loss"] = pinball.mean()

    return row


@app.cell
def _(
    actual_7d,
    baseline_7d,
    context_7d,
    forecast_7d,
    forecast_calendar_7d,
    lower_7d,
    lower_calendar_7d,
    median_7d,
    median_calendar_7d,
    quantiles_7d,
    quantiles_calendar_7d,
    upper_7d,
    upper_calendar_7d,
    weekly_baseline_7d_new,
):
    valid_actual_7d = np.isfinite(actual_7d)

    comparison_7d = pd.DataFrame([
        score_bike_forecast(
            "Timestral — target only",
            actual_7d,
            median_7d,
            context_7d,
            lower=lower_7d,
            upper=upper_7d,
            quantiles=quantiles_7d,
            quantile_levels=forecast_7d.quantile_levels,
        ),
        score_bike_forecast(
            "Timestral — calendar categories",
            actual_7d,
            median_calendar_7d,
            context_7d,
            lower=lower_calendar_7d,
            upper=upper_calendar_7d,
            quantiles=quantiles_calendar_7d,
            quantile_levels=forecast_calendar_7d.quantile_levels,
        ),
        score_bike_forecast(
            "Daily seasonal naive",
            actual_7d,
            baseline_7d,
            context_7d,
        ),
        score_bike_forecast(
            "Weekly seasonal naive",
            actual_7d,
            weekly_baseline_7d_new,
            context_7d,
        ),
    ]).set_index("Approach")

    print("Evaluation observations:", valid_actual_7d.sum())
    comparison_7d.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the calander-aware forecast
    """)
    return


@app.cell
def _(
    actual_7d,
    df_hourly,
    future_index_7d,
    lower_calendar_7d,
    median_7d,
    median_calendar_7d,
    upper_calendar_7d,
    weekly_baseline_7d_new,
):
    def _():
        fig, ax = plt.subplots(figsize=(14, 6))

        history_index_7d = df_hourly.index[-336:-168]
        history_values_7d = df_hourly["cnt"].iloc[-336:-168]

        ax.plot(
            history_index_7d,
            history_values_7d,
            label="Previous week",
        )

        ax.plot(
            future_index_7d,
            actual_7d,
            label="Actual future rentals",
        )

        ax.plot(
            future_index_7d,
            median_7d,
            label="Timestral — target only",
        )

        ax.plot(
            future_index_7d,
            median_calendar_7d,
            label="Timestral — calendar categories",
        )

        ax.plot(
            future_index_7d,
            weekly_baseline_7d_new,
            label="Weekly seasonal naive",
        )

        ax.fill_between(
            future_index_7d,
            lower_calendar_7d,
            upper_calendar_7d,
            alpha=0.2,
            label="Calendar forecast 80% interval",
        )

        ax.axvline(
            future_index_7d[0],
            linestyle="--",
            label="Forecast origin",
        )

        ax.set(
            title="Bike Sharing: calendar-aware seven-day forecast",
            xlabel="Date",
            ylabel="Bikes rented per hour",
        )

        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Next experiment: holiday and working-day indicators
    The UCI dataset includes two useful features that we haven't supplied yet:
    - holiday: whether the date is a holiday.
    - workingday: whether the date is neither a weekend nor a holiday.
      UCI Machine Learning Repository
    These are particularly relevant to our December test period. Unlike future weather observations, the holiday and working-day flags can be known before the forecast starts.

    Timestral also supports known-future numeric covariates. Binary indicator rows containing only 0s and 1s receive special treatment in the implementation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Construct the indicator features
    """)
    return


@app.cell
def _(df, df_hourly):
    # Extract one holiday and workingday value per calendar date.
    daily_calendar = (
        df.assign(calendar_date=df["datetime"].dt.normalize())
        .groupby("calendar_date")[["holiday", "workingday"]]
        .first()
    )

    # Map the date-level flags onto every hourly position.
    calendar_by_hour = daily_calendar.reindex(
        df_hourly.index.normalize()
    ).astype(float)

    # Two numeric indicator rows: holiday and workingday.
    binary_calendar = (
        calendar_by_hour
        .to_numpy(dtype=np.float32)
        .T[None, :, :]
    )

    print("Indicator shape:", binary_calendar.shape)
    print("Missing indicator values:", np.isnan(binary_calendar).sum())

    assert binary_calendar.shape == (1, 2, len(df_hourly))
    return (binary_calendar,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Forecast with the new indicators
    """)
    return


@app.cell
def _(
    HORIZON_7D,
    binary_calendar,
    calendar_categories,
    context_7d,
    forecaster,
):
    # Forecast using holiday and workingday only.
    forecast_flags_7d = forecaster.forecast(
        context_7d,
        prediction_length=HORIZON_7D,
        known_covariates=binary_calendar,
    )

    # Forecast using hour + weekday categories AND the indicators.
    forecast_calendar_flags_7d = forecaster.forecast(
        context_7d,
        prediction_length=HORIZON_7D,
        categorical_known_covariates=calendar_categories,
        known_covariates=binary_calendar,
    )

    print("Holiday/workingday forecast shape:",
          tuple(forecast_flags_7d.median.shape))

    print("Combined forecast shape:",
          tuple(forecast_calendar_flags_7d.median.shape))
    return forecast_calendar_flags_7d, forecast_flags_7d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare all approaches
    """)
    return


@app.cell
def _(
    actual_7d,
    baseline_7d,
    context_7d,
    forecast_7d,
    forecast_calendar_7d,
    forecast_calendar_flags_7d,
    forecast_flags_7d,
    weekly_baseline_7d_new,
):
    def score_timestral_forecast(name, forecast):
        quantiles = (
            forecast.quantiles[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        levels = np.asarray(forecast.quantile_levels)

        q10 = np.where(np.isclose(levels, 0.1))[0][0]
        q90 = np.where(np.isclose(levels, 0.9))[0][0]

        median = (
            forecast.median[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        return score_bike_forecast(
            name,
            actual_7d,
            median,
            context_7d,
            lower=quantiles[:, q10],
            upper=quantiles[:, q90],
            quantiles=quantiles,
            quantile_levels=levels,
        )


    comparison_flags = pd.DataFrame([
        score_timestral_forecast(
            "Timestral — target only",
            forecast_7d,
        ),
        score_timestral_forecast(
            "Timestral — hour + weekday",
            forecast_calendar_7d,
        ),
        score_timestral_forecast(
            "Timestral — holiday + workingday",
            forecast_flags_7d,
        ),
        score_timestral_forecast(
            "Timestral — calendar + indicators",
            forecast_calendar_flags_7d,
        ),
        score_bike_forecast(
            "Daily seasonal naive",
            actual_7d,
            baseline_7d,
            context_7d,
        ),
        score_bike_forecast(
            "Weekly seasonal naive",
            actual_7d,
            weekly_baseline_7d_new,
            context_7d,
        ),
    ]).set_index("Approach")

    daily_naive_mae = comparison_flags.loc[
        "Daily seasonal naive", "MAE"
    ]

    comparison_flags["MAE improvement vs daily naive (%)"] = (
        100
        * (daily_naive_mae - comparison_flags["MAE"])
        / daily_naive_mae
    )

    comparison_flags.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Rolling-origin backtesting
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We'll evaluate four approaches:
    1. Timestral using target history only.
    2. Timestral using calendar categories plus holiday and working-day indicators.
    3. Daily seasonal naive.
    4. Weekly seasonal naive.
    We'll test 20 seven-day windows spaced 21 days apart, covering approximately 13 months from November 2011 through December 2012. Each window starts on the same weekday and at midnight, and the forecast horizon is always 168 hours. This gives us a more varied assessment of seasonality and holiday effects.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Define the backtesting function
    Create one cell containing this function. Variables such as context, actual, forecast, and origin are now local to the function, so they won't conflict with variables in other notebook cells.
    """)
    return


@app.function
def run_bike_rolling_backtest_v2(
    forecaster_arg,
    target_values_arg,
    hourly_frame_arg,
    calendar_categories_arg,
    binary_calendar_arg,
    score_function_arg,
    n_origins_arg=20,
    horizon_hours_arg=168,
    spacing_days_arg=21,
):
    """Run rolling-origin backtesting and return three result tables."""

    timestamps_local = hourly_frame_arg.index
    target_values_local = np.asarray(
        target_values_arg,
        dtype=np.float32,
    )

    if len(target_values_local) != len(timestamps_local):
        raise ValueError("Target and timestamp lengths do not match.")

    if calendar_categories_arg.shape[-1] != len(timestamps_local):
        raise ValueError("Calendar categories do not match the timeline.")

    if binary_calendar_arg.shape[-1] != len(timestamps_local):
        raise ValueError("Calendar indicators do not match the timeline.")

    if n_origins_arg < 1 or spacing_days_arg < 1:
        raise ValueError("Number of origins and spacing must be positive.")

    if horizon_hours_arg < 1:
        raise ValueError("Forecast horizon must be positive.")

    # Generate regularly spaced forecast origins.
    last_origin_local = timestamps_local[-horizon_hours_arg]

    origins_local = [
        last_origin_local
        - pd.Timedelta(
            days=spacing_days_arg * (n_origins_arg - 1 - i)
        )
        for i in range(n_origins_arg)
    ]

    missing_origins_local = [
    origin for origin in origins_local
    if origin not in timestamps_local
]

    if missing_origins_local:
        raise ValueError(
            f"Forecast origins not found in the index: "
            f"{missing_origins_local}"
    )

    backtest_rows_local = []

    def find_complete_pattern_local(
        context_values,
        period,
        step,
    ):
        """Find a complete historical seasonal pattern."""
        for end_idx in range(
            len(context_values),
            period - 1,
            -step,
        ):
            candidate = context_values[end_idx - period:end_idx]

            if (
                len(candidate) == period
                and np.isfinite(candidate).all()
            ):
                return candidate.copy()

        raise ValueError(
            f"No complete historical pattern found for period={period}."
        )

    def add_timestral_result_local(
        approach_name,
        forecast_result,
        context_values,
        actual_values,
        origin_value,
    ):
        """Score one Timestral forecast and append its results."""
        quantiles_local = (
            forecast_result.quantiles[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        levels_local = np.asarray(
            forecast_result.quantile_levels
        )

        median_local = (
            forecast_result.median[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        q10_idx_local = np.where(
            np.isclose(levels_local, 0.1)
        )[0][0]

        q90_idx_local = np.where(
            np.isclose(levels_local, 0.9)
        )[0][0]

        result_row_local = score_function_arg(
            approach_name,
            actual_values,
            median_local,
            context_values,
            lower=quantiles_local[:, q10_idx_local],
            upper=quantiles_local[:, q90_idx_local],
            quantiles=quantiles_local,
            quantile_levels=levels_local,
        )

        result_row_local["origin"] = origin_value
        result_row_local["valid_hours"] = int(
            np.isfinite(actual_values).sum()
        )

        backtest_rows_local.append(result_row_local)

    # Process every forecast window.
    for window_number_local, origin_local in enumerate(
        origins_local,
        start=1,
    ):
        start_idx_local = int(
            timestamps_local.get_loc(origin_local)
        )
        end_idx_local = start_idx_local + horizon_hours_arg

        context_local = target_values_local[:start_idx_local]
        actual_local = target_values_local[
            start_idx_local:end_idx_local
        ]

        if len(actual_local) != horizon_hours_arg:
            raise ValueError(
                f"Incomplete forecast window at {origin_local}."
            )

        valid_count_local = int(
            np.isfinite(actual_local).sum()
        )

        print(
            f"Window {window_number_local}/{n_origins_arg}: "
            f"{origin_local} "
            f"({valid_count_local} valid observations)"
        )

        # The covariates must end exactly at this window's forecast end.
        categories_local = calendar_categories_arg[
            :, :, :end_idx_local
        ]

        indicators_local = binary_calendar_arg[
            :, :, :end_idx_local
        ]

        if not np.isfinite(categories_local).all():
            raise ValueError(
                f"Missing categorical known covariates at {origin_local}."
            )

        if not np.isfinite(indicators_local).all():
            raise ValueError(
                f"Missing numeric known covariates at {origin_local}."
            )

        # 1. Target-only Timestral forecast.
        target_forecast_local = forecaster_arg.forecast(
            context_local,
            prediction_length=horizon_hours_arg,
        )

        add_timestral_result_local(
            "Timestral — target only",
            target_forecast_local,
            context_local,
            actual_local,
            origin_local,
        )

        # 2. Timestral with known-future calendar features.
        calendar_forecast_local = forecaster_arg.forecast(
            context_local,
            prediction_length=horizon_hours_arg,
            categorical_known_covariates=categories_local,
            known_covariates=indicators_local,
        )

        add_timestral_result_local(
            "Timestral — calendar + indicators",
            calendar_forecast_local,
            context_local,
            actual_local,
            origin_local,
        )

        # 3. Daily seasonal-naive baseline.
        daily_pattern_local = find_complete_pattern_local(
            context_local,
            period=24,
            step=24,
        )

        daily_prediction_local = np.tile(
            daily_pattern_local,
            int(np.ceil(horizon_hours_arg / 24)),
        )[:horizon_hours_arg]

        daily_row_local = score_function_arg(
            "Daily seasonal naive",
            actual_local,
            daily_prediction_local,
            context_local,
        )

        daily_row_local["origin"] = origin_local
        daily_row_local["valid_hours"] = valid_count_local
        backtest_rows_local.append(daily_row_local)

        # 4. Weekly seasonal-naive baseline.
        weekly_pattern_local = find_complete_pattern_local(
            context_local,
            period=168,
            step=168,
        )

        weekly_prediction_local = np.tile(
            weekly_pattern_local,
            int(np.ceil(horizon_hours_arg / 168)),
        )[:horizon_hours_arg]

        weekly_row_local = score_function_arg(
            "Weekly seasonal naive",
            actual_local,
            weekly_prediction_local,
            context_local,
        )

        weekly_row_local["origin"] = origin_local
        weekly_row_local["valid_hours"] = valid_count_local
        backtest_rows_local.append(weekly_row_local)

    # Assemble detailed results for all approaches and origins.
    details_local = pd.DataFrame(backtest_rows_local)

    # Summarise performance across forecast windows.
    summary_local = (
        details_local
        .groupby("Approach")
        .agg(
            forecast_windows=("origin", "nunique"),
            total_scored_hours=("valid_hours", "sum"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            mean_RMSE=("RMSE", "mean"),
            mean_sMAPE=("sMAPE (%)", "mean"),
            mean_MASE=("MASE (daily period)", "mean"),
            mean_80_coverage=("80% coverage (%)", "mean"),
            mean_interval_width=("80% interval width", "mean"),
            mean_pinball_loss=("Mean pinball loss", "mean"),
        )
    )

    # Compare each approach against the daily seasonal-naive baseline
    # at the same forecast origins.
    mae_by_origin_local = details_local.pivot(
        index="origin",
        columns="Approach",
        values="MAE",
    )

    daily_mae_local = mae_by_origin_local[
        "Daily seasonal naive"
    ]

    win_rows_local = []

    for approach_local in mae_by_origin_local.columns:
        model_mae_local = mae_by_origin_local[approach_local]

        paired_local = pd.concat(
            [
                daily_mae_local.rename("baseline_mae"),
                model_mae_local.rename("model_mae"),
            ],
            axis=1,
        ).dropna()

        improvement_local = (
            100
            * (
                paired_local["baseline_mae"]
                - paired_local["model_mae"]
            )
            / paired_local["baseline_mae"]
        )

        win_rows_local.append({
            "Approach": approach_local,
            "Mean MAE improvement vs daily naive (%)": (
                improvement_local.mean()
            ),
            "Windows beating daily naive": int(
                (
                    paired_local["model_mae"]
                    < paired_local["baseline_mae"]
                ).sum()
            ),
            "Compared windows": len(paired_local),
        })

    win_summary_local = pd.DataFrame(win_rows_local)

    return details_local, summary_local, win_summary_local


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Call the function and unpack its results
    """)
    return


@app.cell
def _(binary_calendar, calendar_categories, df_hourly, forecaster, target):
    (
        bike_backtest_details_v2,
        bike_backtest_summary_v2,
        bike_backtest_win_summary_v2,
    ) = run_bike_rolling_backtest_v2(
        forecaster_arg=forecaster,
        target_values_arg=target,
        hourly_frame_arg=df_hourly,
        calendar_categories_arg=calendar_categories,
        binary_calendar_arg=binary_calendar,
        score_function_arg=score_bike_forecast,
        n_origins_arg=20,
        horizon_hours_arg=168,
        spacing_days_arg=21,
    )
    return (
        bike_backtest_details_v2,
        bike_backtest_summary_v2,
        bike_backtest_win_summary_v2,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the summary tables
    """)
    return


@app.cell
def _(bike_backtest_summary_v2):
    bike_backtest_summary_v2.round(4)
    return


@app.cell
def _(bike_backtest_win_summary_v2):
    bike_backtest_win_summary_v2.round(3)
    return


@app.cell
def _(bike_backtest_details_v2):
    bike_backtest_details_v2.round(4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
