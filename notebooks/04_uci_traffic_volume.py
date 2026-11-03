import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from ucimlrepo import fetch_ucirepo

    from statsforecast import StatsForecast
    from statsforecast.models import (
        Naive,
        HistoricAverage,
        WindowAverage,
        SeasonalNaive,
        SeasonalWindowAverage,
    )
    from statsmodels.tsa.seasonal import seasonal_decompose

    from utilsforecast.plotting import plot_series
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import (
        mae,
        rmse,
        mape,
        smape,
        mase,
        wape,
        quantile_loss,
        mqloss,
        coverage,
        winkler_score,
    )

    from timestral import TimestralForecaster


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load the Dataset
    """)
    return


@app.cell
def _():
    traffic_dataset_raw = fetch_ucirepo(id=492)

    traffic_features_raw = traffic_dataset_raw.data.features
    traffic_targets_raw = traffic_dataset_raw.data.targets

    traffic_df_raw = pd.concat(objs=[
        traffic_features_raw.reset_index(drop=True),
        traffic_targets_raw.reset_index(drop=True),
        ],
        axis=1
    )

    traffic_df_raw = (
        traffic_df_raw
        .sort_values("date_time")
        .reset_index(drop=True)
    )

    traffic_df_raw.head()
    return (traffic_df_raw,)


@app.cell
def _(traffic_df_raw):
    # Save dataset to csv

    traffic_df_raw.to_csv("traffic_df_raw.csv", sep=",", index=False)
    return


@app.cell
def _(traffic_df_raw):
    # convert "date_time" to datetime format

    traffic_df_raw["date_time"] = pd.to_datetime(traffic_df_raw["date_time"], format="%Y-%m-%d %H:%M:%S")
    return


@app.cell
def _(traffic_df_raw):
    # Reorder columns
    cols = traffic_df_raw.columns[-2]

    traffic_df_raw.insert(0, cols, traffic_df_raw.pop(cols))

    traffic_df_raw.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Inspect the timestamps and data quality
    """)
    return


@app.cell
def _(traffic_df_raw):
    print(f"Rows: {len(traffic_df_raw)}")
    print(f"\nColumns: {traffic_df_raw.columns.tolist()}")

    print(f"\nStart: {traffic_df_raw['date_time'].min()}")
    print(f"\nEnd: {traffic_df_raw['date_time'].max()}")

    print(f"\nMissing cells per column:")
    print(f"{traffic_df_raw.isna().sum()}")

    print(f"\nTotal missing in dataset: {traffic_df_raw.isna().sum().sum():,}")

    print(f"\nDuplicate timestamps: {traffic_df_raw["date_time"].duplicated().sum():,}")

    print("\nObserved time-step frequencies:")
    print(f"{traffic_df_raw['date_time'].diff().value_counts().head(12)}")

    print("\nTraffic volume summary:")
    print(f"{traffic_df_raw['traffic_volume'].describe()}")

    print(f"\nZero traffic-volume observations: {(traffic_df_raw['traffic_volume'] == 0).sum()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize traffic volume
    """)
    return


@app.cell
def _(traffic_df_raw):
    traffic_fig, traffic_ax = plt.subplots(figsize=(13, 5))

    traffic_ax.plot(
        traffic_df_raw['date_time'],
        traffic_df_raw['traffic_volume'],
        linewidth=0.6,
    )

    traffic_ax.set(
        title="Metro Interstate Traffic Volume, 2012-2018",
        xlabel="Date",
        ylabel="Vehicles per hour",
    )

    traffic_fig.tight_layout()
    traffic_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Data Audit
    """)
    return


@app.function
def audit_traffic_irregularities_v1(frame_arg):
    """
    Inspect timestamp duplicates, temporal gaps and
    holiday labels
    """
    work_local = frame_arg.copy()

    work_local = work_local.sort_values(
        "date_time"
    ).reset_index(drop=True)

    # 1. Group records by timestamp and count distinct values
    duplicate_audit_local = (
        work_local
        .groupby("date_time", sort=True)
        .agg(
            row_count=("traffic_volume", "size"),  # Fixed: passed as string "size"
            traffic_unique=(
                "traffic_volume",
                lambda s: s.nunique(dropna=False),
            ),
            temp_unique=(
                "temp",
                lambda s: s.nunique(dropna=False),
            ),
            rain_unique=(
                "rain_1h",
                lambda s: s.nunique(dropna=False),
            ),
            snow_unique=(
                "snow_1h",
                lambda s: s.nunique(dropna=False),
            ),
            clouds_unique=(
                "clouds_all",
                lambda s: s.nunique(dropna=False),
            ),
            holiday_unique=(
                "holiday",
                lambda s: s.nunique(dropna=False),
            ),
            weather_main_unique=(
                "weather_main",
                lambda s: s.nunique(dropna=False),
            ),
            weather_description_unique=(
                "weather_description",
                lambda s: s.nunique(dropna=False),
            ),
        )
        .reset_index()
    )

    duplicate_groups_local = duplicate_audit_local.loc[
        duplicate_audit_local["row_count"] > 1
    ].copy()

    duplicate_summary_local = pd.DataFrame({
        "Check": [
            "Original records",
            "Unique timestamps",
            "Timestamp groups with duplicates",
            "Excess records due to duplicate timestamps",
            "Exact duplicate rows",
            "Duplicate groups with conflicting traffic volume",
            "Duplicate groups with conflicting temperature",
            "Duplicate groups with conflicting weather_main",
            "Duplicate groups with conflicting weather_description",
        ],
        "Value": [
            len(work_local),
            work_local["date_time"].nunique(),
            len(duplicate_groups_local),
            int((duplicate_groups_local["row_count"] - 1).sum()),  # Fixed: removed len() on scalar
            int(work_local.duplicated().sum()),                    # Fixed: removed len() on scalar
            int((duplicate_groups_local["traffic_unique"] > 1).sum()),
            int((duplicate_groups_local["temp_unique"] > 1).sum()),
            int((duplicate_groups_local["weather_main_unique"] > 1).sum()),
            int((duplicate_groups_local["weather_description_unique"] > 1).sum()),
        ],
    })

    # Select five timestamp groups with the most repeated records
    sample_times_local = (
        duplicate_groups_local
        .nlargest(5, "row_count")["date_time"]
    )

    duplicate_examples_local = (
        work_local[
            work_local["date_time"].isin(sample_times_local)
        ]
        .sort_values("date_time")
        [[
            "date_time",
            "traffic_volume",
            "temp",
            "rain_1h",
            "snow_1h",
            "clouds_all",
            "holiday",
            "weather_main",
            "weather_description",
        ]]
    )

    # 2. Inspect gaps between unique timestamps
    unique_times_local = pd.DatetimeIndex(
        work_local["date_time"].dropna().drop_duplicates().sort_values()
    )

    gaps_local = pd.DataFrame({
        "timestamp": unique_times_local,
        "gap_from_previous": unique_times_local.to_series()
        .diff()
        .to_numpy(),
    })

    gaps_local = gaps_local.loc[
        gaps_local["gap_from_previous"] > pd.Timedelta(hours=1)
    ].copy()

    gaps_local["missing_hour_slots"] = (
        gaps_local["gap_from_previous"].dt.total_seconds() / 3600 - 1  # Fixed: corrected column name typo
    )

    gaps_local = gaps_local.sort_values(
        "gap_from_previous",
        ascending=False,
    )

    # 3. Inspect holiday labels by calendar date
    work_local["calendar_date"] = (
        work_local["date_time"].dt.normalize()
    )

    holiday_days_local = (
        work_local
        .groupby("calendar_date")
        .agg(
            total_rows=("holiday", "size"),
            labelled_rows=("holiday", "count"),
            holiday_names=(
                "holiday",
                lambda s: ", ".join(
                    sorted(set(s.dropna().astype(str)))
                ),
            ),
        )
        .reset_index()
    )

    holiday_days_local = holiday_days_local.loc[
        holiday_days_local["labelled_rows"] > 0  # Fixed: corrected spelling of "labelled_rows"
    ]

    # 4. Inspect potentially anomalous measurements
    anomalies_local = pd.DataFrame({
        "Check": [
            "Temperature equal to 0 K",
            "Traffic volume equal to zero",
            "Rainfall greater than or equal to 9000 mm",
        ],
        "Count": [
            int((work_local["temp"] == 0).sum()),
            int((work_local["traffic_volume"] == 0).sum()),
            int((work_local["rain_1h"] >= 9000).sum()),
        ],
    })

    return (
        duplicate_summary_local,
        duplicate_groups_local,
        duplicate_examples_local,
        gaps_local,
        holiday_days_local,
        anomalies_local,
    )


@app.cell
def _(traffic_df_raw):
    (
        traffic_duplicate_summary_v1,
        traffic_duplicate_groups_v1,
        traffic_duplicate_examples_v1,
        traffic_time_gaps_v1,
        traffic_holiday_days_v1,
        traffic_anomalies_v1,
    ) = audit_traffic_irregularities_v1(traffic_df_raw)
    return (
        traffic_anomalies_v1,
        traffic_duplicate_examples_v1,
        traffic_duplicate_summary_v1,
        traffic_holiday_days_v1,
        traffic_time_gaps_v1,
    )


@app.cell
def _(traffic_duplicate_summary_v1):
    traffic_duplicate_summary_v1
    return


@app.cell
def _(traffic_duplicate_examples_v1):
    traffic_duplicate_examples_v1
    return


@app.cell
def _(traffic_time_gaps_v1):
    traffic_time_gaps_v1.head(20)
    return


@app.cell
def _(traffic_holiday_days_v1):
    traffic_holiday_days_v1.head(30)
    return


@app.cell
def _(traffic_anomalies_v1):
    traffic_anomalies_v1
    return


@app.function
def inspect_traffic_cleaning_v2(
    frame_arg,
    segment_start_arg="2015-06-11 20:00:00",
):
    """Audit duplicate conflicts, anomalies and post-gap target coverage."""

    work_local = frame_arg.copy()

    # Explicitly convert timestamps to pandas datetime.
    work_local["date_time"] = pd.to_datetime(
        work_local["date_time"],
        errors="coerce",
    )
    
    # Validate timestamp conversion.
    if work_local["date_time"].isna().any():
        raise ValueError(
            "Invalid or missing timestamps found in date_time."
        )
    
    work_local = work_local.sort_values(
        "date_time"
    ).reset_index(drop=True)

    grouped_local = work_local.groupby(
        "date_time",
        sort=True,
    )

    duplicate_sizes_local = grouped_local.size()
    duplicate_times_local = duplicate_sizes_local[
        duplicate_sizes_local > 1
    ].index

    # Check every column for conflicting values within duplicate groups.
    columns_local = [
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "holiday",
        "weather_main",
        "weather_description",
    ]

    conflict_rows_local = []

    for column_local in columns_local:
        distinct_local = grouped_local[column_local].nunique(
            dropna=False
        )

        conflict_count_local = int(
            distinct_local.loc[duplicate_times_local].gt(1).sum()
        )

        conflict_rows_local.append({
            "Column": column_local,
            "Conflicting timestamp groups": conflict_count_local,
        })

    conflicts_local = pd.DataFrame(conflict_rows_local)

    # Inspect duplicate timestamps with conflicting numeric weather data.
    numeric_weather_local = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    numeric_uniques_local = (
        grouped_local[numeric_weather_local]
        .nunique(dropna=False)
        .loc[duplicate_times_local]
    )

    numeric_conflict_times_local = (
        numeric_uniques_local.index[
            numeric_uniques_local.gt(1).any(axis=1)
        ]
    )

    numeric_examples_local = work_local[
        work_local["date_time"].isin(
            numeric_conflict_times_local[:10]
        )
    ][[
        "date_time",
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
    ]]

    # Inspect suspicious observations without deleting anything.
    anomaly_mask_local = (
        (work_local["temp"] == 0)
        | (work_local["traffic_volume"] == 0)
        | (work_local["rain_1h"] >= 9000)
    )

    anomalies_local = work_local.loc[
        anomaly_mask_local,
        [
            "date_time",
            "traffic_volume",
            "temp",
            "rain_1h",
            "snow_1h",
            "clouds_all",
        ],
    ]

    # Inspect how holiday values are represented by the loader.
    holiday_counts_local = (
        work_local["holiday"]
        .value_counts(dropna=False)
        .rename_axis("Holiday value")
        .reset_index(name="Count")
    )

    # Target values have already been verified as consistent
    # within duplicate timestamp groups.
    target_by_hour_local = (
        grouped_local["traffic_volume"]
        .first()
        .sort_index()
    )

    if len(duplicate_times_local):
        conflicting_target_local = (
            grouped_local["traffic_volume"]
            .nunique(dropna=False)
            .loc[duplicate_times_local]
            .gt(1)
            .sum()
        )

        if conflicting_target_local:
            raise ValueError(
                "Conflicting traffic volumes found; "
                "cannot safely deduplicate target."
            )

    # Select the period following the major historical gap.
    segment_local = target_by_hour_local.loc[
        pd.Timestamp(segment_start_arg):
    ]

    hourly_index_local = pd.date_range(
        start=segment_local.index.min(),
        end=segment_local.index.max(),
        freq="h",
    )

    regular_target_local = segment_local.reindex(
        hourly_index_local
    )

    missing_mask_local = regular_target_local.isna().to_numpy()

    missing_runs_local = (
        pd.Series(missing_mask_local)
        .groupby(
            pd.Series(missing_mask_local).ne(
                pd.Series(missing_mask_local).shift()
            ).cumsum()
        )
        .sum()
    )

    longest_missing_run_local = (
        int(missing_runs_local.max())
        if missing_mask_local.any()
        else 0
    )

    segment_summary_local = pd.DataFrame({
        "Metric": [
            "Segment start",
            "Segment end",
            "Expected hourly positions",
            "Observed traffic observations",
            "Missing hourly positions",
            "Missing percentage",
            "Longest consecutive missing run (hours)",
            "Timestral context limit",
        ],
        "Value": [
            regular_target_local.index.min(),
            regular_target_local.index.max(),
            len(regular_target_local),
            int(regular_target_local.notna().sum()),
            int(regular_target_local.isna().sum()),
            regular_target_local.isna().mean() * 100,
            longest_missing_run_local,
            32768,
        ],
    })

    return (
        conflicts_local,
        numeric_examples_local,
        anomalies_local,
        holiday_counts_local,
        regular_target_local,
        segment_summary_local,
    )


@app.cell
def _(traffic_target_segment_v2):
    print(type(traffic_target_segment_v2))
    print(traffic_target_segment_v2.dtype)
    print(type(traffic_target_segment_v2.index))
    print(traffic_target_segment_v2.index.dtype)
    return


@app.cell
def _(traffic_df_raw):


    (
        traffic_conflicts_v2,
        traffic_numeric_examples_v2,
        traffic_anomalies_v2,
        traffic_holiday_counts_v2,
        traffic_target_segment_v2,
        traffic_segment_summary_v2,
    ) = inspect_traffic_cleaning_v2(traffic_df_raw)
    return (
        traffic_anomalies_v2,
        traffic_conflicts_v2,
        traffic_holiday_counts_v2,
        traffic_numeric_examples_v2,
        traffic_segment_summary_v2,
        traffic_target_segment_v2,
    )


@app.cell
def _(traffic_conflicts_v2):
    traffic_conflicts_v2
    return


@app.cell
def _(traffic_numeric_examples_v2):
    traffic_numeric_examples_v2
    return


@app.cell
def _(traffic_anomalies_v2):
    traffic_anomalies_v2
    return


@app.cell
def _(traffic_holiday_counts_v2):
    traffic_holiday_counts_v2
    return


@app.cell
def _(traffic_segment_summary_v2):
    traffic_segment_summary_v2
    return


@app.cell
def _(traffic_target_segment_v2):
    pd.concat(objs=[
        traffic_target_segment_v2.head(5),
        traffic_target_segment_v2.tail(5),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### First traffic-only forecast
    """)
    return


@app.cell
def _():
    traffic_forecaster_v1 = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M",
        device="cpu",
    )
    return (traffic_forecaster_v1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Create the experiment function

    This function deliberately searches for the latest pair of consecutive complete 24-hour periods. The first day supplies our seasonal-naive baseline; the second day is the test period. This ensures both approaches are evaluated against the same 24 actual observations.
    """)
    return


@app.function
def run_traffic_target_only_v1(
    target_series_arg,
    forecaster_arg,
    horizon_arg=24,
):
    """Run a 24-hour Timestral forecast against a daily-naive baseline."""

    # 1. Validate the incoming time index.
    if not isinstance(target_series_arg.index, pd.DatetimeIndex):
        raise TypeError("Target must have a DatetimeIndex.")
    
    if target_series_arg.index.hasnans:
        raise ValueError("Target index contains NaT timestamps.")
    
    if not target_series_arg.index.is_unique:
        raise ValueError("Target contains duplicate timestamps.")
    
    # 2. Sort the original observations.
    series_local = target_series_arg.sort_index()
    
    # 3. Construct a complete hourly timeline.
    hourly_index_local = pd.date_range(
        start=series_local.index.min(),
        end=series_local.index.max(),
        freq="h",
    )
    
    # 4. Ensure all original timestamps align to the hourly grid.
    off_grid_local = series_local.index.difference(
        hourly_index_local
    )
    
    if len(off_grid_local) > 0:
        raise ValueError(
            f"Found timestamps outside the hourly grid: "
            f"{off_grid_local[:5].tolist()}"
        )
    
    # 5. Reindex to the complete hourly timeline.
    # Existing NaNs remain NaN; missing timestamps become NaN.
    series_local = series_local.reindex(hourly_index_local)
    
    values_local = series_local.to_numpy(dtype=np.float32)
    
    print("Hourly positions:", len(values_local))
    print("Missing target observations:", np.isnan(values_local).sum())

    # Find the latest 48 consecutive observations with no missing values.
    selected_end_local = None

    for end_idx_local in range(
        len(values_local),
        2 * horizon_arg - 1,
        -1,
    ):
        candidate_local = values_local[
            end_idx_local - 2 * horizon_arg:end_idx_local
        ]

        if np.isfinite(candidate_local).all():
            selected_end_local = end_idx_local
            break

    if selected_end_local is None:
        raise ValueError(
            "No complete consecutive baseline/test window found."
        )

    test_start_idx_local = selected_end_local - horizon_arg

    context_local = values_local[:test_start_idx_local]

    actual_local = values_local[
        test_start_idx_local:selected_end_local
    ]

    baseline_local = context_local[-horizon_arg:]

    future_index_local = series_local.index[
        test_start_idx_local:selected_end_local
    ]

    if len(context_local) > 32768:
        raise ValueError("Historical context exceeds Timestral's limit.")

    # Timestral inference.
    forecast_local = forecaster_arg.forecast(
        context_local,
        prediction_length=horizon_arg,
    )

    median_local = (
        forecast_local.median[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    quantiles_local = (
        forecast_local.quantiles[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    levels_local = np.asarray(
        forecast_local.quantile_levels
    )

    q10_local = np.where(
        np.isclose(levels_local, 0.1)
    )[0][0]

    q90_local = np.where(
        np.isclose(levels_local, 0.9)
    )[0][0]

    lower_local = quantiles_local[:, q10_local]
    upper_local = quantiles_local[:, q90_local]

    # Point-forecast metrics.
    errors_local = actual_local - median_local
    baseline_errors_local = actual_local - baseline_local

    mae_local = np.mean(np.abs(errors_local))
    rmse_local = np.sqrt(np.mean(errors_local**2))

    baseline_mae_local = np.mean(
        np.abs(baseline_errors_local)
    )

    baseline_rmse_local = np.sqrt(
        np.mean(baseline_errors_local**2)
    )

    # MAPE is not reliable if any actual value is zero.
    mape_local = (
        np.mean(
            np.abs(errors_local) / np.abs(actual_local)
        ) * 100
        if np.all(np.abs(actual_local) > 1e-8)
        else np.nan
    )

    denominator_local = (
        np.abs(actual_local) + np.abs(median_local)
    )

    smape_local = np.mean(
        np.divide(
            200 * np.abs(errors_local),
            denominator_local,
            out=np.zeros_like(errors_local),
            where=denominator_local > 0,
        )
    )

    # Historical seasonal scaling.
    historical_diffs_local = np.abs(
        context_local[horizon_arg:]
        - context_local[:-horizon_arg]
    )

    historical_diffs_local = historical_diffs_local[
        np.isfinite(historical_diffs_local)
    ]

    seasonal_scale_local = historical_diffs_local.mean()

    mase_local = (
        mae_local / seasonal_scale_local
        if seasonal_scale_local > 0
        else np.nan
    )

    # Probabilistic metrics.
    coverage_local = np.mean(
        (actual_local >= lower_local)
        & (actual_local <= upper_local)
    ) * 100

    interval_width_local = np.mean(
        upper_local - lower_local
    )

    quantile_errors_local = (
        actual_local[:, None] - quantiles_local
    )

    pinball_local = np.maximum(
        levels_local[None, :] * quantile_errors_local,
        (levels_local[None, :] - 1) * quantile_errors_local,
    ).mean()

    metrics_local = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "MAPE (%)",
            "sMAPE (%)",
            "MASE (daily period)",
            "80% coverage (%)",
            "80% interval width",
            "Mean pinball loss",
        ],
        "Timestral": [
            mae_local,
            rmse_local,
            mape_local,
            smape_local,
            mase_local,
            coverage_local,
            interval_width_local,
            pinball_local,
        ],
        "Daily seasonal naive": [
            baseline_mae_local,
            baseline_rmse_local,
            np.mean(
                np.abs(baseline_errors_local)
                / np.abs(actual_local)
            ) * 100
            if np.all(np.abs(actual_local) > 1e-8)
            else np.nan,
            np.mean(
                np.divide(
                    200 * np.abs(baseline_errors_local),
                    np.abs(actual_local) + np.abs(baseline_local),
                    out=np.zeros_like(actual_local),
                    where=(
                        np.abs(actual_local) + np.abs(baseline_local)
                    ) > 0,
                )
            ),
            baseline_mae_local / seasonal_scale_local
            if seasonal_scale_local > 0
            else np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    })

    predictions_local = pd.DataFrame(
        {
            "actual": actual_local,
            "timestral_median": median_local,
            "daily_naive": baseline_local,
            "lower_80": lower_local,
            "upper_80": upper_local,
        },
        index=future_index_local,
    )

    metadata_local = pd.DataFrame({
        "Metric": [
            "Forecast origin",
            "Test start",
            "Test end",
            "Historical context length",
            "Missing historical observations",
            "Historical missing percentage",
        ],
        "Value": [
            future_index_local[0],
            future_index_local[0],
            future_index_local[-1],
            len(context_local),
            int(np.isnan(context_local).sum()),
            float(np.isnan(context_local).mean() * 100),
        ],
    })

    return metrics_local, predictions_local, metadata_local


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Unpack into new variables
    """)
    return


@app.cell
def _(traffic_forecaster_v1, traffic_target_segment_v2):
    (
        traffic_metrics_target_v1,
        traffic_predictions_target_v1,
        traffic_metadata_target_v1,
    ) = run_traffic_target_only_v1(
        target_series_arg=traffic_target_segment_v2,
        forecaster_arg=traffic_forecaster_v1,
        horizon_arg=24,
    )
    return (
        traffic_metadata_target_v1,
        traffic_metrics_target_v1,
        traffic_predictions_target_v1,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Display results
    """)
    return


@app.cell
def _(traffic_metadata_target_v1):
    traffic_metadata_target_v1
    return


@app.cell
def _(traffic_metrics_target_v1):
    traffic_metrics_target_v1.round(4)
    return


@app.cell
def _(traffic_predictions_target_v1):
    fig, ax = plt.subplots(figsize=(12, 5))

    prediction_index_v1 = traffic_predictions_target_v1.index

    ax.plot(
        prediction_index_v1,
        traffic_predictions_target_v1["actual"],
        label="Actual traffic",
    )

    ax.plot(
        prediction_index_v1,
        traffic_predictions_target_v1["timestral_median"],
        label="Timestral median",
    )

    ax.plot(
        prediction_index_v1,
        traffic_predictions_target_v1["daily_naive"],
        label="Daily seasonal naive",
    )

    ax.fill_between(
        prediction_index_v1,
        traffic_predictions_target_v1["lower_80"],
        traffic_predictions_target_v1["upper_80"],
        alpha=0.2,
        label="80% predictive interval",
    )

    ax.set(
        title="Metro Interstate Traffic: 24-hour forecast",
        xlabel="Date and time",
        ylabel="Traffic volume",
    )

    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Weekly seasonal naive comparison (Sunday vs. previous Sunday)
    """)
    return


@app.function
def evaluate_traffic_weekly_baseline_v1(
    target_series_arg,
    predictions_arg,
):
    """Compare Timestral, daily naive and weekly naive forecasts."""

    series_local = target_series_arg.sort_index()
    actual_local = predictions_arg["actual"].to_numpy()

    origin_local = predictions_arg.index[0]
    origin_idx_local = series_local.index.get_loc(origin_local)

    values_local = series_local.to_numpy(dtype=float)

    horizon_local = len(actual_local)
    weekly_pattern_local = None
    source_idx_local = None

    # Find the latest complete same-weekday historical pattern.
    for weeks_back_local in range(
        1,
        origin_idx_local // 168 + 1,
    ):
        candidate_start_local = (
            origin_idx_local - 168 * weeks_back_local
        )

        candidate_local = values_local[
            candidate_start_local:
            candidate_start_local + horizon_local
        ]

        if (
            len(candidate_local) == horizon_local
            and np.isfinite(candidate_local).all()
        ):
            weekly_pattern_local = candidate_local.copy()
            source_idx_local = candidate_start_local
            break

    if weekly_pattern_local is None:
        raise ValueError(
            "No complete same-weekday historical pattern found."
        )

    def point_metrics_local(actual_arg, predicted_arg):
        errors_local = actual_arg - predicted_arg
        abs_errors_local = np.abs(errors_local)

        denominator_local = (
            np.abs(actual_arg) + np.abs(predicted_arg)
        )

        return {
            "MAE": abs_errors_local.mean(),
            "RMSE": np.sqrt(np.mean(errors_local**2)),
            "sMAPE (%)": np.mean(
                np.divide(
                    200 * abs_errors_local,
                    denominator_local,
                    out=np.zeros_like(abs_errors_local),
                    where=denominator_local > 0,
                )
            ),
        }

    timestral_metrics_local = point_metrics_local(
        actual_local,
        predictions_arg["timestral_median"].to_numpy(),
    )

    daily_metrics_local = point_metrics_local(
        actual_local,
        predictions_arg["daily_naive"].to_numpy(),
    )

    weekly_metrics_local = point_metrics_local(
        actual_local,
        weekly_pattern_local,
    )

    comparison_local = pd.DataFrame([
        {"Approach": "Timestral", **timestral_metrics_local},
        {"Approach": "Daily seasonal naive", **daily_metrics_local},
        {"Approach": "Weekly seasonal naive", **weekly_metrics_local},
    ]).set_index("Approach")

    metadata_local = pd.DataFrame({
        "Metric": [
            "Forecast origin",
            "Weekly reference start",
            "Weekly reference end",
            "Weeks back used",
        ],
        "Value": [
            origin_local,
            series_local.index[source_idx_local],
            series_local.index[source_idx_local + horizon_local - 1],
            (origin_idx_local - source_idx_local) // 168,
        ],
    })

    return comparison_local, metadata_local


@app.cell
def _(traffic_predictions_target_v1, traffic_target_segment_v2):
    (
        traffic_weekly_comparison_v1,
        traffic_weekly_metadata_v1,
    ) = evaluate_traffic_weekly_baseline_v1(
        traffic_target_segment_v2,
        traffic_predictions_target_v1,
    )
    return traffic_weekly_comparison_v1, traffic_weekly_metadata_v1


@app.cell
def _(traffic_weekly_metadata_v1):
    traffic_weekly_metadata_v1
    return


@app.cell
def _(traffic_weekly_comparison_v1):
    traffic_weekly_comparison_v1.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Identify contiguous traffic runs
    """)
    return


@app.function
def inspect_contiguous_traffic_runs_v1(
    series_arg,
    horizon_arg=24,
):
    """Find the latest clean training suffix and longest historical run."""

    ordered_local = series_arg.sort_index()

    values_local = ordered_local.to_numpy(dtype=float)

    if len(values_local) <= horizon_arg:
        raise ValueError("Insufficient observations.")

    actual_local = values_local[-horizon_arg:]

    if not np.isfinite(actual_local).all():
        raise ValueError("Final test horizon contains missing values.")

    context_local = values_local[:-horizon_arg]

    # Find the start of the latest uninterrupted training suffix.
    missing_positions_local = np.flatnonzero(
        ~np.isfinite(context_local)
    )

    suffix_start_local = (
        int(missing_positions_local[-1] + 1)
        if len(missing_positions_local)
        else 0
    )

    clean_suffix_local = ordered_local.iloc[
        suffix_start_local:-horizon_arg
    ].copy()

    # Identify every complete historical run.
    valid_mask_local = np.isfinite(context_local)

    padded_local = np.r_[
        False,
        valid_mask_local,
        False,
    ].astype(np.int8)

    transitions_local = np.diff(padded_local)

    starts_local = np.flatnonzero(transitions_local == 1)
    ends_local = np.flatnonzero(transitions_local == -1)

    lengths_local = ends_local - starts_local

    if len(lengths_local) == 0:
        raise ValueError("No complete historical segment found.")

    longest_idx_local = int(np.argmax(lengths_local))

    longest_start_local = starts_local[longest_idx_local]
    longest_end_local = ends_local[longest_idx_local]

    longest_run_local = ordered_local.iloc[
        longest_start_local:longest_end_local
    ].copy()

    metadata_local = pd.DataFrame({
        "Metric": [
            "Forecast start",
            "Forecast end",
            "Latest clean training suffix start",
            "Latest clean training suffix end",
            "Latest clean training suffix length",
            "Longest historical run start",
            "Longest historical run end",
            "Longest historical run length",
            "Minimum history for 4 weekly seasonal cycles",
        ],
        "Value": [
            ordered_local.index[-horizon_arg],
            ordered_local.index[-1],
            clean_suffix_local.index[0] if len(clean_suffix_local) else None,
            clean_suffix_local.index[-1] if len(clean_suffix_local) else None,
            len(clean_suffix_local),
            longest_run_local.index[0],
            longest_run_local.index[-1],
            len(longest_run_local),
            4 * 168,
        ],
    })

    return metadata_local, clean_suffix_local, longest_run_local


@app.cell
def _(traffic_target_segment_v2):
    (
        traffic_run_metadata_v1,
        traffic_clean_suffix_v1,
        traffic_longest_run_v1,
    ) = inspect_contiguous_traffic_runs_v1(
        traffic_target_segment_v2,
        horizon_arg=24,
    )
    return traffic_clean_suffix_v1, traffic_run_metadata_v1


@app.cell
def _(traffic_run_metadata_v1):
    traffic_run_metadata_v1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Decompose the traffic series
    """)
    return


@app.function
def diagnose_traffic_spacing_v1(series_arg):
    """Inspect timestamp spacing in a traffic series."""

    series_local = series_arg.sort_index()
    index_local = series_local.index

    if not isinstance(index_local, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    differences_local = np.diff(index_local.asi8)

    expected_local = pd.Timedelta(hours=1)
    
    irregular_mask_local = (
        differences_local != expected_local
    )
    
    invalid_positions_local = np.flatnonzero(
        irregular_mask_local
    )
    
    # Count actual unique time differences.
    difference_counts_local = (
        pd.Series(differences_local)
        .value_counts()
        .rename_axis("Time difference")
        .reset_index(name="Count")
    )

    irregular_rows_local = []

    for position_local in invalid_positions_local[:20]:
        irregular_rows_local.append({
            "Previous timestamp": index_local[position_local],
            "Next timestamp": index_local[position_local + 1],
            "Observed difference": (
                index_local[position_local + 1]
                - index_local[position_local]
            ),
        })

    spacing_summary_local = pd.DataFrame({
        "Metric": [
            "Number of observations",
            "Index type",
            "Index frequency attribute",
            "Start timestamp",
            "End timestamp",
            "Number of irregular intervals",
            "Unique timestamp differences",
        ],
        "Value": [
            len(series_local),
            type(index_local).__name__,
            str(index_local.freq),
            index_local.min(),
            index_local.max(),
            len(invalid_positions_local),
            difference_counts_local["Time difference"].nunique(),
        ],
    })

    irregular_details_local = pd.DataFrame(
        irregular_rows_local
    )

    return spacing_summary_local, irregular_details_local


@app.cell
def _(traffic_clean_suffix_v1):
    (
        traffic_spacing_summary_v1,
        traffic_spacing_details_v1,
    ) = diagnose_traffic_spacing_v1(
        traffic_clean_suffix_v1
    )
    return traffic_spacing_details_v1, traffic_spacing_summary_v1


@app.cell
def _(traffic_spacing_summary_v1):
    traffic_spacing_summary_v1
    return


@app.cell
def _(traffic_spacing_details_v1):
    traffic_spacing_details_v1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Acutal decompositiion function
    """)
    return


@app.function
def decompose_traffic_series_v1(
    series_arg,
    period_arg,
    model_arg="additive",
):
    """Decompose a complete, regularly sampled hourly series."""

    series_local = series_arg.sort_index()

    if not isinstance(series_local.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    if len(series_local) < 2 * period_arg:
        raise ValueError("At least two complete cycles are required.")

    values_local = series_local.to_numpy(dtype=float)

    if not np.isfinite(values_local).all():
        raise ValueError(
            "Decomposition requires complete observations. "
            "Missing values were found."
        )

    hourly_gap_local = (
        series_local.index[1:] - series_local.index[:-1]
    )
    
    if not np.all(
        hourly_gap_local == pd.Timedelta(hours=1)
    ):
        raise ValueError("Series must have regular hourly spacing.")

    return seasonal_decompose(
        series_local,
        model=model_arg,
        period=period_arg,
        extrapolate_trend="period",
    )


@app.cell
def _(traffic_clean_suffix_v1):
    traffic_decomposition_daily_v1 = decompose_traffic_series_v1(
        traffic_clean_suffix_v1,
        period_arg=24,
    )

    traffic_decomposition_weekly_v1 = decompose_traffic_series_v1(
        traffic_clean_suffix_v1,
        period_arg=168,
    )
    return traffic_decomposition_daily_v1, traffic_decomposition_weekly_v1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Decomposition helper function
    """)
    return


@app.function
def plot_traffic_decomposition_v1(
    result_arg,
    title_arg,
):
    """Plot observed, trend, seasonal and residual components."""

    fig_local = result_arg.plot()

    fig_local.set_size_inches(14, 9)
    fig_local.suptitle(title_arg, fontsize=14, y=1.01)
    fig_local.tight_layout()

    return fig_local


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Daily Decomposition
    """)
    return


@app.cell
def _(traffic_decomposition_daily_v1):
    plot_traffic_decomposition_v1(
        traffic_decomposition_daily_v1,
        "Traffic volume — daily decomposition",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Weekly Decomposition
    """)
    return


@app.cell
def _(traffic_decomposition_weekly_v1):
    plot_traffic_decomposition_v1(
        traffic_decomposition_weekly_v1,
        "Traffic volume — weekly decomposition",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Build statistical forecasting baselines
    """)
    return


@app.function
def prepare_traffic_nixtla_data_v1(
    train_series_arg,
    full_series_arg,
    horizon_arg=24,
):
    """
    Prepare clean StatsForecast training and test DataFrames
    """

    train_local = train_series_arg.sort_index()
    full_local = full_series_arg.sort_index()

    if train_local.isna().any():
        raise ValueError("StatsForecast training data contains missing values.")

    if train_local.index.has_duplicates:
        raise ValueError("Duplicate training timestamps found.")

    train_gaps_local = (
        train_local.index[1:] - train_local.index[:-1]
    )

    if not np.all(
        train_gaps_local == pd.Timedelta(hours=1)
    ):
        raise ValueError("Series must have regular hourly spacing.")

    test_local = full_local.iloc[-horizon_arg:]

    if len(test_local) != horizon_arg:
        raise ValueError("Incorrect test horizon length")

    if not np.isfinite(test_local.to_numpy(dtype=float)).all():
        raise ValueError("Test observations contains missing values.")

    if (
        train_local.index[-1] + pd.Timedelta(hours=1)
        != test_local.index[0]
    ):
        raise ValueError(
            "Training and test periods are not contiguous."
        )

    train_df_local = pd.DataFrame({
        "unique_id": "I94",
        "ds": train_local.index,
        "y": train_local.to_numpy(dtype=float),
    })

    test_df_local = pd.DataFrame({
        "unique_id": "I94",
        "ds": test_local.index,
        "y": test_local.to_numpy(dtype=float),
    })

    metadata_local = pd.DataFrame({
        "Metric": [
            "Training start",
            "Training end",
            "Training observations",
            "Test start",
            "Test end",
            "Test observations",
        ],
        "Values": [
            train_local.index[0],
            train_local.index[-1],
            len(train_local),
            test_local.index[0],
            test_local.index[-1],
            len(test_local),
        ],
    })

    return train_df_local, test_df_local, metadata_local


@app.cell
def _(traffic_clean_suffix_v1, traffic_target_segment_v2):
    (
        traffic_sf_train_v1,
        traffic_sf_test_v1,
        traffic_sf_metadata_v1,
    ) = prepare_traffic_nixtla_data_v1(
        traffic_clean_suffix_v1,
        traffic_target_segment_v2,
        horizon_arg=24,
    )
    return traffic_sf_metadata_v1, traffic_sf_test_v1, traffic_sf_train_v1


@app.cell
def _(traffic_sf_metadata_v1):
    traffic_sf_metadata_v1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define the StatsForecast models
    """)
    return


@app.function
def run_traffic_statsforecast_v1(
    train_df_arg,
    horizon_arg=24,
):
    """
    Fit statistical baseline models and generate forecasts
    """

    models_local = [
        Naive(alias="Naive"),

        HistoricAverage(alias="HistoricAverage"),

        WindowAverage(
            window_size=24,
            alias="WindowAverage=24",
        ),

        SeasonalNaive(
            season_length=24,
            alias="SeasonalNaive-24",
        ),

        SeasonalNaive(
            season_length=164,
            alias="SeasonalNaive-168",
        ),

        SeasonalWindowAverage(
            season_length=24,
            window_size=7,
            alias="SeasonalWindowAverage-24-7",
        ),

        SeasonalWindowAverage(
            season_length=168,
            window_size=4,
            alias="SeasonalWindowAverage-168-4",
        ),
    ]

    statsforecast_local = StatsForecast(
        models=models_local,
        freq="h",
        n_jobs=1,
    )

    statsforecast_local.fit(df=train_df_arg)

    forecasts_local = statsforecast_local.predict(
        h=horizon_arg,
    )

    return forecasts_local


@app.cell
def _(traffic_sf_train_v1):
    traffic_statsforecast_forecasts_v1 = (
        run_traffic_statsforecast_v1(
            traffic_sf_train_v1,
            horizon_arg=24,
        )
    )
    return (traffic_statsforecast_forecasts_v1,)


@app.cell
def _(traffic_statsforecast_forecasts_v1):
    traffic_statsforecast_forecasts_v1.round(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generate a fair Timestral forecast
    """)
    return


@app.function
def run_traffic_timestral_clean_v1(
    train_df_arg,
    test_df_arg,
    forecaster_arg,
):
    """
    Generate a Timestral forecast using the clean training suffix
    """

    context_local = train_df_arg["y"].to_numpy(
        dtype=np.float32
    )

    horizon_local = len(test_df_arg)

    forecast_local = forecaster_arg.forecast(
        context_local,
        prediction_length=horizon_local,
    )

    median_local = (
        forecast_local.median[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    lower_local = (
        forecast_local.quantile(0.1)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    upper_local = (
        forecast_local.quantile(0.9)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    predictions_local = pd.DataFrame({
        "unique_id": test_df_arg["unique_id"].to_numpy(),
        "ds": test_df_arg["ds"].to_numpy(),
        "Timestral-clean": median_local,
        "Timestral-clean-lo-80": lower_local,
        "Timestral-clean-hi-80": upper_local,
    })

    return predictions_local


@app.cell
def _(traffic_forecaster_v1, traffic_sf_test_v1, traffic_sf_train_v1):
    traffic_timestral_clean_forecast_v1 = (
        run_traffic_timestral_clean_v1(
            traffic_sf_train_v1,
            traffic_sf_test_v1,
            traffic_forecaster_v1,
        )
    )
    return (traffic_timestral_clean_forecast_v1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Evaluate everything using UtilsForecast

    Combine the statistical forecasts and Timestral predictions
    """)
    return


@app.cell
def _(traffic_statsforecast_forecasts_v1, traffic_timestral_clean_forecast_v1):
    traffic_all_forecasts_v1 = (
        traffic_statsforecast_forecasts_v1.merge(
            traffic_timestral_clean_forecast_v1,
            on=["unique_id", "ds"],
            validate="one_to_one",
        )
    )
    return (traffic_all_forecasts_v1,)


@app.cell
def _(traffic_all_forecasts_v1, traffic_sf_test_v1):
    # Prepare the evaluation dataframe

    traffic_evaluation_df_v1 = (
        traffic_sf_test_v1.merge(
            traffic_all_forecasts_v1,
            on=["unique_id", "ds"],
            validate="one_to_one",
        )
    )

    traffic_model_names_v1 = [
        column
        for column in traffic_all_forecasts_v1.columns
        if column not in ["unique_id", "ds"]
    ]

    print(f"Models being evaluated: {traffic_model_names_v1}")
    print(f"\nEvaluation observations: {len(traffic_evaluation_df_v1)}")
    return traffic_evaluation_df_v1, traffic_model_names_v1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Run UtilsForecast
    """)
    return


@app.cell
def _(traffic_evaluation_df_v1, traffic_model_names_v1, traffic_sf_train_v1):
    traffic_point_metrics_v1 = evaluate(
        df=traffic_evaluation_df_v1,
        models=traffic_model_names_v1,
        metrics=[
            mae,
            rmse,
            mape,
            smape,
            wape,
        ],
    )

    traffic_mase_metrics_v1 = mase(
        df=traffic_evaluation_df_v1,
        models=traffic_model_names_v1,
        seasonality=24,
        train_df=traffic_sf_train_v1,
    )
    return traffic_mase_metrics_v1, traffic_point_metrics_v1


@app.cell
def _(traffic_point_metrics_v1):
    # Results

    traffic_point_metrics_v1.round(3)
    return


@app.cell
def _(traffic_mase_metrics_v1):
    # Display seasonal results
    traffic_mase_metrics_v1.round(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Visualize the competing forecasts
    """)
    return


@app.cell
def _(traffic_all_forecasts_v1, traffic_sf_train_v1):
    plot_series(
        df=traffic_sf_train_v1,
        forecasts_df=traffic_all_forecasts_v1,
        ids=["I94"],
        models=[
            "Timestral-clean",
            "Naive",
            "SeasonalNaive-24",
            "SeasonalNaive-168",
            "SeasonalWindowAverage-24-7",
            "SeasonalWindowAverage-168-4",
        ],
        plot_random=False,
        max_insample_length=24 * 7,
        engine="matplotlib",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Exclude the lower and upper quantile bounds
    """)
    return


@app.cell
def _(traffic_all_forecasts_v1):
    traffic_point_models_v2 = [
        column
        for column in traffic_all_forecasts_v1.columns
        if column not in ["unique_id", "ds"]
        and not column.endswith("-lo-80")
        and not column.endswith("-hi-80")
    ]

    print("Point forecasting models:")
    print(traffic_point_models_v2)
    return (traffic_point_models_v2,)


@app.cell
def _(traffic_evaluation_df_v1, traffic_point_models_v2, traffic_sf_train_v1):
    traffic_point_metrics_v2 = evaluate(
        df=traffic_evaluation_df_v1,
        models=traffic_point_models_v2,
        metrics=[
            mae,
            rmse,
            mape,
            smape,
            wape,
        ],
    )

    traffic_mase_metrics_v2 = mase(
        df=traffic_evaluation_df_v1,
        models=traffic_point_models_v2,
        seasonality=24,
        train_df=traffic_sf_train_v1,
    )
    return traffic_mase_metrics_v2, traffic_point_metrics_v2


@app.cell
def _(traffic_point_metrics_v2):
    traffic_point_metrics_v2.round(4)
    return


@app.cell
def _(traffic_mase_metrics_v2):
    traffic_mase_metrics_v2.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Evaluate Timestral's prediction interval separately
    """)
    return


@app.function
def evaluate_traffic_interval_v1(
    evaluation_df_arg,
    lower_col_arg,
    upper_col_arg,
):
    """Evaluate an 80% prediction interval."""

    frame_local = evaluation_df_arg.copy()

    valid_local = (
        frame_local["y"].notna()
        & frame_local[lower_col_arg].notna()
        & frame_local[upper_col_arg].notna()
    )

    actual_local = frame_local.loc[
        valid_local, "y"
    ].to_numpy()

    lower_local = frame_local.loc[
        valid_local, lower_col_arg
    ].to_numpy()

    upper_local = frame_local.loc[
        valid_local, upper_col_arg
    ].to_numpy()

    if np.any(lower_local > upper_local):
        raise ValueError("Lower interval exceeds upper interval.")

    return pd.DataFrame({
        "Metric": [
            "Evaluation observations",
            "80% interval coverage (%)",
            "Mean interval width",
        ],
        "Value": [
            len(actual_local),
            np.mean(
                (actual_local >= lower_local)
                & (actual_local <= upper_local)
            ) * 100,
            np.mean(upper_local - lower_local),
        ],
    })


@app.cell
def _(traffic_evaluation_df_v1):
    traffic_interval_metrics_v1 = evaluate_traffic_interval_v1(
        traffic_evaluation_df_v1,
        "Timestral-clean-lo-80",
        "Timestral-clean-hi-80",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How much historical context does Timestral need?
    """)
    return


@app.function
def run_traffic_context_sensitivity_v1(
    series_arg,
    forecaster_arg,
    horizon_arg=24,
    context_lengths_arg=(
        168,
        336,
        672,
        909,
        1680,
        3360,
        6720,
        13440,
        28948,
    ),
):
    """Evaluate Timestral using different historical context lengths."""

    series_local = series_arg.sort_index()

    values_local = series_local.to_numpy(dtype=np.float32)

    if len(values_local) <= horizon_arg:
        raise ValueError("Insufficient observations.")

    actual_local = values_local[-horizon_arg:]

    if not np.isfinite(actual_local).all():
        raise ValueError("Test observations contain missing values.")

    history_local = values_local[:-horizon_arg]

    prediction_rows_local = []
    result_rows_local = []

    for context_length_local in context_lengths_arg:

        if context_length_local > len(history_local):
            raise ValueError(
                f"Context length {context_length_local} exceeds "
                f"available history."
            )

        if context_length_local > 32768:
            raise ValueError("Timestral context limit exceeded.")

        context_local = history_local[-context_length_local:]

        observed_local = int(
            np.isfinite(context_local).sum()
        )

        missing_local = int(
            np.isnan(context_local).sum()
        )

        print(
            f"Forecasting with {context_length_local:,} positions "
            f"({observed_local:,} observed, "
            f"{missing_local:,} missing)"
        )

        forecast_local = forecaster_arg.forecast(
            context_local,
            prediction_length=horizon_arg,
        )

        median_local = (
            forecast_local.median[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        quantiles_local = (
            forecast_local.quantiles[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        levels_local = np.asarray(
            forecast_local.quantile_levels
        )

        q10_idx_local = np.where(
            np.isclose(levels_local, 0.1)
        )[0][0]

        q90_idx_local = np.where(
            np.isclose(levels_local, 0.9)
        )[0][0]

        lower_local = quantiles_local[:, q10_idx_local]
        upper_local = quantiles_local[:, q90_idx_local]

        # Point forecast errors.
        errors_local = actual_local - median_local
        absolute_errors_local = np.abs(errors_local)

        mae_local = absolute_errors_local.mean()

        rmse_local = np.sqrt(
            np.mean(errors_local**2)
        )

        # Percentage metrics.
        nonzero_local = np.abs(actual_local) > 1e-8

        mape_local = (
            np.mean(
                absolute_errors_local[nonzero_local]
                / np.abs(actual_local[nonzero_local])
            ) * 100
            if nonzero_local.any()
            else np.nan
        )

        denominator_local = (
            np.abs(actual_local) + np.abs(median_local)
        )

        smape_local = np.mean(
            np.divide(
                200 * absolute_errors_local,
                denominator_local,
                out=np.zeros_like(absolute_errors_local),
                where=denominator_local > 0,
            )
        )

        wape_local = (
            100 * absolute_errors_local.sum()
            / np.abs(actual_local).sum()
        )

        # Historical seasonal scaling.
        seasonal_differences_local = np.abs(
            context_local[24:] - context_local[:-24]
        )

        valid_differences_local = np.isfinite(
            seasonal_differences_local
        )

        seasonal_scale_local = (
            seasonal_differences_local[
                valid_differences_local
            ].mean()
            if valid_differences_local.any()
            else np.nan
        )

        mase_local = (
            mae_local / seasonal_scale_local
            if seasonal_scale_local > 0
            else np.nan
        )

        # Probabilistic metrics.
        coverage_local = np.mean(
            (actual_local >= lower_local)
            & (actual_local <= upper_local)
        ) * 100

        interval_width_local = np.mean(
            upper_local - lower_local
        )

        quantile_errors_local = (
            actual_local[:, None] - quantiles_local
        )

        pinball_local = np.maximum(
            levels_local[None, :] * quantile_errors_local,
            (levels_local[None, :] - 1) * quantile_errors_local,
        ).mean()

        result_rows_local.append({
            "Context positions": context_length_local,
            "Observed hours": observed_local,
            "Missing hours": missing_local,
            "Missing (%)": (
                100 * missing_local / context_length_local
            ),
            "MAE": mae_local,
            "RMSE": rmse_local,
            "MAPE (%)": mape_local,
            "sMAPE (%)": smape_local,
            "WAPE (%)": wape_local,
            "MASE": mase_local,
            "80% coverage (%)": coverage_local,
            "80% interval width": interval_width_local,
            "Mean pinball loss": pinball_local,
        })

        # Store forecasts for selected plotting experiments.
        prediction_rows_local.append({
            "context_length": context_length_local,
            "median": median_local,
            "lower": lower_local,
            "upper": upper_local,
        })

    results_local = pd.DataFrame(result_rows_local)

    predictions_local = {
        item_local["context_length"]: item_local
        for item_local in prediction_rows_local
    }

    return results_local, predictions_local


@app.cell
def _(traffic_forecaster_v1, traffic_target_segment_v2):
    (
        traffic_context_results_v1,
        traffic_context_predictions_v1,
    ) = run_traffic_context_sensitivity_v1(
        series_arg=traffic_target_segment_v2,
        forecaster_arg=traffic_forecaster_v1,
        horizon_arg=24,
    )
    return (traffic_context_results_v1,)


@app.cell
def _(traffic_context_results_v1):
    traffic_context_results_v1.round(3)
    return


@app.cell
def _(traffic_context_results_v1):
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].plot(
            traffic_context_results_v1["Context positions"],
            traffic_context_results_v1["MAE"],
            marker="o",
        )

        axes[0].set(
            title="Historical context vs MAE",
            xlabel="Historical positions (hours)",
            ylabel="MAE",
            xscale="log",
        )

        axes[1].plot(
            traffic_context_results_v1["Context positions"],
            traffic_context_results_v1["80% interval width"],
            marker="o",
        )

        axes[1].set(
            title="Historical context vs interval width",
            xlabel="Historical positions (hours)",
            ylabel="Mean 80% interval width",
            xscale="log",
        )

        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Experiment with past covariates - Weather data
    """)
    return


@app.function
def prepare_traffic_weather_covariates_v1(
    raw_frame_arg,
    target_series_arg,
):
    """Deduplicate, clean and align numeric weather covariates."""

    work_local = raw_frame_arg.copy()

    work_local["date_time"] = pd.to_datetime(
        work_local["date_time"],
        errors="coerce",
    )

    if work_local["date_time"].isna().any():
        raise ValueError("Invalid timestamps in weather data.")

    feature_columns_local = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    # Ensure numeric data types.
    for column_local in feature_columns_local:
        work_local[column_local] = pd.to_numeric(
            work_local[column_local],
            errors="coerce",
        )

    # Replace known invalid measurements.
    invalid_temp_local = (
        work_local["temp"] <= 0
    )

    invalid_rain_local = (
        work_local["rain_1h"] >= 9000
    )

    work_local.loc[
        invalid_temp_local, "temp"
    ] = np.nan

    work_local.loc[
        invalid_rain_local, "rain_1h"
    ] = np.nan

    work_local[feature_columns_local] = (
        work_local[feature_columns_local]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Aggregate numeric weather observations by timestamp.
    weather_grouped_local = (
        work_local
        .groupby("date_time")[feature_columns_local]
        .median()
        .sort_index()
    )

    # Align precisely with the deduplicated hourly target timeline.
    weather_aligned_local = (
        weather_grouped_local
        .reindex(target_series_arg.index)
        .astype(np.float32)
    )

    if not weather_aligned_local.index.equals(
        target_series_arg.index
    ):
        raise ValueError("Weather and target indexes do not match.")

    # Feature-level quality summary.
    quality_local = pd.DataFrame({
        "Feature": feature_columns_local,
        "Missing observations": [
            int(weather_aligned_local[col].isna().sum())
            for col in feature_columns_local
        ],
        "Missing (%)": [
            float(weather_aligned_local[col].isna().mean() * 100)
            for col in feature_columns_local
        ],
        "Observed values": [
            int(weather_aligned_local[col].notna().sum())
            for col in feature_columns_local
        ],
    })

    metadata_local = pd.DataFrame({
        "Metric": [
            "Original weather records",
            "Unique weather timestamps",
            "Duplicate excess records",
            "Invalid temperature records",
            "Invalid rainfall records",
            "Aligned hourly positions",
            "Weather feature columns",
        ],
        "Value": [
            len(work_local),
            work_local["date_time"].nunique(),
            len(work_local) - work_local["date_time"].nunique(),
            int(invalid_temp_local.sum()),
            int(invalid_rain_local.sum()),
            len(weather_aligned_local),
            len(feature_columns_local),
        ],
    })

    return (
        weather_aligned_local,
        quality_local,
        metadata_local,
    )


@app.cell
def _(traffic_df_raw, traffic_target_segment_v2):
    (
        traffic_weather_hourly_v1,
        traffic_weather_quality_v1,
        traffic_weather_metadata_v1,
    ) = prepare_traffic_weather_covariates_v1(
        traffic_df_raw,
        traffic_target_segment_v2,
    )
    return (
        traffic_weather_hourly_v1,
        traffic_weather_metadata_v1,
        traffic_weather_quality_v1,
    )


@app.cell
def _(traffic_weather_metadata_v1):
    traffic_weather_metadata_v1
    return


@app.cell
def _(traffic_weather_quality_v1):
    traffic_weather_quality_v1.round(4)
    return


@app.cell
def _(traffic_weather_hourly_v1):
    traffic_weather_hourly_v1.head(10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Prepare a common forecasting origin

    We'll use September 23, 2018, at 00:00 as the forecast origin, with a maximum horizon of 192 hours.

    The historical context will therefore contain 28,780 hourly positions. The weather covariate array will contain only the weather values corresponding to those historical positions.
    """)
    return


@app.function
def prepare_traffic_horizon_study_v1(
    target_series_arg,
    weather_frame_arg,
    max_horizon_arg=192,
):
    """Prepare common target, weather and clean-baseline inputs."""

    target_local = target_series_arg.sort_index()
    weather_local = weather_frame_arg.sort_index()

    if not target_local.index.equals(weather_local.index):
        raise ValueError(
            "Target and weather timestamps do not match."
        )

    index_local = target_local.index

    if not isinstance(index_local, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    if index_local.has_duplicates:
        raise ValueError("Duplicate timestamps found.")

    if not np.all(
        index_local[1:] - index_local[:-1]
        == pd.Timedelta(hours=1)
    ):
        raise ValueError("Expected a regular hourly timeline.")

    if len(target_local) <= max_horizon_arg:
        raise ValueError("Insufficient historical observations.")

    target_values_local = target_local.to_numpy(
        dtype=np.float32
    )

    # Common forecast origin.
    context_length_local = (
        len(target_values_local) - max_horizon_arg
    )

    target_context_local = target_values_local[
        :context_length_local
    ]

    actual_future_local = target_values_local[
        context_length_local:
    ]

    future_index_local = index_local[
        context_length_local:
    ]

    # Only historical weather information is supplied.
    weather_columns_local = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    weather_context_local = (
        weather_local[weather_columns_local]
        .iloc[:context_length_local]
        .to_numpy(dtype=np.float32)
    )

    weather_context_local = np.where(
        np.isfinite(weather_context_local),
        weather_context_local,
        np.nan,
    ).astype(np.float32)

    weather_covariates_local = (
        weather_context_local.T[None, :, :]
    )

    # Locate the latest complete target suffix.
    missing_positions_local = np.flatnonzero(
        ~np.isfinite(target_context_local)
    )

    clean_start_local = (
        int(missing_positions_local[-1] + 1)
        if len(missing_positions_local)
        else 0
    )

    clean_context_local = pd.Series(
        target_context_local[clean_start_local:],
        index=index_local[
            clean_start_local:context_length_local
        ],
        name="y",
        dtype=float,
    )

    if len(clean_context_local) < 672:
        raise ValueError(
            "Clean context is shorter than four weekly cycles."
        )

    if clean_context_local.isna().any():
        raise ValueError("Clean context unexpectedly contains NaN.")

    # StatsForecast-compatible training data.
    stats_train_local = pd.DataFrame({
        "unique_id": "I94",
        "ds": clean_context_local.index,
        "y": clean_context_local.to_numpy(),
    })

    metadata_local = pd.DataFrame({
        "Metric": [
            "Forecast origin",
            "Forecast end",
            "Historical context positions",
            "Missing target observations",
            "Historical target missing (%)",
            "Weather covariate shape",
            "Missing temperature observations",
            "Missing rainfall observations",
            "Missing snowfall observations",
            "Missing cloud-cover observations",
            "Latest complete context length",
            "Latest complete context start",
            "Latest complete context end",
            "Valid future target observations",
            "Missing future target observations",
        ],
        "Value": [
            future_index_local[0],
            future_index_local[-1],
            len(target_context_local),
            int(np.isnan(target_context_local).sum()),
            float(np.isnan(target_context_local).mean() * 100),
            str(weather_covariates_local.shape),
            int(np.isnan(weather_context_local[:, 0]).sum()),
            int(np.isnan(weather_context_local[:, 1]).sum()),
            int(np.isnan(weather_context_local[:, 2]).sum()),
            int(np.isnan(weather_context_local[:, 3]).sum()),
            len(clean_context_local),
            clean_context_local.index[0],
            clean_context_local.index[-1],
            int(np.isfinite(actual_future_local).sum()),
            int(np.isnan(actual_future_local).sum()),
        ],
    })

    return (
        target_context_local,
        actual_future_local,
        weather_covariates_local,
        future_index_local,
        clean_context_local,
        stats_train_local,
        metadata_local,
    )


@app.cell
def _(traffic_target_segment_v2, traffic_weather_hourly_v1):
    (
        traffic_horizon_context_v1,
        traffic_horizon_actual_v1,
        traffic_horizon_weather_cov_v1,
        traffic_horizon_index_v1,
        traffic_horizon_clean_context_v1,
        traffic_horizon_stats_train_v1,
        traffic_horizon_metadata_v1,
    ) = prepare_traffic_horizon_study_v1(
        traffic_target_segment_v2,
        traffic_weather_hourly_v1,
        max_horizon_arg=192,
    )
    return (
        traffic_horizon_actual_v1,
        traffic_horizon_clean_context_v1,
        traffic_horizon_context_v1,
        traffic_horizon_index_v1,
        traffic_horizon_metadata_v1,
        traffic_horizon_stats_train_v1,
        traffic_horizon_weather_cov_v1,
    )


@app.cell
def _(traffic_horizon_metadata_v1):
    traffic_horizon_metadata_v1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generate the four forecasts
    """)
    return


@app.function
def run_traffic_horizon_forecasts_v1(
    context_arg,
    actual_arg,
    weather_cov_arg,
    future_index_arg,
    clean_context_arg,
    stats_train_arg,
    forecaster_arg,
    horizon_arg=192,
):
    """Generate full-horizon forecasts for four approaches."""

    # A. Timestral using full target history.
    target_forecast_local = forecaster_arg.forecast(
        context_arg,
        prediction_length=horizon_arg,
    )

    # B. Timestral using full target history and all four
    # historical weather features.
    weather_forecast_local = forecaster_arg.forecast(
        context_arg,
        prediction_length=horizon_arg,
        past_covariates=weather_cov_arg,
    )

    # C. Timestral using only the latest complete target history.
    clean_context_local = clean_context_arg.to_numpy(
        dtype=np.float32
    )

    clean_forecast_local = forecaster_arg.forecast(
        clean_context_local,
        prediction_length=horizon_arg,
    )

    # D. Strong statistical baseline.
    sf_local = StatsForecast(
        models=[
            SeasonalWindowAverage(
                season_length=168,
                window_size=4,
                alias="SeasonalWindowAverage-168-4",
            ),
        ],
        freq="h",
        n_jobs=1,
    )

    sf_local.fit(df=stats_train_arg)

    sf_predictions_local = sf_local.predict(
        h=horizon_arg,
    )

    # Extract median forecasts.
    target_median_local = (
        target_forecast_local.median[0, 0]
        .detach().cpu().numpy()
    )

    weather_median_local = (
        weather_forecast_local.median[0, 0]
        .detach().cpu().numpy()
    )

    clean_median_local = (
        clean_forecast_local.median[0, 0]
        .detach().cpu().numpy()
    )

    # Extract all quantiles for probabilistic evaluation.
    quantiles_local = {
        "Timestral-full-target": (
            target_forecast_local.quantiles[0, 0]
            .detach().cpu().numpy()
        ),
        "Timestral-full-weather": (
            weather_forecast_local.quantiles[0, 0]
            .detach().cpu().numpy()
        ),
        "Timestral-clean-context": (
            clean_forecast_local.quantiles[0, 0]
            .detach().cpu().numpy()
        ),
    }

    levels_local = tuple(
        target_forecast_local.quantile_levels
    )

    # Combine all point forecasts.
    predictions_local = pd.DataFrame({
        "unique_id": "I94",
        "ds": future_index_arg,
        "y": actual_arg,
        "Timestral-full-target": target_median_local,
        "Timestral-full-weather": weather_median_local,
        "Timestral-clean-context": clean_median_local,
    })

    predictions_local = predictions_local.merge(
        sf_predictions_local,
        on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )

    forecast_metadata_local = pd.DataFrame({
        "Forecast": [
            "Full target",
            "Full target + weather",
            "Clean target context",
            "SeasonalWindowAverage-168-4",
        ],
        "Historical observations": [
            len(context_arg),
            len(context_arg),
            len(clean_context_local),
            len(stats_train_arg),
        ],
    })

    return (
        predictions_local,
        quantiles_local,
        levels_local,
        forecast_metadata_local,
    )


@app.cell
def _(
    traffic_forecaster_v1,
    traffic_horizon_actual_v1,
    traffic_horizon_clean_context_v1,
    traffic_horizon_context_v1,
    traffic_horizon_index_v1,
    traffic_horizon_stats_train_v1,
    traffic_horizon_weather_cov_v1,
):
    (
        traffic_horizon_predictions_v1,
        traffic_horizon_quantiles_v1,
        traffic_horizon_quantile_levels_v1,
        traffic_horizon_forecast_metadata_v1,
    ) = run_traffic_horizon_forecasts_v1(
        context_arg=traffic_horizon_context_v1,
        actual_arg=traffic_horizon_actual_v1,
        weather_cov_arg=traffic_horizon_weather_cov_v1,
        future_index_arg=traffic_horizon_index_v1,
        clean_context_arg=traffic_horizon_clean_context_v1,
        stats_train_arg=traffic_horizon_stats_train_v1,
        forecaster_arg=traffic_forecaster_v1,
        horizon_arg=192,
    )
    return (
        traffic_horizon_forecast_metadata_v1,
        traffic_horizon_predictions_v1,
        traffic_horizon_quantile_levels_v1,
        traffic_horizon_quantiles_v1,
    )


@app.cell
def _(traffic_horizon_forecast_metadata_v1):
    traffic_horizon_forecast_metadata_v1
    return


@app.cell
def _(traffic_horizon_predictions_v1):
    traffic_horizon_predictions_v1.head(10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Evaluate forecast accuracy across horizons
    """)
    return


@app.function
def evaluate_traffic_horizons_v1(
    predictions_arg,
    quantiles_arg,
    levels_arg,
    stats_train_arg,
    horizons_arg=(24, 48, 72, 168, 192),
):
    """Evaluate cumulative horizons and individual lead-time bands."""

    model_columns_local = [
        "Timestral-full-target",
        "Timestral-full-weather",
        "Timestral-clean-context",
        "SeasonalWindowAverage-168-4",
    ]

    predictions_local = predictions_arg.copy()

    point_frames_local = []
    probabilistic_rows_local = []
    lead_time_rows_local = []

    levels_local = np.asarray(levels_arg)

    q10_idx_local = np.where(
        np.isclose(levels_local, 0.1)
    )[0][0]

    q90_idx_local = np.where(
        np.isclose(levels_local, 0.9)
    )[0][0]

    def score_probabilistic_local(
        actual_values,
        quantile_values,
    ):
        valid_local = np.isfinite(actual_values)

        actual_local = actual_values[valid_local]
        quantiles_local = quantile_values[valid_local]

        if len(actual_local) == 0:
            raise ValueError("No valid actual observations.")

        lower_local = quantiles_local[:, q10_idx_local]
        upper_local = quantiles_local[:, q90_idx_local]

        errors_local = (
            actual_local[:, None] - quantiles_local
        )

        pinball_local = np.maximum(
            levels_local[None, :] * errors_local,
            (levels_local[None, :] - 1) * errors_local,
        )

        return {
            "Valid observations": len(actual_local),
            "80% coverage (%)": np.mean(
                (actual_local >= lower_local)
                & (actual_local <= upper_local)
            ) * 100,
            "80% interval width": np.mean(
                upper_local - lower_local
            ),
            "Mean pinball loss": pinball_local.mean(),
        }

    # --------------------------------------------------
    # A. Cumulative horizon evaluation.
    # --------------------------------------------------

    for horizon_local in horizons_arg:

        evaluation_local = (
            predictions_local.iloc[:horizon_local]
            .dropna(subset=["y"])
            .copy()
        )

        if evaluation_local.empty:
            raise ValueError(
                f"No valid observations for horizon {horizon_local}."
            )

        # UtilsForecast point metrics.
        point_local = evaluate(
            df=evaluation_local,
            models=model_columns_local,
            metrics=[
                mae,
                rmse,
                mape,
                smape,
                wape,
            ],
        )

        # UtilsForecast MASE.
        mase_local = mase(
            df=evaluation_local,
            models=model_columns_local,
            seasonality=24,
            train_df=stats_train_arg,
        )

        if "metric" not in mase_local.columns:
            mase_local.insert(1, "metric", "mase")

        # Convert ratio-based percentage metrics into percentages.
        metric_percentage_mask_local = (
            point_local["metric"].isin(["mape", "smape", "wape"])
        )

        point_local.loc[
            metric_percentage_mask_local,
            model_columns_local,
        ] *= 100

        combined_local = pd.concat(
            [point_local, mase_local],
            ignore_index=True,
        )

        combined_local["horizon_hours"] = horizon_local

        point_frames_local.append(combined_local)

        # Probabilistic metrics for the three Timestral variants.
        actual_prefix_local = predictions_local[
            "y"
        ].iloc[:horizon_local].to_numpy(dtype=float)

        for model_local, quantile_values_local in quantiles_arg.items():

            score_local = score_probabilistic_local(
                actual_prefix_local,
                quantile_values_local[:horizon_local],
            )

            probabilistic_rows_local.append({
                "Approach": model_local,
                "Horizon hours": horizon_local,
                **score_local,
            })

    # --------------------------------------------------
    # B. Individual lead-time bands from the 192-hour forecast.
    # --------------------------------------------------

    bands_local = [
        ("Hours 1-24", 0, 24),
        ("Hours 25-48", 24, 48),
        ("Hours 49-72", 48, 72),
        ("Hours 73-168", 72, 168),
        ("Hours 169-192", 168, 192),
    ]

    for band_name_local, start_local, end_local in bands_local:

        band_local = predictions_local.iloc[
            start_local:end_local
        ].dropna(subset=["y"])

        actual_local = band_local["y"].to_numpy()

        for model_local in model_columns_local:

            predicted_local = band_local[
                model_local
            ].to_numpy()

            errors_local = actual_local - predicted_local

            lead_time_rows_local.append({
                "Lead-time band": band_name_local,
                "Approach": model_local,
                "Valid observations": len(actual_local),
                "MAE": np.mean(np.abs(errors_local)),
                "RMSE": np.sqrt(np.mean(errors_local**2)),
                "WAPE (%)": (
                    100 * np.abs(errors_local).sum()
                    / np.abs(actual_local).sum()
                ),
            })

    point_results_local = pd.concat(
        point_frames_local,
        ignore_index=True,
    )

    probabilistic_results_local = pd.DataFrame(
        probabilistic_rows_local
    )

    lead_time_results_local = pd.DataFrame(
        lead_time_rows_local
    )

    return (
        point_results_local,
        probabilistic_results_local,
        lead_time_results_local,
    )


@app.cell
def _(
    traffic_horizon_predictions_v1,
    traffic_horizon_quantile_levels_v1,
    traffic_horizon_quantiles_v1,
    traffic_horizon_stats_train_v1,
):
    (
        traffic_horizon_point_metrics_v1,
        traffic_horizon_prob_metrics_v1,
        traffic_horizon_lead_metrics_v1,
    ) = evaluate_traffic_horizons_v1(
        predictions_arg=traffic_horizon_predictions_v1,
        quantiles_arg=traffic_horizon_quantiles_v1,
        levels_arg=traffic_horizon_quantile_levels_v1,
        stats_train_arg=traffic_horizon_stats_train_v1,
    )
    return (
        traffic_horizon_lead_metrics_v1,
        traffic_horizon_point_metrics_v1,
        traffic_horizon_prob_metrics_v1,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare cumulative forecast horizons
    """)
    return


@app.cell
def _(traffic_horizon_point_metrics_v1):
    traffic_horizon_point_metrics_v1.round(4)
    return


@app.cell
def _(traffic_horizon_prob_metrics_v1):
    traffic_horizon_prob_metrics_v1.round(4)
    return


@app.cell
def _(traffic_horizon_lead_metrics_v1):
    traffic_horizon_lead_metrics_v1.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the forecast
    """)
    return


@app.cell
def _(traffic_horizon_predictions_v1, traffic_horizon_stats_train_v1):
    traffic_horizon_plot_fig_v2 = plot_series(
        df=traffic_horizon_stats_train_v1,
        forecasts_df=traffic_horizon_predictions_v1,
        ids=["I94"],
        models=[
            "Timestral-full-target",
            "Timestral-full-weather",
            "Timestral-clean-context",
            "SeasonalWindowAverage-168-4",
        ],
        plot_random=False,
        max_insample_length=24 * 7,
        engine="matplotlib",
    )

    traffic_horizon_plot_fig_v2
    return


@app.function
def build_traffic_interval_plot_data_v2(
    predictions_arg,
    quantiles_arg,
    levels_arg,
):
    """Add Timestral 80% interval columns for UtilsForecast plotting."""

    plot_data_local = predictions_arg.copy()

    levels_local = np.asarray(levels_arg)

    q10_local = np.where(
        np.isclose(levels_local, 0.1)
    )[0][0]

    q90_local = np.where(
        np.isclose(levels_local, 0.9)
    )[0][0]

    for model_local, values_local in quantiles_arg.items():
        plot_data_local[f"{model_local}-lo-80"] = (
            values_local[:, q10_local]
        )

        plot_data_local[f"{model_local}-hi-80"] = (
            values_local[:, q90_local]
        )

    return plot_data_local


@app.cell
def _(
    traffic_horizon_predictions_v1,
    traffic_horizon_quantile_levels_v1,
    traffic_horizon_quantiles_v1,
):
    traffic_horizon_interval_plot_data_v2 = (
        build_traffic_interval_plot_data_v2(
            traffic_horizon_predictions_v1,
            traffic_horizon_quantiles_v1,
            traffic_horizon_quantile_levels_v1,
        )
    )
    return (traffic_horizon_interval_plot_data_v2,)


@app.cell
def _(traffic_horizon_interval_plot_data_v2, traffic_horizon_stats_train_v1):
    traffic_horizon_interval_fig_v2 = plot_series(
        df=traffic_horizon_stats_train_v1,
        forecasts_df=traffic_horizon_interval_plot_data_v2,
        ids=["I94"],
        models=[
            "Timestral-full-target",
            "Timestral-full-weather",
            "Timestral-clean-context",
        ],
        level=[80],
        plot_random=False,
        max_insample_length=24 * 7,
        engine="matplotlib",
    )

    traffic_horizon_interval_fig_v2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
