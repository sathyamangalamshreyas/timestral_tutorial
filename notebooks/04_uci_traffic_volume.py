import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from ucimlrepo import fetch_ucirepo

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
