import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from ucimlrepo import fetch_ucirepo


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
