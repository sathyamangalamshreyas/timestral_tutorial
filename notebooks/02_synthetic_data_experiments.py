import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from timestral import TimestralForecaster


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Load Timestral-209M checkpoint from HuggingFace
    """)
    return


@app.cell
def _():
    forecaster = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M",
        device="cpu",
    )
    return (forecaster,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Set Conetxt Length and Horizon

    - CONTEXT_LENGTH is for 14 days of hourly observations
    - HORIZON is Forecast the next 48 hours
    """)
    return


@app.cell
def _():
    CONTEXT_LENGTH = 24 * 14
    HORIZON = 48

    TOTAL_LENGTH = CONTEXT_LENGTH + HORIZON
    return CONTEXT_LENGTH, HORIZON, TOTAL_LENGTH


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Generate Synthetic Data
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH):
    def make_synthetic_series(
        noise_scale: float = 0.8,
        level_shift: float = 0.0,
        seed: int = 42,
    ) -> np.ndarray:
        t = np.arange(TOTAL_LENGTH)

        # Reusing the seed keeps the underlying noise pattern consistent
        rng = np.random.default_rng(seed=seed)
        noise = rng.normal(size=TOTAL_LENGTH)

        trend = 0.015 * t
        daily_seasonality = 4.0 * np.sin(2 * np.pi * t /24)

        shift = np.where(t >= CONTEXT_LENGTH, level_shift, 0.0)

        return (
            30
            + trend
            + daily_seasonality
            + noise_scale * noise
            + shift
        )

    return (make_synthetic_series,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Define Scenarios
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, make_synthetic_series):
    SCENARIOS = {
        "Low Noise": {
            "noise_scale": 0.2,
            "level_shift": 0.0,
        },
        "Reference": {
            "noise_scale": 0.8,
            "level_shift": 0.0,
        },
        "High Noise": {
            "noise_scale": 2.0,
            "level_shift": 0.0,
        },
        "Future level shift": {
            "noise_scale": 0.8,
            "level_shift": 5.0,
        },
    }

    scenario_data = {}

    for name, parameters in SCENARIOS.items():
        series = make_synthetic_series(**parameters)

        scenario_data[name] = {
            "context": series[:CONTEXT_LENGTH],
            "actual_future": series[CONTEXT_LENGTH:],
            "series": series,
        }
    return SCENARIOS, scenario_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Visualize the data
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, SCENARIOS, TOTAL_LENGTH, scenario_data):
    fig, axes = plt.subplots(
        len(SCENARIOS),
        1,
        figsize=(12, 10),
        sharex=True,
    )

    t = np.arange(TOTAL_LENGTH)

    for ax, (name_viz, data) in zip(axes, scenario_data.items()):
        ax.plot(t, data["series"], label=name_viz)
        ax.axvline(
            CONTEXT_LENGTH,
            linestyle="--",
            color='r',
            label="Forecast Origin",
        )
        ax.set_ylabel("Value")
        ax.set_title(name_viz)
        ax.legend()

    axes[-1].set_xlabel("Hour")

    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Function to evaluate model performance
    """)
    return


@app.function
def evaluate_scenario(
    context: np.ndarray,
    actual: np.ndarray,
    forecast,
) -> dict:
    predicted = (
        forecast.median[0, 0].detach().cpu().numpy()
    )
    quantiles = (
        forecast.quantiles[0, 0].detach().cpu().numpy()
    )
    levels = np.asarray(forecast.quantile_levels)

    errors = actual - predicted
    absolute_errors = np.abs(errors)

    mae = absolute_errors.mean()
    rmse = np.sqrt(np.mean(errors**2))

    mape = (
        np.mean(absolute_errors / np.abs(actual)) * 100
        if np.all(np.abs(actual) > 1e-8)
        else np.nan
    )

    period = 24
    seasonal_differences = np.abs(
    context[period:] - context[:-period]
    )

    valid_differences = np.isfinite(seasonal_differences)

    seasonal_scale = (
    seasonal_differences[valid_differences].mean()
    if valid_differences.any()
    else np.nan
    )

    seasonal_naive = np.tile(
        context[-period:],
        int(np.ceil(len(actual) / period)),
    )[:len(actual)]

    baseline_mae = np.mean(np.abs(actual - seasonal_naive))

    q10_idx = np.where(np.isclose(levels, 0.1))[0][0]
    q90_idx = np.where(np.isclose(levels, 0.9))[0][0]

    lower = quantiles[:, q10_idx]
    upper = quantiles[:, q90_idx]

    coverage = np.mean(
        (actual >= lower) & (actual <= upper)
    ) * 100

    quantile_errors = actual[:, None] - quantiles

    pinball = np.maximum(
        levels[None, :] * quantile_errors,
        (levels[None, :] - 1) * quantile_errors,
    ).mean()

    return {
        "timestral_mae": mae,
        "Seasonal Naive MAE": baseline_mae,
        "MAE Improvement (%)": (
            100 * (baseline_mae - mae) / baseline_mae
            if baseline_mae > 0
            else np.nan
        ),
        "timestral_mape": mape,
        "timestral_rmse": rmse,
        "80% coverage (%)": coverage,
        "80% interval width": np.mean(upper - lower),
        "Mean Pinball Loss": pinball,
    }


@app.cell
def _(HORIZON, forecaster, scenario_data):
    def _():
        forecasts = {}
        results = []

        for name, data in scenario_data.items():
            forecast = forecaster.forecast(
                data["context"],
                prediction_length=HORIZON,
            )

            forecasts[name] = forecast

            results.append({
                "Scenario": name,
                **evaluate_scenario(
                    data["context"],
                    data["actual_future"],
                    forecast,
                ),
            })

        results_df = pd.DataFrame(results).set_index("Scenario")
        return results_df.round(4)

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generate New Scenarios
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH):
    def make_seasonality_shift_series(
        future_amplitude: float = 4.0,
        future_period: int = 24,
        seed: int = 42,
    ) -> np.ndarray:
        t = np.arange(TOTAL_LENGTH)
        rng = np.random.default_rng(seed=seed)

        noise = rng.normal(0, 0.8, size=TOTAL_LENGTH)
        seasonality = 4.0 * np.sin(2 * np.pi * t / 24)

        future = t >= CONTEXT_LENGTH
        seasonality[future] = (
            future_amplitude * np.sin(2 * np.pi * t[future] / future_period)
        )

        return 30 + 0.015 * t + seasonality + noise

    seasonality_scenarios = {
        "Stable daily cycle": make_seasonality_shift_series(),
        "Reduced seasonality amplitude": make_seasonality_shift_series(
            future_amplitude=1.5,
            future_period=24,
        ),
        "Changed seasonal period": make_seasonality_shift_series(
            future_amplitude=4.0,
            future_period=12,
        ),
    }
    return (seasonality_scenarios,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Plot the scenarios
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH, seasonality_scenarios):
    def _():
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        for ax, (name, series) in zip(
            axes, seasonality_scenarios.items()
        ):
            ax.plot(
                np.arange(CONTEXT_LENGTH),
                series[:CONTEXT_LENGTH],
                label="Historical Context",
            )
            ax.plot(
                np.arange(CONTEXT_LENGTH, TOTAL_LENGTH),
                series[CONTEXT_LENGTH:],
                label="Acutal Future",
            )
            ax.axvline(CONTEXT_LENGTH, linestyle="--", color='g')
            ax.set_title(name)
            ax.legend()

        axes[-1].set_xlabel("Hour")
        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Forecast and Evaluate
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, HORIZON, forecaster, seasonality_scenarios):
    def _():
        seasonality_results = []

        for name, series in seasonality_scenarios.items():
            context = series[:CONTEXT_LENGTH]
            acutal_future = series[CONTEXT_LENGTH:]

            forecast = forecaster.forecast(
                context=context,
                prediction_length=HORIZON,
            )

            seasonality_results.append({
                "Scenario": name,
                **evaluate_scenario(
                    context=context,
                    actual=acutal_future,
                    forecast=forecast,
                ),
            })

        seasonality_results_df = pd.DataFrame(
            seasonality_results
        ).set_index("Scenario")
        return seasonality_results_df.round(4)


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Test three things:
    1. Noise with temporal dependence
    2. Isolated outliers
    3. Missing values in the historical context
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##### Generate new scenarios
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH, make_synthetic_series):
    def make_ar1_series(
        phi: float = 0.8,
        noise_scale: float = 0.8,
        seed: int = 42,
    ) -> np.ndarray:

        t = np.arange(TOTAL_LENGTH)
        rng = np.random.default_rng(seed=seed)

        # Scale innovations so the stationary noise standard
        # deviation is approximately noise_scale
        innovation_scale = noise_scale * np.sqrt(1 - phi**2)
        innovations = rng.normal(
            0,
            innovation_scale,
            size=TOTAL_LENGTH,
        )

        noise = np.empty(TOTAL_LENGTH)
        noise[0] = rng.normal(0, noise_scale)

        for i in range(1, TOTAL_LENGTH):
            noise[i] = phi * noise[i - 1] + innovations[i]

        trend = 0.015 * t
        seasonality = 4.0 * np.sin(2 * np.pi * t / 24)

        return (
            30
            + trend
            + seasonality
            + noise
        )
    reference_series = make_synthetic_series(
        noise_scale=0.8,
        seed=42,
    )

    # Scenario 1: Temporally correlated noise
    ar1_series = make_ar1_series(phi=0.8, seed=42)

    # Scenario 2: One outlier in the historical context
    historical_outlier_series = reference_series.copy()
    historical_outlier_series[CONTEXT_LENGTH - 36] += 10

    # Scenario 3: Two outliers in the future
    future_outlier_series = reference_series.copy()
    future_outlier_series[CONTEXT_LENGTH + 12] += 10
    future_outlier_series[CONTEXT_LENGTH + 34] -= 8

    # Scenario 4: Approximately 10% of the older historical
    # observations are missing. Keep the latest 24 hours intact
    missing_series = reference_series.copy()

    missing_rng = np.random.default_rng(202)
    eligible_indices = np.arange(CONTEXT_LENGTH - 24)
    num_missing = int(0.10 * len(eligible_indices))

    missing_indices = missing_rng.choice(
        eligible_indices,
        size=num_missing,
        replace=False,
    )

    missing_series[missing_indices] = np.nan

    experiment_scenarios = {
        "Rerference (white noise)": reference_series,
        "AR(1) noise": ar1_series,
        "Historical outliers": historical_outlier_series,
        "Future outliers": future_outlier_series,
        "10% missing history": missing_series,
    }
    return (experiment_scenarios,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Forecast and evaluate scenarios
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, HORIZON, experiment_scenarios, forecaster):
    def _():
        experiment_results = []
        experiment_forecasts = {}

        for name, series in experiment_scenarios.items():
            context = series[:CONTEXT_LENGTH]
            actual_future = series[CONTEXT_LENGTH:]

            forecast = forecaster.forecast(
                context=context,
                prediction_length=HORIZON,
            )

            experiment_forecasts[name] = forecast

            metrics = evaluate_scenario(
                context=context,
                actual=actual_future,
                forecast=forecast,
            )

            experiment_results.append({
                "Scenario": name,
                **metrics,
            })

            experiment_results_df = (
                pd.DataFrame(experiment_results).set_index("Scenario")
            )
        return experiment_results_df.round(4)


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Test forecast accuracy with known-future covariate
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We'll compare two forecasts on exactly the same target data:
    - Baseline: Timestral receives the target history only
    - Covariate-aware: Timestral receives the same target history plus the promotion schedule, including future promotions
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH):
    promo_t = np.arange(TOTAL_LENGTH)
    promo_rng = np.random.default_rng(seed=42)

    # Promotion schedule: 0 = no promotion, 1 = promotion
    promo_indicator = np.zeros(TOTAL_LENGTH, dtype=float)

    # Historical promotions: varying days and starting hours
    historical_promotions = [
        (2, 12),
        (5, 19),
        (8, 9),
        (11, 18),
        (13, 13),
    ]

    for day, start_hour in historical_promotions:
        start = day * 24 + start_hour
        promo_indicator[start:start + 6] = 1.0

    # Future promotions: these are known in advance
    promo_indicator[CONTEXT_LENGTH + 12:
        CONTEXT_LENGTH + 18] = 1.0

    promo_indicator[CONTEXT_LENGTH + 36:
        CONTEXT_LENGTH + 42] = 1.0

    # Demand is influenced by trend, daily seasonality
    # promotions and random noise
    promo_target = (
        30
        + 0.015 * promo_t
        + 4.0 + np.sin(2 * np.pi * promo_t / 24)
        + 6.0 * promo_indicator
        + promo_rng.normal(0, 0.8, size=TOTAL_LENGTH)
    )

    promo_context = promo_target [:CONTEXT_LENGTH]
    promo_actual = promo_target[CONTEXT_LENGTH:]

    print("Historical observations:", len(promo_context))
    print("Future observations:", len(promo_actual))
    print("Future promotion hours:", int(promo_indicator[CONTEXT_LENGTH:].sum()))
    return promo_actual, promo_context, promo_indicator


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the covariate data
    """)
    return


@app.cell
def _(
    CONTEXT_LENGTH,
    TOTAL_LENGTH,
    promo_actual,
    promo_context,
    promo_indicator,
):
    def _():
        fig, (ax, ax_promo) = plt.subplots(
            2, 1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ax.plot(
            np.arange(CONTEXT_LENGTH),
            promo_context,
            label="Historical demand",
        )

        ax.plot(
            np.arange(CONTEXT_LENGTH, TOTAL_LENGTH),
            promo_actual,
            label="Actual Future Demand",
        )

        ax.axvline(CONTEXT_LENGTH, linestyle="--", color="r", label="Forecast Origin")
        ax.set_ylabel("Demand")
        ax.legend()

        ax_promo.step(
            np.arange(CONTEXT_LENGTH, TOTAL_LENGTH),
            promo_indicator[CONTEXT_LENGTH:],
            where="post",
            label="Known-future promotion",
        )

        ax_promo.set_ylabel("Promotion")
        return ax_promo.set_xlabel("Hour")
        ax_promo.set_ylim(-0.1, 1.1)
        ax_promo.legend()

        fig.tight_layout()
        fig

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Forecast with and without the covariates
    """)
    return


@app.cell
def _(HORIZON, TOTAL_LENGTH, forecaster, promo_context, promo_indicator):
    # Baseline: target history only
    promo_forecast_baseline = forecaster.forecast(
        promo_context,
        prediction_length=HORIZON,
    )

    # The known-covariate array includes BOTH historical
    # and future values. Shape: (batch, covariates, time)
    known_promotion = promo_indicator[None, None, :]

    assert known_promotion.shape == (1, 1, TOTAL_LENGTH)

    # Covariate-aware forecast
    promo_forecast_known = forecaster.forecast(
        promo_context,
        prediction_length=HORIZON,
        known_covariates=known_promotion,
    )

    print(
        "Baseline forecase shape:",
        tuple(promo_forecast_baseline.median.shape),
    )

    print(
        "Covariate forecast shape:",
        tuple(promo_forecast_known.median.shape),
    )
    return promo_forecast_baseline, promo_forecast_known


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare the forecasts
    """)
    return


@app.cell
def _(
    CONTEXT_LENGTH,
    TOTAL_LENGTH,
    promo_actual,
    promo_forecast_baseline,
    promo_forecast_known,
):
    def _():
        promo_baseline_median = (
            promo_forecast_baseline.median[0, 0]
                .detach().cpu().numpy()
        )

        promo_known_median = (
            promo_forecast_known.median[0, 0]
                .detach().cpu().numpy()
        )

        promo_lower = (
            promo_forecast_known.quantile(0.1)[0, 0]
                .detach().cpu().numpy()
        )

        promo_upper = (
            promo_forecast_known.quantile(0.9)[0, 0]
                .detach().cpu().numpy()
        )

        future_t = np.arange(CONTEXT_LENGTH, TOTAL_LENGTH)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(future_t, promo_actual, label="Actual future demand")
        ax.plot(future_t, promo_baseline_median, label="Target-only forecast")
        ax.plot(future_t, promo_known_median, label="Forecast with promotion covariate")

        ax.fill_between(
            future_t,
            promo_lower,
            promo_upper,
            alpha=0.2,
            label="Covariate-aware 80% interval",
        )

        ax.set(
            title="Effect of known-future promotion covariate",
            xlabel="Hour",
            ylabel="Demand",
        )

        ax.legend()
        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare the errors
    """)
    return


@app.cell
def _(
    promo_actual,
    promo_context,
    promo_forecast_baseline,
    promo_forecast_known,
):
    promo_comparison = pd.DataFrame(
        [
            {
                "Approach": "Target only",
                **evaluate_scenario(
                    promo_context,
                    promo_actual,
                    promo_forecast_baseline,
                ),
            },
            {
                "Approach": "Target + known-future promotion",
                **evaluate_scenario(
                    promo_context,
                    promo_actual,
                    promo_forecast_known,
                ),
            },
        ]
    ).set_index("Approach")

    promo_comparison.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Numeric Past-only Covariates
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Test a different situation: a useful numeric feature is available historically, but its future values are not known when the forecast is made.
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH):
    cov_t = np.arange(TOTAL_LENGTH)
    cov_rng = np.random.default_rng(2026)

    # Simulate a persistant upstream sensor signal
    innovations = cov_rng.normal(0, 0.35, size=TOTAL_LENGTH)
    sensor_state = np.zeros(TOTAL_LENGTH)

    for i in range(1, TOTAL_LENGTH):
        sensor_state[i] = 0.95 * sensor_state[i - 1] + innovations[i]

    upstream_sensor = (
        10
        + 1.5 * np.sin(2 * np.pi * cov_t / 72)
        + sensor_state
    )

    # The target responds to the upstream signal with a
    # 24-hour delay
    lag = 24
    lagged_sensor = np.empty(TOTAL_LENGTH)
    lagged_sensor[:lag] = upstream_sensor[0]
    lagged_sensor[lag:] = upstream_sensor[:-lag]

    sensor_noise = cov_rng.normal(0, 0.8, size=TOTAL_LENGTH)

    sensor_target = (
        30
        + 0.015 * cov_t
        + 4.0 * np.sin(2 * np.pi * cov_t / 24)
        + 1.5 * (lagged_sensor - 10)
        + sensor_noise
    )

    sensor_context = sensor_target[:CONTEXT_LENGTH]
    sensor_actual = sensor_target[CONTEXT_LENGTH:]

    print("Historical Observations:", len(sensor_context))
    print("Future Observations:", len(sensor_actual))
    return sensor_actual, sensor_context, upstream_sensor


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the past-known covariate data
    """)
    return


@app.cell
def _(
    CONTEXT_LENGTH,
    TOTAL_LENGTH,
    sensor_actual,
    sensor_context,
    upstream_sensor,
):
    def _():
        def visualize_sensor_forecast():
            fig, (ax, ax_upstream) = plt.subplots(
                2, 1,
                figsize=(12, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            # 1. Top Panel: Target Sensor (Historical vs. Actual Future)
            ax.plot(
                np.arange(CONTEXT_LENGTH),
                sensor_context,
                label="Historical Sensor Target",
                color="royalblue"
            )

            ax.plot(
                np.arange(CONTEXT_LENGTH, TOTAL_LENGTH),
                sensor_actual,
                label="Actual Future Sensor Target",
                color="darkorange"
            )

            ax.axvline(CONTEXT_LENGTH, linestyle="--", color="r", label="Forecast Origin")
            ax.set_ylabel("Sensor Target")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            # 2. Bottom Panel: Upstream Sensor Signal
            ax_upstream.plot(
                np.arange(TOTAL_LENGTH),
                upstream_sensor,
                label="Upstream Sensor Signal",
                color="seagreen",
                alpha=0.8
            )
        
            ax_upstream.axvline(CONTEXT_LENGTH, linestyle="--", color="r", alpha=0.5, label="Forecast Origin")
            ax_upstream.set_ylabel("Upstream")
            ax_upstream.set_xlabel("Hour")
            ax_upstream.legend(loc="upper left")
            ax_upstream.grid(True, alpha=0.3)

            fig.tight_layout()
            return fig

        # Execute the visualization
        fig = visualize_sensor_forecast()
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Forecast with and without the sensor
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, HORIZON, forecaster, sensor_context, upstream_sensor):
    # Forecast using the target history only
    sensor_forecast_baseline = forecaster.forecast(
        sensor_context,
        prediction_length=HORIZON,
    )

    # Only the historical sensor readings are supplied
    past_sensor = upstream_sensor[:CONTEXT_LENGTH][None, None, :]

    assert past_sensor.shape == (1, 1, CONTEXT_LENGTH)

    sensor_forecast_cov = forecaster.forecast(
        sensor_context,
        prediction_length=HORIZON,
        past_covariates=past_sensor,
    )

    print(
        "Past-covariate shape:",
        past_sensor.shape,
    )

    print(
        "Forecast shape:",
        tuple(sensor_forecast_cov.median.shape),
    )
    return sensor_forecast_baseline, sensor_forecast_cov


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare the two forecasts
    """)
    return


@app.cell
def _(
    sensor_actual,
    sensor_context,
    sensor_forecast_baseline,
    sensor_forecast_cov,
):
    sensor_comparison = pd.DataFrame(
        [
            {
                "Approach": "Target Only",
                **evaluate_scenario(
                    sensor_context,
                    sensor_actual,
                    sensor_forecast_baseline,
                ),
            },
            {
                "Approach": "Target + past sensor",
                **evaluate_scenario(
                    sensor_context,
                    sensor_actual,
                    sensor_forecast_cov,
                ),
            },
        ]
    ).set_index("Approach")

    sensor_comparison.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Categorical covariates: day of week
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Test whether Timestral can use a calendar category to anticipate systematic differences in demand.
    """)
    return


@app.cell
def _(CONTEXT_LENGTH):
    DAY_LENGTH = 24
    CATEGORY_HORIZON = 7 * DAY_LENGTH
    CATEGORY_TOTAL_LENGTH = CONTEXT_LENGTH + CATEGORY_HORIZON

    cat_t = np.arange(CATEGORY_TOTAL_LENGTH)
    cat_rng = np.random.default_rng(314)

    # Assume t=0 is Monday: Monday=0,..., Sunday=6
    day_of_week = (cat_t // DAY_LENGTH) % 7

    # Synthetic Demand effect for each weekday
    weekday_effects = np.array([
        0.0, # Monday
        0.5, # Tuesday
        0.8, # Wednesday
        1.0, # Thursday
        1.5, # Friday
        -3.0, # Saturday
        -2.0, # Sunday
    ])

    weekday_effect = weekday_effects[day_of_week]

    categorical_target = (
        30
        + 0.015 * cat_t
        + 4.0 * np.sin(2 * np.pi * cat_t / 24)
        + weekday_effect
        + cat_rng.normal(0, 0.8, size=CATEGORY_TOTAL_LENGTH)
    )

    categorical_context = categorical_target[:CONTEXT_LENGTH]
    categorical_actual = categorical_target[CONTEXT_LENGTH:]

    # Category codes for the entire historical + future timeline
    # Shape: (batch, categorical covariates, time)
    weekday_codes = day_of_week.astype(float)[None, None, :]

    print(f"Context shape: {categorical_context.shape}")
    print(f"Actual future shape: {categorical_actual.shape}")
    print(f"Categorical covariate shape: {weekday_codes.shape}")
    return (
        CATEGORY_HORIZON,
        CATEGORY_TOTAL_LENGTH,
        DAY_LENGTH,
        categorical_actual,
        categorical_context,
        weekday_codes,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Plot the historical and future pattern
    """)
    return


@app.cell
def _(
    CATEGORY_TOTAL_LENGTH,
    CONTEXT_LENGTH,
    DAY_LENGTH,
    categorical_actual,
    categorical_context,
):
    def _():
        fig, ax = plt.subplots(figsize=(12, 5))

        # Convert hour indices to day indices
        day_context = np.arange(CONTEXT_LENGTH) / DAY_LENGTH
        day_actual = np.arange(CONTEXT_LENGTH, CATEGORY_TOTAL_LENGTH) / DAY_LENGTH
        forecast_origin_day = CONTEXT_LENGTH / DAY_LENGTH

        ax.plot(
            day_context,
            categorical_context,
            label="Historical demand",
        )

        ax.plot(
            day_actual,
            categorical_actual,
            label="Actual future demand",
        )

        ax.axvline(
            forecast_origin_day,
            linestyle="--",
            color='r',
            label="Forecast Origin",
        )

        ax.set(
            title="Synthetic demand with weekday effects",
            xlabel="Day",
            ylabel="Demand",
        )

        ax.legend()
        fig.tight_layout()
        return fig

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare the forecasts with and without the category effect
    """)
    return


@app.cell
def _(CATEGORY_HORIZON, categorical_context, forecaster, weekday_codes):
    # Target-only forecast
    weekday_forecast_baseline = forecaster.forecast(
        categorical_context,
        prediction_length=CATEGORY_HORIZON,
    )

    # Forecast with a known-future categorical feature
    weekday_forecast_known = forecaster.forecast(
        categorical_context,
        prediction_length=CATEGORY_HORIZON,
        categorical_known_covariates=weekday_codes,
    )

    print(
        "Target-only median shape:",
        tuple(weekday_forecast_baseline.median.shape),
    )

    print(
        "Categorical median shape:",
        tuple(weekday_forecast_known.median.shape),
    )
    return weekday_forecast_baseline, weekday_forecast_known


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the forecasts
    """)
    return


@app.cell
def _(
    CATEGORY_TOTAL_LENGTH,
    CONTEXT_LENGTH,
    categorical_actual,
    weekday_forecast_baseline,
    weekday_forecast_known,
):
    def _():
        weekday_baseline_median = (
            weekday_forecast_baseline.median[0, 0]
            .detach().cpu().numpy()
        )

        weekly_known_median = (
            weekday_forecast_known.median[0, 0]
            .detach().cpu().numpy()
        )

        future_t = np.arange(CONTEXT_LENGTH, CATEGORY_TOTAL_LENGTH)

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(
            future_t,
            categorical_actual,
            label="Actual future demand",
        )

        ax.plot(
            future_t,
            weekday_baseline_median,
            label="Target Only",
        )

        ax.plot(
            future_t,
            weekly_known_median,
            label="Target + Weekday category",
        )

        ax.set(
            title="Weekly demand forecast",
            xlabel="Hour",
            ylabel="Demand",
        )

        ax.legend()
        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Evaluate the metrics
    """)
    return


@app.cell
def _(
    DAY_LENGTH,
    categorical_actual,
    categorical_context,
    weekday_forecast_baseline,
    weekday_forecast_known,
):
    def _():
        weekly_naive = np.tile(
            categorical_context[-7 * DAY_LENGTH:],
            int(np.ceil(len(categorical_actual) / (7 * DAY_LENGTH))),
        )[:len(categorical_actual)]

        weekly_naive_mae = np.mean(
            np.abs(categorical_actual - weekly_naive)
        )

        weekday_results = []

        for name, forecast in [
            ("Target only", weekday_forecast_baseline),
            ("Target + weekday category", weekday_forecast_known),
        ]:
            row = {
                "Approach": name,
                **evaluate_scenario(
                    categorical_context,
                    categorical_actual,
                    forecast,
                ),
            }

            row["Weekly seasonal naive MAE"] = weekly_naive_mae
            weekday_results.append(row)

        weekday_comparison = (
            pd.DataFrame(weekday_results)
            .set_index("Approach")
        )
        return weekday_comparison.round(4)


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Known-past-only categorical covariates
    """)
    return


@app.cell
def _(CONTEXT_LENGTH, TOTAL_LENGTH):
    state_t = np.arange(TOTAL_LENGTH)
    state_rng = np.random.default_rng(2718)

    # 0 = idle, 1 = running.
    machine_state = np.zeros(TOTAL_LENGTH, dtype=float)

    machine_state[120:216] = 1
    machine_state[216:288] = 0
    machine_state[288:] = 1

    # Operating state affects both the level and the amplitude
    # of the daily pattern.
    daily_amplitude = np.where(machine_state == 1, 5.0, 2.0)

    state_target = (
        30
        + 0.015 * state_t
        + daily_amplitude * np.sin(2 * np.pi * state_t / 24)
        + 5.0 * machine_state
        + state_rng.normal(0, 0.8, size=TOTAL_LENGTH)
    )

    state_context = state_target[:CONTEXT_LENGTH]
    state_actual = state_target[CONTEXT_LENGTH:]

    # Only historical operating-state codes are supplied.
    past_state_codes = machine_state[:CONTEXT_LENGTH][None, None, :]

    print("Historical target:", state_context.shape)
    print("Future target:", state_actual.shape)
    print("Past-only category:", past_state_codes.shape)
    return machine_state, past_state_codes, state_actual, state_context


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the data
    """)
    return


@app.cell
def _(
    CONTEXT_LENGTH,
    TOTAL_LENGTH,
    machine_state,
    state_actual,
    state_context,
):
    def _():
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 7),
            sharex=True,
        )

        ax1.plot(
            np.arange(CONTEXT_LENGTH),
            state_context,
            label="Historical target",
        )
        ax1.plot(
            np.arange(CONTEXT_LENGTH, TOTAL_LENGTH),
            state_actual,
            label="Actual future target",
        )
        ax1.axvline(CONTEXT_LENGTH, linestyle="--", label="Forecast origin")
        ax1.set_ylabel("Target")
        ax1.legend()

        ax2.step(
            np.arange(CONTEXT_LENGTH),
            machine_state[:CONTEXT_LENGTH],
            where="post",
            label="Historical operating state",
        )
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("State (0/1)")
        ax2.legend()

        fig.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compare the forecasts
    """)
    return


@app.cell
def _(HORIZON, forecaster, past_state_codes, state_actual, state_context):
    # Target-only forecast.
    state_forecast_baseline = forecaster.forecast(
        state_context,
        prediction_length=HORIZON,
    )

    # Target + historical categorical covariate.
    state_forecast_cat = forecaster.forecast(
        state_context,
        prediction_length=HORIZON,
        categorical_past_covariates=past_state_codes,
    )

    state_comparison = pd.DataFrame(
        [
            {
                "Approach": "Target only",
                **evaluate_scenario(
                    state_context,
                    state_actual,
                    state_forecast_baseline,
                ),
            },
            {
                "Approach": "Target + past operating state",
                **evaluate_scenario(
                    state_context,
                    state_actual,
                    state_forecast_cat,
                ),
            },
        ]
    ).set_index("Approach")

    state_comparison.round(4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
