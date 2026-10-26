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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
