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
    seasonal_scale = np.mean(
        np.abs(context[period:] - context[:-period])
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
def _(TOTAL_LENGTH, make_synthetic_series):
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
    return


if __name__ == "__main__":
    app.run()
