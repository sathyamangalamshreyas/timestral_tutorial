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
    #### Load the pre-trained Timestral 209M model
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
    #### Generate simple synthetic time-series data
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(seed=42)

    context_length = 24 * 14 # 14 days of hourly observations

    horizon = 48 # Forecast for the next 48 hours

    t = np.arange(context_length + horizon)

    trend = 0.015 * t

    daily_seasonality = 4.0 * np.sin(2 * np.pi * t / 24)

    noise = rng.normal(0, 0.8, size=t.size)

    series = 30 + trend + daily_seasonality + noise

    context = series[:context_length]

    actual_future = series[context_length:]

    print(f"Length Historical Observations: {len(context)}")
    print(f"Length Future Observations: {len(actual_future)}")
    return actual_future, context, context_length, horizon


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generate the Forecast for next 48 hours
    """)
    return


@app.cell
def _(context, forecaster, horizon):
    forecast = forecaster.forecast(
        context=context,
        prediction_length=horizon,
    )

    print(f"Quantile Levels: {forecast.quantile_levels}")
    print(f"(Num Series, Num Variates, Num Steps, Num Quantiles): {tuple(forecast.quantiles.shape)}")
    print(f"(Num Series, Num Median Predictions per Future Hour): {tuple(forecast.median.shape)}")
    return (forecast,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Visualize the Forecast
    """)
    return


@app.cell
def _(actual_future, context, context_length, forecast, horizon):
    median = forecast.median[0, 0].detach().cpu().numpy()
    lower = forecast.quantile(0.1)[0, 0].detach().cpu().numpy()
    upper = forecast.quantile(0.9)[0, 0].detach().cpu().numpy()

    history_x = np.arange(context_length)
    future_x = np.arange(context_length, context_length + horizon)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history_x, context, label="Historical Observations")
    ax.plot(future_x, actual_future, label="Actual Future")
    ax.plot(future_x, median, label="Timestral Median Forecast")

    ax.fill_between(
        future_x,
        lower,
        upper,
        alpha=0.2,
        label="10th-90th percentile interval",
    )

    ax.axvline(context_length - 0.5, linestyle="--", color='r', label="Forecast Origin")

    ax.set(
        title="Timestral: 48-hour Forecast",
        xlabel="Hour",
        ylabel="Synthetic Value",
    )

    ax.legend()
    fig.tight_layout()
    fig
    return lower, median, upper


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Compute:
    - MAE - Mean Absolute Error
    - RMSE - Root Mean Squared Error
    - MAPE - Mean Absolute Percentage Error
    - MASE - Mean Absolute Scaled Error
    """)
    return


@app.cell
def _(actual_future, context, median):
    y_true = np.asarray(actual_future, dtype=float)
    y_pred = np.asarray(median, dtype=float)

    errors = y_true - y_pred

    absolute_errors = np.abs(errors)

    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(errors**2))

    # MAPE is undefined when actual values are zero
    # Avoid reporting it when values are effectively zero
    if np.all(np.abs(y_true)>1e-8):
        mape = np.mean(absolute_errors/np.abs(y_true)) * 100
    else:
        mape = np.nan

    # sMAPE: Handle the case where both acutal and predicted values are zero
    smape_denominator = np.abs(y_true) + np.abs(y_pred)

    smape_values = np.divide(
        200 * absolute_errors,
        smape_denominator,
        out=np.zeros_like(absolute_errors),
        where=smape_denominator > 0,
    )
    smape = np.mean(smape_values)

    # Seasonal-naive baseline: repeat the latest 24 hours
    seasonal_period = 24

    seasonal_naive = np.tile(
        context[-seasonal_period:],
        int(np.ceil(len(y_true) / seasonal_period)),
    )[:len(y_true)]

    seasonal_naive_mae = np.mean(
        np.abs(y_true - seasonal_naive)
    )

    # MASE scaling comes from historical observations only
    seasonal_scale = np.mean(
        np.abs(context[seasonal_period:] - context[:-seasonal_period])
    )

    mase = mae / seasonal_scale if seasonal_scale > 0 else np.nan

    metrics = pd.DataFrame(
        {
            "Metric": [
                "MAE - Timestral",
                "RMSE - Timestral",
                "MAPE - Timestral",
                "sMAPE - Timestral",
                "MASE - Timestral",
                "MAE - Seasonal naive baseline",
            ],
            "Value": [
                mae,
                rmse,
                mape,
                smape,
                mase,
                seasonal_naive_mae,
            ],
        }
    )

    metrics.round(4)
    return (y_true,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Evaluate the Probabilistic Forecast

    - **Coverage**: What percentage of the actual future values fell between the 10th and 90th percentile predictions?
    - **Mean interval width**: How wide was that interval on average? A very wide interval can achieve high coverage without being particularly informative.
    - **Mean pinball loss**: How well did Timestral predict its entire collection of quantiles? This metric rewards both accuracy and appropriate quantile placement. Lower is better.
    """)
    return


@app.cell
def _(actual_future, forecast, lower, upper, y_true):
    coverage_80 = np.mean(
        (actual_future >= lower) & (actual_future <= upper)
    ) * 100

    mean_interval_width = np.mean(upper - lower)

    # Shape: (48 forecast steps, 21 quantiles)
    predicted_quantiles = (
        forecast.quantiles[0, 0].detach().cpu().numpy()
    )

    quantile_levels = np.asarray(forecast.quantile_levels)

    # Positive error means the acutal value exceeds the quantile forecast
    quantile_errors = (
        y_true[:, None] - predicted_quantiles
    )

    pinball_losses = np.maximum(
        quantile_levels[None, :] * quantile_errors,
        (quantile_levels[None, :] - 1) * quantile_errors,
    )

    mean_pinball_loss = np.mean(pinball_losses)

    probabalistic_metrics = pd.DataFrame(
        {
            "Metric": [
                "80% interval empirical coverage",
                "Mean 80% interval width",
                "Mean pinball loss across 21 quantiles",
            ],
            "Values": [
                coverage_80,
                mean_interval_width,
                mean_pinball_loss,
            ],
            "Unit": [
                "%",
                "Target units",
                "Target units"
            ],
        }
    )

    probabalistic_metrics.round(4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
