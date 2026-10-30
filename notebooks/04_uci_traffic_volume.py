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
def _():
    return


if __name__ == "__main__":
    app.run()
