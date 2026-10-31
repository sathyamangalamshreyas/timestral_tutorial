import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import pandas as pd

    from data_profiling import ProfileReport
    import pygwalker as pyg

    from skimpy import skim


@app.cell
def _():
    # read the csv file, convert date_time to datetime and re-order columns

    df = pd.read_csv("data/traffic_df_raw.csv", sep=",")

    df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")

    col_name = df.columns[-2]

    df.insert(0, col_name, df.pop(col_name))


    df.head()
    return (df,)


@app.cell
def _(df):
    df['year'] = df['date_time'].dt.year

    df['month'] = df['date_time'].dt.month

    df.head()
    return


@app.cell
def _(df):
    # Profile using data_profiling

    df_profile = ProfileReport(df=df)

    df_profile.to_file("traffic_data.html")
    return (df_profile,)


@app.cell
def _(df_profile):
    df_profile.to_notebook_iframe()
    return


@app.cell
def _(df):
    # Time-series report
    ts_traffic_profile = ProfileReport(
        df=df,
        title="Traffic time-series report",
        tsmode=True,
        sortby="date_time"
    )

    ts_traffic_profile.to_file("time_series_report.html")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### PygWalker profiling
    """)
    return


@app.cell
def _(df):
    pyg.walk(df)
    return


@app.cell
def _(df):
    skim(df_in=df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
