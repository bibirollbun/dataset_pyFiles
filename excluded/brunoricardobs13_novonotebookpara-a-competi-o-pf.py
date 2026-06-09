import os
import numpy as np
import pandas as pd
# ===== List input files =====
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ===== Imports =====
import time
import os
import psutil
import polars as pl
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np






base = "/kaggle/input/predict-energy-behavior-of-prosumers"
files = {
    "train": f"{base}/train.csv",
    "client": f"{base}/client.csv",
    "gas_prices": f"{base}/gas_prices.csv",
    "electricity_prices": f"{base}/electricity_prices.csv",
    "weather_station_mapping": f"{base}/weather_station_to_county_mapping.csv",
    "historical_weather": f"{base}/historical_weather.csv",
    "forecast_weather": f"{base}/forecast_weather.csv",
    "sample_submission": f"{base}/example_test_files/sample_submission.csv",
    "revealed_targets": f"{base}/example_test_files/revealed_targets.csv",
    "test": f"{base}/example_test_files/test.csv",
}


def summarize(df: pd.DataFrame, name: str):
    print(f"\n=== Dataset: {name} ===")
    print("Schema & unique dtypes:", df.dtypes.unique().tolist())
    print("Shape:", df.shape)
    nonzero = df.isnull().sum()[lambda x: x > 0]
    if not nonzero.empty:
        print("\nNull counts (non-zero):")
        print(nonzero)


def summarize_train(df: pd.DataFrame):
    summarize(df, "train")
    print("\nUnique counts:")
    print(df.nunique())
    by_cons = (
        df.groupby("is_consumption")["target"]
        .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    by_cons.index = by_cons.index.map({0: "ProduÃ§Ã£o", 1: "Consumo"})
    print("\nNumeric summary of target by is_consumption:")
    print(by_cons)


def summarize_client(df: pd.DataFrame):
    summarize(df, "client")
    print("\nNumeric summary (eic_count, installed_capacity):")
    print(df[["eic_count", "installed_capacity"]].describe().T)


def summarize_gas(df: pd.DataFrame):
    summarize(df, "gas_prices")
    print("\nNumeric summary (gas prices):")
    print(df[["lowest_price_per_mwh", "highest_price_per_mwh"]].describe().T)


def summarize_elec(df: pd.DataFrame):
    summarize(df, "electricity_prices")
    print("\nNumeric summary (euros_per_mwh):")
    print(df["euros_per_mwh"].describe())


def summarize_ws(df: pd.DataFrame):
    print("\n=== Dataset: weather_station_mapping ===")
    print("Schema & dtypes:")
    print(df.dtypes)


def summarize_hist(df: pd.DataFrame):
    print("\n=== Dataset: historical_weather ===")
    print("Unique dtypes in this dataset:")
    print(df.dtypes.unique().tolist())


def summarize_fcast(df: pd.DataFrame):
    print("\n=== Dataset: forecast_weather ===")
    print("Schema & unique dtypes:", df.dtypes.unique().tolist())
    nonzero = df.isnull().sum()[lambda x: x > 0]
    if not nonzero.empty:
        print("\nNull counts (non-zero):")
        print(nonzero)
    print("\nVariation of temperature:")
    print(df["temperature"].agg(["count", "mean", "std"]))


if __name__ == "__main__":
    df_train = pd.read_csv(files["train"], parse_dates=["datetime"])
    df_client = pd.read_csv(files["client"], parse_dates=["date"])
    df_gas = pd.read_csv(files["gas_prices"], parse_dates=["forecast_date", "origin_date"])
    df_elec = pd.read_csv(
        files["electricity_prices"],
        parse_dates=["forecast_date", "origin_date"],
    )
    df_ws = pd.read_csv(files["weather_station_mapping"])
    df_hist = pd.read_csv(files["historical_weather"], parse_dates=["datetime"])
    df_fcast = pd.read_csv(
        files["forecast_weather"],
        parse_dates=["origin_datetime", "forecast_datetime"],
    )

    summarize_train(df_train)
    summarize_client(df_client)
    summarize_gas(df_gas)
    summarize_elec(df_elec)
    summarize_ws(df_ws)
    summarize_hist(df_hist)
    summarize_fcast(df_fcast)



import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict

# Constantes
PRODUCT_NAMES: Dict[int, str] = {
    0: "Combined",
    1: "Fixed",
    2: "General service",
    3: "Spot",
}

# Carregamento de Dados
def load_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["date"] = df["datetime"].dt.date
    return df

def load_client(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])

def load_gas(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["forecast_date", "origin_date"])

def load_electricity(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["forecast_date", "origin_date"])

def load_hist_weather(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"])

# AgregaÃ§Ã£o
def get_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .groupby(["date", "is_consumption"])["target"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={False: "ProduÃ§Ã£o", True: "Consumo"})
        .reset_index()
    )

def get_by_county(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_county = (
        df
        .groupby(["county", "is_consumption"])["target"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={False: "ProduÃ§Ã£o", True: "Consumo"})
        .reset_index()
    )

    by_county_prod = (
        df
        .groupby(["county", "product_type", "is_consumption"])["target"]
        .sum()
        .reset_index()
        .pivot_table(
            index=["county", "product_type"],
            columns="is_consumption",
            values="target",
            fill_value=0
        )
        .rename(columns={False: "ProduÃ§Ã£o", True: "Consumo"})
        .reset_index()
    )
    by_county_prod["product_name"] = by_county_prod["product_type"].map(PRODUCT_NAMES)
    return by_county, by_county_prod

# Plotagem principal
def plot_dual_axis(daily: pd.DataFrame, y_max: int = 1_200_000) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=daily.date, y=daily.ProduÃ§Ã£o, name="ProduÃ§Ã£o"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=daily.date, y=daily.Consumo, name="Consumo", line_color="firebrick"),
        secondary_y=True
    )
    fig.update_layout(
        title="EvoluÃ§Ã£o DiÃ¡ria: ProduÃ§Ã£o vs Consumo",
        xaxis_title="Data",
        legend=dict(x=0.01, y=0.99),
        template="simple_white"
    )
    fig.update_yaxes(
        title_text="ProduÃ§Ã£o (MWh)",
        range=[0, y_max],
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Consumo   (MWh)",
        range=[0, y_max],
        secondary_y=True
    )
    return fig

def plot_scatter_trend(daily: pd.DataFrame) -> go.Figure:
    df = daily.dropna(subset=["ProduÃ§Ã£o", "Consumo"])
    fig = px.scatter(
        df,
        x="ProduÃ§Ã£o",
        y="Consumo",
        trendline="ols",
        trendline_color_override="black",
        title="RelaÃ§Ã£o DiÃ¡ria: ProduÃ§Ã£o vs Consumo",
        labels={"ProduÃ§Ã£o": "ProduÃ§Ã£o (MWh)", "Consumo": "Consumo (MWh)"}
    )
    corr = df.ProduÃ§Ã£o.corr(df.Consumo)
    fig.update_layout(
        title=f"RelaÃ§Ã£o DiÃ¡ria: ProduÃ§Ã£o vs Consumo<br>CorrelaÃ§Ã£o: {corr:.3f}",
        template="simple_white"
    )
    return fig

def plot_by_county(by_county: pd.DataFrame) -> go.Figure:
    dfm = by_county.melt(
        id_vars="county",
        value_vars=["ProduÃ§Ã£o", "Consumo"],
        var_name="Tipo",
        value_name="MWh"
    )
    fig = px.bar(
        dfm,
        x="county",
        y="MWh",
        color="Tipo",
        barmode="group",
        title="ProduÃ§Ã£o vs Consumo por Condado",
        labels={"county": "Condado", "MWh": "Total (MWh)"},
        template="simple_white"
    )
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return fig

def plot_by_county_prod(by_county_prod: pd.DataFrame) -> go.Figure:
    dfm = by_county_prod.melt(
        id_vars=["county", "product_name"],
        value_vars=["ProduÃ§Ã£o", "Consumo"],
        var_name="Tipo",
        value_name="MWh"
    )
    fig = px.bar(
        dfm,
        x="county",
        y="MWh",
        color="Tipo",
        facet_col="product_name",
        facet_col_wrap=2,
        barmode="group",
        title="ProduÃ§Ã£o vs Consumo por Condado e Tipo de Produto",
        template="simple_white"
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    return fig

# GrÃ¡ficos avanÃ§ados
def plot_client_relationship_grid(client: pd.DataFrame, train: pd.DataFrame) -> go.Figure:
    summary = (
        train
        .groupby(["data_block_id", "is_consumption"])["target"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={False: "ProduÃ§Ã£o", True: "Consumo"})
    )
    df = client.merge(summary, on="data_block_id")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "EIC Count vs ProduÃ§Ã£o",
            "EIC Count vs Consumo",
            "Installed Cap vs ProduÃ§Ã£o",
            "Installed Cap vs Consumo"
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )

    fig.add_trace(
        go.Scatter(x=df["eic_count"], y=df["ProduÃ§Ã£o"], mode="markers"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["eic_count"], y=df["Consumo"], mode="markers", marker_color="firebrick"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df["installed_capacity"], y=df["ProduÃ§Ã£o"], mode="markers", marker_symbol="diamond"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["installed_capacity"],
            y=df["Consumo"],
            mode="markers",
            marker_symbol="diamond",
            marker_color="firebrick"
        ),
        row=2, col=2
    )

    axes = [
        ("EIC Count", "ProduÃ§Ã£o (MWh)"),
        ("EIC Count", "Consumo (MWh)"),
        ("Installed Capacity", "ProduÃ§Ã£o (MWh)"),
        ("Installed Capacity", "Consumo (MWh)"),
    ]
    idx = 0
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=axes[idx][0], row=row, col=col)
            fig.update_yaxes(title_text=axes[idx][1], row=row, col=col)
            idx += 1

    fig.update_layout(
        title="RelaÃ§Ãµes Cliente vs Energia",
        height=700,
        width=1000,
        template="simple_white"
    )
    return fig

def plot_price_histograms(gas: pd.DataFrame, elec: pd.DataFrame) -> go.Figure:
    g = gas.copy()
    g["mean_price_per_mwh"] = (g.lowest_price_per_mwh + g.highest_price_per_mwh) / 2

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["GÃ¡s: PreÃ§o MÃ©dio (â‚¬/MWh)", "Eletricidade: PreÃ§o (â‚¬/MWh)"],
        horizontal_spacing=0.15
    )
    fig.add_trace(
        go.Histogram(x=g.mean_price_per_mwh, nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=elec.euros_per_mwh, nbinsx=50, marker_color="firebrick"),
        row=1, col=2
    )
    fig.update_layout(
        title="DistribuiÃ§Ã£o de PreÃ§os de Energia",
        height=450,
        width=900,
        template="simple_white"
    )
    fig.update_xaxes(title_text="â‚¬ï¼�MWh", row=1, col=1)
    fig.update_xaxes(title_text="â‚¬ï¼�MWh", row=1, col=2)
    fig.update_yaxes(title_text="Contagem", row=1, col=1)
    fig.update_yaxes(title_text="Contagem", row=1, col=2)
    return fig

def plot_weather_correlation(daily: pd.DataFrame, hist: pd.DataFrame) -> go.Figure:
    tmp = hist.copy()
    tmp["date"] = tmp.datetime.dt.date
    weather = tmp.groupby("date")["temperature"].mean().rename("Temp_MÃ©dia").reset_index()
    df = daily.merge(weather, on="date")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["ProduÃ§Ã£o vs Temp MÃ©dia", "Consumo vs Temp MÃ©dia"],
        horizontal_spacing=0.12
    )
    fig.add_trace(
        go.Scatter(x=df["Temp_MÃ©dia"], y=df.ProduÃ§Ã£o, mode="markers", marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Temp_MÃ©dia"], y=df.Consumo, mode="markers", marker=dict(size=4, color="firebrick")),
        row=1, col=2
    )

    t1 = px.scatter(df, x="Temp_MÃ©dia", y="ProduÃ§Ã£o", trendline="ols").data[1]
    t2 = px.scatter(
        df, x="Temp_MÃ©dia", y="Consumo", trendline="ols", trendline_color_override="firebrick"
    ).data[1]
    fig.add_trace(t1, row=1, col=1)
    fig.add_trace(t2, row=1, col=2)

    corr_p = df.ProduÃ§Ã£o.corr(df["Temp_MÃ©dia"])
    corr_c = df.Consumo.corr(df["Temp_MÃ©dia"])
    fig.add_annotation(
        text=f"corr={corr_p:.3f}",
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.05,
        row=1,
        col=1,
        showarrow=False
    )
    fig.add_annotation(
        text=f"corr={corr_c:.3f}",
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.05,
        row=1,
        col=2,
        showarrow=False
    )

    fig.update_layout(
        title="Temperatura MÃ©dia DiÃ¡ria vs Energia",
        height=450,
        width=950,
        showlegend=False,
        template="simple_white"
    )
    fig.update_xaxes(title_text="Temp MÃ©dia (K)", row=1, col=1)
    fig.update_xaxes(title_text="Temp MÃ©dia (K)", row=1, col=2)
    fig.update_yaxes(title_text="ProduÃ§Ã£o (MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Consumo   (MWh)", row=1, col=2)
    return fig

# Fluxo Principal
def main():
    paths = {
        'train': '/kaggle/input/predict-energy-behavior-of-prosumers/train.csv',
        'client': '/kaggle/input/predict-energy-behavior-of-prosumers/client.csv',
        'gas': '/kaggle/input/predict-energy-behavior-of-prosumers/gas_prices.csv',
        'electricity': '/kaggle/input/predict-energy-behavior-of-prosumers/electricity_prices.csv',
        'hist_weather': '/kaggle/input/predict-energy-behavior-of-prosumers/historical_weather.csv',
    }

    train = load_train(paths['train'])
    client = load_client(paths['client'])
    gas = load_gas(paths['gas'])
    elec = load_electricity(paths['electricity'])
    hist = load_hist_weather(paths['hist_weather'])

    daily = get_daily(train)
    by_c, by_cp = get_by_county(train)

    plot_dual_axis(daily).show()
    plot_scatter_trend(daily).show()
    plot_by_county(by_c).show()
    plot_by_county_prod(by_cp).show()

    plot_client_relationship_grid(client, train).show()
    plot_price_histograms(gas, elec).show()
    plot_weather_correlation(daily, hist).show()

if __name__ == "__main__":
    main()



import os
import polars as pl

# DefiniÃ§Ãµes de caminho e colunas
root = "/kaggle/input/predict-energy-behavior-of-prosumers"

data_cols = [
    "target", "county", "is_business", "product_type",
    "is_consumption", "datetime", "row_id"
]
client_cols = [
    "product_type", "county", "eic_count",
    "installed_capacity", "is_business", "date"
]
gas_prices_cols = [
    "forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"
]
electricity_prices_cols = [
    "forecast_date", "euros_per_mwh"
]
forecast_weather_cols = [
    "latitude", "longitude", "hours_ahead", "temperature", "dewpoint",
    "cloudcover_high", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
    "10_metre_u_wind_component", "10_metre_v_wind_component",
    "forecast_datetime", "direct_solar_radiation",
    "surface_solar_radiation_downwards", "snowfall", "total_precipitation"
]
historical_weather_cols = [
    "datetime", "temperature", "dewpoint", "rain", "snowfall",
    "surface_pressure", "cloudcover_total", "cloudcover_low", "cloudcover_mid",
    "cloudcover_high", "windspeed_10m", "winddirection_10m",
    "shortwave_radiation", "direct_solar_radiation", "diffuse_radiation",
    "latitude", "longitude"
]
location_cols = ["longitude", "latitude", "county"]
target_cols = [
    "target", "county", "is_business",
    "product_type", "is_consumption", "datetime"
]



# Carregamento seletivo com Polars
df_data = pl.read_csv(
    os.path.join(root, "train.csv"),
    columns=data_cols,
    try_parse_dates=True
)
df_client = pl.read_csv(
    os.path.join(root, "client.csv"),
    columns=client_cols,
    try_parse_dates=True
)
df_gas_prices = pl.read_csv(
    os.path.join(root, "gas_prices.csv"),
    columns=gas_prices_cols,
    try_parse_dates=True
)
df_elec_prices = pl.read_csv(
    os.path.join(root, "electricity_prices.csv"),
    columns=electricity_prices_cols,
    try_parse_dates=True
)
df_fcast_weather = pl.read_csv(
    os.path.join(root, "forecast_weather.csv"),
    columns=forecast_weather_cols,
    try_parse_dates=True
)
df_hist_weather = pl.read_csv(
    os.path.join(root, "historical_weather.csv"),
    columns=historical_weather_cols,
    try_parse_dates=True
)
df_loc_map = pl.read_csv(
    os.path.join(root, "weather_station_to_county_mapping.csv"),
    columns=location_cols,
    try_parse_dates=True
)

df_target = df_data.select(target_cols)



# ConfirmaÃ§Ã£o dos schemas
print("df_data schema:", df_data.schema)
print("df_client schema:", df_client.schema)
print("df_gas_prices schema:", df_gas_prices.schema)
print("df_elec_prices schema:", df_elec_prices.schema)
print("df_fcast_weather schema:", df_fcast_weather.schema)
print("df_hist_weather schema:", df_hist_weather.schema)
print("df_loc_map schema:", df_loc_map.schema)
print("df_target schema:", df_target.schema)



def generate_features2(
        df_data, 
        df_client, 
        df_gas_prices, 
        df_electricity_prices, 
        df_forecast_weather, 
        df_historical_weather, 
        df_weather_station_to_county_mapping, 
        df_target
):
    df_data = (
        df_data
        .with_columns(
            pl.col("datetime").cast(pl.Date).alias("date"),
        )
    )
    
    df_gas_prices = (
        df_gas_prices
        .rename({"forecast_date": "date"})
    )
    
    df_electricity_prices = (
        df_electricity_prices
        .rename({"forecast_date": "datetime"})
    )
    
    df_weather_station_to_county_mapping = (
        df_weather_station_to_county_mapping
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32)
        )
    )
    
    # sum of all product_type targets related to ["datetime", "county", "is_business", "is_consumption"]
    df_target_all_type_sum = (
        df_target
        .group_by(["datetime", "county", "is_business", "is_consumption"]).sum()
        .drop("product_type")
    )
    
    df_forecast_weather = (
        df_forecast_weather
        .rename({"forecast_datetime": "datetime"})
        .filter(pl.col("hours_ahead") >= 24) # we don't need forecast for today
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
            # datetime for forecast in a different timezone
            pl.col('datetime').dt.replace_time_zone(None).cast(pl.Datetime("us"))
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
    )
    
    df_historical_weather = (
        df_historical_weather
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
#            pl.col("datetime") + pl.duration(hours=37)
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
    )
    
    # creating average forecast characteristics for all weather stations
    df_forecast_weather_date = (
        df_forecast_weather
        .group_by("datetime").mean()
        .drop("county")
    )
    
    # creating average forecast characteristics for weather stations related to county
    df_forecast_weather_local = (
        df_forecast_weather
        .filter(pl.col("county").is_not_null())
        .group_by("county", "datetime").mean()
    )
    
    # creating average historical characteristics for all weather stations
    df_historical_weather_date = (
        df_historical_weather
        .group_by("datetime").mean()
        .drop("county")
    )
    
    # creating average historical characteristics for weather stations related to county
    df_historical_weather_local = (
        df_historical_weather
        .filter(pl.col("county").is_not_null())
        .group_by("county", "datetime").mean()
    )
    
    df_data = (
        df_data
        # pl.duration(days=1) shifts datetime to join lag features (usually we join last available values)
        .join(df_gas_prices.with_columns((pl.col("date") + pl.duration(days=1)).cast(pl.Date)), on="date", how="left")
        .join(df_client.with_columns((pl.col("date") + pl.duration(days=2)).cast(pl.Date)), on=["county", "is_business", "product_type", "date"], how="left")
        .join(df_electricity_prices.with_columns(pl.col("datetime") + pl.duration(days=1)), on="datetime", how="left")
        
        # lag forecast_weather features (24 hours * days)
        .join(df_forecast_weather_date, on="datetime", how="left", suffix="_fd")
        .join(df_forecast_weather_local, on=["county", "datetime"], how="left", suffix="_fl")
        .join(df_forecast_weather_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_fd_7d")
        .join(df_forecast_weather_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_fl_7d")

        # lag historical_weather features (24 hours * days)
        .join(df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(days=2)), on="datetime", how="left", suffix="_hd_2d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(days=2)), on=["county", "datetime"], how="left", suffix="_hl_2d")
        .join(df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_hd_7d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_hl_7d")
        
        # lag target features (24 hours * days)
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=2)).rename({"target": "target_1"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=3)).rename({"target": "target_2"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=4)).rename({"target": "target_3"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=5)).rename({"target": "target_4"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=6)).rename({"target": "target_5"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=7)).rename({"target": "target_6"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=14)).rename({"target": "target_7"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=2)).rename({"target": "target_1"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=3)).rename({"target": "target_2"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=7)).rename({"target": "target_6"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=14)).rename({"target": "target_7"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        
        
        .with_columns(
            pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.year().alias("year"),
        )
        
        .with_columns(
            pl.concat_str("county", "is_business", "product_type", "is_consumption", separator="_").alias("segment"),
        )
        
        # cyclical features encoding https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
        .with_columns(
            (np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"),
            (np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"),
            (np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"),
            (np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"),
        )
        
        .with_columns(
            pl.col(pl.Float64).cast(pl.Float32),
        )
        
        .drop("date", "hour", "dayofyear")
    )
    
    return df_data



def generate_features(
        df_data, 
        df_client, 
        df_gas_prices, 
        df_electricity_prices, 
        df_forecast_weather, 
        df_historical_weather, 
        df_weather_station_to_county_mapping, 
        df_target
):
    df_data = (
        df_data
        .with_columns(
            pl.col("datetime").cast(pl.Date).alias("date"),
        )
    )
    
    df_gas_prices = (
        df_gas_prices
        .rename({"forecast_date": "date"})
    )
    
    df_electricity_prices = (
        df_electricity_prices
        .rename({"forecast_date": "datetime"})
    )
    
    df_weather_station_to_county_mapping = (
        df_weather_station_to_county_mapping
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32)
        )
    )
    
    df_target_all_type_sum = (
        df_target
        .group_by(["datetime", "county", "is_business", "is_consumption"])
        .sum()
        .drop("product_type")
    )
    
    df_forecast_weather = (
        df_forecast_weather
        .rename({"forecast_datetime": "datetime"})
        .filter(pl.col("hours_ahead") >= 24)
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
            pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime("us"))
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
    )
    
    df_historical_weather = (
        df_historical_weather
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
    )
    
    df_forecast_weather_date = (
        df_forecast_weather.group_by("datetime").mean().drop("county")
    )
    df_forecast_weather_local = (
        df_forecast_weather.filter(pl.col("county").is_not_null())
                          .group_by("county", "datetime").mean()
    )
    df_historical_weather_date = (
        df_historical_weather.group_by("datetime").mean().drop("county")
    )
    df_historical_weather_local = (
        df_historical_weather.filter(pl.col("county").is_not_null())
                             .group_by("county", "datetime").mean()
    )
    
    df_data = (
        df_data
        .join(df_gas_prices.with_columns((pl.col("date") + pl.duration(days=1)).cast(pl.Date)),
              on="date", how="left")
        .join(df_client.with_columns((pl.col("date") + pl.duration(days=2)).cast(pl.Date)),
              on=["county","is_business","product_type","date"], how="left")
        .join(df_electricity_prices.with_columns(pl.col("datetime") + pl.duration(days=1)),
              on="datetime", how="left")
        
        # lag forecast weather
        .join(df_forecast_weather_date, on="datetime", how="left", suffix="_fd")
        .join(df_forecast_weather_local, on=["county","datetime"], how="left", suffix="_fl")
        .join(df_forecast_weather_date.with_columns(pl.col("datetime")+pl.duration(days=7)),
              on="datetime", how="left", suffix="_fd_7d")
        .join(df_forecast_weather_local.with_columns(pl.col("datetime")+pl.duration(days=7)),
              on=["county","datetime"], how="left", suffix="_fl_7d")

        # lag historical weather
        .join(df_historical_weather_date.with_columns(pl.col("datetime")+pl.duration(days=2)),
              on="datetime", how="left", suffix="_hd_2d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime")+pl.duration(days=2)),
              on=["county","datetime"], how="left", suffix="_hl_2d")
        .join(df_historical_weather_date.with_columns(pl.col("datetime")+pl.duration(days=7)),
              on="datetime", how="left", suffix="_hd_7d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime")+pl.duration(days=7)),
              on=["county","datetime"], how="left", suffix="_hl_7d")
        
        # lag target (days)
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=2))
              .rename({"target":"target_2d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=3))
              .rename({"target":"target_3d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=4))
              .rename({"target":"target_4d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=5))
              .rename({"target":"target_5d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=6))
              .rename({"target":"target_6d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=7))
              .rename({"target":"target_7d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=14))
              .rename({"target":"target_14d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        
        # additional weekly lags
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=28))
              .rename({"target":"target_28d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=42))
              .rename({"target":"target_42d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=56))
              .rename({"target":"target_56d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=70))
              .rename({"target":"target_70d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=84))
              .rename({"target":"target_84d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=98))
              .rename({"target":"target_98d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=112))
              .rename({"target":"target_112d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=126))
              .rename({"target":"target_126d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=140))
              .rename({"target":"target_140d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime")+pl.duration(days=364))
              .rename({"target":"target_364d"}),
              on=["county","is_business","product_type","is_consumption","datetime"], how="left")
        
        # lag all-type-sum (days)
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=2))
              .rename({"target":"sum_2d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=3))
              .rename({"target":"sum_3d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=7))
              .rename({"target":"sum_7d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=14))
              .rename({"target":"sum_14d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=28))
              .rename({"target":"sum_28d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=42))
              .rename({"target":"sum_42d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=56))
              .rename({"target":"sum_56d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=70))
              .rename({"target":"sum_70d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=84))
              .rename({"target":"sum_84d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=98))
              .rename({"target":"sum_98d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=112))
              .rename({"target":"sum_112d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=126))
              .rename({"target":"sum_126d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=140))
              .rename({"target":"sum_140d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime")+pl.duration(days=364))
              .rename({"target":"sum_364d"}),
              on=["county","is_business","is_consumption","datetime"], suffix="_all_type_sum", how="left")
        
        .with_columns(
            pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.year().alias("year"),
        )
        .with_columns(
            pl.concat_str("county", "is_business", "product_type", "is_consumption", separator="_")
              .alias("segment"),
        )
        .with_columns(
            (np.pi * pl.col("dayofyear")/183).sin().alias("sin(dayofyear)"),
            (np.pi * pl.col("dayofyear")/183).cos().alias("cos(dayofyear)"),
            (np.pi * pl.col("hour")/12).sin().alias("sin(hour)"),
            (np.pi * pl.col("hour")/12).cos().alias("cos(hour)"),
        )
        .with_columns(
            pl.col(pl.Float64).cast(pl.Float32),
        )
        .drop("date", "hour", "dayofyear")
    )
    
    return df_data


# â”€â”€â”€ 1) FunÃ§Ã£o de limpeza â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def clean_data(
    df_data, df_client, df_gas_prices, df_elec_prices,
    df_fcast_weather, df_hist_weather, df_loc_map
):
    # 1.1) df_data: remove nulls, linhas com target == 0, forÃ§a target >= 0
    df_data = (
        df_data
        .filter(pl.col("target").is_not_null())
        .filter(pl.col("target") != 0)
        .with_columns(
            pl.when(pl.col("target") < 0)
              .then(0.0)
              .otherwise(pl.col("target"))
              .alias("target")
        )
    )

    # 1.2) df_client: corrige negativos e remove zeros
    df_client = (
        df_client
        .with_columns(
            pl.when(pl.col("installed_capacity") < 0)
              .then(0.0)
              .otherwise(pl.col("installed_capacity"))
              .alias("installed_capacity")
        )
        .filter(pl.col("installed_capacity") > 0)
    )

    # 1.3) df_gas_prices: zerar â‰¤0 e imputar mÃ©dia dos positivos
    gas_low_mean = (
        df_gas_prices
        .filter(pl.col("lowest_price_per_mwh") > 0)
        .select(pl.col("lowest_price_per_mwh").mean())
        .item()
    )
    gas_high_mean = (
        df_gas_prices
        .filter(pl.col("highest_price_per_mwh") > 0)
        .select(pl.col("highest_price_per_mwh").mean())
        .item()
    )
    df_gas_prices = df_gas_prices.with_columns([
        pl.when(pl.col("lowest_price_per_mwh") <= 0)
          .then(gas_low_mean)
          .otherwise(pl.col("lowest_price_per_mwh"))
          .alias("lowest_price_per_mwh"),
        pl.when(pl.col("highest_price_per_mwh") <= 0)
          .then(gas_high_mean)
          .otherwise(pl.col("highest_price_per_mwh"))
          .alias("highest_price_per_mwh"),
    ])

    # 1.4) df_elec_prices: zerar â‰¤0 e imputar mÃ©dia dos positivos
    elec_mean = (
        df_elec_prices
        .filter(pl.col("euros_per_mwh") > 0)
        .select(pl.col("euros_per_mwh").mean())
        .item()
    )
    df_elec_prices = df_elec_prices.with_columns(
        pl.when(pl.col("euros_per_mwh") <= 0)
          .then(elec_mean)
          .otherwise(pl.col("euros_per_mwh"))
          .alias("euros_per_mwh")
    )

    # 1.5) df_loc_map: garantir Float32 em lat/lon
    df_loc_map = df_loc_map.with_columns([
        pl.col("latitude").cast(pl.Float32),
        pl.col("longitude").cast(pl.Float32),
    ])

    # 1.6) identifica colunas string em df_data
    string_cols = [c for (c, t) in df_data.schema.items() if t == pl.Utf8]
    if string_cols:
        print("Colunas string em df_data:", string_cols)

    return (
        df_data,
        df_client,
        df_gas_prices,
        df_elec_prices,
        df_fcast_weather,
        df_hist_weather,
        df_loc_map
    )



# â”€â”€â”€ 2) Limpeza e GeraÃ§Ã£o de Features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Aplica a funÃ§Ã£o de limpeza
(
    df_data_clean,
    df_client_clean,
    df_gas_clean,
    df_elec_clean,
    df_fcast_clean,
    df_hist_clean,
    df_loc_clean
) = clean_data(
    df_data,
    df_client,
    df_gas_prices,
    df_elec_prices,
    df_fcast_weather,
    df_hist_weather,
    df_loc_map
)

# Mede tempo e memÃ³ria antes de gerar as features
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss  # bytes
t0 = time.perf_counter()

# GeraÃ§Ã£o de features
df_enriched = generate_features(
    df_data_clean,
    df_client_clean,
    df_gas_clean,
    df_elec_clean,
    df_fcast_clean,
    df_hist_clean,
    df_loc_clean,
    df_target
)

# Mede tempo e memÃ³ria apÃ³s gerar as features
t1 = time.perf_counter()
mem_after = process.memory_info().rss  # bytes

# â”€â”€â”€ 3) VerificaÃ§Ã£o de Tempo e MemÃ³ria â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"â�± GeraÃ§Ã£o de features levou {(t1 - t0) * 1000:.2f} ms")
print(f"ğŸ�� MemÃ³ria antes: {mem_before / 1024**2:.2f} MiB")
print(f"ğŸ�� MemÃ³ria depois: {mem_after / 1024**2:.2f} MiB")
print(f"âˆ† MemÃ³ria: {(mem_after - mem_before) / 1024**2:.2f} MiB\n")

# â”€â”€â”€ 4) VerificaÃ§Ã£o do Resultado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Schema apÃ³s limpeza + geraÃ§Ã£o de features:\n", df_enriched.schema)
print("Linhas totais:", df_enriched.height)



import numpy as np
import pandas as pd
import polars as pl

# â”€â”€â”€ 1) Quantos segmentos Ãºnicos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
num_segments = df_enriched.select(pl.col("segment")).unique().height
print(f"Total de segmentos Ãºnicos: {num_segments}")

# â”€â”€â”€ 2) Quais e quantas colunas sÃ£o do tipo string (Utf8) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
string_cols = [name for name, dtype in df_enriched.schema.items() if dtype == pl.Utf8]
print(f"Colunas string ({len(string_cols)}): {string_cols}")

# Cast de 'segment' para categÃ³rica
df_enriched = df_enriched.with_columns(
    pl.col("segment").cast(pl.Categorical).alias("segment")
)

# â”€â”€â”€ 3) ConversÃ£o para pandas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pdf = df_enriched.to_pandas()

# â”€â”€â”€ 4) Targets ordenados por is_consumption â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
target_cons = (
    pdf.loc[pdf['is_consumption'] == 1, 'target']
    .sort_values()
    .reset_index(drop=True)
)
target_prod = (
    pdf.loc[pdf['is_consumption'] == 0, 'target']
    .sort_values()
    .reset_index(drop=True)
)

# â”€â”€â”€ 5) Amostra das variÃ¡veis cÃ­clicas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cyclic_sample = pdf[['sin(dayofyear)', 'cos(dayofyear)', 'sin(hour)', 'cos(hour)']].head(10)

# â”€â”€â”€ 6) Segmentos Ãºnicos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
segments = pdf['segment'].unique()

# â”€â”€â”€ 7) Determina tamanho mÃ¡ximo para padding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
max_len = max(len(target_cons), len(target_prod), len(cyclic_sample), len(segments))

# â”€â”€â”€ 8) FunÃ§Ã£o de padding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def pad_series(s, length, fill_value=np.nan):
    lst = list(s)
    return pd.Series(lst + [fill_value] * (length - len(lst)))

# â”€â”€â”€ 9) ConstrÃ³i DataFrame combinado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df_combined = pd.DataFrame({
    'target_consumo': pad_series(target_cons, max_len),
    'target_producao': pad_series(target_prod, max_len),
    'sin(dayofyear)': pad_series(cyclic_sample['sin(dayofyear)'], max_len),
    'cos(dayofyear)': pad_series(cyclic_sample['cos(dayofyear)'], max_len),
    'sin(hour)':      pad_series(cyclic_sample['sin(hour)'], max_len),
    'cos(hour)':      pad_series(cyclic_sample['cos(hour)'], max_len),
    'segment':        pad_series(segments, max_len, fill_value=None)
})

# â”€â”€â”€ 10) Exibe o DataFrame combinado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df_combined



import matplotlib.pyplot as plt

# Converte Polars â†’ pandas
pdf = df_enriched.to_pandas()

# Separa valores
cons_values = pdf[pdf['is_consumption'] == 1]['target']
prod_values = pdf[pdf['is_consumption'] == 0]['target']

# Histograma Consumo
plt.figure()
plt.hist(cons_values, bins=100)
plt.title('DistribuiÃ§Ã£o de Target para Consumo')
plt.xlabel('Target')
plt.ylabel('FrequÃªncia')
plt.show()

# Histograma ProduÃ§Ã£o
plt.figure()
plt.hist(prod_values, bins=100)
plt.title('DistribuiÃ§Ã£o de Target para ProduÃ§Ã£o')
plt.xlabel('Target')
plt.ylabel('FrequÃªncia')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# â”€â”€â”€ 6) Carregar dados originais para PACF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train = pd.read_csv(
    "/kaggle/input/predict-energy-behavior-of-prosumers/train.csv",
    parse_dates=["datetime"]
)

# â”€â”€â”€ 7) Agregar por dia e por tipo (0=ProduÃ§Ã£o, 1=Consumo) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train["date"] = train["datetime"].dt.date
daily = (
    train
    .groupby(["date", "is_consumption"])["target"]
    .sum()
    .unstack(fill_value=0)
    .rename(columns={0: "ProduÃ§Ã£o", 1: "Consumo"})
)
daily.index = pd.to_datetime(daily.index)  # transforma o Ã­ndice em DateTimeIndex

# â”€â”€â”€ 8) Plotar PACF atÃ© lag 12 para ProduÃ§Ã£o (95% de confianÃ§a) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(
    daily["ProduÃ§Ã£o"],
    lags=14,
    alpha=0.05,      # 95% de confianÃ§a
    method="ywm",    # mÃ©todo â€œYuleâ€“Walker modificadoâ€�
    ax=ax
)
ax.set_title("Partial Autocorrelation â€“ ProduÃ§Ã£o\n(95% de confianÃ§a em azul)")
ax.set_xlabel("Lag")
ax.set_ylabel("PACF")
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from IPython.display import display

df = df_enriched.to_pandas()

features = ["installed_capacity"]
records = []
for feat in features:
    for tipo, mask in [
        ("ProduÃ§Ã£o", df["is_consumption"] == 0),
        ("Consumo",  df["is_consumption"] == 1)
    ]:
        vals = df.loc[mask, feat].dropna()
        records.append({
            "Feature":  feat,
            "Tipo":     tipo,
            "MÃ©dia":    vals.mean(),
            "Mediana":  vals.median(),
            "Std":      vals.std(),
            "Skewness": vals.skew(),
            "Kurtosis": vals.kurtosis()
        })

stats_df = pd.DataFrame.from_records(records)

# â”€â”€â”€ 2) Formatar stats_df: arredondar todos os valores numÃ©ricos para 2 casas â”€â”€â”€
stats_df[['MÃ©dia', 'Mediana', 'Std', 'Skewness', 'Kurtosis']] = \
    stats_df[['MÃ©dia', 'Mediana', 'Std', 'Skewness', 'Kurtosis']].round(2)

# â”€â”€â”€ 3) Construir pivot table para comparaÃ§Ã£o lado a lado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pivot = stats_df.pivot(index="Feature", columns="Tipo")
# â€œDesempacotarâ€� o MultiIndex das colunas
pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
pivot = pivot.round(2)


# â”€â”€â”€ 4) Exibir de forma mais â€œinterativaâ€� â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=== EstatÃ­sticas por Feature e Tipo (arredondadas) ===")
display(stats_df)

print("\n=== Matriz Pivot (Feature Ã— Tipo, arredondada) ===")
display(pivot)

# â”€â”€â”€ 5) Exemplo de estilizaÃ§Ã£o com gradiente â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n=== EstatÃ­sticas estilizadas com gradiente ===")
styled_stats = (
    stats_df.style
            .background_gradient(subset=['MÃ©dia','Std','Skewness','Kurtosis'], cmap='Blues')
            .format({col: "{:.2f}" for col in ['MÃ©dia','Mediana','Std','Skewness','Kurtosis']})
)
display(styled_stats)

print("\n=== Pivot Table estilizada com gradiente ===")
styled_pivot = (
    pivot.style
         .background_gradient(subset=[col for col in pivot.columns if "MÃ©dia" in col], cmap='Reds')
         .format("{:.2f}")
)
display(styled_pivot)



import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import holidays  #Feriados da EstÃ³nia

# 1) Converter Polars â†’ pandas (se necessÃ¡rio)
if isinstance(df_enriched, pl.DataFrame):
    df_enriched_pd = df_enriched.to_pandas()
else:
    df_enriched_pd = df_enriched.copy()

# 2) Garantir dtype datetime
df_enriched_pd['datetime'] = pd.to_datetime(df_enriched_pd['datetime'])

# 3) VariÃ¡veis temporais bÃ¡sicas
df_enriched_pd['date'] = df_enriched_pd['datetime'].dt.date
estonian_holidays = holidays.country_holidays("EE", years=range(df_enriched_pd['datetime'].dt.year.min(),
                                                                 df_enriched_pd['datetime'].dt.year.max() + 1))

# 4) Adicionar variÃ¡veis de feriado/fim de semana
df_enriched_pd['is_holiday']       = df_enriched_pd['date'].isin(estonian_holidays).astype(int)
df_enriched_pd['is_weekend']       = df_enriched_pd['datetime'].dt.weekday.isin([5, 6]).astype(int)
df_enriched_pd['is_holiday_prev']  = df_enriched_pd['date'].apply(lambda d: (d - timedelta(days=1)) in estonian_holidays).astype(int)
df_enriched_pd['is_holiday_next']  = df_enriched_pd['date'].apply(lambda d: (d + timedelta(days=1)) in estonian_holidays).astype(int)

# 5) Definir corte para os Ãºltimos 2 meses
t_max   = df_enriched_pd['datetime'].max()
t_corte = t_max - relativedelta(months=2)

# 6) Particionar treino e teste
train_df = df_enriched_pd[df_enriched_pd['datetime'] <= t_corte].copy()
test_df  = df_enriched_pd[df_enriched_pd['datetime'] >  t_corte].copy()
print(f"Treino: atÃ© {t_corte.date()} â†’ {len(train_df)} linhas")
print(f"Teste:  de {(t_corte + pd.Timedelta(seconds=1)).date()} atÃ© {t_max.date()} â†’ {len(test_df)} linhas")
assert len(train_df) + len(test_df) == len(df_enriched_pd), "Perda de observaÃ§Ãµes!"

# 7) Verificar e alinhar variÃ¡vel categÃ³rica 'segment'
for name, df in [('train_df', train_df), ('test_df', test_df)]:
    dtype = df['segment'].dtype
    print(f"\n--- {name} ---")
    print(f"Tipo de dados de 'segment': {dtype}")
    print("Categorias (top 10):")
    print(df['segment'].value_counts().head(10))

train_df['segment'] = train_df['segment'].astype('category')
test_df['segment']  = test_df['segment'].astype('category')
test_df['segment']  = test_df['segment'].cat.set_categories(train_df['segment'].cat.categories)

# 8) Codificar 'segment' numericamente e remover a original
train_df['segment_code'] = train_df['segment'].cat.codes
test_df['segment_code']  = test_df['segment'].cat.codes
train_df.drop(columns='segment', inplace=True)
test_df.drop(columns='segment',  inplace=True)

# 9) Verificar consistÃªncia
print("\nTreino â€” cÃ³digos mÃ­nimos e mÃ¡ximos:",
      train_df['segment_code'].min(), train_df['segment_code'].max())
print("Teste  â€” cÃ³digos mÃ­nimos e mÃ¡ximos:",
      test_df['segment_code'].min(), test_df['segment_code'].max())

# 10) Particionar treino em Consumo vs ProduÃ§Ã£o
train_consumption_df = train_df[train_df["is_consumption"] == 1].copy()
train_production_df  = train_df[train_df["is_consumption"] == 0].copy()
assert len(train_consumption_df) + len(train_production_df) == len(train_df), "Erro na separaÃ§Ã£o de consumo/produÃ§Ã£o"
print(f"\nTreino Consumo:  {len(train_consumption_df)} linhas")
print(f"Treino ProduÃ§Ã£o: {len(train_production_df)} linhas")

# 11) Preparar df_enriched_pd com os mesmos cÃ³digos
df_enriched_pd['segment'] = df_enriched_pd['segment'].astype('category')
df_enriched_pd['segment_code'] = df_enriched_pd['segment'].cat.codes
df_enriched_pd.drop(columns='segment', inplace=True)

# 12) VisualizaÃ§Ã£o do dataset final completo
print("\nDataFrame Completo (df_enriched_pd):")
print(df_enriched_pd[['datetime', 'target', 'is_consumption', 'segment_code', 'installed_capacity', 'is_holiday']].head())
print(f"Tamanho total: {len(df_enriched_pd)} linhas")

# 13) SeparaÃ§Ã£o total: consumo vs produÃ§Ã£o (tudo)
df_consumption = df_enriched_pd[df_enriched_pd["is_consumption"] == 1].copy()
df_production  = df_enriched_pd[df_enriched_pd["is_consumption"] == 0].copy()
assert len(df_consumption) + len(df_production) == len(df_enriched_pd), "Erro na partiÃ§Ã£o final"
print(f"\nTotal Consumo:  {len(df_consumption)}")
print(f"Total ProduÃ§Ã£o: {len(df_production)}")


df_enriched_pd


import pandas as pd

# Mostrar todas as linhas e colunas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)       # largura total da tela
pd.set_option('display.max_colwidth', None)  # mostra colunas longas completamente

# Exibir o DataFrame
df_enriched_pd.head(5)
print(f"NÃºmero total de colunas (features): {df_enriched_pd.shape[1]}")


# â”€â”€â”€ CÃ©lula 1: Imports, parÃ¢metros e preparaÃ§Ã£o dos conjuntos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb
from tqdm import tqdm


# 1) DefiniÃ§Ãµes de alvos
cons_targets = {'raw': 'target'}
prod_targets = {'norm': 'target_por_capacidade'}

# 2) HiperparÃ¢metros XGBoost (para consumo e produÃ§Ã£o)
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 2500,          # mais Ã¡rvores
    'learning_rate': 0.006,        # aprendizagem mais lenta
    'max_depth': 7,                # mantÃ©m profundidade moderada
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'reg_alpha': 2.0,
    'reg_lambda': 3.0,
    'min_child_weight': 15,        # folhas mais robustas
    'gamma': 0.3,                  # exige mais ganho para splits
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}

# 3) HiperparÃ¢metros LightGBM (para consumo e produÃ§Ã£o)
lgb_params_cons = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 2500,              # mais iteraÃ§Ãµes
    'learning_rate': 0.015,            # aprendizagem mais lenta
    'colsample_bytree': 0.85,
    'colsample_bynode': 0.55,
    'lambda_l1': 4.0,
    'lambda_l2': 2.0,
    'max_depth': 12,
    'num_leaves': 128,                 # mais folhas â†’ maior capacidade
    'min_child_samples': 60,           # previne overfitting
    'device': 'gpu',
    'verbosity': -1,
    'seed': 42
}

lgb_params_prod = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 2500,
    'learning_rate': 0.015,
    'colsample_bytree': 0.85,
    'colsample_bynode': 0.55,
    'lambda_l1': 4.0,
    'lambda_l2': 2.0,
    'max_depth': 12,
    'num_leaves': 128,
    'min_child_samples': 60,
    'device': 'gpu',
    'verbosity': -1,
    'seed': 42
}


# 4) PreparaÃ§Ã£o dos dados de Consumo
df_train_cons = train_df[train_df.is_consumption == 1].copy()
df_test_cons  = test_df[test_df.is_consumption == 1].copy()
DROP_COLS = ['datetime', 'date', 'row_id', 'is_consumption']
FEATURE_COLS_CONS = [
    c for c in df_train_cons.columns
    if c not in DROP_COLS + list(cons_targets.values())
]

# 5) PreparaÃ§Ã£o dos dados de ProduÃ§Ã£o
df_train_prod = train_df[train_df.is_consumption == 0].copy()
df_test_prod  = test_df[test_df.is_consumption == 0].copy()
df_train_prod['installed_capacity'] = df_train_prod['installed_capacity'].fillna(0.0)
df_test_prod['installed_capacity']  = df_test_prod['installed_capacity'].fillna(0.0)
# Criar alvo normalizado no treino
df_train_prod['target_por_capacidade'] = np.where(
    df_train_prod['installed_capacity'] > 0,
    df_train_prod['target'] / df_train_prod['installed_capacity'],
    0.0
)
FEATURE_COLS_PROD = [
    c for c in df_train_prod.columns
    if c not in DROP_COLS + ['target'] + list(prod_targets.values())
]



from dateutil.relativedelta import relativedelta
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. ParÃ¢metros e preparaÃ§Ã£o
window_options = list(range(2, 11))  # Janelas de 2 a 10 meses
t_max = df_train_prod['datetime'].max()
t_holdout_ini = t_max - relativedelta(months=2)

# Dados de teste
X_test = df_test_prod[FEATURE_COLS_PROD].reset_index(drop=True)
y_test = df_test_prod['target'].values
cap_test = df_test_prod['installed_capacity'].fillna(0.0).values

# InicializaÃ§Ã£o
results = []
voting_models = []

# 2. Treinamento por janelas mÃ³veis
for months in tqdm(window_options, desc="Treinando janelas mÃºltiplas"):
    t_ini = t_holdout_ini - relativedelta(months=months)
    train_window = df_train_prod[
        (df_train_prod['datetime'] >= t_ini) & (df_train_prod['datetime'] < t_holdout_ini)
    ].copy()

    # Alvo normalizado por capacidade instalada
    train_window['target_por_capacidade'] = np.where(
        train_window['installed_capacity'] > 0,
        train_window['target'] / train_window['installed_capacity'],
        0.0
    )

    # Dados de treino
    X_train = train_window[FEATURE_COLS_PROD]
    y_train = train_window['target_por_capacidade']

    # Modelo LightGBM (GPU se ativado no ambiente)
    model = LGBMRegressor(**lgb_params_prod)
    model.fit(X_train, y_train)

    # PrevisÃ£o (normalizada â†’ desnormalizada)
    y_pred_norm = model.predict(X_test)
    y_pred = np.where(cap_test > 0, y_pred_norm * cap_test, 0.0)
    mae = mean_absolute_error(y_test, y_pred)

    # Armazenar resultados
    results.append({'months': months, 'mae': mae, 'model': model})
    voting_models.append((f'model_{months}m', model))

# 3. Melhor modelo individual
best_model_info = min(results, key=lambda r: r['mae'])
best_model = best_model_info['model']
print(f"\nMelhor janela: {best_model_info['months']} meses â†’ MAE = {best_model_info['mae']:.3f}")

# 4. Voting Regressor (ensemble de todos)
voting_reg = VotingRegressor(estimators=voting_models)
voting_reg.fit(X_train, y_train)

# 5. PrevisÃ£o com ensemble
y_pred_vote_norm = voting_reg.predict(X_test)
y_pred_vote = np.where(cap_test > 0, y_pred_vote_norm * cap_test, 0.0)

# 6. AvaliaÃ§Ã£o
mae_vote = mean_absolute_error(y_test, y_pred_vote)

print(f"\nMAE VotingRegressor .............: {mae_vote:.3f}")


from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

# 1. Organizar os dados
df_train_prod_sorted = df_train_prod.sort_values("datetime").reset_index(drop=True)
df_train_prod_sorted["installed_capacity"] = df_train_prod_sorted["installed_capacity"].fillna(0.0)
df_train_prod_sorted["target_por_capacidade"] = np.where(
    df_train_prod_sorted["installed_capacity"] > 0,
    df_train_prod_sorted["target"] / df_train_prod_sorted["installed_capacity"],
    0.0
)

# 2. Criar datas para divisÃ£o
min_date = df_train_prod_sorted["datetime"].min().date()
max_date = df_train_prod_sorted["datetime"].max().date()
test_start = max_date - relativedelta(months=2)
train_end = test_start - timedelta(days=1)
train_start = train_end - relativedelta(months=3)

# 3. Treino e Teste
train_data = df_train_prod_sorted[
    (df_train_prod_sorted["datetime"].dt.date >= train_start) &
    (df_train_prod_sorted["datetime"].dt.date <= train_end)
].copy()
test_data = df_train_prod_sorted[df_train_prod_sorted["datetime"].dt.date > train_end].copy()

X_train = train_data[FEATURE_COLS_PROD]
y_train = train_data["target_por_capacidade"]
cap_train = train_data["installed_capacity"].values

X_test = test_data[FEATURE_COLS_PROD].reset_index(drop=True)
cap_test = test_data["installed_capacity"].fillna(0.0).values
y_true = test_data["target"].values

# 4. Modelo base com 3 meses
print("Treinando modelo base...")
model_base = LGBMRegressor(**lgb_params_prod)
model_base.fit(X_train, y_train)

# 5. PrevisÃ£o no conjunto de teste
y_pred_base = model_base.predict(X_test)
y_pred_base_denorm = y_pred_base * cap_test

# 6. AvaliaÃ§Ã£o
mae_base = mean_absolute_error(y_true, y_pred_base_denorm)
print(f"MAE base .............: {mae_base:.3f}")


import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm

# â”€â”€â”€ Modelo LightGBM para Consumo com validaÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models_cons_lgb, scores_cons_lgb = {}, {}
training_logs = {}

with tqdm(cons_targets.items(), desc="Modelos consumo (LightGBM)") as pbar:
    for name, tgt_col in pbar:
        X = df_train_cons[FEATURE_COLS_CONS]
        y = df_train_cons[tgt_col]

        mask = y.notna() & np.isfinite(y)
        X = X.loc[mask]
        y = y.loc[mask]

        # DivisÃ£o 80% treino / 20% validaÃ§Ã£o
        split_idx = int(0.8 * len(X))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Modelo
        eval_log = {}
        model = lgb.LGBMRegressor(**lgb_params_cons)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.record_evaluation(eval_log)
            ]
        )

        # AvaliaÃ§Ã£o no conjunto de teste
        y_pred = model.predict(df_test_cons[FEATURE_COLS_CONS])
        mae = mean_absolute_error(df_test_cons['target'], y_pred)

        models_cons_lgb[name] = model
        scores_cons_lgb[name] = mae
        training_logs[name] = eval_log['valid_0']['l1']
        pbar.set_postfix(mae=f"{mae:.3f}")

# â”€â”€â”€ Resumo dos resultados â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("MAE por modelo de consumo (LightGBM):")
for name, mae in scores_cons_lgb.items():
    print(f" â€¢ {name:<12s} â†’ MAE = {mae:.3f}")

# â”€â”€â”€ Ensemble: mÃ©dia das prediÃ§Ãµes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_preds_lgb = np.column_stack([
    models_cons_lgb[n].predict(df_test_cons[FEATURE_COLS_CONS])
    for n in cons_targets
])
ensemble_pred_lgb = all_preds_lgb.mean(axis=1)
ensemble_mae_lgb = mean_absolute_error(df_test_cons['target'], ensemble_pred_lgb)
print(f"Ensemble consumo (LightGBM) MAE: {ensemble_mae_lgb:.3f}")

# â”€â”€â”€ GrÃ¡fico 1: Real vs Previsto (ensemble) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 5))
plt.plot(df_test_cons['datetime'], df_test_cons['target'], label='Real', linewidth=2)
plt.plot(df_test_cons['datetime'], ensemble_pred_lgb, label='Previsto (ensemble)', alpha=0.8)
plt.title("Consumo: Real vs Previsto (Ensemble LightGBM)")
plt.xlabel("Datetime")
plt.ylabel("Consumo [MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# â”€â”€â”€ GrÃ¡fico 2: EvoluÃ§Ã£o do MAE (validaÃ§Ã£o) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12, 5))
for name, loss_vals in training_logs.items():
    plt.plot(loss_vals, label=f'{name} MAE (val)')
plt.title("EvoluÃ§Ã£o do MAE na ValidaÃ§Ã£o durante o Treinamento")
plt.xlabel("IteraÃ§Ã£o")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# â”€â”€â”€ LightGBM para ProduÃ§Ã£o com curva de overfitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models_prod_lgb, scores_prod_lgb = {}, {}
train_logs = {}
val_logs = {}

with tqdm(prod_targets.items(), desc="Modelos produÃ§Ã£o (LightGBM)") as pbar:
    for name, tgt_col in pbar:
        X = df_train_prod[FEATURE_COLS_PROD]
        y = df_train_prod[tgt_col]

        mask = y.notna() & np.isfinite(y)
        X = X.loc[mask]
        y = y.loc[mask]

        # Split: 80% treino, 20% validaÃ§Ã£o
        split_idx = int(0.8 * len(X))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Modelo com log
        eval_log = {}
        model = lgb.LGBMRegressor(**lgb_params_prod)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='mae',
            callbacks=[
                lgb.record_evaluation(eval_log),
                lgb.early_stopping(stopping_rounds=300)
            ]
        )

        # PrevisÃ£o no conjunto de teste
        X_te = df_test_prod[FEATURE_COLS_PROD]
        capa_te = df_test_prod['installed_capacity'].values
        y_pred_norm = model.predict(X_te)
        y_pred_orig = np.where(capa_te > 0, y_pred_norm * capa_te, 0.0)
        mae = mean_absolute_error(df_test_prod['target'], y_pred_orig)

        models_prod_lgb[name] = model
        scores_prod_lgb[name] = mae
        train_logs[name] = eval_log['training']['l1']
        val_logs[name] = eval_log['valid_1']['l1']
        pbar.set_postfix(mae=f"{mae:.3f}")

# â”€â”€â”€ Resumo dos Resultados â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("MAE por modelo de produÃ§Ã£o (LightGBM):")
for name, mae in scores_prod_lgb.items():
    print(f" â€¢ {name:<12s} â†’ MAE = {mae:.3f}")

# â”€â”€â”€ Ensemble (mÃ©dia dos modelos treinados) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_preds_norm = np.column_stack([
    models_prod_lgb[n].predict(df_test_prod[FEATURE_COLS_PROD])
    for n in prod_targets
])
all_preds_orig = [
    np.where(
        df_test_prod['installed_capacity'] > 0,
        preds_norm * df_test_prod['installed_capacity'],
        0.0
    )
    for preds_norm in all_preds_norm.T
]
ensemble_pred_prod = np.mean(all_preds_orig, axis=0)
ensemble_mae_prod = mean_absolute_error(df_test_prod['target'], ensemble_pred_prod)
print(f"Ensemble produÃ§Ã£o (LightGBM) MAE: {ensemble_mae_prod:.3f}")

# â”€â”€â”€ GrÃ¡fico 1: ProduÃ§Ã£o Real vs Prevista â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 5))
plt.plot(df_test_prod['datetime'], df_test_prod['target'], label='Real', linewidth=2)
plt.plot(df_test_prod['datetime'], ensemble_pred_prod, label='Previsto', alpha=0.8)
plt.title("ProduÃ§Ã£o: Real vs Previsto (Ensemble LightGBM)")
plt.xlabel("Datetime")
plt.ylabel("ProduÃ§Ã£o [MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# â”€â”€â”€ GrÃ¡fico 2: Curva de Overfitting (Treino vs ValidaÃ§Ã£o MAE) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12, 5))
for name in train_logs:
    plt.plot(train_logs[name], label=f'{name} Train MAE', linestyle='--')
    plt.plot(val_logs[name], label=f'{name} Val MAE', linestyle='-')
plt.title("Curva de Overfitting: Erro Absoluto MÃ©dio por IteraÃ§Ã£o")
plt.xlabel("IteraÃ§Ã£o")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm

# â”€â”€â”€ CÃ©lula 4: Modelo XGBoost para Consumo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models_cons_xgb, scores_cons_xgb = {}, {}
training_logs_xgb = {}

with tqdm(cons_targets.items(), desc="Modelos consumo (XGBoost)") as pbar:
    for name, tgt_col in pbar:
        # SeparaÃ§Ã£o dos dados de treino e teste
        X_tr = df_train_cons[FEATURE_COLS_CONS]
        y_tr = df_train_cons[tgt_col]
        X_te = df_test_cons[FEATURE_COLS_CONS]
        y_te = df_test_cons['target']

        # RemoÃ§Ã£o de valores nulos/invÃ¡lidos
        mask = y_tr.notna() & np.isfinite(y_tr)
        X_tr_clean = X_tr.loc[mask]
        y_tr_clean = y_tr.loc[mask]

        # InicializaÃ§Ã£o do modelo
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_tr_clean, y_tr_clean,
            eval_set=[(X_tr_clean, y_tr_clean)],
            eval_metric='mae',
            verbose=False
        )

        # Obter histÃ³rico de treino
        evals_result = model.evals_result()

        # PrevisÃ£o e avaliaÃ§Ã£o
        y_pred = model.predict(X_te)
        mae    = mean_absolute_error(y_te, y_pred)

        # Armazenamento dos resultados
        models_cons_xgb[name] = model
        scores_cons_xgb[name] = mae
        training_logs_xgb[name] = evals_result['validation_0']['mae']
        pbar.set_postfix(mae=f"{mae:.3f}")

# â”€â”€â”€ Resumo dos Resultados â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("MAE por modelo de consumo (XGBoost):")
for name, mae in scores_cons_xgb.items():
    print(f" â€¢ {name:<12s} â†’ MAE = {mae:.3f}")

# â”€â”€â”€ Ensemble (mÃ©dia) das previsÃµes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_preds_xgb = np.column_stack([
    models_cons_xgb[n].predict(df_test_cons[FEATURE_COLS_CONS])
    for n in cons_targets
])
ensemble_pred_xgb = all_preds_xgb.mean(axis=1)
ensemble_mae_xgb = mean_absolute_error(df_test_cons['target'], ensemble_pred_xgb)
print(f"Ensemble consumo (XGBoost) MAE: {ensemble_mae_xgb:.3f}")

# â”€â”€â”€ GrÃ¡fico 1: Real vs Previsto (Ensemble) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 5))
plt.plot(df_test_cons['datetime'], df_test_cons['target'], label='Real', linewidth=2)
plt.plot(df_test_cons['datetime'], ensemble_pred_xgb, label='Previsto (ensemble)', alpha=0.8)
plt.title("Consumo: Real vs Previsto (Ensemble XGBoost)")
plt.xlabel("Datetime")
plt.ylabel("Consumo [MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# â”€â”€â”€ GrÃ¡fico 2: EvoluÃ§Ã£o do MAE no treino â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12, 5))
for name, loss_vals in training_logs_xgb.items():
    plt.plot(loss_vals, label=f'{name} MAE')
plt.title("EvoluÃ§Ã£o do Erro Absoluto MÃ©dio (MAE) no Treinamento â€” Consumo (XGBoost)")
plt.xlabel("IteraÃ§Ã£o")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm

# â”€â”€â”€ CÃ©lula 5: Modelo XGBoost para ProduÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models_prod_xgb, scores_prod_xgb = {}, {}
training_logs_prod_xgb = {}

with tqdm(prod_targets.items(), desc="Modelos produÃ§Ã£o (XGBoost)") as pbar:
    for name, tgt_col in pbar:
        X_tr = df_train_prod[FEATURE_COLS_PROD]
        y_tr = df_train_prod[tgt_col]
        X_te = df_test_prod[FEATURE_COLS_PROD]
        capa_te = df_test_prod['installed_capacity'].values

        # AtribuiÃ§Ã£o de pesos temporais
        date_diff = (df_train_prod['datetime'].max() - df_train_prod['datetime']).dt.days
        weights = np.where(
            date_diff <= 15, 0.5,
            np.where(date_diff <= 15 + 90, 0.3, 0.2)
        )

        # Definir eval_metric dentro do construtor
        xgb_params['eval_metric'] = 'mae'
        model = XGBRegressor(**xgb_params)

        model.fit(
            X_tr, y_tr,
            sample_weight=weights,
            eval_set=[(X_tr, y_tr)],
            verbose=False
        )

        evals_result = model.evals_result()

        y_pred_norm = model.predict(X_te)
        y_pred_orig = np.where(capa_te > 0, y_pred_norm * capa_te, 0.0)
        mae = mean_absolute_error(df_test_prod['target'], y_pred_orig)

        models_prod_xgb[name] = model
        scores_prod_xgb[name] = mae
        training_logs_prod_xgb[name] = evals_result['validation_0']['mae']
        pbar.set_postfix(mae=f"{mae:.3f}")

# â”€â”€â”€ Resumo dos resultados â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("MAE por modelo de produÃ§Ã£o (XGBoost):")
for name, mae in scores_prod_xgb.items():
    print(f" â€¢ {name:<12s} â†’ MAE = {mae:.3f}")

# â”€â”€â”€ Ensemble (mÃ©dia) das previsÃµes de produÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_preds_norm_xgb = np.column_stack([
    models_prod_xgb[n].predict(df_test_prod[FEATURE_COLS_PROD])
    for n in prod_targets
])
all_preds_orig_xgb = [
    np.where(
        df_test_prod['installed_capacity'] > 0,
        preds_norm * df_test_prod['installed_capacity'],
        0.0
    )
    for preds_norm in all_preds_norm_xgb.T
]
ensemble_pred_prod_xgb = np.mean(all_preds_orig_xgb, axis=0)
ensemble_mae_prod_xgb = mean_absolute_error(df_test_prod['target'], ensemble_pred_prod_xgb)
print(f"Ensemble produÃ§Ã£o (XGBoost) MAE: {ensemble_mae_prod_xgb:.3f}")

# â”€â”€â”€ GrÃ¡fico 1: Real vs Previsto (Ensemble XGBoost ProduÃ§Ã£o) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 5))
plt.plot(df_test_prod['datetime'], df_test_prod['target'], label='Real', linewidth=2)
plt.plot(df_test_prod['datetime'], ensemble_pred_prod_xgb, label='Previsto (ensemble)', alpha=0.8)
plt.title("ProduÃ§Ã£o: Real vs Previsto (Ensemble XGBoost)")
plt.xlabel("Datetime")
plt.ylabel("ProduÃ§Ã£o [MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# â”€â”€â”€ GrÃ¡fico 2: EvoluÃ§Ã£o do MAE no treino â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(12, 5))
for name, loss_vals in training_logs_prod_xgb.items():
    plt.plot(loss_vals, label=f'{name} MAE')
plt.title("EvoluÃ§Ã£o do Erro Absoluto MÃ©dio (MAE) no Treinamento â€” ProduÃ§Ã£o (XGBoost)")
plt.xlabel("IteraÃ§Ã£o")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# â”€â”€â”€ CÃ¡lculo do target normalizado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Se ainda nÃ£o existir, cria a coluna com o target normalizado pela capacidade instalada
if 'target_por_capacidade' not in df_enriched_pd.columns:
    df_enriched_pd['target_por_capacidade'] = np.where(
        df_enriched_pd['installed_capacity'] > 0,
        df_enriched_pd['target'] / df_enriched_pd['installed_capacity'],
        0.0
    )

# â”€â”€â”€ HiperparÃ¢metros XGBoost para produÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
xgb_params_prod = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'n_estimators': 1500,
    'learning_rate': 0.015,
    'max_depth': 7,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'reg_alpha': 1.5,
    'reg_lambda': 2.5,
    'min_child_weight': 10,
    'gamma': 0.2,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}

# â”€â”€â”€ PreparaÃ§Ã£o da janela deslizante â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
window_size = timedelta(days=14)
prod_dates = sorted(df_enriched_pd['date'].unique())
start_date = prod_dates[0] + window_size
test_dates = [d for d in prod_dates if d >= start_date]

drop_cols_prod = [
    'target', 'target_por_capacidade',
    'datetime', 'date', 'row_id', 'is_consumption'
]
feature_cols_prod = [
    c for c in df_enriched_pd.columns
    if c not in drop_cols_prod
]

# â”€â”€â”€ Loop principal de treino/teste â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=== Sliding Window: PRODUÃ‡ÃƒO (XGBoost) ===")
prod_dt, prod_true, prod_pred = [], [], []

with tqdm(test_dates, desc='ProduÃ§Ã£o') as pbar:
    fold = 0
    for day in pbar:
        train_start = day - window_size
        train_win = df_enriched_pd[
            (df_enriched_pd['date'] >= train_start) &
            (df_enriched_pd['date'] < day)
        ]
        test_day = df_enriched_pd[df_enriched_pd['date'] == day]

        train_p = train_win[train_win['is_consumption'] == 0]
        test_p  = test_day[test_day['is_consumption'] == 0]
        if train_p.empty or test_p.empty:
            continue

        fold += 1
        X_tr = train_p[feature_cols_prod]
        y_tr = train_p['target_por_capacidade']
        X_te = test_p[feature_cols_prod]
        y_true = test_p['target'].values
        cap    = test_p['installed_capacity'].values

        # Treina o modelo
        model = XGBRegressor(**xgb_params_prod)
        model.fit(X_tr, y_tr)

        # PrevisÃ£o normalizada
        y_pred_norm = model.predict(X_te)

        # Reescalonamento para produÃ§Ã£o real (em MWh)
        y_pred = np.where(cap > 0, y_pred_norm * cap, 0.0)
        y_pred = np.maximum(0, y_pred)

        prod_dt.extend(test_p['datetime'])
        prod_true.extend(y_true)
        prod_pred.extend(y_pred)

        fold_mae = np.mean(np.abs(y_true - y_pred))
        pbar.set_postfix(fold=fold, mae=f"{fold_mae:.3f}")

# â”€â”€â”€ CriaÃ§Ã£o do DataFrame de Resultados â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df_prod = pd.DataFrame({
    'datetime': prod_dt,
    'y_true':   prod_true,
    'y_pred':   prod_pred
}).sort_values('datetime').reset_index(drop=True)
df_prod['abs_err'] = np.abs(df_prod['y_true'] - df_prod['y_pred'])
mae_prod = df_prod['abs_err'].mean()

print(f"\n=== MAE GERAL (ProduÃ§Ã£o): {mae_prod:.3f} â€” Total previsÃµes: {len(df_prod)} ===")

# â”€â”€â”€ GrÃ¡fico 1: Real vs Previsto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 5))
plt.plot(df_prod['datetime'], df_prod['y_true'], label='Real', linewidth=2)
plt.plot(df_prod['datetime'], df_prod['y_pred'], label='Previsto', alpha=0.8)
plt.title("ProduÃ§Ã£o: Real vs Previsto (Sliding Window XGBoost)")
plt.xlabel("Datetime")
plt.ylabel("ProduÃ§Ã£o [MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# â”€â”€â”€ GrÃ¡fico 2: Erro Absoluto por Timestamp â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 4))
plt.plot(df_prod['datetime'], df_prod['abs_err'], label='Erro Absoluto')
plt.title("Erro Absoluto por Timestamp (ProduÃ§Ã£o)")
plt.xlabel("Datetime")
plt.ylabel("Erro [MWh]")
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# â”€â”€â”€ CÃ©lula: Sliding Window XGBoost para CONSUMO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

xgb_params_cons = {
    'objective': 'reg:squarederror',   # funÃ§Ã£o de perda para regressÃ£o
    'eval_metric': 'mae',              # mÃ©trica de avaliaÃ§Ã£o baseada em erro absoluto
    'n_estimators': 400,               # â†“ menos Ã¡rvores = treino mais rÃ¡pido
    'learning_rate': 0.1,              # â†‘ aprendizagem mais rÃ¡pida â†’ convergÃªncia mais cedo
    'max_depth': 4,                    # â†“ Ã¡rvores mais rasas â†’ menor tempo por Ã¡rvore
    'subsample': 0.7,                  # menos amostras por Ã¡rvore â†’ menos custo
    'colsample_bytree': 0.7,           # menos features por Ã¡rvore â†’ acelera split
    'reg_alpha': 0.5,                  # regularizaÃ§Ã£o L1 moderada
    'reg_lambda': 1.0,                 # regularizaÃ§Ã£o L2 moderada
    'min_child_weight': 20,            # exige mais dados por folha â†’ menos splits
    'gamma': 0.3,                      # sÃ³ aceita splits com maior ganho
    'tree_method': 'hist',             # mÃ©todo rÃ¡pido de construÃ§Ã£o de Ã¡rvores
    'random_state': 42,
    'verbosity': 0
}
window_size = timedelta(days=14)
cons_dates  = sorted(df_consumption['date'].unique())
start_date  = cons_dates[0] + window_size
test_dates  = [d for d in cons_dates if d >= start_date]

drop_cols_cons = ['target', 'target_por_capacidade', 'datetime', 'date', 'row_id', 'is_consumption']
feature_cols_cons = [c for c in df_consumption.columns if c not in drop_cols_cons]

print("=== Sliding Window: CONSUMO (XGBoost) ===")
cons_dt, cons_true, cons_pred = [], [], []

with tqdm(test_dates, desc='Consumo') as pbar:
    fold = 0
    for day in pbar:
        train_start = day - window_size
        train_win   = df_consumption[
            (df_consumption['date'] >= train_start) &
            (df_consumption['date'] < day)
        ]
        test_day = df_consumption[df_consumption['date'] == day]

        train_c = train_win[train_win['is_consumption'] == 1]
        test_c  = test_day[test_day['is_consumption']    == 1]
        if train_c.empty or test_c.empty:
            continue

        fold += 1
        X_tr   = train_c[feature_cols_cons]
        y_tr   = train_c['target']
        X_te   = test_c[feature_cols_cons]
        y_true = test_c['target'].values

        model  = XGBRegressor(**xgb_params_cons)
        model.fit(X_tr, y_tr)
        y_pred = np.maximum(0, model.predict(X_te))

        cons_dt.extend(test_c['datetime'])
        cons_true.extend(y_true)
        cons_pred.extend(y_pred)

        fold_mae = np.mean(np.abs(y_true - y_pred))
        pbar.set_postfix(fold=fold, mae=f"{fold_mae:.3f}")

df_cons = pd.DataFrame({
    'datetime': cons_dt,
    'y_true':   cons_true,
    'y_pred':   cons_pred
}).sort_values('datetime').reset_index(drop=True)
df_cons['abs_err'] = np.abs(df_cons['y_true'] - df_cons['y_pred'])
mae_cons = df_cons['abs_err'].mean()

print(f"\n=== MAE GERAL (Consumo): {mae_cons:.3f} â€” Total previsÃµes: {len(df_cons)} ===")

# â”€â”€â”€ GrÃ¡fico 1: Real vs Previsto (Sliding Window XGBoost - Consumo) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 5))
plt.plot(df_cons['datetime'], df_cons['y_true'], label='Real', linewidth=2)
plt.plot(df_cons['datetime'], df_cons['y_pred'], label='Previsto', alpha=0.8)
plt.title("Consumo: Real vs Previsto (Sliding Window XGBoost)")
plt.xlabel("Datetime")
plt.ylabel("Consumo [MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# â”€â”€â”€ GrÃ¡fico 2: Erro Absoluto por Timestamp â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(14, 4))
plt.plot(df_cons['datetime'], df_cons['abs_err'], label='Erro Absoluto')
plt.title("Erro Absoluto por Timestamp (Consumo)")
plt.xlabel("Datetime")
plt.ylabel("Erro [MWh]")
plt.grid(True)
plt.tight_layout()
plt.show()




