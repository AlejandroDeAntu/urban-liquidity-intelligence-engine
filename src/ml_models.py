"""Segmentación de mercado (K-Means) y pronóstico de precio/m² (Holt-Winters) con backtest."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CLUSTER_FEATURES = ["price_m2", "days_on_market", "dist_transit_km"]
SEGMENT_LABELS = [  # ordenados de menor a mayor precio/m² promedio
    "Periferia / Baja Liquidez",
    "Corredor Emergente (Alta Oportunidad)",
    "Consolidado / Liquidez Media",
    "Prime / Alta Demanda",
]
HOLDOUT_MONTHS = 3


def run_market_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Devuelve una copia de df con cluster_id y segmento_inversion."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    ids = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    out = df.assign(cluster_id=ids)
    order = out.groupby("cluster_id")["price_m2"].mean().sort_values().index
    return out.assign(segmento_inversion=out["cluster_id"].map(dict(zip(order, SEGMENT_LABELS))))


def monthly_price_series(df: pd.DataFrame, alcaldia: str) -> pd.Series:
    sub = df[df["alcaldia"] == alcaldia]
    month = pd.to_datetime(sub["transaction_date"]).dt.to_period("M").dt.to_timestamp()
    return sub.groupby(month)["price_m2"].mean().asfreq("MS").interpolate().rename("price_m2")


def _fit(series: pd.Series):
    return ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated").fit()


def backtest_mape(series: pd.Series, holdout: int = HOLDOUT_MONTHS) -> float:
    """MAPE (%) entrenando sin los últimos `holdout` meses y comparando contra ellos."""
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    pred = _fit(train).forecast(holdout)
    return float(np.mean(np.abs((test.values - pred.values) / test.values)) * 100)


def forecast_alcaldia_prices(df: pd.DataFrame, alcaldia: str, steps: int = 6) -> tuple[pd.DataFrame, float]:
    """Histórico + pronóstico a `steps` meses, y MAPE del backtest."""
    series = monthly_price_series(df, alcaldia)
    forecast = _fit(series).forecast(steps)
    hist = pd.DataFrame({"Fecha": series.index, "Precio_m2": series.values, "Tipo": "Histórico"})
    fut = pd.DataFrame({"Fecha": forecast.index, "Precio_m2": forecast.values, "Tipo": f"Pronóstico ({steps}M)"})
    return pd.concat([hist, fut], ignore_index=True), backtest_mape(series)
