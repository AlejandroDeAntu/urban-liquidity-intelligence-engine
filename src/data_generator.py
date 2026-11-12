"""Genera un dataset sintético y reproducible de transacciones inmobiliarias en CDMX y lo persiste en SQLite."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "urban_real_estate.db"

ALCALDIAS = {
    "Cuauhtémoc": {"lat": 19.4326, "lon": -99.1332, "base_m2": 58000, "liq_base": 45},
    "Benito Juárez": {"lat": 19.3800, "lon": -99.1620, "base_m2": 54000, "liq_base": 50},
    "Miguel Hidalgo": {"lat": 19.4320, "lon": -99.2000, "base_m2": 75000, "liq_base": 65},
    "Coyoacán": {"lat": 19.3467, "lon": -99.1617, "base_m2": 44000, "liq_base": 60},
    "Álvaro Obregón": {"lat": 19.3587, "lon": -99.2033, "base_m2": 48000, "liq_base": 70},
    "Tlalpan": {"lat": 19.2886, "lon": -99.1670, "base_m2": 36000, "liq_base": 80},
}

# Nodos de transporte masivo / centros financieros (lat, lon)
TRANSIT_HUBS = np.array([
    (19.4270, -99.1677),  # Reforma / Ángel
    (19.3907, -99.1736),  # Insurgentes Sur / WTC
    (19.4352, -99.1945),  # Polanco
    (19.3435, -99.1511),  # Taxqueña
])

EARTH_RADIUS_KM = 6371.0
HISTORY_DAYS = 730
START_DATE = pd.Timestamp("2024-01-01")
ANNUAL_TREND = 0.07  # ~7% de plusvalía anual simulada


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia geodésica en km (vectorizada)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def min_distance_to_hubs(lat, lon):
    """Distancia (km) al nodo de transporte más cercano para arrays de lat/lon."""
    lat, lon = np.asarray(lat, float)[:, None], np.asarray(lon, float)[:, None]
    return haversine_km(lat, lon, TRANSIT_HUBS[:, 0], TRANSIT_HUBS[:, 1]).min(axis=1)


def generate_dataframe(n_records: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(ALCALDIAS)
    alc = rng.choice(names, n_records)
    info = pd.DataFrame(ALCALDIAS).T.loc[alc].astype(float).reset_index(drop=True)

    lat = info["lat"] + rng.normal(0, 0.015, n_records)
    lon = info["lon"] + rng.normal(0, 0.015, n_records)
    dist = min_distance_to_hubs(lat, lon)

    days_offset = rng.integers(0, HISTORY_DAYS, n_records)
    trend = 1 + (days_offset / 365) * ANNUAL_TREND
    area = np.round(rng.uniform(45, 220, n_records), 1)
    proximity = np.maximum(0.75, 1.15 - dist * 0.04)  # el precio cae al alejarse del transporte
    price_m2 = info["base_m2"] * proximity * trend * rng.normal(1, 0.08, n_records)
    days_on_market = np.maximum(15, info["liq_base"] * (1 + dist * 0.08) * rng.normal(1, 0.2, n_records)).astype(int)

    return pd.DataFrame({
        "property_id": [f"PROP-{10000 + i}" for i in range(n_records)],
        "alcaldia": alc,
        "latitude": lat.round(6),
        "longitude": lon.round(6),
        "dist_transit_km": dist.round(2),
        "area_m2": area,
        "price_m2": price_m2.round(2),
        "total_price": (price_m2 * area).round(2),
        "days_on_market": days_on_market,
        "transaction_date": (START_DATE + pd.to_timedelta(days_offset, unit="D")).strftime("%Y-%m-%d"),
    })


def generate_dataset(n_records: int = 2500, db_path: Path = DB_PATH) -> pd.DataFrame:
    df = generate_dataframe(n_records)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
    print(f"Base de datos creada con {len(df):,} registros en {db_path}")
    return df


def load_transactions(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        generate_dataset(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM transactions", conn)


if __name__ == "__main__":
    generate_dataset()
