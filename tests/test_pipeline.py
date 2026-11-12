import pytest

from data_generator import generate_dataframe, generate_dataset, haversine_km, load_transactions
from ml_models import SEGMENT_LABELS, forecast_alcaldia_prices, run_market_clustering


@pytest.fixture(scope="module")
def df():
    return generate_dataframe()


def test_haversine_known_distance():
    # Ángel de la Independencia -> Zócalo ≈ 3.7 km
    assert haversine_km(19.4270, -99.1677, 19.4326, -99.1332) == pytest.approx(3.67, abs=0.1)


def test_dataset_schema_and_reproducibility(df):
    assert len(df) == 2500
    assert df["property_id"].is_unique
    assert (df["price_m2"] > 0).all() and (df["days_on_market"] >= 15).all()
    assert df.equals(generate_dataframe())


def test_sqlite_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    generate_dataset(200, db_path=db)
    assert len(load_transactions(db)) == 200


def test_clustering_labels_ordered_by_price(df):
    out = run_market_clustering(df)
    assert "cluster_id" not in df.columns  # no muta la entrada
    means = out.groupby("segmento_inversion")["price_m2"].mean()
    assert list(means.sort_values().index) == SEGMENT_LABELS


def test_forecast_shape_and_mape(df):
    fc, mape = forecast_alcaldia_prices(df, "Coyoacán", steps=6)
    assert (fc["Tipo"] != "Histórico").sum() == 6
    assert fc["Precio_m2"].notna().all()
    assert 0 <= mape < 20
