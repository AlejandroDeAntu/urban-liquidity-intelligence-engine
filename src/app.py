"""Dashboard Streamlit: Motor de Inteligencia de Inversión Inmobiliaria y Liquidez Urbana."""
import plotly.express as px
import streamlit as st

from data_generator import load_transactions
from ml_models import forecast_alcaldia_prices, run_market_clustering

st.set_page_config(page_title="Urban Liquidity Intelligence Engine", page_icon="🏙️", layout="wide")


@st.cache_data
def load_data():
    return run_market_clustering(load_transactions())


df = load_data()

# --- Filtros ---
st.sidebar.header("Filtros de inversión")
alcaldias = sorted(df["alcaldia"].unique())
alcaldia = st.sidebar.selectbox("Alcaldía objetivo", alcaldias, index=alcaldias.index("Benito Juárez"))
max_budget = st.sidebar.slider("Presupuesto máximo (MXN)", 1_500_000, 20_000_000, 8_500_000, 500_000, format="$%d")
max_days = st.sidebar.slider("Liquidez máxima tolerada (días en mercado)", 20, 160, 90)
st.sidebar.caption("Datos sintéticos generados para fines demostrativos.")

filtered = df[(df["alcaldia"] == alcaldia) & (df["total_price"] <= max_budget) & (df["days_on_market"] <= max_days)]

# --- Encabezado y KPIs ---
st.title("🏙️ Motor de Inteligencia de Inversión Inmobiliaria y Liquidez Urbana")
st.markdown(f"**Análisis geoespacial y modelado predictivo de absorción de mercado — {alcaldia}, CDMX**")

forecast_df, mape = forecast_alcaldia_prices(df, alcaldia, steps=6)
empty = filtered.empty
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Propiedades viables", f"{len(filtered):,}")
c2.metric("Precio promedio / m²", "N/A" if empty else f"${filtered['price_m2'].mean():,.0f}")
c3.metric("Tiempo promedio de venta", "N/A" if empty else f"{filtered['days_on_market'].mean():.0f} días")
c4.metric("Distancia media a transporte", "N/A" if empty else f"{filtered['dist_transit_km'].mean():.2f} km")
c5.metric("Error del pronóstico (MAPE)", f"{mape:.1f}%", help="Backtest: entrenado sin los últimos 3 meses.")

st.divider()
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Mapa de clústeres de inversión (K-Means)")
    if empty:
        st.warning("Ninguna propiedad cumple los filtros. Amplía el presupuesto o los días de liquidez.")
    else:
        fig_map = px.scatter_map(
            filtered, lat="latitude", lon="longitude", color="segmento_inversion", size="area_m2",
            hover_data=["property_id", "price_m2", "days_on_market", "dist_transit_km"],
            color_discrete_sequence=px.colors.qualitative.Bold, zoom=12, height=450, map_style="carto-positron",
        )
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, legend={"orientation": "h", "y": -0.05})
        st.plotly_chart(fig_map, width="stretch")

with right:
    st.subheader(f"Pronóstico de precio/m² a 6 meses ({alcaldia})")
    fig_ts = px.line(
        forecast_df, x="Fecha", y="Precio_m2", color="Tipo", markers=True,
        labels={"Precio_m2": "Precio / m² (MXN)"}, height=450,
    )
    st.plotly_chart(fig_ts, width="stretch")

st.subheader("Elasticidad precio–distancia a transporte (toda la CDMX)")
fig_el = px.scatter(
    df, x="dist_transit_km", y="price_m2", color="segmento_inversion", opacity=0.5, trendline="ols",
    trendline_scope="overall", labels={"dist_transit_km": "Distancia a transporte (km)", "price_m2": "Precio / m²"},
)
st.plotly_chart(fig_el, width="stretch")

st.subheader("Top 10 oportunidades: mayor liquidez y mejor precio/m²")
if not empty:
    cols = ["property_id", "segmento_inversion", "area_m2", "price_m2", "total_price", "days_on_market", "dist_transit_km"]
    st.dataframe(filtered.sort_values(["days_on_market", "price_m2"]).head(10)[cols], width="stretch", hide_index=True)
