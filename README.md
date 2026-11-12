# 🏙️ Motor de Inteligencia de Inversión Inmobiliaria y Pronóstico de Liquidez Urbana

Sistema end-to-end de analítica geoespacial, segmentación con Machine Learning y pronóstico de series de tiempo para identificar oportunidades de inversión inmobiliaria y estimar la velocidad de venta (liquidez) en la Ciudad de México.

🔗 **Demo en vivo:** https://urban-liquidity-engine.streamlit.app
👤 **Autor:** Alejandro De Antuñano

> **Nota sobre los datos:** el dataset (2,500 transacciones en 6 alcaldías, ene-2024 a dic-2025) es **sintético y reproducible** (`seed=42`). Se generó con supuestos realistas de precio/m² y tiempo de venta por alcaldía, más una penalización por distancia al transporte. El pipeline está pensado para sustituirlo por datos reales (p. ej., portales inmobiliarios o catastro) sin cambiar los modelos.

---

## 1. Problema identificado
Los inversionistas y desarrolladores suelen evaluar zonas solo por el **precio promedio histórico**. Con eso pasan por alto dos variables que determinan el retorno:
1. **Liquidez:** cuántos días tarda una propiedad en venderse, es decir, cuánto tiempo queda inmovilizado el capital.
2. **Infraestructura urbana:** cuánto cambia el precio según la cercanía a los nodos de transporte y a los centros financieros.

Sin estas variables, se invierte en zonas sobrevaluadas o se dejan pasar corredores emergentes.

## 2. Solución desarrollada
| Etapa | Implementación |
|---|---|
| Ingeniería de datos | Generación vectorizada de transacciones; distancia **Haversine** al nodo de transporte más cercano; persistencia en **SQLite** |
| Segmentación no supervisada | **K-Means** (k=4) sobre `precio/m²`, `días en mercado` y `distancia a transporte`, estandarizados; los clústeres reciben etiquetas de negocio ordenadas por precio |
| Pronóstico | **Holt-Winters** (tendencia aditiva) del precio/m² mensual por alcaldía a 6 meses, validado con **backtest holdout** de 3 meses (MAPE) |
| Producto de datos | Dashboard **Streamlit** con filtros de presupuesto y liquidez, mapa de clústeres, curva de pronóstico con su error, elasticidad precio–distancia y ranking de las 10 mejores oportunidades |
| Calidad | Tests con **pytest** y CI en **GitHub Actions** en cada push |

## 3. Stack
Python · pandas · NumPy · SQLite · scikit-learn · statsmodels · Streamlit · Plotly · pytest · GitHub Actions

## 4. Resultados (medidos sobre el dataset)
**Segmentos de inversión detectados**

| Segmento | Propiedades | Precio/m² prom. | Días en mercado | Dist. a transporte |
|---|---:|---:|---:|---:|
| Prime / Alta Demanda | 422 | $85,391 | 75 | 1.9 km |
| Consolidado / Liquidez Media | 1,050 | $58,399 | 59 | 2.3 km |
| Corredor Emergente (Alta Oportunidad) | 653 | $49,259 | 88 | 4.6 km |
| Periferia / Baja Liquidez | 375 | $34,659 | 130 | 7.1 km |

**Error del pronóstico (MAPE, backtest de 3 meses):** Miguel Hidalgo 0.6% · Benito Juárez 0.9% · Cuauhtémoc 1.0% · Álvaro Obregón 1.4% · Coyoacán 1.7% · Tlalpan 2.4%.
_El error es bajo porque los datos sintéticos tienen una tendencia suave. Con datos reales se espera un error mayor, y el backtest está incluido justamente para medirlo._

**Elasticidad:** correlación precio/m² vs. distancia a transporte = **−0.64**. La cercanía al transporte es el principal factor de precio después de la alcaldía.

**Valor de negocio**
- Filtra miles de propiedades por presupuesto, liquidez tolerada y alcaldía en segundos.
- Separa los activos **Prime** (caros y no necesariamente los más líquidos) de los **Consolidados** (que son los que se venden más rápido). Así el inversionista puede priorizar rotación de capital o plusvalía.
- Cada pronóstico se muestra junto con su error medido, para que la decisión se tome conociendo la incertidumbre.

## 5. Ejecución local
```bash
git clone https://github.com/AlejandroDeAntu/urban-liquidity-intelligence-engine.git
cd urban-liquidity-intelligence-engine
pip install -r requirements.txt
python src/data_generator.py      # opcional: la app genera la BD si no existe
streamlit run src/app.py
pip install pytest && pytest -v   # tests
```

## 6. Estructura
```
├── .github/workflows/ci.yml   # pytest en cada push / PR
├── src/
│   ├── data_generator.py      # dataset sintético + Haversine + SQLite
│   ├── ml_models.py           # K-Means, Holt-Winters, backtest MAPE
│   └── app.py                 # dashboard Streamlit
├── tests/test_pipeline.py
├── requirements.txt
└── pytest.ini
```

## 7. Siguientes pasos
- Reemplazar los datos sintéticos por datos reales (scraping de portales, datos abiertos CDMX).
- Agregar variables urbanas: amenidades, densidad comercial, índice de seguridad.
- Modelo supervisado de días en mercado (gradient boosting) para estimar la liquidez de cada propiedad.
