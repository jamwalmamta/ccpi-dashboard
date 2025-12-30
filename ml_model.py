import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
import numpy as np

DB_PATH = "ccpi.db"


def predict_indicator(years, values, target_year):
    """
    Robust indicator prediction using bounded linear trend
    """
    X = np.array(years).reshape(-1, 1)
    y = np.array(values)

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict([[target_year]])[0]

    # Soft bounds using recent history
    lower = min(values[-3:])
    upper = max(values[-3:])

    return max(lower, min(upper, pred))


def predict_future(country, target_year):
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT year,
               ghg_score,
               renewable_energy,
               energy_use,
               climate_policy,
               overall_ccpi
        FROM ccpi
        WHERE country = ?
        ORDER BY year
    """, conn, params=(country,))

    conn.close()

    indicators = [
        "ghg_score",
        "renewable_energy",
        "energy_use",
        "climate_policy"
    ]

    predictions = {}
    predicted_indicators = []

    for ind in indicators:
        sub = df[["year", ind]].dropna()

        # If too little data → use last known value
        if len(sub) < 4:
            pred = sub[ind].iloc[-1]
        else:
            pred = predict_indicator(
                sub["year"].tolist(),
                sub[ind].tolist(),
                target_year
            )

        predictions[ind] = round(pred, 2)
        predicted_indicators.append(pred)

    # -------- CCPI prediction (from indicators) --------
    valid_rows = df[indicators + ["overall_ccpi"]].dropna()

    ccpi_model = LinearRegression()
    ccpi_model.fit(valid_rows[indicators], valid_rows["overall_ccpi"])

    predicted_ccpi = ccpi_model.predict(
        [predicted_indicators]
    )[0]

    predicted_ccpi = max(0, min(100, predicted_ccpi))

    return {
        "country": country,
        "year": target_year,
        "ghg_score": predictions["ghg_score"],
        "renewable_energy": predictions["renewable_energy"],
        "energy_use": predictions["energy_use"],
        "climate_policy": predictions["climate_policy"],
        "overall_ccpi": round(predicted_ccpi, 2)
    }
