from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for Flask
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import seaborn as sns
import plotly.express as px
from ml_model import predict_future
from flask import send_file
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import plotly.io as pio
if pio.kaleido.scope is not None:
    pio.kaleido.scope.default_format = "png"




app = Flask(__name__)

DB_PATH = "ccpi.db"
CHART_DIR = "static/charts"

os.makedirs(CHART_DIR, exist_ok=True)

# ------------------ DATABASE CONNECTION ------------------

def get_db():
    return sqlite3.connect(DB_PATH)

def generate_choropleth_image(df, year, indicator, label):
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color=indicator,
        hover_name="country",
        color_continuous_scale="YlGnBu",
        title=f"{label} – {year}"
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    path = f"static/charts/map_{indicator}_{year}.png"

    # Export as image (THIS is the key)
    fig.write_image(path, width=900, height=500)

    return path


# ------------------ HOME DASHBOARD ------------------

@app.route("/", methods=["GET", "POST"])
def index():
    year = request.form.get("year", 2021)

    conn = get_db()

    years = pd.read_sql("SELECT DISTINCT year FROM ccpi ORDER BY year", conn)["year"].tolist()

    top_df = pd.read_sql("""
        SELECT country, overall_ccpi
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi DESC
        LIMIT 10
    """, conn, params=(year,))

    bottom_df = pd.read_sql("""
        SELECT country, overall_ccpi
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi ASC
        LIMIT 10
    """, conn, params=(year,))

    conn.close()

    # --------- TOP 10 CHART ---------
    plt.figure(figsize=(8, 5))
    plt.barh(top_df['country'], top_df['overall_ccpi'])
    plt.title(f"Top 10 Countries ({year})")
    plt.xlabel("CCPI Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    top_path = f"{CHART_DIR}/top10.png"
    plt.savefig(top_path)
    plt.close()

    # --------- BOTTOM 10 CHART ---------
    plt.figure(figsize=(8, 5))
    plt.barh(bottom_df['country'], bottom_df['overall_ccpi'], color="red")
    plt.title(f"Bottom 10 Countries ({year})")
    plt.xlabel("CCPI Score")
    plt.tight_layout()
    bottom_path = f"{CHART_DIR}/bottom10.png"
    plt.savefig(bottom_path)
    plt.close()

    return render_template(
        "index.html",
        years=years,
        selected_year=int(year),
        top_chart="/" + top_path,
        bottom_chart="/" + bottom_path
    )


#Visualize

@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    conn = get_db()

    # Dropdown data
    years = pd.read_sql(
        "SELECT DISTINCT year FROM ccpi ORDER BY year", conn
    )["year"].tolist()

    countries = pd.read_sql(
        "SELECT DISTINCT country FROM ccpi ORDER BY country", conn
    )["country"].tolist()

    # Defaults
    selected_year = int(request.form.get("year", years[0]))
    selected_country = request.form.get("country")

    top_chart = None
    bottom_chart = None
    trend_chart = None
    insight = None
    active_tab = request.form.get("active_tab", "top10")


    # ---------------- TOP 10 ----------------
    top_df = pd.read_sql("""
        SELECT country, overall_ccpi
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi DESC
        LIMIT 10
    """, conn, params=(selected_year,))

    plt.figure(figsize=(7.5, 4))
    plt.barh(top_df["country"], top_df["overall_ccpi"], color="#4C72B0")
    plt.gca().invert_yaxis()
    plt.title(f"Top 10 Countries – {selected_year}")
    plt.xlabel("Overall CCPI Score")
    plt.tight_layout()

    top_path = "static/charts/top10.png"
    plt.savefig(top_path)
    plt.close()
    top_chart = "/" + top_path

    # ---------------- BOTTOM 10 ----------------
    bottom_df = pd.read_sql("""
        SELECT country, overall_ccpi
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi ASC
        LIMIT 10
    """, conn, params=(selected_year,))

    plt.figure(figsize=(7.5, 4))
    plt.barh(bottom_df["country"], bottom_df["overall_ccpi"], color="#C44E52")
    plt.title(f"Bottom 10 Countries – {selected_year}")
    plt.xlabel("Overall CCPI Score")
    plt.tight_layout()

    bottom_path = "static/charts/bottom10.png"
    plt.savefig(bottom_path)
    plt.close()
    bottom_chart = "/" + bottom_path

    # ---------------- YEARLY TREND (YOUR EXACT PLOT) ----------------
    if selected_country:
        df = pd.read_sql("""
            SELECT year, overall_ccpi
            FROM ccpi
            WHERE country = ?
            ORDER BY year
        """, conn, params=(selected_country,))

        plt.figure(figsize=(8.5, 4.5))  # ideal for 2018–2026 span

        plt.plot(
            df['year'],
            df['overall_ccpi'],
            color="#4C72B0",
            linewidth=1.8,
            marker='o',
            markersize=4
        )

        plt.xticks(df['year'])

        best_idx = df['overall_ccpi'].idxmax()
        worst_idx = df['overall_ccpi'].idxmin()

        plt.scatter(
            df.loc[best_idx, 'year'],
            df.loc[best_idx, 'overall_ccpi'],
            color="#2E8B57",
            s=45,
            zorder=5
        )

        plt.scatter(
            df.loc[worst_idx, 'year'],
            df.loc[worst_idx, 'overall_ccpi'],
            color="#8B0000",
            s=45,
            zorder=5
        )

        plt.annotate(
            "Highest",
            (df.loc[best_idx, 'year'], df.loc[best_idx, 'overall_ccpi']),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=9
        )

        plt.annotate(
            "Lowest",
            (df.loc[worst_idx, 'year'], df.loc[worst_idx, 'overall_ccpi']),
            textcoords="offset points",
            xytext=(0, -14),
            ha='center',
            fontsize=9
        )

        y_min = df['overall_ccpi'].min()
        y_max = df['overall_ccpi'].max()
        padding = (y_max - y_min) * 0.12
        plt.ylim(y_min - padding, y_max + padding)

        plt.title(f"Year-wise CCPI Trend for {selected_country}", fontsize=13)
        plt.xlabel("Year", fontsize=11)
        plt.ylabel("Overall CCPI Score", fontsize=11)

        plt.grid(True, linestyle=':', alpha=0.4)

        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)

        plt.tight_layout()

        trend_path = "static/charts/trend.png"
        plt.savefig(trend_path)
        plt.close()

        trend_chart = "/" + trend_path
        insight = (
            "Improving trend"
            if df.iloc[-1]['overall_ccpi'] > df.iloc[0]['overall_ccpi']
            else "Declining or stagnant trend"
        )

    conn.close()

    return render_template(
        "visualize.html",
        years=years,
        countries=countries,
        selected_year=selected_year,
        selected_country=selected_country,
        top_chart=top_chart,
        bottom_chart=bottom_chart,
        trend_chart=trend_chart,
        insight=insight,
        active_tab=active_tab
    )




# ------------------ COUNTRY COMPARISON ------------------

@app.route("/compare", methods=["GET", "POST"])
def compare():
    conn = get_db()

    countries = pd.read_sql(
        "SELECT DISTINCT country FROM ccpi ORDER BY country",
        conn
    )["country"].tolist()

    data = None
    chart_path = None
    line_chart = None
    selected_year = None

    if request.method == "POST":
        c1 = request.form.get("country1")
        c2 = request.form.get("country2")
        selected_year = int(request.form.get("year"))

        # -------- TABLE DATA --------
        data = pd.read_sql("""
            SELECT country,
                   ghg_score,
                   renewable_energy,
                   energy_use,
                   climate_policy,
                   overall_ccpi
            FROM ccpi
            WHERE year = ? AND country IN (?, ?)
        """, conn, params=(selected_year, c1, c2))

        # -------- BAR CHART (EXISTING FEATURE) --------
        indicators = [
            "ghg_score",
            "renewable_energy",
            "energy_use",
            "climate_policy",
            "overall_ccpi"
        ]

        melted = data.melt(
            id_vars="country",
            value_vars=indicators,
            var_name="Indicator",
            value_name="Score"
        )

        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=melted,
            x="Indicator",
            y="Score",
            hue="country"
        )
        plt.title(f"Indicator-wise Comparison ({selected_year})")
        plt.xlabel("")
        plt.ylabel("Score")
        plt.xticks(rotation=30)
        plt.tight_layout()

        chart_path = "static/charts/compare_bar.png"
        plt.savefig(chart_path)
        plt.close()

        # -------- LINE CHART (NEW FEATURE – TREND) --------
        trend_df = pd.read_sql("""
            SELECT year, country, overall_ccpi
            FROM ccpi
            WHERE country IN (?, ?)
            ORDER BY year
        """, conn, params=(c1, c2))

        plt.figure(figsize=(6, 4))
        for country in trend_df["country"].unique():
            sub = trend_df[trend_df["country"] == country]
            plt.plot(
                sub["year"],
                sub["overall_ccpi"],
                linewidth=2,
                marker="o",
                label=country
            )

        plt.xticks(sorted(trend_df["year"].unique()))
        plt.xlabel("Year")
        plt.ylabel("Overall CCPI")
        plt.title("Overall CCPI Trend (2018–2026)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        line_chart = "static/charts/compare_trend.png"
        plt.savefig(line_chart)
        plt.close()

    conn.close()

    return render_template(
        "compare.html",
        countries=countries,
        data=data,
        chart_path="/" + chart_path if chart_path else None,
        line_chart="/" + line_chart if line_chart else None,
        selected_year=selected_year
    )




# ------------------ PREDICTIVE ANALYTICS ------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    conn = get_db()

    countries = pd.read_sql(
        "SELECT DISTINCT country FROM ccpi ORDER BY country",
        conn
    )["country"].tolist()

    conn.close()

    prediction = None

    if request.method == "POST":
        country = request.form.get("country")
        year = int(request.form.get("year"))

        prediction = predict_future(country, year)

        print("POST received:", country, year)


    return render_template(
        "predict.html",
        countries=countries,
        prediction=prediction
    )




#CHoropleth

@app.route("/map", methods=["GET", "POST"])
def choropleth():
    conn = get_db()

    # Available years
    years = pd.read_sql(
        "SELECT DISTINCT year FROM ccpi ORDER BY year", conn
    )["year"].tolist()

    # Climate indicators
    indicators = {
        "ghg_score": "GHG Emissions Score",
        "renewable_energy": "Renewable Energy Score",
        "energy_use": "Energy Use Score",
        "climate_policy": "Climate Policy Score",
        "overall_ccpi": "Overall CCPI Score"
    }

    # Defaults
    selected_year = request.form.get("year", years[0])
    selected_indicator = request.form.get("indicator", "overall_ccpi")

    # Fetch data
    df = pd.read_sql("""
        SELECT country, year, ghg_score, renewable_energy,
               energy_use, climate_policy, overall_ccpi
        FROM ccpi
        WHERE year = ?
    """, conn, params=(selected_year,))

    conn.close()

    # Ensure numeric values
    df[selected_indicator] = pd.to_numeric(
        df[selected_indicator], errors="coerce"
    )

    # Drop rows where selected indicator is missing
    df = df.dropna(subset=[selected_indicator])

    if df.empty:
        return render_template(
            "map.html",
            map_html=None,
            years=years,
            indicators=indicators,
            selected_year=int(selected_year),
            selected_indicator=selected_indicator,
            message="No data available for the selected indicator and year."
        )



    # Choropleth map
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color=selected_indicator,
        hover_name="country",
        color_continuous_scale="YlGnBu",
        range_color=(df[selected_indicator].min(), df[selected_indicator].max()),
        title=f"{indicators[selected_indicator]} – {selected_year}"
    )


    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Score")
    )

    map_html = fig.to_html(full_html=False)

    return render_template(
        "map.html",
        map_html=map_html,
        years=years,
        indicators=indicators,
        selected_year=int(selected_year),
        selected_indicator=selected_indicator
    )

@app.route("/map_test")
def map_test():
    import plotly.express as px
    df = px.data.gapminder().query("year==2007")
    fig = px.choropleth(df, locations="country", locationmode="country names",
                        color="lifeExp", title="Test Map")
    return fig.to_html()

#EXPORT

@app.route("/reports", methods=["GET"])
def reports():
    conn = get_db()

    years = pd.read_sql(
        "SELECT DISTINCT year FROM ccpi ORDER BY year", conn
    )["year"].tolist()

    conn.close()

    return render_template("reports.html", years=years)

@app.route("/export/excel", methods=["POST"])
def export_excel():
    year = int(request.form.get("year"))

    conn = get_db()
    df = pd.read_sql("""
        SELECT country, year,
               ghg_score, renewable_energy,
               energy_use, climate_policy,
               overall_ccpi, rank
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi DESC
    """, conn, params=(year,))
    conn.close()

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=f"CCPI_{year}")

    output.seek(0)

    return send_file(
        output,
        download_name=f"CCPI_Report_{year}.xlsx",
        as_attachment=True
    )

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    year = int(request.form.get("year"))

    conn = get_db()

    # -------- FETCH DATA --------
    top_df = pd.read_sql("""
        SELECT country, overall_ccpi
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi DESC
        LIMIT 10
    """, conn, params=(year,))

    bottom_df = pd.read_sql("""
        SELECT country, overall_ccpi
        FROM ccpi
        WHERE year = ?
        ORDER BY overall_ccpi ASC
        LIMIT 10
    """, conn, params=(year,))

    # -------- FETCH DATA FOR MAPS --------
    map_df = pd.read_sql("""
        SELECT country, ghg_score, renewable_energy,
            energy_use, climate_policy, overall_ccpi
        FROM ccpi
        WHERE year = ?
    """, conn, params=(year,))


    conn.close()

    os.makedirs("static/reports", exist_ok=True)

    # -------- GENERATE CHARTS FOR THIS YEAR --------
    top_chart = f"static/reports/top10_{year}.png"
    bottom_chart = f"static/reports/bottom10_{year}.png"

    # Top 10
    plt.figure(figsize=(7, 4))
    plt.barh(top_df["country"], top_df["overall_ccpi"], color="#4C72B0")
    plt.gca().invert_yaxis()
    plt.title(f"Top 10 Countries – {year}")
    plt.xlabel("Overall CCPI Score")
    plt.tight_layout()
    plt.savefig(top_chart)
    plt.close()

    # Bottom 10
    plt.figure(figsize=(7, 4))
    plt.barh(bottom_df["country"], bottom_df["overall_ccpi"], color="#C44E52")
    plt.title(f"Bottom 10 Countries – {year}")
    plt.xlabel("Overall CCPI Score")
    plt.tight_layout()
    plt.savefig(bottom_chart)
    plt.close()

    # -------- GENERATE CHOROPLETH MAPS --------
    indicators = {
        "ghg_score": "GHG Emissions Score",
        "renewable_energy": "Renewable Energy Score",
        "energy_use": "Energy Use Score",
        "climate_policy": "Climate Policy Score",
        "overall_ccpi": "Overall CCPI Score"
    }

    map_paths = []

    for key, label in indicators.items():
        path = generate_choropleth_image(
            map_df,
            year,
            key,
            label
        )
        map_paths.append((label, path))




    # -------- BUILD PDF --------
    pdf_path = f"static/reports/CCPI_Report_{year}.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        f"<b>Climate Change Performance Index (CCPI) Report – {year}</b>",
        styles["Title"]
    ))

    elements.append(Paragraph(
        "This report presents an analysis of country-wise climate performance "
        "based on the Climate Change Performance Index (CCPI).",
        styles["Normal"]
    ))

    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph("Top 10 Performing Countries", styles["Heading2"]))
    elements.append(Image(top_chart, width=400, height=250))

    elements.append(Paragraph("<br/>Bottom 10 Performing Countries", styles["Heading2"]))
    elements.append(Image(bottom_chart, width=400, height=250))

    elements.append(Paragraph("<br/>Global Climate Performance Maps", styles["Heading2"]))

    for label, path in map_paths:
        elements.append(Paragraph(label, styles["Heading3"]))
        elements.append(Image(path, width=450, height=250))


    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)






# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

