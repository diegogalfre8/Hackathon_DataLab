import json
import urllib.request
import io
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ============================================================
# CONFIGURATION INITIALE
# ============================================================

# Charger les données
print("📥 Chargement des données...")

# --- Données de couverture vaccinale (SPF) ---
REG_URL = "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-region/exports/csv"
df_grippe = pd.read_csv(REG_URL, sep=";")

# Nettoyage colonnes
df_grippe.columns = (
    df_grippe.columns
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("é", "e", regex=False)
    .str.replace("è", "e", regex=False)
    .str.replace("à", "a", regex=False)
)

code_col = 'reg'
name_col = 'reglib'
df_grippe = df_grippe.drop_duplicates()
df_grippe["an_mesure"] = df_grippe["an_mesure"].replace("NC", np.nan).astype(float).astype("Int64")

GRIPPE_COLUMNS = [
    c for c in ['grip_moins65', 'grip_65plus', 'grip_6574', 'grip_75plus', 'grip_resid', 'grip_pro']
    if c in df_grippe.columns
]

# --- Données doses/actes ---
URL_DOSES = "https://www.data.gouv.fr/api/1/datasets/r/848e3e48-4971-4dc5-97c7-d856cdfde2f6"
data_doses = requests.get(URL_DOSES).content
df_doses = pd.read_csv(io.BytesIO(data_doses), sep=None, engine="python")

df_doses.columns = df_doses.columns.str.strip().str.lower()
df_doses = df_doses.rename(columns={
    "region": "region_name",
    "code": "region_code",
    "variable": "variable",
    "groupe": "groupe",
    "valeur": "valeur"
})

df_doses_pivot = df_doses.pivot_table(
    index=["region_code", "region_name", "groupe"],
    columns="variable",
    values="valeur",
    aggfunc="first"
).reset_index()

df_doses_pivot = df_doses_pivot.rename(columns={
    "DOSES(J07E1)": "doses",
    "ACTE(VGP)": "actes"
})

df_doses_pivot["non_utilisees"] = df_doses_pivot["doses"] - df_doses_pivot["actes"]
df_doses_pivot["region_code"] = df_doses_pivot["region_code"].astype(str).str.zfill(2)

# --- GeoJSON ---
GEOJSON_URL = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson"
with urllib.request.urlopen(GEOJSON_URL) as resp:
    geojson_admin = json.load(io.TextIOWrapper(resp, encoding="utf-8"))

METRO_REGION_CODES = ['11','24','27','28','32','44','52','53','75','76','84','93','94']

latest_year_grippe = df_grippe['an_mesure'].dropna().max()

def normalize_code_series(s):
    return s.astype(str).str.replace(r"\\.0$", "", regex=True)

def compute_stats_grippe(year):
    d = df_grippe[df_grippe['an_mesure'] == year].copy()
    stats = d.groupby([code_col, name_col])[GRIPPE_COLUMNS].mean().reset_index()
    stats[code_col] = normalize_code_series(stats[code_col])
    return stats[stats[code_col].isin(METRO_REGION_CODES)]

print("✅ Données chargées avec succès!")

# ============================================================
# FONCTIONS DE GRAPHIQUES
# ============================================================

def make_grippe_map(column, year, color_scale="RdYlGn"):
    """Carte choroplèthe - Vaccination grippe"""
    data = compute_stats_grippe(year)
    
    filtered_geojson = {
        **geojson_admin,
        'features': [
            f for f in geojson_admin['features']
            if f.get('properties', {}).get('code') in METRO_REGION_CODES
        ]
    }
    
    fig = px.choropleth_mapbox(
        data,
        geojson=filtered_geojson,
        locations=code_col,
        featureidkey="properties.code",
        color=column,
        color_continuous_scale=color_scale,
        mapbox_style="carto-positron",
        center={"lat": 46.5, "lon": 2.2},
        zoom=4.15,
        opacity=0.85,
        hover_name=name_col,
        hover_data={code_col: True, column: ":.1f"},
        title=f"Taux de vaccination grippe ({year})"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Taux (%)", ticksuffix=" %"),
        title=dict(x=0.5, font=dict(size=14))
    )
    return fig

def make_grippe_bars(year):
    """Graphique comparatif - Vaccination grippe"""
    data = compute_stats_grippe(year)
    fig = go.Figure()
    
    for col in GRIPPE_COLUMNS:
        fig.add_bar(
            x=data[name_col],
            y=data[col],
            name=col.replace("grip_", "").replace("_", " ").title()
        )
    
    fig.update_layout(
        barmode="group",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Comparatif par région - Vaccination grippe ({year})",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    return fig

def make_doses_map(column="non_utilisees", groupe="65 ans et plus", palette="YlOrRd"):
    """Carte choroplèthe - Doses/actes"""
    data = df_doses_pivot[df_doses_pivot["groupe"] == groupe].copy()
    data = data[data["region_code"].isin(METRO_REGION_CODES)]
    
    filtered_geojson = {
        **geojson_admin,
        'features': [
            f for f in geojson_admin['features']
            if f.get('properties', {}).get('code') in METRO_REGION_CODES
        ]
    }
    
    fig = px.choropleth_mapbox(
        data,
        geojson=filtered_geojson,
        locations="region_code",
        featureidkey="properties.code",
        color=column,
        color_continuous_scale=palette,
        mapbox_style="carto-positron",
        center={"lat": 46.5, "lon": 2.2},
        zoom=4.15,
        opacity=0.85,
        hover_name="region_name",
        hover_data={
            "region_code": True,
            "doses": ":.0f",
            "actes": ":.0f",
            "non_utilisees": ":.0f"
        },
        title=f"{column.replace('_',' ').title()} – {groupe}"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Nombre/100k"),
        title=dict(x=0.5, font=dict(size=14))
    )
    return fig

def make_doses_bars(groupe="65 ans et plus"):
    """Graphique comparatif - Doses/actes"""
    data = df_doses_pivot[df_doses_pivot["groupe"] == groupe].copy()
    data = data[data["region_code"].isin(METRO_REGION_CODES)]
    
    fig = go.Figure()
    
    fig.add_bar(
        x=data["region_name"],
        y=data["doses"],
        name="Doses administrables",
        hovertemplate="%{x}<br>Doses: %{y:.0f}<extra></extra>"
    )
    fig.add_bar(
        x=data["region_name"],
        y=data["actes"],
        name="Actes réalisés",
        hovertemplate="%{x}<br>Actes: %{y:.0f}<extra></extra>"
    )
    fig.add_bar(
        x=data["region_name"],
        y=data["non_utilisees"],
        name="Doses non utilisées",
        hovertemplate="%{x}<br>Non utilisées: %{y:.0f}<extra></extra>"
    )
    
    fig.update_layout(
        barmode="group",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Comparatif doses/actes – {groupe}",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    return fig

# ============================================================
# CRÉATION DE L'APPLICATION DASH
# ============================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📊 Dashboard Hackathon - Vaccination", className="text-center mb-4 mt-4")
        ])
    ]),
    
    # ===== SECTION 1 : VACCINATION GRIPPE =====
    dbc.Row([
        dbc.Col([
            html.H3("🩹 Vaccination Grippe", className="mt-4 mb-3")
        ])
    ]),
    
    # Contrôles pour la grippe
    dbc.Row([
        dbc.Col([
            html.Label("Tranche d'âge:", className="fw-bold"),
            dcc.Dropdown(
                id="grippe-column-dropdown",
                options=[{"label": col.replace("grip_", "").replace("_", " ").title(), "value": col} 
                        for col in GRIPPE_COLUMNS],
                value="grip_65plus",
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Année:", className="fw-bold"),
            dcc.Dropdown(
                id="grippe-year-dropdown",
                options=[{"label": str(int(year)), "value": year} 
                        for year in sorted(df_grippe['an_mesure'].dropna().unique())],
                value=latest_year_grippe,
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Palette de couleurs:", className="fw-bold"),
            dcc.Dropdown(
                id="grippe-palette-dropdown",
                options=[{"label": p, "value": p} for p in ["RdYlGn", "Viridis", "Plasma", "YlOrRd", "Blues"]],
                value="RdYlGn",
                clearable=False
            )
        ], md=3),
    ], className="mb-3"),
    
    # Graphiques grippe
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-grippe-map",
                type="default",
                children=[dcc.Graph(id="grippe-map")]
            )
        ], md=6),
        dbc.Col([
            dcc.Loading(
                id="loading-grippe-bars",
                type="default",
                children=[dcc.Graph(id="grippe-bars")]
            )
        ], md=6),
    ], className="mb-4"),
    
    # ===== SECTION 2 : DOSES/ACTES =====
    dbc.Row([
        dbc.Col([
            html.H3("💉 Doses et Actes Réalisés", className="mt-4 mb-3")
        ])
    ]),
    
    # Contrôles pour les doses
    dbc.Row([
        dbc.Col([
            html.Label("Indicateur:", className="fw-bold"),
            dcc.Dropdown(
                id="doses-column-dropdown",
                options=[
                    {"label": "Doses administrables", "value": "doses"},
                    {"label": "Actes réalisés", "value": "actes"},
                    {"label": "Doses non utilisées", "value": "non_utilisees"}
                ],
                value="non_utilisees",
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Tranche d'âge:", className="fw-bold"),
            dcc.Dropdown(
                id="doses-groupe-dropdown",
                options=[{"label": g, "value": g} for g in sorted(df_doses_pivot["groupe"].unique())],
                value="65 ans et plus",
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Palette de couleurs:", className="fw-bold"),
            dcc.Dropdown(
                id="doses-palette-dropdown",
                options=[{"label": p, "value": p} for p in ["YlOrRd", "Viridis", "Plasma", "Blues", "Greens"]],
                value="YlOrRd",
                clearable=False
            )
        ], md=3),
    ], className="mb-3"),
    
    # Graphiques doses
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-doses-map",
                type="default",
                children=[dcc.Graph(id="doses-map")]
            )
        ], md=6),
        dbc.Col([
            dcc.Loading(
                id="loading-doses-bars",
                type="default",
                children=[dcc.Graph(id="doses-bars")]
            )
        ], md=6),
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Dashboard créé à partir des données de Santé Publique France et Data.gouv.fr", 
                   className="text-center text-muted small")
        ])
    ])
    
], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# ============================================================
# CALLBACKS POUR LES GRAPHIQUES
# ============================================================

@app.callback(
    Output("grippe-map", "figure"),
    [Input("grippe-column-dropdown", "value"),
     Input("grippe-year-dropdown", "value"),
     Input("grippe-palette-dropdown", "value")]
)
def update_grippe_map(column, year, palette):
    return make_grippe_map(column, year, palette)

@app.callback(
    Output("grippe-bars", "figure"),
    Input("grippe-year-dropdown", "value")
)
def update_grippe_bars(year):
    return make_grippe_bars(year)

@app.callback(
    Output("doses-map", "figure"),
    [Input("doses-column-dropdown", "value"),
     Input("doses-groupe-dropdown", "value"),
     Input("doses-palette-dropdown", "value")]
)
def update_doses_map(column, groupe, palette):
    return make_doses_map(column, groupe, palette)

@app.callback(
    Output("doses-bars", "figure"),
    Input("doses-groupe-dropdown", "value")
)
def update_doses_bars(groupe):
    return make_doses_bars(groupe)

# ============================================================
# LANCEMENT DE L'APPLICATION
# ============================================================

if __name__ == "__main__":
    print("🚀 Démarrage du dashboard...")
    print("📍 Accédez au dashboard via l'URL affichée ci-dessous")
    app.run(debug=False, host="0.0.0.0", port=8050)
