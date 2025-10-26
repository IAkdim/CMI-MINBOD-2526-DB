import pandas as pd
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# DATA LOADING AND PREPARATION
BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "dataset" / "pwt110_imputed.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {CSV_PATH}. Make sure the `dataset/` folder is present.")

df = pd.read_csv(CSV_PATH)
df = df.copy()

if 'year' in df.columns:
    df['year'] = df['year'].astype(int)

if 'gdp_pc' not in df.columns or df['gdp_pc'].isna().all():
    if {'rgdpe', 'pop'}.issubset(df.columns):
        df['gdp_pc'] = df['rgdpe'] / df['pop']
    else:
        for col in ['rgdpo', 'rgdpe']:
            if col in df.columns:
                df['gdp_pc'] = df[col]
                break

if 'gdp_pc_growth' not in df.columns:
    if 'gdp_pc' in df.columns:
        df = df.sort_values(['country', 'year'])
        df['gdp_pc_growth'] = df.groupby('country')['gdp_pc'].pct_change() * 100

df['gdp_pc'] = pd.to_numeric(df['gdp_pc'], errors='coerce')
if 'gdp_pc_growth' in df.columns:
    df['gdp_pc_growth'] = pd.to_numeric(df['gdp_pc_growth'], errors='coerce')

# PRECOMPUTE CAPITAL ANALYSIS COLUMNS
# K/L (capital per worker), K/Y (capital-output ratio)
if 'rnna' in df.columns and 'emp' in df.columns:
    df['kl'] = df['rnna'] / df['emp']
if 'rnna' in df.columns and 'rgdpo' in df.columns:
    df['ky'] = df['rnna'] / df['rgdpo']

# Log10 transformations for capital analysis
for col in ['rgdpo', 'rnna', 'kl', 'ky']:
    if col in df.columns:
        df[f'log10_{col}'] = np.log10(df[col].replace(0, np.nan))

# PRECOMPUTE LABOR ANALYSIS COLUMNS
# Labor productivity (output per worker)
if 'rgdpo' in df.columns and 'emp' in df.columns:
    df['labprod'] = df['rgdpo'] / df['emp']

# Log10 transformations for labor analysis
if 'rgdpo' in df.columns:
    df['log10_rgdpo'] = np.log10(df['rgdpo'].replace(0, np.nan))
if 'emp' in df.columns:
    df['log10_emp'] = np.log10(df['emp'].replace(0, np.nan))
if 'hc' in df.columns:
    df['log10_hc'] = np.log10(df['hc'].replace(0, np.nan))
if 'labsh' in df.columns:
    df['log10_labsh'] = np.log10(df['labsh'].clip(lower=1e-6).replace(0, np.nan))
if 'labprod' in df.columns:
    df['log10_labprod'] = np.log10(df['labprod'].replace(0, np.nan))

# PRECOMPUTE PRODUCTIVITY ANALYSIS COLUMNS
# Productivity measures
if 'rgdpo' in df.columns and 'rnna' in df.columns:
    df['capprod'] = df['rgdpo'] / df['rnna']  # capital productivity
if 'rtfpna' in df.columns:
    df['tfp'] = df['rtfpna']  # total factor productivity

# Log10 transformations for productivity analysis
for col in ['capprod', 'tfp']:
    if col in df.columns:
        df[f'log10_{col}'] = np.log10(df[col].clip(lower=1e-8))

# Growth rates for decomposition
df = df.sort_values(['countrycode', 'year'])
if 'rgdpo' in df.columns:
    df['gdp_growth'] = df.groupby('countrycode')['rgdpo'].pct_change() * 100
if 'tfp' in df.columns:
    df['tfp_growth'] = df.groupby('countrycode')['tfp'].pct_change() * 100
if 'labprod' in df.columns:
    df['labprod_growth'] = df.groupby('countrycode')['labprod'].pct_change() * 100
if 'capprod' in df.columns:
    df['capprod_growth'] = df.groupby('countrycode')['capprod'].pct_change() * 100

YEARS = sorted(int(y) for y in df['year'].unique())
MIN_YEAR, MAX_YEAR = min(YEARS), max(YEARS)
COUNTRIES = df['country'].fillna(df['countrycode']).unique()

latest = df[df['year'] == MAX_YEAR]
default_countries = ['Kuwait', 'Ukraine', 'Türkiye']

KEY_INDICATORS = {
    'gdp_pc': 'GDP per capita (2021 USD)',
    'rgdpe': 'Real GDP - expenditure (mil. 2021 USD)',
    'rgdpo': 'Real GDP - output (mil. 2021 USD)',
    'pop': 'Population (millions)',
    'emp': 'Employment (millions)',
    'avh': 'Average hours worked (hours/year)',
    'hc': 'Human capital index (scale)',
    'labsh': 'Labor share of income (fraction)',
    'csh_c': 'Consumption share of GDP (fraction)',
    'csh_i': 'Investment share of GDP (fraction)',
    'csh_g': 'Government share of GDP (fraction)',
    'csh_x': 'Export share of GDP (fraction)',
    'csh_m': 'Import share of GDP (fraction)'
}

AVAILABLE_INDICATORS = {k: v for k, v in KEY_INDICATORS.items() if k in df.columns}

INDICATOR_DESCRIPTIONS = {
    'countrycode': '3-letter ISO country code',
    'country': 'Country name',
    'currency_unit': 'Currency unit',
    'year': 'Year',
    'rgdpe': 'Expenditure-side real GDP at chained PPPs (in mil. 2021US$)',
    'rgdpo': 'Output-side real GDP at chained PPPs (in mil. 2021US$)',
    'pop': 'Population (in millions)',
    'emp': 'Number of persons engaged (in millions)',
    'avh': 'Average annual hours worked by persons engaged',
    'hc': 'Human capital index, based on years of schooling and returns to education',
    'ccon': 'Real consumption of households and government, at current PPPs (in mil. 2021US$)',
    'cda': 'Real domestic absorption (real consumption plus investment), at current PPPs (in mil. 2021US$)',
    'cgdpe': 'Expenditure-side real GDP at current PPPs (in mil. 2021US$)',
    'cgdpo': 'Output-side real GDP at current PPPs (in mil. 2021US$)',
    'cn': 'Capital stock at current PPPs (in mil. 2021US$)',
    'ck': 'Capital services levels at current PPPs (USA=1)',
    'ctfp': 'TFP level at current PPPs (USA=1)',
    'cwtfp': 'Welfare-relevant TFP levels at current PPPs (USA=1)',
    'rgdpna': 'Real GDP at constant 2021 national prices (in mil. 2021US$)',
    'rconna': 'Real consumption at constant 2021 national prices (in mil. 2021US$)',
    'rdana': 'Real domestic absorption at constant 2021 national prices (in mil. 2021US$)',
    'rnna': 'Capital stock at constant 2021 national prices (in mil. 2021US$)',
    'rkna': 'Capital services at constant 2021 national prices (2021=1)',
    'rtfpna': 'TFP at constant national prices (2021=1)',
    'rwtfpna': 'Welfare-relevant TFP at constant national prices (2021=1)',
    'labsh': 'Share of labor compensation in GDP at current national prices',
    'irr': 'Real internal rate of return',
    'delta': 'Average depreciation rate of the capital stock',
    'xr': 'Exchange rate, national currency/USD (market+estimated)',
    'pl_con': 'Price level of CCON (PPP/XR), price level of USA GDPo in 2021=1',
    'pl_da': 'Price level of CDA (PPP/XR), price level of USA GDPo in 2021=1',
    'pl_gdpo': 'Price level of CGDPo (PPP/XR), price level of USA GDPo in 2021=1',
    'i_cig': '0/1/2/3/4: relative price data for consumption, investment and government (extrapolated/benchmark/interpolated/etc.)',
    'i_xm': '0/1/2: relative price data for exports and imports is extrapolated (0), benchmark (1) or interpolated (2)',
    'i_xr': '0/1: the exchange rate is market-based (0) or estimated (1)',
    'i_outlier': '0/1: observation on pl_gdpe or pl_gdpo is not an outlier (0) or is an outlier (1)',
    'i_irr': '0/1/2/3: irr observation not an outlier (0), biased (1), lower bound (2), or outlier (3)',
    'cor_exp': 'Correlation between expenditure shares of the country and the US (benchmark observations only)',
    'csh_c': 'Share of household consumption at current PPPs',
    'csh_i': 'Share of gross capital formation at current PPPs',
    'csh_g': 'Share of government consumption at current PPPs',
    'csh_x': 'Share of merchandise exports at current PPPs',
    'csh_m': 'Share of merchandise imports at current PPPs',
    'csh_r': 'Share of residual trade and GDP statistical discrepancy at current PPPs',
    'pl_c': 'Price level of household consumption, price level of USA GDPo in 2021=1',
    'pl_i': 'Price level of capital formation, price level of USA GDPo in 2021=1',
    'pl_g': 'Price level of government consumption, price level of USA GDPo in 2021=1',
    'pl_x': 'Price level of exports, price level of USA GDPo in 2021=1',
    'pl_m': 'Price level of imports, price level of USA GDPo in 2021=1',
    'pl_n': 'Price level of the capital stock, price level of USA in 2021=1',
    'pl_k': 'Price level of the capital services, price level of USA=1'
}

# APP CONFIGURATION
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
CHART_LAYOUT = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, color='#222'),
    xaxis=dict(
        gridcolor='#e0e0e0',
        linecolor='#333',
        linewidth=2,
        showgrid=True,
        zeroline=True,
        zerolinecolor='#999',
        zerolinewidth=1
    ),
    yaxis=dict(
        gridcolor='#e0e0e0',
        linecolor='#333',
        linewidth=2,
        showgrid=True,
        zeroline=True,
        zerolinecolor='#999',
        zerolinewidth=1
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#ccc',
        borderwidth=1
    )
)

COLORS = px.colors.qualitative.Bold

# DASHBOARD LAYOUT
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("PWT 11.0 | Exploratory Data Analysis", className='text-center mb-2')
        ])
    ]),

    # Tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="GDP", tab_id="tab-gdp", children=[
                    html.Div([
                        html.H4("GDP Analysis", style={'marginTop': '10px', 'marginBottom': '15px'}),

                        # Controls
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Countries", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='gdp-country-dropdown',
                                    options=[{'label': c, 'value': c} for c in COUNTRIES],
                                    value=default_countries,
                                    multi=True,
                                    placeholder='Select countries...'
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Select Year", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id='gdp-year-slider',
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    value=MAX_YEAR,
                                    marks={y: str(y) for y in range(MIN_YEAR, MAX_YEAR+1, 10)},
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], width=6)
                        ], style={'marginBottom': '20px'}),

                        # Visualizations Grid
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='gdp-world-map', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='gdp-distribution', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ], className='mb-2'),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='gdp-scatter-log', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='gdp-timeseries', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ])
                    ], style={'padding': '10px'})
                ]),
                dbc.Tab(label="Capital", tab_id="tab-capital", children=[
                    html.Div([
                        html.H4("Capital Analysis", style={'marginTop': '10px', 'marginBottom': '15px'}),

                        # Controls
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Year", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id='capital-year-slider',
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    value=MAX_YEAR,
                                    marks={y: str(y) for y in range(MIN_YEAR, MAX_YEAR+1, 10)},
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Select Countries (for time series)", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='capital-country-dropdown',
                                    options=[{'label': c, 'value': c} for c in COUNTRIES],
                                    value=['United States', 'China', 'India', 'Germany', 'Japan'] if all(c in COUNTRIES for c in ['United States', 'China', 'India', 'Germany', 'Japan']) else default_countries,
                                    multi=True,
                                    placeholder='Select countries...'
                                ),
                            ], width=6)
                        ], style={'marginBottom': '20px'}),

                        # Visualizations Grid
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='capital-gdp-scatter', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='capital-kl-ky-timeseries', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ], className='mb-2'),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='capital-linearity-diagnostic', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='capital-residuals-plot', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ])
                    ], style={'padding': '10px'})
                ]),
                dbc.Tab(label="Labor", tab_id="tab-labor", children=[
                    html.Div([
                        html.H4("Labor Analysis", style={'marginTop': '10px', 'marginBottom': '15px'}),

                        # Controls
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Labor Indicator", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='labor-indicator-dropdown',
                                    options=[
                                        {'label': 'Employment', 'value': 'emp'},
                                        {'label': 'Human Capital', 'value': 'hc'},
                                        {'label': 'Labor Share', 'value': 'labsh'},
                                        {'label': 'Labor Productivity', 'value': 'labprod'}
                                    ],
                                    value='emp',
                                    clearable=False
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Select Year", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id='labor-year-slider',
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    value=MAX_YEAR,
                                    marks={y: str(y) for y in range(MIN_YEAR, MAX_YEAR+1, 10)},
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Select Countries (for time series)", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='labor-country-dropdown',
                                    options=[{'label': c, 'value': c} for c in COUNTRIES],
                                    value=['United States', 'China', 'India', 'Germany', 'Japan'] if all(c in COUNTRIES for c in ['United States', 'China', 'India', 'Germany', 'Japan']) else default_countries,
                                    multi=True,
                                    placeholder='Select countries...'
                                ),
                            ], width=4)
                        ], style={'marginBottom': '20px'}),

                        # Visualizations Grid (reordered to match Capital layout)
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='labor-gdp-scatter', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='labor-timeseries', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ], className='mb-2'),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='labor-linearity-diagnostic', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='labor-residuals-plot', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ])
                    ], style={'padding': '10px'})
                ]),
                dbc.Tab(label="Productivity", tab_id="tab-productivity", children=[
                    html.Div([
                        html.H4("Productivity Analysis", style={'marginTop': '10px', 'marginBottom': '15px'}),

                        # Controls
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Productivity Measure", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='productivity-indicator-dropdown',
                                    options=[
                                        {'label': 'Labor Productivity (GDP per worker)', 'value': 'labprod'},
                                        {'label': 'Capital Productivity (GDP per unit capital)', 'value': 'capprod'},
                                        {'label': 'Total Factor Productivity', 'value': 'tfp'}
                                    ],
                                    value='labprod',
                                    clearable=False
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Select Year", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id='productivity-year-slider',
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    value=MAX_YEAR,
                                    marks={y: str(y) for y in range(MIN_YEAR, MAX_YEAR+1, 10)},
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Select Countries", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='productivity-country-dropdown',
                                    options=[{'label': c, 'value': c} for c in COUNTRIES],
                                    value=['United States', 'China', 'India', 'Germany', 'Japan'] if all(c in COUNTRIES for c in ['United States', 'China', 'India', 'Germany', 'Japan']) else default_countries,
                                    multi=True,
                                    placeholder='Select countries...'
                                ),
                            ], width=4)
                        ], style={'marginBottom': '20px'}),

                        # Visualizations Grid
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='productivity-gdp-scatter', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='productivity-timeseries', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ], className='mb-2'),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='productivity-linearity-diagnostic', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='productivity-growth-decomp', config={'displayModeBar': False},
                                         style={'height': '360px'})
                            ], width=6)
                        ])
                    ], style={'padding': '10px'})
                ])
            ], id="main-tabs", active_tab="tab-gdp")
        ], width=12)
    ])
], fluid=True, style={'paddingTop': '10px', 'paddingBottom': '10px'})

# CALLBACKS

# GDP TAB CALLBACK
@app.callback(
    Output('gdp-world-map', 'figure'),
    Output('gdp-distribution', 'figure'),
    Output('gdp-scatter-log', 'figure'),
    Output('gdp-timeseries', 'figure'),
    Input('gdp-country-dropdown', 'value'),
    Input('gdp-year-slider', 'value')
)
def update_gdp_tab(selected_countries, selected_year):
    """Update all GDP visualizations based on selected countries and year."""
    if not selected_countries:
        selected_countries = []

    # Filter data for selected year
    year_data = df[df['year'] == selected_year].copy()

    # Use rgdpe (Real GDP - expenditure) as the main GDP indicator
    gdp_col = 'rgdpe'

    # 1. WORLD MAP
    iso_col = None
    for c in ['iso_a3', 'iso3', 'countrycode', 'country_code', 'countrycode3']:
        if c in year_data.columns:
            iso_col = c
            break

    if gdp_col in year_data.columns and not year_data[gdp_col].isna().all():
        if iso_col is not None and year_data[iso_col].notna().any():
            fig_map = px.choropleth(
                year_data,
                locations=iso_col,
                color=gdp_col,
                hover_name='country',
                locationmode='ISO-3',
                color_continuous_scale='Viridis',
                labels={gdp_col: 'Real GDP (mil. 2021 USD)'},
                title=f'Global GDP Distribution ({selected_year})'
            )
        else:
            fig_map = px.choropleth(
                year_data,
                locations='country',
                color=gdp_col,
                hover_name='country',
                color_continuous_scale='Viridis',
                labels={gdp_col: 'Real GDP (mil. 2021 USD)'},
                title=f'Global GDP Distribution ({selected_year})'
            )

        fig_map.update_layout(
            **CHART_LAYOUT,
            margin={'t': 40, 'b': 0, 'l': 0, 'r': 0},
            coloraxis_colorbar={'title': 'GDP<br>(mil. USD)', 'thickness': 15, 'len': 0.7},
            geo=dict(
                showframe=True,
                framecolor='#333',
                framewidth=2,
                showcoastlines=True,
                coastlinecolor='#666',
                projection_type='natural earth'
            )
        )
    else:
        fig_map = go.Figure()
        fig_map.add_annotation(text='GDP data not available', xref='paper', yref='paper',
                              x=0.5, y=0.5, showarrow=False)
        fig_map.update_layout(**CHART_LAYOUT)

    # 2. DISTRIBUTION GRAPH (LOG SCALE)
    if gdp_col in year_data.columns:
        gdp_values = year_data[gdp_col].dropna()
        # Filter out zero or negative values for log scale
        gdp_values = gdp_values[gdp_values > 0]

        if len(gdp_values) > 0:
            # Convert to log10
            log_gdp_values = np.log10(gdp_values)

            fig_dist = go.Figure()

            # Histogram
            fig_dist.add_trace(go.Histogram(
                x=log_gdp_values,
                name='Frequency',
                marker=dict(color=COLORS[0], opacity=0.7, line=dict(width=1, color='white')),
                nbinsx=40,
                histnorm='probability density'
            ))

            # KDE
            from scipy import stats as scipy_stats
            if len(log_gdp_values) > 2:
                kde = scipy_stats.gaussian_kde(log_gdp_values)
                x_range = np.linspace(log_gdp_values.min(), log_gdp_values.max(), 200)
                y_kde = kde(x_range)
                fig_dist.add_trace(go.Scatter(
                    x=x_range, y=y_kde, mode='lines', name='KDE',
                    line=dict(color='#333', width=3)
                ))

            # Mark selected countries
            if selected_countries:
                for idx, country in enumerate(selected_countries):
                    country_data = year_data[year_data['country'] == country]
                    if not country_data.empty and pd.notna(country_data[gdp_col].iloc[0]):
                        value = country_data[gdp_col].iloc[0]
                        if value > 0:  # Only plot if positive
                            log_value = np.log10(value)
                            color_idx = idx % len(COLORS)
                            fig_dist.add_vline(
                                x=log_value,
                                line_dash="dash",
                                line_color=COLORS[color_idx],
                                line_width=2,
                                annotation_text=country[:15],
                                annotation_position="top"
                            )

            fig_dist.update_layout(
                **CHART_LAYOUT,
                title=f'GDP Distribution - log₁₀ scale ({selected_year})',
                xaxis_title='log₁₀(Real GDP)',
                yaxis_title='Density',
                showlegend=True,
                bargap=0.05
            )
        else:
            fig_dist = go.Figure()
            fig_dist.add_annotation(text='No data available', xref='paper', yref='paper',
                                   x=0.5, y=0.5, showarrow=False)
            fig_dist.update_layout(**CHART_LAYOUT)
    else:
        fig_dist = go.Figure()
        fig_dist.add_annotation(text='GDP data not available', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        fig_dist.update_layout(**CHART_LAYOUT)

    # 3. SCATTERPLOT (LOG GDP vs LOG POPULATION) - Labeled
    if gdp_col in year_data.columns and 'pop' in year_data.columns:
        scatter_data = year_data.dropna(subset=[gdp_col, 'pop']).copy()
        scatter_data['log_gdp'] = np.log10(scatter_data[gdp_col])
        scatter_data['log_pop'] = np.log10(scatter_data['pop'].replace(0, np.nan))
        scatter_data = scatter_data.dropna(subset=['log_pop'])
        scatter_data['selected'] = scatter_data['country'].isin(selected_countries)

        fig_scatter = go.Figure()

        # Other countries (not selected)
        other = scatter_data[~scatter_data['selected']]
        if len(other) > 0:
            fig_scatter.add_trace(go.Scatter(
                x=other['log_pop'],
                y=other['log_gdp'],
                mode='markers',
                name='Other countries',
                text=other['country'],
                marker=dict(size=8, color='rgba(200, 200, 200, 0.5)', line=dict(width=0.5, color='white')),
                hovertemplate='<b>%{text}</b><br>log₁₀(Population): %{x:.2f}<br>log₁₀(GDP): %{y:.2f}<extra></extra>',
                showlegend=True
            ))

        # Selected countries (labeled)
        selected = scatter_data[scatter_data['selected']]
        if len(selected) > 0:
            for idx, (_, row) in enumerate(selected.iterrows()):
                color_idx = idx % len(COLORS)
                fig_scatter.add_trace(go.Scatter(
                    x=[row['log_pop']],
                    y=[row['log_gdp']],
                    mode='markers+text',
                    name=row['country'],
                    text=[row['country']],
                    textposition='top center',
                    textfont=dict(size=10, color='#222'),
                    marker=dict(size=12, color=COLORS[color_idx], line=dict(width=2, color='white')),
                    hovertemplate=f'<b>{row["country"]}</b><br>log₁₀(Population): {row["log_pop"]:.2f}<br>log₁₀(GDP): {row["log_gdp"]:.2f}<extra></extra>',
                    showlegend=True
                ))

        fig_scatter.update_layout(
            **CHART_LAYOUT,
            title=f'GDP vs Population ({selected_year})',
            showlegend=True,
            hovermode='closest'
        )
        fig_scatter.update_xaxes(title='log₁₀(Population)')
        fig_scatter.update_yaxes(title='log₁₀(Real GDP)')
    else:
        fig_scatter = go.Figure()
        fig_scatter.add_annotation(text='GDP data not available', xref='paper', yref='paper',
                                  x=0.5, y=0.5, showarrow=False)
        fig_scatter.update_layout(**CHART_LAYOUT)

    # 4. TIME SERIES for selected countries
    if selected_countries and gdp_col in df.columns:
        ts_data = df[df['country'].isin(selected_countries)].copy()

        fig_ts = px.line(
            ts_data,
            x='year',
            y=gdp_col,
            color='country',
            markers=True,
            labels={gdp_col: 'Real GDP (mil. 2021 USD)', 'year': 'Year'},
            title='GDP Over Time - Selected Countries',
            color_discrete_sequence=COLORS
        )

        fig_ts.update_traces(line=dict(width=3), marker=dict(size=6))
        fig_ts.update_layout(**CHART_LAYOUT)
        fig_ts.update_yaxes(title='Real GDP (mil. 2021 USD)')

        # Add vertical line for selected year
        fig_ts.add_vline(
            x=selected_year,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Year: {selected_year}",
            annotation_position="top"
        )
    else:
        fig_ts = go.Figure()
        if not selected_countries:
            fig_ts.add_annotation(text='Please select countries to view time series',
                                 xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        else:
            fig_ts.add_annotation(text='GDP data not available', xref='paper', yref='paper',
                                 x=0.5, y=0.5, showarrow=False)
        fig_ts.update_layout(**CHART_LAYOUT)

    return fig_map, fig_dist, fig_scatter, fig_ts


# CAPITAL TAB CALLBACK
@app.callback(
    Output('capital-gdp-scatter', 'figure'),
    Output('capital-kl-ky-timeseries', 'figure'),
    Output('capital-linearity-diagnostic', 'figure'),
    Output('capital-residuals-plot', 'figure'),
    Input('capital-year-slider', 'value'),
    Input('capital-country-dropdown', 'value')
)
def update_capital_tab(selected_year, selected_countries):
    """Update all Capital visualizations based on selected year and countries."""
    if not selected_countries:
        selected_countries = []

    # Filter data for selected year
    year_data = df[df['year'] == selected_year].copy()

    # Compute axis ranges from full dataset for consistency
    log10_rnna_range = [df['log10_rnna'].min(), df['log10_rnna'].max()]
    log10_rgdpo_range = [df['log10_rgdpo'].min(), df['log10_rgdpo'].max()]

    # =====================================================================
    # FIGURE 1: Capital vs GDP (log-log scatter with OLS fit)
    # =====================================================================
    if 'log10_rnna' in year_data.columns and 'log10_rgdpo' in year_data.columns:
        scatter_data = year_data.dropna(subset=['log10_rnna', 'log10_rgdpo', 'country']).copy()

        fig1 = go.Figure()

        if len(scatter_data) > 0:
            # Size by population if available
            if 'pop' in scatter_data.columns:
                sizes = scatter_data['pop'].fillna(1)
                sizes = (sizes / sizes.max()) * 30 + 5  # Scale to 5-35
            else:
                sizes = 10

            # Mark selected countries
            scatter_data['is_selected'] = scatter_data['country'].isin(selected_countries)

            # Plot non-selected countries
            non_selected = scatter_data[~scatter_data['is_selected']]
            if len(non_selected) > 0:
                non_sel_sizes = sizes[~scatter_data['is_selected']] if 'pop' in scatter_data.columns else 10
                fig1.add_trace(go.Scatter(
                    x=non_selected['log10_rnna'],
                    y=non_selected['log10_rgdpo'],
                    mode='markers',
                    name='Other countries',
                    text=non_selected['country'],
                    marker=dict(
                        size=non_sel_sizes,
                        color='rgba(150, 150, 150, 0.4)',
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>log₁₀(Capital): %{x:.2f}<br>log₁₀(GDP): %{y:.2f}<extra></extra>',
                    showlegend=True
                ))

            # Plot selected countries with highlights
            selected_data = scatter_data[scatter_data['is_selected']]
            if len(selected_data) > 0:
                for idx, (row_idx, row) in enumerate(selected_data.iterrows()):
                    color_idx = idx % len(COLORS)

                    # Add dotted lines to axes
                    fig1.add_shape(type="line",
                        x0=log10_rnna_range[0], y0=row['log10_rgdpo'],
                        x1=row['log10_rnna'], y1=row['log10_rgdpo'],
                        line=dict(color=COLORS[color_idx], width=1, dash="dot"),
                        layer='below'
                    )
                    fig1.add_shape(type="line",
                        x0=row['log10_rnna'], y0=log10_rgdpo_range[0],
                        x1=row['log10_rnna'], y1=row['log10_rgdpo'],
                        line=dict(color=COLORS[color_idx], width=1, dash="dot"),
                        layer='below'
                    )

                    # Add marker
                    if 'pop' in scatter_data.columns and row_idx in sizes.index:
                        marker_size = sizes.loc[row_idx] * 1.2
                    else:
                        marker_size = 15

                    fig1.add_trace(go.Scatter(
                        x=[row['log10_rnna']],
                        y=[row['log10_rgdpo']],
                        mode='markers+text',
                        name=row['country'],
                        text=[row['country']],
                        textposition='top center',
                        textfont=dict(size=9, color='#222'),
                        marker=dict(
                            size=marker_size,
                            color=COLORS[color_idx],
                            opacity=0.9,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=f'<b>{row["country"]}</b><br>log₁₀(Capital): {row["log10_rnna"]:.2f}<br>log₁₀(GDP): {row["log10_rgdpo"]:.2f}<extra></extra>',
                        showlegend=True
                    ))

            # OLS fit
            if len(scatter_data) > 2:
                X = scatter_data[['log10_rnna']].dropna()
                y = scatter_data.loc[X.index, 'log10_rgdpo']
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()

                # Fitted line
                x_fit = np.linspace(scatter_data['log10_rnna'].min(), scatter_data['log10_rnna'].max(), 100)
                y_fit = model.params['const'] + model.params['log10_rnna'] * x_fit

                fig1.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'OLS Fit (R²={model.rsquared:.3f})',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='OLS Fit<extra></extra>'
                ))

            fig1.update_layout(
                **CHART_LAYOUT,
                title=f'Capital Stock vs Real GDP (log-log), Year: {selected_year}',
                hovermode='closest'
            )
            fig1.update_xaxes(title='log₁₀(Capital Stock - rnna)', range=log10_rnna_range)
            fig1.update_yaxes(title='log₁₀(Real GDP - rgdpo)', range=log10_rgdpo_range)
            fig1.add_annotation(
                text="",
                xref='paper', yref='paper',
                x=0.5, y=1.05,
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color='#666')
            )
        else:
            fig1.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig1.update_layout(**CHART_LAYOUT)
    else:
        fig1 = go.Figure()
        fig1.add_annotation(text='Capital/GDP data not available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig1.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 2: K/L and K/Y time series
    # =====================================================================
    if selected_countries and 'kl' in df.columns and 'ky' in df.columns:
        ts_data = df[df['country'].isin(selected_countries)].copy()

        fig2 = go.Figure()

        # K/L traces
        for idx, country in enumerate(selected_countries):
            country_data = ts_data[ts_data['country'] == country]
            if not country_data.empty:
                color_idx = idx % len(COLORS)
                # K/L
                fig2.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data['kl'],
                    mode='lines+markers',
                    name=f'{country} - K/L',
                    line=dict(color=COLORS[color_idx], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>K/L: %{{y:.2f}}<extra></extra>'
                ))

        fig2.update_layout(
            **CHART_LAYOUT,
            title='Capital Intensity (K/L) and Capital-Output Ratio (K/Y)',
            xaxis_title='Year',
            yaxis_title='K/L (Capital per Worker)',
            hovermode='x unified',
            showlegend=True
        )

        # Add button to toggle between K/L and K/Y
        fig2.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            label="K/L",
                            method="update",
                            args=[{"visible": [True] * len(selected_countries) + [False] * len(selected_countries)},
                                  {"yaxis": {"title": "K/L (Capital per Worker)"}}]
                        ),
                        dict(
                            label="K/Y",
                            method="update",
                            args=[{"visible": [False] * len(selected_countries) + [True] * len(selected_countries)},
                                  {"yaxis": {"title": "K/Y (Capital-Output Ratio)"}}]
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

        # Add K/Y traces (initially hidden)
        for idx, country in enumerate(selected_countries):
            country_data = ts_data[ts_data['country'] == country]
            if not country_data.empty:
                color_idx = idx % len(COLORS)
                fig2.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data['ky'],
                    mode='lines+markers',
                    name=f'{country} - K/Y',
                    line=dict(color=COLORS[color_idx], width=2, dash='dot'),
                    marker=dict(size=4),
                    visible=False,
                    hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>K/Y: %{{y:.2f}}<extra></extra>'
                ))
    else:
        fig2 = go.Figure()
        if not selected_countries:
            fig2.add_annotation(text='Please select countries', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        else:
            fig2.add_annotation(text='K/L or K/Y data not available', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        fig2.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 3: Linearity Diagnostic with LOWESS
    # =====================================================================
    if 'log10_rnna' in year_data.columns and 'log10_rgdpo' in year_data.columns:
        diag_data = year_data.dropna(subset=['log10_rnna', 'log10_rgdpo']).copy()

        fig3 = go.Figure()

        if len(diag_data) > 3:
            # Model 1: Linear
            X = diag_data[['log10_rnna']]
            y = diag_data['log10_rgdpo']
            X_const = sm.add_constant(X)
            model1 = sm.OLS(y, X_const).fit()

            # Model 2: Quadratic
            diag_data['log10_rnna_sq'] = diag_data['log10_rnna'] ** 2
            X2 = diag_data[['log10_rnna', 'log10_rnna_sq']]
            X2_const = sm.add_constant(X2)
            model2 = sm.OLS(y, X2_const).fit()

            # Print diagnostics to console
            print(f"\n=== CAPITAL LINEARITY DIAGNOSTICS (Year {selected_year}) ===")
            print(f"Linear Model R²: {model1.rsquared:.4f}")
            print(f"Linear Model Slope: {model1.params['log10_rnna']:.4f}")
            print(f"Quadratic Model R²: {model2.rsquared:.4f}")
            print(f"Squared Term p-value: {model2.pvalues['log10_rnna_sq']:.4f}")
            print("=" * 60)

            # Scatter points
            fig3.add_trace(go.Scatter(
                x=diag_data['log10_rnna'],
                y=diag_data['log10_rgdpo'],
                mode='markers',
                name='Data',
                marker=dict(size=6, color='rgba(100, 100, 200, 0.5)', line=dict(width=0.5, color='white')),
                hovertemplate='log₁₀(Capital): %{x:.2f}<br>log₁₀(GDP): %{y:.2f}<extra></extra>'
            ))

            # OLS line
            x_fit = np.linspace(diag_data['log10_rnna'].min(), diag_data['log10_rnna'].max(), 100)
            y_fit = model1.params['const'] + model1.params['log10_rnna'] * x_fit
            fig3.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name='Linear Fit',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Linear Fit<extra></extra>'
            ))

            # LOWESS curve
            if len(diag_data) > 10:
                lowess_result = lowess(diag_data['log10_rgdpo'], diag_data['log10_rnna'], frac=0.3)
                fig3.add_trace(go.Scatter(
                    x=lowess_result[:, 0],
                    y=lowess_result[:, 1],
                    mode='lines',
                    name='LOWESS',
                    line=dict(color='green', width=2),
                    hovertemplate='LOWESS<extra></extra>'
                ))

            fig3.update_layout(
                **CHART_LAYOUT,
                title=f'Linearity Check: log₁₀(rgdpo) vs log₁₀(rnna), Year {selected_year}',
                xaxis_title='log₁₀(Capital Stock - rnna)',
                yaxis_title='log₁₀(Real GDP - rgdpo)',
                hovermode='closest',
                showlegend=True
            )

            # Annotation with diagnostics
            annotation_text = (
                f"<b>Linear Model</b><br>"
                f"R²: {model1.rsquared:.3f}<br>"
                f"Slope: {model1.params['log10_rnna']:.3f}<br>"
                f"<b>Quadratic Model</b><br>"
                f"R²: {model2.rsquared:.3f}<br>"
                f"p(x²): {model2.pvalues['log10_rnna_sq']:.4f}"
            )
            fig3.add_annotation(
                text=annotation_text,
                xref='paper', yref='paper',
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#999',
                borderwidth=1,
                borderpad=6,
                font=dict(size=9)
            )
        else:
            fig3.add_annotation(text='Insufficient data for diagnostics', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig3.update_layout(**CHART_LAYOUT)
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text='Data not available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig3.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 4: Residuals vs Fitted
    # =====================================================================
    if 'log10_rnna' in year_data.columns and 'log10_rgdpo' in year_data.columns:
        resid_data = year_data.dropna(subset=['log10_rnna', 'log10_rgdpo']).copy()

        fig4 = go.Figure()

        if len(resid_data) > 2:
            # Fit linear model
            X = resid_data[['log10_rnna']]
            y = resid_data['log10_rgdpo']
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()

            # Compute fitted values and residuals
            fitted = model.fittedvalues
            residuals = model.resid

            # Scatter plot
            fig4.add_trace(go.Scatter(
                x=fitted,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, color='rgba(150, 100, 150, 0.5)', line=dict(width=0.5, color='white')),
                hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ))

            # Zero line
            fig4.add_hline(y=0, line_dash="solid", line_color="#666", line_width=2)

            # Add LOWESS trend to residuals
            if len(fitted) > 10:
                lowess_resid = lowess(residuals, fitted, frac=0.3)
                fig4.add_trace(go.Scatter(
                    x=lowess_resid[:, 0],
                    y=lowess_resid[:, 1],
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2),
                    hovertemplate='Trend<extra></extra>'
                ))

            fig4.update_layout(
                **CHART_LAYOUT,
                title=f'Residuals vs Fitted (Linear Model), Year {selected_year}',
                xaxis_title='Fitted Values',
                yaxis_title='Residuals',
                hovermode='closest',
                showlegend=True
            )
        else:
            fig4.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig4.update_layout(**CHART_LAYOUT)
    else:
        fig4 = go.Figure()
        fig4.add_annotation(text='Data not available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig4.update_layout(**CHART_LAYOUT)

    return fig1, fig2, fig3, fig4


# LABOR TAB CALLBACK
@app.callback(
    Output('labor-gdp-scatter', 'figure'),
    Output('labor-timeseries', 'figure'),
    Output('labor-linearity-diagnostic', 'figure'),
    Output('labor-residuals-plot', 'figure'),
    Input('labor-indicator-dropdown', 'value'),
    Input('labor-year-slider', 'value'),
    Input('labor-country-dropdown', 'value')
)
def update_labor_tab(selected_indicator, selected_year, selected_countries):
    """Update all Labor visualizations based on selected indicator, year, and countries."""

    if not selected_countries:
        selected_countries = []

    # Indicator mapping for labels
    indicator_labels = {
        'emp': 'Employment (millions)',
        'hc': 'Human Capital Index',
        'labsh': 'Labor Share',
        'labprod': 'Labor Productivity (GDP/Worker)'
    }

    indicator_label = indicator_labels.get(selected_indicator, selected_indicator)
    log_indicator = f'log10_{selected_indicator}'

    # Filter data for selected year
    year_data = df[df['year'] == selected_year].copy()

    # Check if required columns exist
    if selected_indicator not in df.columns or log_indicator not in df.columns or 'log10_rgdpo' not in df.columns:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f'Data not available for {indicator_label}',
                                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(**CHART_LAYOUT)
        return empty_fig, empty_fig, empty_fig, empty_fig

    # =====================================================================
    # FIGURE 1: GDP vs Selected Labour Indicator (log-log scatter)
    # =====================================================================
    scatter_data = year_data.dropna(subset=[log_indicator, 'log10_rgdpo', 'country']).copy()

    # Compute axis ranges from full dataset for consistency
    log_indicator_range = [df[log_indicator].min(), df[log_indicator].max()]
    log10_rgdpo_range = [df['log10_rgdpo'].min(), df['log10_rgdpo'].max()]

    fig1 = go.Figure()

    if len(scatter_data) > 2:
        # Size by population if available
        if 'pop' in scatter_data.columns:
            sizes = scatter_data['pop'].fillna(1)
            sizes = (sizes / sizes.max()) * 30 + 5  # Scale to 5-35
        else:
            sizes = 10

        # Mark selected countries
        scatter_data['is_selected'] = scatter_data['country'].isin(selected_countries)

        # Plot non-selected countries
        non_selected = scatter_data[~scatter_data['is_selected']]
        if len(non_selected) > 0:
            non_sel_sizes = sizes[~scatter_data['is_selected']] if 'pop' in scatter_data.columns else 10
            fig1.add_trace(go.Scatter(
                x=non_selected[log_indicator],
                y=non_selected['log10_rgdpo'],
                mode='markers',
                name='Other countries',
                text=non_selected['country'],
                marker=dict(
                    size=non_sel_sizes,
                    color='rgba(150, 150, 150, 0.4)',
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    f'{indicator_label}: %{{customdata[0]:.2f}}<br>' +
                    'GDP: %{customdata[1]:.2f}<br>' +
                    'Population: %{customdata[2]:.1f}M<extra></extra>'
                ),
                customdata=np.column_stack([
                    non_selected[selected_indicator],
                    non_selected['rgdpo'],
                    non_selected.get('pop', np.zeros(len(non_selected)))
                ]),
                showlegend=True
            ))

        # Plot selected countries with highlights
        selected_data = scatter_data[scatter_data['is_selected']]
        if len(selected_data) > 0:
            for idx, (row_idx, row) in enumerate(selected_data.iterrows()):
                color_idx = idx % len(COLORS)

                # Add dotted lines to axes
                fig1.add_shape(type="line",
                    x0=log_indicator_range[0], y0=row['log10_rgdpo'],
                    x1=row[log_indicator], y1=row['log10_rgdpo'],
                    line=dict(color=COLORS[color_idx], width=1, dash="dot"),
                    layer='below'
                )
                fig1.add_shape(type="line",
                    x0=row[log_indicator], y0=log10_rgdpo_range[0],
                    x1=row[log_indicator], y1=row['log10_rgdpo'],
                    line=dict(color=COLORS[color_idx], width=1, dash="dot"),
                    layer='below'
                )

                # Add marker
                if 'pop' in scatter_data.columns and row_idx in sizes.index:
                    marker_size = sizes.loc[row_idx] * 1.2
                else:
                    marker_size = 15

                fig1.add_trace(go.Scatter(
                    x=[row[log_indicator]],
                    y=[row['log10_rgdpo']],
                    mode='markers+text',
                    name=row['country'],
                    text=[row['country']],
                    textposition='top center',
                    textfont=dict(size=9, color='#222'),
                    marker=dict(
                        size=marker_size,
                        color=COLORS[color_idx],
                        opacity=0.9,
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=(
                        f'<b>{row["country"]}</b><br>' +
                        f'{indicator_label}: {row[selected_indicator]:.2f}<br>' +
                        f'GDP: {row["rgdpo"]:.2f}<br>' +
                        f'Population: {row.get("pop", 0):.1f}M<extra></extra>'
                    ),
                    showlegend=True
                ))

        # OLS fit
        X = scatter_data[[log_indicator]].dropna()
        y = scatter_data.loc[X.index, 'log10_rgdpo']
        X_const = sm.add_constant(X)
        model_linear = sm.OLS(y, X_const).fit()

        # Quadratic model for non-linearity test
        scatter_data['log_indicator_sq'] = scatter_data[log_indicator] ** 2
        X2 = scatter_data[[log_indicator, 'log_indicator_sq']].dropna()
        y2 = scatter_data.loc[X2.index, 'log10_rgdpo']
        X2_const = sm.add_constant(X2)
        model_quad = sm.OLS(y2, X2_const).fit()

        # Print diagnostics
        print(f"\n=== LABOR ANALYSIS DIAGNOSTICS ===")
        print(f"Year {selected_year}, Indicator={indicator_label}:")
        print(f"  R²={model_linear.rsquared:.3f}")
        print(f"  slope={model_linear.params[log_indicator]:.3f}")
        print(f"  p²-term={model_quad.pvalues['log_indicator_sq']:.4f}")
        print("=" * 60)

        # Fitted line
        x_fit = np.linspace(scatter_data[log_indicator].min(), scatter_data[log_indicator].max(), 100)
        y_fit = model_linear.params['const'] + model_linear.params[log_indicator] * x_fit

        fig1.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f'OLS (R²={model_linear.rsquared:.3f})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='OLS Fit<extra></extra>'
        ))

        # LOWESS curve
        if len(scatter_data) > 10:
            lowess_result = lowess(scatter_data['log10_rgdpo'], scatter_data[log_indicator], frac=0.3)
            fig1.add_trace(go.Scatter(
                x=lowess_result[:, 0],
                y=lowess_result[:, 1],
                mode='lines',
                name='LOWESS',
                line=dict(color='green', width=2),
                hovertemplate='LOWESS<extra></extra>'
            ))

        fig1.update_layout(
            **CHART_LAYOUT,
            title=f'GDP vs {indicator_label}, Year {selected_year}',
            hovermode='closest',
            showlegend=True
        )
        fig1.update_xaxes(title=f'log₁₀({indicator_label})', range=log_indicator_range)
        fig1.update_yaxes(title='log₁₀(Real GDP)', range=log10_rgdpo_range)
    else:
        fig1.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig1.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 2: Linearity Diagnostic
    # =====================================================================
    diag_data = year_data.dropna(subset=[log_indicator, 'log10_rgdpo']).copy()

    fig2 = go.Figure()

    if len(diag_data) > 3:
        # Scatter points
        fig2.add_trace(go.Scatter(
            x=diag_data[log_indicator],
            y=diag_data['log10_rgdpo'],
            mode='markers',
            name='Data',
            marker=dict(size=6, color='rgba(100, 150, 200, 0.5)', line=dict(width=0.5, color='white')),
            hovertemplate=f'log₁₀({indicator_label}): %{{x:.2f}}<br>log₁₀(GDP): %{{y:.2f}}<extra></extra>'
        ))

        # Recompute models for this figure
        X = diag_data[[log_indicator]].dropna()
        y = diag_data.loc[X.index, 'log10_rgdpo']
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        # Quadratic
        diag_data['log_indicator_sq'] = diag_data[log_indicator] ** 2
        X2 = diag_data[[log_indicator, 'log_indicator_sq']].dropna()
        y2 = diag_data.loc[X2.index, 'log10_rgdpo']
        X2_const = sm.add_constant(X2)
        model2 = sm.OLS(y2, X2_const).fit()

        # OLS line
        x_fit = np.linspace(diag_data[log_indicator].min(), diag_data[log_indicator].max(), 100)
        y_fit = model.params['const'] + model.params[log_indicator] * x_fit

        fig2.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name='OLS Fit',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='OLS Fit<extra></extra>'
        ))

        # LOWESS
        if len(diag_data) > 10:
            lowess_result = lowess(diag_data['log10_rgdpo'], diag_data[log_indicator], frac=0.3)
            fig2.add_trace(go.Scatter(
                x=lowess_result[:, 0],
                y=lowess_result[:, 1],
                mode='lines',
                name='LOWESS',
                line=dict(color='green', width=2),
                hovertemplate='LOWESS<extra></extra>'
            ))

        fig2.update_layout(
            **CHART_LAYOUT,
            title=f'Linearity Check: GDP vs {indicator_label}, Year {selected_year}',
            hovermode='closest',
            showlegend=True
        )
        fig2.update_xaxes(title=f'log₁₀({indicator_label})')
        fig2.update_yaxes(title='log₁₀(Real GDP)')

        # Annotation with diagnostics
        annotation_text = (
            f"<b>Diagnostics</b><br>"
            f"R²: {model.rsquared:.3f}<br>"
            f"Slope: {model.params[log_indicator]:.3f}<br>"
            f"p(x²): {model2.pvalues['log_indicator_sq']:.4f}"
        )
        fig2.add_annotation(
            text=annotation_text,
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#999',
            borderwidth=1,
            borderpad=6,
            font=dict(size=9)
        )
    else:
        fig2.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig2.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 3: Residuals vs Fitted
    # =====================================================================
    resid_data = year_data.dropna(subset=[log_indicator, 'log10_rgdpo']).copy()

    fig3 = go.Figure()

    if len(resid_data) > 2:
        # Fit model
        X = resid_data[[log_indicator]].dropna()
        y = resid_data.loc[X.index, 'log10_rgdpo']
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        # Compute residuals
        fitted = model.fittedvalues
        residuals = model.resid

        # Scatter
        fig3.add_trace(go.Scatter(
            x=fitted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=6, color='rgba(150, 100, 150, 0.5)', line=dict(width=0.5, color='white')),
            hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ))

        # Zero line
        fig3.add_hline(y=0, line_dash="solid", line_color="#666", line_width=2)

        # LOWESS trend
        if len(fitted) > 10:
            lowess_resid = lowess(residuals, fitted, frac=0.3)
            fig3.add_trace(go.Scatter(
                x=lowess_resid[:, 0],
                y=lowess_resid[:, 1],
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2),
                hovertemplate='Trend<extra></extra>'
            ))

        fig3.update_layout(
            **CHART_LAYOUT,
            title=f'Residuals vs Fitted ({indicator_label}), Year {selected_year}',
            xaxis_title='Fitted Values',
            yaxis_title='Residuals',
            hovermode='closest',
            showlegend=True
        )
    else:
        fig3.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig3.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 4: Time Series Trends
    # =====================================================================
    fig4 = go.Figure()

    if selected_countries and selected_indicator in df.columns:
        ts_data = df[df['country'].isin(selected_countries)].copy()

        if len(ts_data) > 0:
            for idx, country in enumerate(selected_countries):
                country_data = ts_data[ts_data['country'] == country].sort_values('year')
                if not country_data.empty:
                    color_idx = idx % len(COLORS)
                    fig4.add_trace(go.Scatter(
                        x=country_data['year'],
                        y=country_data[selected_indicator],
                        mode='lines+markers',
                        name=country,
                        line=dict(color=COLORS[color_idx], width=2),
                        marker=dict(size=4),
                        hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>{indicator_label}: %{{y:.2f}}<extra></extra>'
                    ))

            # Add vertical line for selected year
            fig4.add_vline(
                x=selected_year,
                line_dash="dash",
                line_color="gray",
                line_width=2,
                annotation_text=f"Year: {selected_year}",
                annotation_position="top"
            )

            fig4.update_layout(
                **CHART_LAYOUT,
                title=f'Evolution of {indicator_label} over Time',
                xaxis_title='Year',
                yaxis_title=indicator_label,
                hovermode='x unified',
                showlegend=True
            )
        else:
            fig4.add_annotation(text='No data for selected countries', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig4.update_layout(**CHART_LAYOUT)
    else:
        if not selected_countries:
            fig4.add_annotation(text='Please select countries to view time series', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        else:
            fig4.add_annotation(text='Data not available', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        fig4.update_layout(**CHART_LAYOUT)

    return fig1, fig4, fig2, fig3


# PRODUCTIVITY TAB CALLBACK
@app.callback(
    Output('productivity-gdp-scatter', 'figure'),
    Output('productivity-timeseries', 'figure'),
    Output('productivity-growth-decomp', 'figure'),
    Output('productivity-linearity-diagnostic', 'figure'),
    Input('productivity-indicator-dropdown', 'value'),
    Input('productivity-year-slider', 'value'),
    Input('productivity-country-dropdown', 'value')
)
def update_productivity_tab(selected_indicator, selected_year, selected_countries):
    """Update all Productivity visualizations based on selected indicator, year, and countries."""

    if not selected_countries:
        selected_countries = []

    # Indicator mapping for labels
    indicator_labels = {
        'labprod': 'Labor Productivity (GDP/Worker)',
        'capprod': 'Capital Productivity (GDP/Capital)',
        'tfp': 'Total Factor Productivity'
    }

    indicator_label = indicator_labels.get(selected_indicator, selected_indicator)
    log_indicator = f'log10_{selected_indicator}'

    # Filter data for selected year
    year_data = df[df['year'] == selected_year].copy()

    # Check if required columns exist
    if selected_indicator not in df.columns or log_indicator not in df.columns or 'log10_rgdpo' not in df.columns:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f'Data not available for {indicator_label}',
                                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(**CHART_LAYOUT)
        return empty_fig, empty_fig, empty_fig, empty_fig

    # =====================================================================
    # FIGURE 1: GDP vs Selected Productivity (log-log scatter)
    # =====================================================================
    scatter_data = year_data.dropna(subset=[log_indicator, 'log10_rgdpo', 'country']).copy()

    fig1 = go.Figure()

    if len(scatter_data) > 2:
        # Mark selected countries
        scatter_data['is_selected'] = scatter_data['country'].isin(selected_countries)

        # Calculate axis ranges with padding
        log_indicator_range = [scatter_data[log_indicator].min() - 0.1, scatter_data[log_indicator].max() + 0.1]
        log10_rgdpo_range = [scatter_data['log10_rgdpo'].min() - 0.1, scatter_data['log10_rgdpo'].max() + 0.1]

        # Size by population if available
        if 'pop' in scatter_data.columns:
            sizes = scatter_data['pop'].fillna(1)
            sizes = (sizes / sizes.max()) * 30 + 5  # Scale to 5-35
        else:
            sizes = pd.Series(10, index=scatter_data.index)

        # Plot non-selected countries in gray
        non_selected = scatter_data[~scatter_data['is_selected']]
        if len(non_selected) > 0:
            if isinstance(sizes, pd.Series):
                non_selected_sizes = sizes.loc[non_selected.index]
            else:
                non_selected_sizes = sizes

            fig1.add_trace(go.Scatter(
                x=non_selected[log_indicator],
                y=non_selected['log10_rgdpo'],
                mode='markers',
                name='Other Countries',
                text=non_selected['country'],
                marker=dict(
                    size=non_selected_sizes,
                    color='rgba(150, 150, 150, 0.3)',
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    f'{indicator_label}: %{{customdata[0]:.4f}}<br>' +
                    'GDP: %{customdata[1]:.2f}<br>' +
                    'Population: %{customdata[2]:.1f}M<extra></extra>'
                ),
                customdata=np.column_stack([
                    non_selected[selected_indicator],
                    non_selected['rgdpo'],
                    non_selected.get('pop', np.zeros(len(non_selected)))
                ])
            ))

        # Plot selected countries with dotted reference lines
        selected_data = scatter_data[scatter_data['is_selected']]
        for idx, (row_idx, row) in enumerate(selected_data.iterrows()):
            color_idx = idx % len(COLORS)

            # Dotted lines to axes
            fig1.add_shape(type="line",
                x0=log_indicator_range[0], y0=row['log10_rgdpo'],
                x1=row[log_indicator], y1=row['log10_rgdpo'],
                line=dict(color=COLORS[color_idx], width=1, dash="dot"),
                layer='below'
            )
            fig1.add_shape(type="line",
                x0=row[log_indicator], y0=log10_rgdpo_range[0],
                x1=row[log_indicator], y1=row['log10_rgdpo'],
                line=dict(color=COLORS[color_idx], width=1, dash="dot"),
                layer='below'
            )

            # Marker sizing with safety check
            if 'pop' in scatter_data.columns and row_idx in sizes.index:
                marker_size = sizes.loc[row_idx] * 1.2
            else:
                marker_size = 15

            # Marker
            fig1.add_trace(go.Scatter(
                x=[row[log_indicator]],
                y=[row['log10_rgdpo']],
                mode='markers+text',
                name=row['country'],
                marker=dict(size=marker_size, color=COLORS[color_idx], line=dict(width=2, color='white')),
                text=row['country'],
                textposition='top center',
                textfont=dict(size=10, color=COLORS[color_idx]),
                hovertemplate=(
                    f'<b>{row["country"]}</b><br>' +
                    f'{indicator_label}: {row[selected_indicator]:.4f}<br>' +
                    f'GDP: {row["rgdpo"]:.2f}<br>' +
                    f'Population: {row.get("pop", 0):.1f}M<extra></extra>'
                )
            ))

        # OLS fit
        X = scatter_data[[log_indicator]].dropna()
        y = scatter_data.loc[X.index, 'log10_rgdpo']
        X_const = sm.add_constant(X)
        model_linear = sm.OLS(y, X_const).fit()

        # Quadratic model for non-linearity test
        scatter_data['log_indicator_sq'] = scatter_data[log_indicator] ** 2
        X2 = scatter_data[[log_indicator, 'log_indicator_sq']].dropna()
        y2 = scatter_data.loc[X2.index, 'log10_rgdpo']
        X2_const = sm.add_constant(X2)
        model_quad = sm.OLS(y2, X2_const).fit()

        # Print diagnostics
        print(f"\n=== PRODUCTIVITY ANALYSIS DIAGNOSTICS ===")
        print(f"Year {selected_year}, Indicator={indicator_label}:")
        print(f"  R²={model_linear.rsquared:.3f}")
        print(f"  slope={model_linear.params[log_indicator]:.3f}")
        print(f"  p²-term={model_quad.pvalues['log_indicator_sq']:.4f}")
        print("=" * 60)

        # Fitted line
        x_fit = np.linspace(scatter_data[log_indicator].min(), scatter_data[log_indicator].max(), 100)
        y_fit = model_linear.params['const'] + model_linear.params[log_indicator] * x_fit

        fig1.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f'OLS (R²={model_linear.rsquared:.3f})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='OLS Fit<extra></extra>'
        ))

        # LOWESS curve
        if len(scatter_data) > 10:
            lowess_result = lowess(scatter_data['log10_rgdpo'], scatter_data[log_indicator], frac=0.3)
            fig1.add_trace(go.Scatter(
                x=lowess_result[:, 0],
                y=lowess_result[:, 1],
                mode='lines',
                name='LOWESS',
                line=dict(color='green', width=2),
                hovertemplate='LOWESS<extra></extra>'
            ))

        fig1.update_layout(
            **CHART_LAYOUT,
            title=f'GDP vs {indicator_label}, Year {selected_year}',
            hovermode='closest',
            showlegend=True
        )
        fig1.update_xaxes(title=f'log₁₀({indicator_label})', range=log_indicator_range)
        fig1.update_yaxes(title='log₁₀(Real GDP)', range=log10_rgdpo_range)
    else:
        fig1.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig1.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 2: Time Series Trends
    # =====================================================================
    fig2 = go.Figure()

    if selected_countries and selected_indicator in df.columns:
        ts_data = df[df['country'].isin(selected_countries)].copy()

        if len(ts_data) > 0:
            for idx, country in enumerate(selected_countries):
                country_data = ts_data[ts_data['country'] == country].sort_values('year')
                if not country_data.empty:
                    color_idx = idx % len(COLORS)
                    fig2.add_trace(go.Scatter(
                        x=country_data['year'],
                        y=country_data[selected_indicator],
                        mode='lines+markers',
                        name=country,
                        line=dict(color=COLORS[color_idx], width=2),
                        marker=dict(size=4),
                        hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>{indicator_label}: %{{y:.4f}}<extra></extra>'
                    ))

            # Add vertical line for selected year
            fig2.add_vline(
                x=selected_year,
                line_dash="dash",
                line_color="gray",
                line_width=2,
                annotation_text=f"Year: {selected_year}",
                annotation_position="top"
            )

            fig2.update_layout(
                **CHART_LAYOUT,
                title=f'Evolution of {indicator_label} over Time',
                xaxis_title='Year',
                yaxis_title=indicator_label,
                hovermode='x unified',
                showlegend=True
            )
        else:
            fig2.add_annotation(text='No data for selected countries', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig2.update_layout(**CHART_LAYOUT)
    else:
        if not selected_countries:
            fig2.add_annotation(text='Please select countries to view time series', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        else:
            fig2.add_annotation(text='Data not available', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
        fig2.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 3: GDP Growth Decomposition
    # =====================================================================
    fig3 = go.Figure()

    if selected_countries and 'gdp_growth' in df.columns:
        # For simplicity, show growth decomposition for the first selected country
        if len(selected_countries) > 0:
            target_country = selected_countries[0]
            decomp_data = df[df['country'] == target_country].copy()
            decomp_data = decomp_data.sort_values('year')

            # Approximate Cobb-Douglas: g_Y ≈ α*g_K + (1-α)*g_L + g_TFP
            # where α = capital share = 1 - labsh
            if all(col in decomp_data.columns for col in ['gdp_growth', 'tfp_growth', 'labsh']):
                decomp_data = decomp_data.dropna(subset=['gdp_growth', 'tfp_growth', 'labsh'])

                if len(decomp_data) > 1:
                    # Compute contributions (simplified)
                    # TFP contribution
                    tfp_contrib = decomp_data['tfp_growth'].fillna(0)

                    # For visualization purposes, show TFP, Labor, and Capital contributions
                    # This is a simplified decomposition
                    years = decomp_data['year']

                    fig3.add_trace(go.Scatter(
                        x=years,
                        y=tfp_contrib,
                        mode='lines',
                        name='TFP Contribution',
                        stackgroup='one',
                        line=dict(width=0.5, color=COLORS[0]),
                        fillcolor=COLORS[0],
                        hovertemplate='<b>TFP</b><br>Year: %{x}<br>Contribution: %{y:.2f}%<extra></extra>'
                    ))

                    # Labor contribution (approximation)
                    if 'labprod_growth' in decomp_data.columns:
                        labor_contrib = decomp_data['gdp_growth'] - tfp_contrib
                        labor_contrib = labor_contrib.fillna(0).clip(lower=-20, upper=20)  # Clip extremes

                        fig3.add_trace(go.Scatter(
                            x=years,
                            y=labor_contrib,
                            mode='lines',
                            name='Other Factors',
                            stackgroup='one',
                            line=dict(width=0.5, color=COLORS[1]),
                            fillcolor=COLORS[1],
                            hovertemplate='<b>Other</b><br>Year: %{x}<br>Contribution: %{y:.2f}%<extra></extra>'
                        ))

                    # Add GDP growth line for reference
                    fig3.add_trace(go.Scatter(
                        x=years,
                        y=decomp_data['gdp_growth'],
                        mode='lines+markers',
                        name='Total GDP Growth',
                        line=dict(color='black', width=2, dash='dot'),
                        marker=dict(size=4),
                        hovertemplate='<b>GDP Growth</b><br>Year: %{x}<br>Growth: %{y:.2f}%<extra></extra>'
                    ))

                    fig3.update_layout(
                        **CHART_LAYOUT,
                        title=f'GDP Growth Decomposition — {target_country}',
                        xaxis_title='Year',
                        yaxis_title='Contribution to Growth (%)',
                        hovermode='x unified',
                        showlegend=True
                    )
                else:
                    fig3.add_annotation(text='Insufficient data for decomposition', xref='paper', yref='paper',
                                       x=0.5, y=0.5, showarrow=False)
                    fig3.update_layout(**CHART_LAYOUT)
            else:
                fig3.add_annotation(text='Required variables not available', xref='paper', yref='paper',
                                   x=0.5, y=0.5, showarrow=False)
                fig3.update_layout(**CHART_LAYOUT)
        else:
            fig3.add_annotation(text='Please select a country', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig3.update_layout(**CHART_LAYOUT)
    else:
        fig3.add_annotation(text='Please select countries', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig3.update_layout(**CHART_LAYOUT)

    # =====================================================================
    # FIGURE 4: Linearity Diagnostic (GDP vs Selected Indicator)
    # =====================================================================
    fig4 = go.Figure()

    if log_indicator in year_data.columns and 'log10_rgdpo' in year_data.columns:
        diag_data = year_data.dropna(subset=[log_indicator, 'log10_rgdpo']).copy()

        if len(diag_data) > 3:
            # Scatter points
            fig4.add_trace(go.Scatter(
                x=diag_data[log_indicator],
                y=diag_data['log10_rgdpo'],
                mode='markers',
                name='Data',
                text=diag_data['country'],
                marker=dict(size=6, color='rgba(100, 100, 200, 0.5)', line=dict(width=0.5, color='white')),
                hovertemplate=f'<b>%{{text}}</b><br>log₁₀({indicator_label}): %{{x:.2f}}<br>log₁₀(GDP): %{{y:.2f}}<extra></extra>'
            ))

            # OLS model
            X = diag_data[[log_indicator]].dropna()
            y = diag_data.loc[X.index, 'log10_rgdpo']
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()

            # Quadratic
            diag_data['log_indicator_sq'] = diag_data[log_indicator] ** 2
            X2 = diag_data[[log_indicator, 'log_indicator_sq']].dropna()
            y2 = diag_data.loc[X2.index, 'log10_rgdpo']
            X2_const = sm.add_constant(X2)
            model2 = sm.OLS(y2, X2_const).fit()

            # OLS line
            x_fit = np.linspace(diag_data[log_indicator].min(), diag_data[log_indicator].max(), 100)
            y_fit = model.params['const'] + model.params[log_indicator] * x_fit

            fig4.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name='OLS Fit',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='OLS Fit<extra></extra>'
            ))

            # LOWESS
            if len(diag_data) > 10:
                lowess_result = lowess(diag_data['log10_rgdpo'], diag_data[log_indicator], frac=0.3)
                fig4.add_trace(go.Scatter(
                    x=lowess_result[:, 0],
                    y=lowess_result[:, 1],
                    mode='lines',
                    name='LOWESS',
                    line=dict(color='green', width=2),
                    hovertemplate='LOWESS<extra></extra>'
                ))

            fig4.update_layout(
                **CHART_LAYOUT,
                title=f'Linearity Check: GDP vs {indicator_label}',
                hovermode='closest',
                showlegend=True
            )
            fig4.update_xaxes(title=f'log₁₀({indicator_label})')
            fig4.update_yaxes(title='log₁₀(Real GDP)')

            # Annotation with diagnostics
            annotation_text = (
                f"<b>Diagnostics (Year {selected_year})</b><br>"
                f"R²: {model.rsquared:.3f}<br>"
                f"Slope: {model.params[log_indicator]:.3f}<br>"
                f"p(x²): {model2.pvalues['log_indicator_sq']:.4f}"
            )
            fig4.add_annotation(
                text=annotation_text,
                xref='paper', yref='paper',
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#999',
                borderwidth=1,
                borderpad=6,
                font=dict(size=9)
            )
        else:
            fig4.add_annotation(text='Insufficient data', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False)
            fig4.update_layout(**CHART_LAYOUT)
    else:
        fig4.add_annotation(text=f'{indicator_label} data not available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
        fig4.update_layout(**CHART_LAYOUT)

    return fig1, fig2, fig3, fig4


# OLD CALLBACKS COMMENTED OUT - Components removed during restructure
# The old chart update callback code has been removed.
# Reference the git history if you need to restore any of the visualization logic.

"""
# CALLBACK - UPDATE ALL CHARTS (DEPRECATED)
@app.callback(
    Output('gdp-world-map', 'figure'),
    Output('gdp-timeseries', 'figure'),
    Output('gdp-growth', 'figure'),
    Output('gdp-rel-world', 'figure'),
    Output('scatter-latest', 'figure'),
    Output('residual-plot', 'figure'),
    Output('distribution-chart', 'figure'),
    Output('qq-plot', 'figure'),
    Output('data-table-container', 'children'),
    Output('summary-stats', 'children'),
    Output('map-selected-info', 'children'),
    Input('country-dropdown', 'value'),
    Input('year-slider', 'value'),
    Input('log-scale', 'value'),
    Input('primary-indicator', 'value'),
    Input('comparison-indicator', 'value'),
    Input('gdp-world-map', 'clickData'),
    Input('scatter-log-x', 'value'),
    Input('scatter-log-y', 'value'),
    Input('scatter-show-labels', 'value'),
    Input('dist-log-scale', 'value')
)
def update_old(country_list, year_range, log_opt, primary_indicator, comparison_indicator, clickData,
           scatter_log_x, scatter_log_y, scatter_show_labels, dist_log_scale):
    if not country_list:
        country_list = [COUNTRIES[0]]
    y0, y1 = int(year_range[0]), int(year_range[1])
    mask = (df['country'].isin(country_list)) & (df['year'] >= y0) & (df['year'] <= y1)
    sub = df.loc[mask].copy()
    indicator_label = AVAILABLE_INDICATORS.get(primary_indicator, primary_indicator)

    # TIMESERIES CHART
    if primary_indicator in sub.columns:
        fig_gdp = px.line(sub, x='year', y=primary_indicator, color='country',
                         markers=True,
                         labels={primary_indicator: indicator_label, 'year': 'Year'},
                         title=f'{indicator_label} | Time Series',
                         color_discrete_sequence=COLORS)
        fig_gdp.update_traces(line=dict(width=3), marker=dict(size=6))
        fig_gdp.update_layout(**CHART_LAYOUT)
        if 'log' in (log_opt or []):
            fig_gdp.update_yaxes(type='log', title=f'{indicator_label} (log scale)')
        else:
            fig_gdp.update_yaxes(title=indicator_label)
    else:
        fig_gdp = go.Figure()
        fig_gdp.add_annotation(text='Indicator not available', xref='paper', yref='paper', showarrow=False)
        fig_gdp.update_layout(**CHART_LAYOUT)

    # GROWTH CHART
    sub_sorted = sub.sort_values(['country', 'year'])
    if primary_indicator in sub_sorted.columns:
        sub_sorted[f'{primary_indicator}_growth'] = sub_sorted.groupby('country')[primary_indicator].pct_change() * 100
        fig_growth = px.bar(sub_sorted, x='year', y=f'{primary_indicator}_growth', color='country',
                           barmode='group',
                           labels={f'{primary_indicator}_growth': 'Year-over-year growth (%)'},
                           title=f'{indicator_label} | Year-over-Year Growth',
                           color_discrete_sequence=COLORS)
        fig_growth.update_layout(**CHART_LAYOUT, hovermode='x unified')
        fig_growth.update_traces(marker_line_width=0)
        fig_growth.update_yaxes(title='Growth Rate (%)')
    else:
        fig_growth = go.Figure()
        fig_growth.add_annotation(text='Growth data not available', xref='paper', yref='paper', showarrow=False)
        fig_growth.update_layout(**CHART_LAYOUT)

    # RELATIVE TO WORLD AVERAGE CHART
    if primary_indicator in df.columns:
        world_mean = df.groupby('year')[primary_indicator].mean().reset_index().rename(columns={primary_indicator: 'world_mean'})
        rel = sub.merge(world_mean, on='year', how='left')
        rel['rel_to_world'] = rel[primary_indicator] / rel['world_mean']
        fig_rel = px.line(rel, x='year', y='rel_to_world', color='country',
                         labels={'rel_to_world': 'Ratio to world mean', 'year': 'Year'},
                         title=f'{indicator_label} | Relative to World Average',
                         color_discrete_sequence=COLORS)
        fig_rel.update_traces(line=dict(width=3), marker=dict(size=6))
        fig_rel.update_layout(**CHART_LAYOUT)
        fig_rel.update_yaxes(title='Ratio (country/world mean)')
        fig_rel.add_hline(y=1.0, line_dash="dash", line_color="#666", line_width=2,
                         annotation_text="World average = 1.0", annotation_position="right")
    else:
        fig_rel = go.Figure()
        fig_rel.add_annotation(text='Relative data not available', xref='paper', yref='paper', showarrow=False)
        fig_rel.update_layout(**CHART_LAYOUT)

    # SCATTER PLOT WITH CORRELATION ANALYSIS
    latest_year = min(y1, df['year'].max())
    all_latest = df[df['year'] == latest_year].copy()

    if comparison_indicator == primary_indicator:
        fig_scatter = go.Figure()
        fig_scatter.add_annotation(
            text='Cannot compare indicator with itself<br>Please select different indicators',
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color='#666')
        )
        fig_scatter.update_layout(**CHART_LAYOUT)
        regression_stats = {}
        valid_data = pd.DataFrame()
    elif comparison_indicator and comparison_indicator in all_latest.columns and primary_indicator in all_latest.columns:
        comparison_label = AVAILABLE_INDICATORS.get(comparison_indicator, comparison_indicator)
        all_latest['selected'] = all_latest['country'].isin(country_list)
        all_latest['display_name'] = all_latest['country']
        scatter_data = all_latest.dropna(subset=[comparison_indicator, primary_indicator])
        fig_scatter = go.Figure()
        from scipy import stats
        valid_data = scatter_data[[comparison_indicator, primary_indicator]].dropna()
        regression_stats = {}

        if len(valid_data) > 1:
            pearson_r, pearson_p = stats.pearsonr(valid_data[comparison_indicator],
                                                   valid_data[primary_indicator])
            spearman_r, spearman_p = stats.spearmanr(valid_data[comparison_indicator],
                                                      valid_data[primary_indicator])
            slope, intercept, r_value, _, _ = stats.linregress(
                valid_data[comparison_indicator],
                valid_data[primary_indicator]
            )
            def to_scalar(val):
                arr = np.asarray(val)
                if arr.ndim == 0:
                    return arr.item()
                elif arr.size == 1:
                    return float(arr.flat[0])
                else:
                    return float(val)

            r_squared = to_scalar(r_value**2)
            regression_stats = {
                'pearson_r': to_scalar(pearson_r),
                'pearson_p': to_scalar(pearson_p),
                'spearman_r': to_scalar(spearman_r),
                'spearman_p': to_scalar(spearman_p),
                'r_squared': r_squared,
                'slope': to_scalar(slope),
                'intercept': to_scalar(intercept),
                'n': len(valid_data)
            }

            x_range = np.linspace(valid_data[comparison_indicator].min(),
                                 valid_data[comparison_indicator].max(), 100)
            y_pred = regression_stats['slope'] * x_range + regression_stats['intercept']

            fig_scatter.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f'Linear fit (R²={r_squared:.3f})',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=2, dash='dash'),
                hovertemplate=f'y = {regression_stats["slope"]:.2e}x + {regression_stats["intercept"]:.2e}<br>R² = {r_squared:.3f}<extra></extra>'
            ))

        other_countries = scatter_data[~scatter_data['selected']]
        if len(other_countries) > 0:
            fig_scatter.add_trace(go.Scatter(
                x=other_countries[comparison_indicator],
                y=other_countries[primary_indicator],
                mode='markers',
                name='Other countries',
                text=other_countries['display_name'],
                marker=dict(size=6, color='rgba(200, 200, 200, 0.4)', line=dict(width=0.5, color='white')),
                hovertemplate='<b>%{text}</b><br>' + f'{comparison_label}: %{{x:,.2f}}<br>' + f'{indicator_label}: %{{y:,.2f}}<extra></extra>',
                showlegend=True
            ))
        selected_countries = scatter_data[scatter_data['selected']]
        show_labels = 'show' in (scatter_show_labels or [])
        if len(selected_countries) > 0:
            for idx, (_, row) in enumerate(selected_countries.iterrows()):
                color = COLORS[idx % len(COLORS)]
                mode = 'markers+text' if show_labels else 'markers'
                fig_scatter.add_trace(go.Scatter(
                    x=[row[comparison_indicator]],
                    y=[row[primary_indicator]],
                    mode=mode,
                    name=row['display_name'],
                    text=[row['display_name']] if show_labels else None,
                    textposition='top center' if show_labels else None,
                    textfont=dict(size=11, color='#222') if show_labels else None,
                    marker=dict(size=14, color=color, line=dict(width=2, color='white'), opacity=0.9),
                    hovertemplate=f'<b>{row["display_name"]}</b><br>' + f'{comparison_label}: {row[comparison_indicator]:,.2f}<br>' + f'{indicator_label}: {row[primary_indicator]:,.2f}<extra></extra>',
                    showlegend=True
                ))

        fig_scatter.update_layout(**CHART_LAYOUT)

        # Set axis titles with units
        if 'log' in (scatter_log_x or []):
            xaxis_title = f'{comparison_label} (log scale)'
            fig_scatter.update_xaxes(type='log', title=xaxis_title)
        else:
            fig_scatter.update_xaxes(title=comparison_label)

        if 'log' in (scatter_log_y or []):
            yaxis_title = f'{indicator_label} (log scale)'
            fig_scatter.update_yaxes(type='log', title=yaxis_title)
        else:
            fig_scatter.update_yaxes(title=indicator_label)
        if regression_stats:
            sig_marker = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            annotation_text = (
                f"<b>Correlation (n={regression_stats['n']})</b><br>"
                f"Pearson r: {regression_stats['pearson_r']:.3f} {sig_marker(regression_stats['pearson_p'])}<br>"
                f"Spearman ρ: {regression_stats['spearman_r']:.3f} {sig_marker(regression_stats['spearman_p'])}<br>"
                f"R²: {regression_stats['r_squared']:.3f}<br>"
                f"<i>* p<0.05, ** p<0.01, *** p<0.001</i>"
            )

            fig_scatter.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor='right', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#999',
                borderwidth=1,
                borderpad=6,
                font=dict(size=9)
            )

        fig_scatter.update_layout(
            title=f'{indicator_label} vs {comparison_label} ({latest_year})',
            hovermode='closest'
        )
    else:
        fig_scatter = go.Figure()
        fig_scatter.add_annotation(text='Comparison data not available', xref='paper', yref='paper', showarrow=False)
        fig_scatter.update_layout(**CHART_LAYOUT)
        regression_stats = {}
        valid_data = pd.DataFrame()

    # RESIDUAL PLOT - Linearity Diagnostics
    # Tests: Random scatter = linear, patterns = non-linear, fan shape = heteroscedasticity (non-constant variance)
    if len(valid_data) > 1 and regression_stats:
        y_pred_all = regression_stats['slope'] * valid_data[comparison_indicator] + regression_stats['intercept']
        residuals = valid_data[primary_indicator] - y_pred_all
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(
            x=y_pred_all,
            y=residuals,
            mode='markers',
            marker=dict(size=6, color='rgba(100, 100, 200, 0.6)', line=dict(width=0.5, color='white')),
            name='Residuals',
            hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ))
        fig_residual.add_hline(y=0, line_dash="solid", line_color="#666", line_width=2)

        if len(y_pred_all) > 10:
            sorted_indices = np.argsort(y_pred_all)
            x_sorted = y_pred_all.iloc[sorted_indices]
            y_sorted = residuals.iloc[sorted_indices]
            window = max(3, len(x_sorted) // 10)
            smoothed = pd.Series(y_sorted).rolling(window=window, center=True, min_periods=1).mean()
            fig_residual.add_trace(go.Scatter(
                x=x_sorted, y=smoothed, mode='lines',
                line=dict(color='red', width=3), name='Trend',
                hovertemplate='Trend<extra></extra>'
            ))

        fig_residual.update_layout(**CHART_LAYOUT, title='Residual Plot | Linearity Check',
                                   xaxis_title=f'Fitted Values ({indicator_label})', yaxis_title='Residuals', showlegend=True)
        fig_residual.add_annotation(
            text="<b>Linearity Diagnostics</b><br>If linear: residuals random around 0<br>Red trend line should be flat<br>Variance should be constant",
            xref="paper", yref="paper", x=0.98, y=0.98, xanchor='right', yanchor='top', showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='#ccc', borderwidth=1, borderpad=8, font=dict(size=9)
        )
    else:
        fig_residual = go.Figure()
        fig_residual.add_annotation(text='Insufficient data for residual analysis', xref='paper', yref='paper', showarrow=False)
        fig_residual.update_layout(**CHART_LAYOUT)

    # Q-Q PLOT - Normality of Residuals
    # Tests: Points follow line = normal, deviation = non-normal, Shapiro-Wilk p>0.05 = normal
    if len(valid_data) > 1 and regression_stats:
        from scipy import stats as scipy_stats
        y_pred_all = regression_stats['slope'] * valid_data[comparison_indicator] + regression_stats['intercept']
        residuals = valid_data[primary_indicator] - y_pred_all
        standardized_residuals = (residuals - residuals.mean()) / residuals.std()
        (theoretical_quantiles, ordered_residuals), (slope_qq, intercept_qq, _) = scipy_stats.probplot(
            standardized_residuals, dist='norm', plot=None
        )
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=ordered_residuals,
            mode='markers',
            marker=dict(size=6, color='rgba(100, 200, 100, 0.6)', line=dict(width=0.5, color='white')),
            name='Residuals',
            hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
        ))
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles, y=slope_qq * theoretical_quantiles + intercept_qq,
            mode='lines', line=dict(color='red', width=2, dash='dash'),
            name='Normal distribution', hovertemplate='Expected if normal<extra></extra>'
        ))

        shapiro_stat, shapiro_p = scipy_stats.shapiro(standardized_residuals)
        fig_qq.update_layout(**CHART_LAYOUT, title='Q-Q Plot | Normality of Residuals',
                            xaxis_title='Theoretical Quantiles (standard normal)', yaxis_title='Sample Quantiles (standardized)', showlegend=True)

        normal_text = f"<b>Normality Test</b><br>Shapiro-Wilk: {shapiro_stat:.4f}<br>p-value: {shapiro_p:.4f}<br>"
        normal_text += "<b style='color:green'>✓ Residuals appear normal</b>" if shapiro_p > 0.05 else "<b style='color:orange'>⚠ Non-normal residuals</b>"
        fig_qq.add_annotation(
            text=normal_text, xref="paper", yref="paper", x=0.02, y=0.98, xanchor='left', yanchor='top', showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='#ccc', borderwidth=1, borderpad=8, font=dict(size=10)
        )
    else:
        fig_qq = go.Figure()
        fig_qq.add_annotation(text='Insufficient data for Q-Q plot', xref='paper', yref='paper', showarrow=False)
        fig_qq.update_layout(**CHART_LAYOUT)

    # WORLD MAP - Geographic Distribution
    map_df = df[df['year'] == latest_year].copy()
    iso_col = None
    for c in ['iso_a3', 'iso3', 'countrycode', 'country_code', 'countrycode3']:
        if c in map_df.columns:
            iso_col = c
            break

    if primary_indicator in map_df.columns:
        if iso_col is not None and map_df[iso_col].notna().any():
            fig_map = px.choropleth(map_df,
                                   locations=iso_col,
                                   color=primary_indicator,
                                   hover_name='country',
                                   locationmode='ISO-3',
                                   color_continuous_scale='RdYlGn',
                                   labels={primary_indicator: indicator_label},
                                   title=f'{indicator_label} by country ({latest_year})')
        else:
            fig_map = px.choropleth(map_df,
                                   locations='country',
                                   color=primary_indicator,
                                   hover_name='country',
                                   color_continuous_scale='RdYlGn',
                                   labels={primary_indicator: indicator_label},
                                   title=f'{indicator_label} by country ({latest_year})')
        fig_map.update_layout(
            margin={'t': 40, 'b': 0, 'l': 0, 'r': 0},
            coloraxis_colorbar={'title': indicator_label, 'thickness': 15, 'len': 0.7},
            geo=dict(
                showframe=True,
                framecolor='#333',
                framewidth=2,
                showcoastlines=True,
                coastlinecolor='#666',
                projection_type='natural earth'
            ),
            paper_bgcolor='white',
            font=dict(size=12, color='#222')
        )
    else:
        fig_map = go.Figure()
        fig_map.add_annotation(text='Map data not available', xref='paper', yref='paper', showarrow=False)
        fig_map.update_layout(paper_bgcolor='white')

    # DISTRIBUTION CHART - Histogram with KDE (kernel density estimate)
    # Shows: Data distribution shape, selected countries' positions, skewness, outliers
    latest_year = min(y1, df['year'].max())
    dist_data = df[df['year'] == latest_year].copy()

    if primary_indicator in dist_data.columns:
        dist_values = dist_data[primary_indicator].dropna()

        if len(dist_values) > 0:
            if 'log' in (dist_log_scale or []):
                dist_values_plot = np.log10(dist_values[dist_values > 0])
                x_label = f'{indicator_label} (log₁₀)'
                title_suffix = 'log scale'
            else:
                dist_values_plot = dist_values
                x_label = indicator_label
                title_suffix = ''

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=dist_values_plot,
                name='Frequency',
                marker=dict(color=COLORS[0], opacity=0.7, line=dict(width=1, color='white')),
                nbinsx=30,
                histnorm='probability density'
            ))

            from scipy import stats as scipy_stats
            if len(dist_values_plot) > 2:
                kde = scipy_stats.gaussian_kde(dist_values_plot)
                x_range = np.linspace(dist_values_plot.min(), dist_values_plot.max(), 200)
                y_kde = kde(x_range)
                fig_dist.add_trace(go.Scatter(
                    x=x_range, y=y_kde, mode='lines', name='KDE',
                    line=dict(color='#333', width=3), hovertemplate='Density: %{y:.4f}<extra></extra>'
                ))

            selected_data = dist_data[dist_data['country'].isin(country_list)]
            if not selected_data.empty:
                for idx, row in selected_data.iterrows():
                    if pd.notna(row[primary_indicator]):
                        value = row[primary_indicator]
                        if 'log' in (dist_log_scale or []) and value > 0:
                            value = np.log10(value)
                        elif 'log' in (dist_log_scale or []):
                            continue
                        color_idx = list(country_list).index(row['country']) % len(COLORS)
                        fig_dist.add_vline(x=value, line_dash="dash", line_color=COLORS[color_idx],
                                          line_width=2, annotation_text=row['country'][:15], annotation_position="top")

            fig_dist.update_layout(**CHART_LAYOUT,
                title=f'{indicator_label} | Distribution ({latest_year}){" | " + title_suffix if title_suffix else ""}',
                xaxis_title=x_label, yaxis_title='Density (probability)', showlegend=True, bargap=0.05)
        else:
            fig_dist = go.Figure()
            fig_dist.add_annotation(text='No data available', xref='paper', yref='paper', showarrow=False)
            fig_dist.update_layout(**CHART_LAYOUT)
    else:
        fig_dist = go.Figure()
        fig_dist.add_annotation(text='Indicator not available', xref='paper', yref='paper', showarrow=False)
        fig_dist.update_layout(**CHART_LAYOUT)

    # MAP CLICK INTERACTION - Show Country Details
    selected_info = ''
    if clickData and isinstance(clickData, dict) and 'points' in clickData and len(clickData['points']) > 0:
        loc = clickData['points'][0].get('location')
        clicked_country = None
        if iso_col is not None and loc is not None:
            row = map_df[map_df[iso_col] == loc]
            if not row.empty:
                clicked_country = row.iloc[0]['country']
        if clicked_country is None and loc is not None:
            clicked_country = loc

        if clicked_country is not None:
            cdf = df[df['country'] == clicked_country].sort_values('year')
            if not cdf.empty and primary_indicator in cdf.columns:
                latest_data = cdf[cdf['year'] == latest_year]
                if not latest_data.empty:
                    latest_value = latest_data[primary_indicator].iloc[0]
                    # Compute growth
                    cdf[f'{primary_indicator}_growth'] = cdf[primary_indicator].pct_change() * 100
                    growth_data = cdf[f'{primary_indicator}_growth'].dropna()
                    if not growth_data.empty:
                        latest_growth = growth_data.iloc[-1]
                        # Format value with units
                        formatted_value = f"{latest_value:,.2f}"
                        selected_info = html.Div([
                            html.Strong(f"{clicked_country}"),
                            html.Br(),
                            f"{indicator_label}: {formatted_value}",
                            html.Br(),
                            f"Year: {latest_year}",
                            html.Br(),
                            f"YoY Growth: {latest_growth:+.2f}%"
                        ])
                    else:
                        selected_info = f"{clicked_country}: {indicator_label} = {latest_value:,.2f} ({latest_year})"
                else:
                    selected_info = f"{clicked_country}: No data for {latest_year}"
            else:
                selected_info = f"{clicked_country}: No {indicator_label} data"
    else:
        selected_info = 'Click any country on the map to see detailed information.'

    # SUMMARY STATISTICS PANEL
    # Shows: Count, mean, median, std, min, max, missing values for primary indicator
    if primary_indicator in sub.columns and not sub.empty:
        indicator_data = sub[primary_indicator].dropna()
        if len(indicator_data) > 0:
            stats_content = html.Div([
                html.P([html.Strong("N: "), f"{len(indicator_data):,}"], style={'marginBottom': '4px'}),
                html.P([html.Strong("Mean: "), f"{indicator_data.mean():,.2f}"], style={'marginBottom': '4px'}),
                html.P([html.Strong("Median: "), f"{indicator_data.median():,.2f}"], style={'marginBottom': '4px'}),
                html.P([html.Strong("Std: "), f"{indicator_data.std():,.2f}"], style={'marginBottom': '4px'}),
                html.P([html.Strong("Min: "), f"{indicator_data.min():,.2f}"], style={'marginBottom': '4px'}),
                html.P([html.Strong("Max: "), f"{indicator_data.max():,.2f}"], style={'marginBottom': '4px'}),
                html.P([html.Strong("Missing: "), f"{sub[primary_indicator].isna().sum()}"], style={'marginBottom': '0'})
            ])
        else:
            stats_content = html.P("No data available", style={'color': '#999', 'fontSize': '0.85rem'})
    else:
        stats_content = html.P("No data available", style={'color': '#999', 'fontSize': '0.85rem'})

    # DATA TABLE - Interactive table showing filtered data
    if not sub.empty:
        table_cols = ['country', 'year', primary_indicator]
        if comparison_indicator and comparison_indicator != primary_indicator and comparison_indicator in sub.columns:
            table_cols.append(comparison_indicator)

        table_data = sub[table_cols].sort_values(['country', 'year'], ascending=[True, False])
        table_data = table_data.head(1000)

        columns = [{'name': col, 'id': col} for col in table_data.columns]
        data_table = dash_table.DataTable(
            data=table_data.to_dict('records'),
            columns=columns,
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '8px',
                'fontSize': '0.9rem',
                'fontFamily': 'Inter, sans-serif'
            },
            style_header={
                'backgroundColor': '#f0f0f0',
                'fontWeight': '600',
                'borderBottom': '2px solid #333'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}
            ],
            page_size=20,
            sort_action='native',
            filter_action='native',
            page_action='native'
        )
    else:
        data_table = html.P("No data to display", style={'color': '#999', 'padding': '20px'})

    return fig_map, fig_gdp, fig_growth, fig_rel, fig_scatter, fig_residual, fig_dist, fig_qq, data_table, stats_content, selected_info
"""

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
