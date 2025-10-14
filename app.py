import pandas as pd
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# DATA LOADING AND PREPARATION
BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "dataset" / "pwt110_cleaned.csv"

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
    'labsh': 'Share of labour compensation in GDP at current national prices',
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
    dbc.Row(dbc.Col(html.H2("PWT 11.0 | Exploratory Data Analysis", className='app-header'))),
    dbc.Row([
    dbc.Col([
            html.Div([
                html.Label("Countries"),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': c, 'value': c} for c in COUNTRIES],
                    value=default_countries,
                    multi=True,
                    placeholder='Select countries...'
                ),
                html.Br(),
                html.Label("Year range"),
                dcc.RangeSlider(
                    id='year-slider',
                    min=MIN_YEAR,
                    max=MAX_YEAR,
                    value=[MIN_YEAR, MAX_YEAR],
                    marks={y: str(y) for y in range(MIN_YEAR, MAX_YEAR+1, max(1, (MAX_YEAR-MIN_YEAR)//8))},
                    step=1
                ),
                html.Br(),
                html.Label('Primary indicator/dependent variable'),
                dcc.Dropdown(
                    id='primary-indicator',
                    options=[{'label': v, 'value': k} for k, v in AVAILABLE_INDICATORS.items()],
                    value='gdp_pc' if 'gdp_pc' in AVAILABLE_INDICATORS else list(AVAILABLE_INDICATORS.keys())[0],
                    clearable=False
                ),
                html.Div(id='primary-indicator-desc', style={'fontSize': '0.85rem', 'color': '#444', 'marginTop': '6px', 'marginBottom': '8px'}),
                html.Br(),
                html.Label('Comparison indicator/independent variable'),
                dcc.Dropdown(
                    id='comparison-indicator',
                    options=[{'label': v, 'value': k} for k, v in AVAILABLE_INDICATORS.items()],
                    value='pop' if 'pop' in AVAILABLE_INDICATORS else list(AVAILABLE_INDICATORS.keys())[1] if len(AVAILABLE_INDICATORS) > 1 else None,
                    clearable=False
                ),
                html.Div(id='comparison-indicator-desc', style={'fontSize': '0.85rem', 'color': '#444', 'marginTop': '6px', 'marginBottom': '8px'}),
                html.Br(),
                dbc.Checklist(
                    id='log-scale',
                    options=[{'label': 'Use log scale for time series', 'value': 'log'}],
                    value=[],
                    switch=True
                ),
                html.Hr(),
                html.Label('Scatter plot settings', style={'fontWeight': '600', 'fontSize': '0.95rem'}),
                html.Br(),
                dbc.Checklist(
                    id='scatter-log-x',
                    options=[{'label': 'Log scale X-axis', 'value': 'log'}],
                    value=[],
                    switch=True
                ),
                dbc.Checklist(
                    id='scatter-log-y',
                    options=[{'label': 'Log scale Y-axis', 'value': 'log'}],
                    value=[],
                    switch=True
                ),
                html.Br(),
                dbc.Checklist(
                    id='scatter-show-labels',
                    options=[{'label': 'Show country labels', 'value': 'show'}],
                    value=['show'],
                    switch=True
                ),
                html.Hr(),
                html.Label('Distribution chart settings', style={'fontWeight': '600', 'fontSize': '0.95rem'}),
                html.Br(),
                dbc.Checklist(
                    id='dist-log-scale',
                    options=[{'label': 'Log scale', 'value': 'log'}],
                    value=[],
                    switch=True
                ),
                html.Hr(),
                dbc.Card([dbc.CardBody([
                    html.H6("Summary Statistics", className='card-title'),
                    html.Div(id='summary-stats', style={'fontSize': '0.85rem'})
                ])], style={'marginBottom': '12px'}),
                html.Br(),
                html.Div(id='map-selected-info', style={'fontSize': '0.9rem', 'color': '#222', 'padding': '8px', 'backgroundColor': '#f0f0f0', 'borderRadius': '4px'}),
            ], className='sidebar')
    ], width=2),

    dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='gdp-world-map', config={'displayModeBar': False}, style={'height': '520px'}), className='dash-graph'), width=6),
                dbc.Col(html.Div(dcc.Graph(id='gdp-timeseries', config={'displayModeBar': False}, style={'height': '520px'}), className='dash-graph'), width=6),
            ], className='chart-row g-2'),

            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='gdp-growth', config={'displayModeBar': False}, style={'height': '400px'}), className='dash-graph'), width=6),
                dbc.Col(html.Div(dcc.Graph(id='gdp-rel-world', config={'displayModeBar': False}, style={'height': '400px'}), className='dash-graph'), width=6)
            ], className='chart-row g-2'),

            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='scatter-latest', config={'displayModeBar': False}, style={'height': '400px'}), className='dash-graph'), width=6),
                dbc.Col(html.Div(dcc.Graph(id='residual-plot', config={'displayModeBar': False}, style={'height': '400px'}), className='dash-graph'), width=6)
            ], className='chart-row g-2'),

            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='distribution-chart', config={'displayModeBar': False}, style={'height': '350px'}), className='dash-graph'), width=6),
                dbc.Col(html.Div(dcc.Graph(id='qq-plot', config={'displayModeBar': False}, style={'height': '350px'}), className='dash-graph'), width=6)
            ], className='chart-row g-2'),

            dbc.Row([
                # missing-data-heatmap removed
            ], className='chart-row g-2'),

            dbc.Row([
                dbc.Col([
                    html.H5("Data Table | Filtered View", style={'marginTop': '8px', 'marginBottom': '8px'}),
                    html.Div(id='data-table-container')
                ], width=12)
            ], className='chart-row g-2'),

        ], width=10)
    ])
], fluid=True, style={'paddingTop': '8px', 'paddingBottom': '20px'})

# CALLBACK - INDICATOR DESCRIPTIONS
@app.callback(
    Output('primary-indicator-desc', 'children'),
    Output('comparison-indicator-desc', 'children'),
    Input('primary-indicator', 'value'),
    Input('comparison-indicator', 'value')
)
def update_indicator_descriptions(primary, comparison):
    """Return human-readable descriptions for selected indicators."""
    def make_desc(key):
        if not key:
            return ''
        label = AVAILABLE_INDICATORS.get(key, key)
        long = INDICATOR_DESCRIPTIONS.get(key)
        if long:
            return html.Div([html.Strong(f"{label}: "), html.Span(long)])
        return html.Div(html.Span(label))

    return make_desc(primary), make_desc(comparison)

# CALLBACK - UPDATE ALL CHARTS
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
def update(country_list, year_range, log_opt, primary_indicator, comparison_indicator, clickData,
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


if __name__ == '__main__':
    app.run(debug=True, port=8000)
