import dash
from dash import dcc, html, Input, Output, State
import requests
import pandas as pd
import plotly.graph_objs as go
from time import time

# Initialize the Dash app
app = dash.Dash(__name__)

# Cache to store API responses and prevent rate-limiting
api_cache = {}

# Cache timeout in seconds
CACHE_TIMEOUT = 60  # Cache data for 1 minute

# Fetch list of cryptocurrencies from CoinGecko API
def fetch_crypto_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        crypto_data = response.json()
        return {crypto['id']: crypto['name'] for crypto in crypto_data}
    except Exception as e:
        print(f"Error fetching crypto list: {e}")
        return {}

# Fetch cryptocurrency prices using the CoinGecko API with caching
def fetch_crypto_prices(crypto_ids):
    cache_key = f"prices:{','.join(crypto_ids)}"
    if cache_key in api_cache and time() - api_cache[cache_key]['timestamp'] < CACHE_TIMEOUT:
        return api_cache[cache_key]['data']
    url = "https://api.coingecko.com/api/v3/simple/price"
    try:
        params = {"ids": ','.join(crypto_ids), "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        api_cache[cache_key] = {'data': data, 'timestamp': time()}
        return data
    except Exception as e:
        print(f"Error fetching crypto prices: {e}")
        return {}

# Fetch cryptocurrency historical data
def fetch_crypto_historical_data(crypto_id):
    cache_key = f"historical:{crypto_id}"
    if cache_key in api_cache and time() - api_cache[cache_key]['timestamp'] < CACHE_TIMEOUT:
        return api_cache[cache_key]['data']
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    try:
        params = {"vs_currency": "usd", "days": "30"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        api_cache[cache_key] = {'data': prices, 'timestamp': time()}
        return prices
    except Exception as e:
        print(f"Error fetching historical data for {crypto_id}: {e}")
        return pd.DataFrame(columns=["timestamp", "price"])

# Generate buy/sell signals
def generate_signals(prices, short_window, long_window):
    if prices.empty:
        return prices
    
    # Calculate short and long moving averages
    prices['SMA_Short'] = prices['price'].rolling(window=short_window).mean()
    prices['SMA_Long'] = prices['price'].rolling(window=long_window).mean()
    
    # Initialize Signal column
    prices['Signal'] = 0

    # Detect crossover points
    prices['Prev_SMA_Short'] = prices['SMA_Short'].shift(1)
    prices['Prev_SMA_Long'] = prices['SMA_Long'].shift(1)

    # Buy signal: Short SMA crosses above Long SMA
    prices.loc[
        (prices['SMA_Short'] > prices['SMA_Long']) & 
        (prices['Prev_SMA_Short'] <= prices['Prev_SMA_Long']),
        'Signal'
    ] = 1

    # Sell signal: Short SMA crosses below Long SMA
    prices.loc[
        (prices['SMA_Short'] < prices['SMA_Long']) & 
        (prices['Prev_SMA_Short'] >= prices['Prev_SMA_Long']),
        'Signal'
    ] = -1

    # Clean up temporary columns
    prices.drop(columns=['Prev_SMA_Short', 'Prev_SMA_Long'], inplace=True)
    
    return prices

# Calculate RSI
def calculate_rsi(prices, window=14):
    if prices.empty:
        return prices
    delta = prices['price'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    prices['RSI'] = 100 - (100 / (1 + rs))
    return prices

# Fetch cryptocurrency list
CRYPTO_LIST = fetch_crypto_list()

# Layout of the app
app.layout = html.Div([
    html.H1("Cryptocurrency Portfolio Tracker", style={'text-align': 'center'}),
    
    html.Div([
        html.Label("Select Cryptocurrencies for Portfolio:"),
        dcc.Dropdown(
            id='crypto-dropdown',
            options=[{'label': name, 'value': crypto} for crypto, name in CRYPTO_LIST.items()],
            multi=True,
            value=['bitcoin', 'ethereum', 'cardano']  # Default selected cryptos
        ),
        html.Button('Update Portfolio', id='update-button', n_clicks=0),
    ], style={'margin-bottom': '20px'}),
    
    html.Div(id='investment-inputs-container', style={'margin-bottom': '20px'}),
    
    dcc.Graph(id='portfolio-value-chart'),
    
    html.Div([
        html.H3("Portfolio Allocation"),
        html.Ul(id='allocation-list'),
    ], style={'margin-top': '20px'}),
    
    html.Div([
        html.Label("Select Cryptocurrency for RSI and Moving Averages:"),
        dcc.Dropdown(
            id='graph-crypto-dropdown',
            options=[{'label': name, 'value': crypto} for crypto, name in CRYPTO_LIST.items()],
            value='bitcoin',  # Default
            style={'margin-bottom': '20px'}
        ),
        html.Label("Adjust Moving Averages:"),
        html.Div([
            html.Label("Short Moving Average:"),
            dcc.Input(id='short-sma-input', type='number', value=5, min=1, step=1, style={'margin-right': '20px'}),
            html.Label("Long Moving Average:"),
            dcc.Input(id='long-sma-input', type='number', value=15, min=1, step=1),
        ], style={'margin-bottom': '20px'}),
        dcc.Graph(id='signals-chart'),
        dcc.Graph(id='rsi-chart'),
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 60 seconds
        n_intervals=0
    )
])

# Callback for investment inputs
@app.callback(
    Output('investment-inputs-container', 'children'),
    Input('crypto-dropdown', 'value')
)
def update_investment_inputs(selected_cryptos):
    if not selected_cryptos:
        return html.Div("No cryptocurrencies selected.")
    return html.Div([
        html.Div([
            html.Label(f"Investment for {CRYPTO_LIST.get(crypto, 'Unknown')}:"),
            dcc.Input(
                id={'type': 'investment-input', 'index': crypto},
                type='number',
                placeholder="Enter amount in USD",
                value=100
            )
        ], style={'margin-bottom': '10px'})
        for crypto in selected_cryptos
    ])

# Callback for portfolio and allocation
@app.callback(
    [Output('portfolio-value-chart', 'figure'),
     Output('allocation-list', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('update-button', 'n_clicks')],
    [State('crypto-dropdown', 'value'),
     State({'type': 'investment-input', 'index': dash.ALL}, 'value')]
)
def update_portfolio(n_intervals, n_clicks, selected_cryptos, investments):
    if not selected_cryptos or not investments:
        return {}, []

    investments_dict = {crypto: inv for crypto, inv in zip(selected_cryptos, investments) if inv is not None}
    prices = fetch_crypto_prices(selected_cryptos)
    if not prices:
        return {}, []

    portfolio_value = {}
    total_value = 0

    for crypto in selected_cryptos:
        if crypto in prices and crypto in investments_dict:
            value = investments_dict[crypto]
            portfolio_value[crypto] = value
            total_value += value

    bar_chart = go.Figure(
        data=[
            go.Bar(
                x=list(portfolio_value.keys()),
                y=list(portfolio_value.values()),
                text=[f"${v:.2f}" for v in portfolio_value.values()],
                textposition='auto'
            )
        ]
    )
    bar_chart.update_layout(
        title="Portfolio Value by Cryptocurrency",
        xaxis_title="Cryptocurrency",
        yaxis_title="Value in USD",
        template="plotly_dark"
    )

    allocation_list = []
    for crypto, value in portfolio_value.items():
        percentage = (value / total_value) * 100
        allocation_list.append(
            html.Li(f"{CRYPTO_LIST.get(crypto, 'Unknown')}: ${value:.2f} ({percentage:.2f}%)")
        )

    return bar_chart, allocation_list

# Callback for RSI and Moving Average graphs
@app.callback(
    [Output('signals-chart', 'figure'),
     Output('rsi-chart', 'figure')],
    [Input('graph-crypto-dropdown', 'value'),
     Input('short-sma-input', 'value'),
     Input('long-sma-input', 'value')]
)
def update_graphs(selected_crypto, short_window, long_window):
    if not selected_crypto or not short_window or not long_window:
        return {}, {}

    historical_data = fetch_crypto_historical_data(selected_crypto)
    if historical_data.empty:
        return {}, {}

    historical_data = generate_signals(historical_data, short_window, long_window)
    historical_data = calculate_rsi(historical_data)

    # Signals Chart
    signals_chart = go.Figure()
    signals_chart.add_trace(go.Scatter(x=historical_data['timestamp'], y=historical_data['price'], mode='lines', name='Price'))
    signals_chart.add_trace(go.Scatter(x=historical_data['timestamp'], y=historical_data['SMA_Short'], mode='lines', name='Short SMA'))
    signals_chart.add_trace(go.Scatter(x=historical_data['timestamp'], y=historical_data['SMA_Long'], mode='lines', name='Long SMA'))
    signals_chart.add_trace(
        go.Scatter(
            x=historical_data.loc[historical_data['Signal'] == 1, 'timestamp'],
            y=historical_data.loc[historical_data['Signal'] == 1, 'price'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=7, symbol='triangle-up')
        )
    )
    signals_chart.add_trace(
        go.Scatter(
            x=historical_data.loc[historical_data['Signal'] == -1, 'timestamp'],
            y=historical_data.loc[historical_data['Signal'] == -1, 'price'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=7, symbol='triangle-down')
        )
    )
    signals_chart.update_layout(
        title=f"Buy/Sell Signals for {CRYPTO_LIST.get(selected_crypto, 'Unknown')}",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    # RSI Chart
    rsi_chart = go.Figure()
    rsi_chart.add_trace(go.Scatter(x=historical_data['timestamp'], y=historical_data['RSI'], mode='lines', name='RSI'))
    rsi_chart.update_layout(
        title=f"RSI for {CRYPTO_LIST.get(selected_crypto, 'Unknown')}",
        xaxis_title="Time",
        yaxis_title="RSI",
        template="plotly_dark",
        shapes=[
            dict(type='line', x0=historical_data['timestamp'].min(), x1=historical_data['timestamp'].max(), y0=70, y1=70, line=dict(color="red", dash="dash")),
            dict(type='line', x0=historical_data['timestamp'].min(), x1=historical_data['timestamp'].max(), y0=30, y1=30, line=dict(color="green", dash="dash"))
        ]
    )

    return signals_chart, rsi_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
