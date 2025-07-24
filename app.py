import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    def objective(sigma):
        if sigma <= 0:
            return np.inf
        price = black_scholes_call(S, K, T, r, sigma) if option_type == 'call' else black_scholes_put(S, K, T, r, sigma)
        return abs(price - market_price)
    
    result = minimize_scalar(objective, bounds=(1e-4, 5.0), method='bounded')
    return result.x if result.success else np.nan


st.title("Implied Volatility Surface")

ticker = st.text_input("Enter a ticker (e.g. AAPL, MSFT)", "AAPL").upper()
r = st.number_input("Risk-free rate (annualized)", value=0.05, step=0.01)

stock = yf.Ticker(ticker)
spot_price = stock.history(period='1d')['Close'].iloc[-1]

option_type = st.radio("Option Type", ["Calls", "Puts"]).lower()
expirations = stock.options
max_expiries = st.slider("Number of expirations", 1, len(expirations), 5)

iv_records = []

for expiry in expirations[:max_expiries]:
    try:
        opt_chain = stock.option_chain(expiry)
        options = opt_chain.calls if option_type == 'calls' else opt_chain.puts
        options = options.dropna(subset=['strike', 'bid', 'ask'])
        options['mid'] = (options['bid'] + options['ask']) / 2
        T = (pd.to_datetime(expiry) - pd.Timestamp.now()).days / 365.0
        
        for _, row in options.iterrows():
            K = row['strike']
            market_price = row['mid']
            iv = implied_volatility(market_price, spot_price, K, T, r, option_type=option_type[:-1])  # strip 's'
            if 0 < iv < 5:  # sanity check
                iv_records.append({'strike': K, 'days': T * 365, 'iv': iv})
    except:
        continue


if iv_records:
    df_iv = pd.DataFrame(iv_records)

    xi = np.linspace(df_iv['strike'].min(), df_iv['strike'].max(), 50)
    yi = np.linspace(df_iv['days'].min(), df_iv['days'].max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((df_iv['strike'], df_iv['days']), df_iv['iv'], (xi, yi), method='cubic')

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale='Viridis')])
    fig.update_layout(
        title=f"Implied Volatility Surface for {ticker} ({option_type.title()})",
        scene=dict(xaxis_title='Strike', yaxis_title='Days to Expiry', zaxis_title='IV'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No implied volatility data could be calculated.")