import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from tqdm import tqdm
from plotly.subplots import make_subplots
import datetime
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(
    page_title="Stock Pattern Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        height: 2.5rem;
        font-weight: 500;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .stDataFrame {
        border-radius: 5px;
        border: 1px solid #f0f2f6;
    }
    .css-18e3th9 {
        padding-top: 0.5rem;
    }
    .css-1kyxreq {
        justify-content: center;
        align-items: center;
    }
    .stAlert {
        border-radius: 4px;
    }
    .stSelectbox label, .stDateInput label {
        font-weight: 500;
    }
    .css-1v0mbdj {
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        font-weight: 600;
    }
    /* Remove extra padding */
    .css-12oz5g7 {
        padding-top: 1rem;
    }
    /* Tighten spacing in sidebar */
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Reduce spacing between widgets */
    .css-ocqkz7 {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'selected_pattern' not in st.session_state:
    st.session_state.selected_pattern = None
if 'scan_cancelled' not in st.session_state:
    st.session_state.scan_cancelled = False
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'min_confidence': 0.7,
        'max_patterns': 5
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_cached(symbol, start_date, end_date, forecast_days):
    return fetch_stock_data(symbol, start_date, end_date, forecast_days)

@st.cache_data
def load_stock_symbols():
    try:
        with open("stock_symbols.txt", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("stock_symbols.txt not found")
        return []
    
    

def fetch_stock_data(symbol, start_date, end_date, forecast_days=30):
    try:
        # Convert dates to string format for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_str, end=end_str)
        
        if df.empty:
            st.warning(f"No data found for {symbol} in the selected date range")
            return None
            
        # Reset index and ensure Date column exists
        df = df.reset_index()
        if 'Date' not in df.columns:
            st.error(f"Date column missing for {symbol}")
            return None
            
        # Handle missing data
        if df.isnull().values.any():
            df = df.ffill().bfill()
            
        # Forecast and indicators
        df = forecast_future_prices(df, forecast_days)
        df = calculate_moving_average(df)
        df = calculate_rsi(df)
        
        return df
    
    except yf.YFinanceError as yf_error:
        st.error(f"Yahoo Finance error for {symbol}: {str(yf_error)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching {symbol}: {str(e)}")
        return None

def calculate_moving_average(df, window=50):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=50):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def find_extrema(df, order=5):
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    return peaks, troughs

def detect_head_and_shoulders(df):
    prices = df['Close']
    peaks = argrelextrema(prices.values, np.greater, order=10)[0]
    patterns = []

    for i in range(len(peaks) - 2):
        LS, H, RS = peaks[i], peaks[i + 1], peaks[i + 2]

        # Check if the head is higher than the shoulders
        if prices.iloc[H] > prices.iloc[LS] and prices.iloc[H] > prices.iloc[RS]:
            # Check if the shoulders are roughly equal (within 5% tolerance)
            shoulder_diff = abs(prices.iloc[LS] - prices.iloc[RS]) / max(prices.iloc[LS], prices.iloc[RS])
            if shoulder_diff <= 0.05:  # 5% tolerance
                # Find neckline (troughs between shoulders and head)
                T1 = prices.iloc[LS:H + 1].idxmin()  # Trough between left shoulder and head
                T2 = prices.iloc[H:RS + 1].idxmin()  # Trough between head and right shoulder
                patterns.append({
                    "left_shoulder": LS,
                    "head": H,
                    "right_shoulder": RS,
                    "neckline": (T1, T2)
                })

    return patterns

# ================SIR2
# def find_peaks(data):
#     """Find all peaks in the close price data with additional smoothing."""
#     peaks = []
#     for i in range(2, len(data) - 2):  # Extended window for better peak detection
#         if (data['Close'].iloc[i] > data['Close'].iloc[i-1] and 
#             data['Close'].iloc[i] > data['Close'].iloc[i+1] and
#             data['Close'].iloc[i] > data['Close'].iloc[i-2] and  # Additional checks
#             data['Close'].iloc[i] > data['Close'].iloc[i+2]):
#             peaks.append(i)
#     return peaks

# def find_valleys(data):
#     """Find all valleys in the close price data with additional smoothing."""
#     valleys = []
#     for i in range(2, len(data) - 2):  # Extended window for better valley detection
#         if (data['Close'].iloc[i] < data['Close'].iloc[i-1] and 
#             data['Close'].iloc[i] < data['Close'].iloc[i+1] and
#             data['Close'].iloc[i] < data['Close'].iloc[i-2] and  # Additional checks
#             data['Close'].iloc[i] < data['Close'].iloc[i+2]):
#             valleys.append(i)
#     return valleys

# def detect_head_and_shoulders(data, tolerance=0.03, min_pattern_length=20, volume_ratio=1.2):
#     """
#     Enhanced Head & Shoulders detection with:
#     - Volume analysis
#     - Trend confirmation
#     - Neckline validation
#     - Breakout confirmation
#     """
#     peaks = find_peaks(data)
#     valleys = find_valleys(data)
#     patterns = []
    
#     for i in range(len(peaks) - 2):
#         LS, H, RS = peaks[i], peaks[i+1], peaks[i+2]
        
#         # 1. Basic structure validation
#         if not (data['Close'].iloc[LS] < data['Close'].iloc[H] > data['Close'].iloc[RS]):
#             continue
            
#         # 2. Shoulder symmetry (price)
#         shoulder_diff = abs(data['Close'].iloc[LS] - data['Close'].iloc[RS]) / max(data['Close'].iloc[LS], data['Close'].iloc[RS])
#         if shoulder_diff > tolerance:
#             continue
            
#         # 3. Time symmetry
#         time_diff = abs((H - LS) - (RS - H)) / max(H - LS, RS - H)
#         if time_diff > 0.3:  # Allow 30% time difference
#             continue
            
#         # 4. Minimum pattern duration
#         if (RS - LS) < min_pattern_length:
#             continue
            
#         # 5. Neckline points
#         valley1 = min([v for v in valleys if LS < v < H], key=lambda x: data['Close'].iloc[x], default=None)
#         valley2 = min([v for v in valleys if H < v < RS], key=lambda x: data['Close'].iloc[x], default=None)
#         if not valley1 or not valley2:
#             continue
            
#         # 6. Neckline slope validation
#         neckline_slope = (data['Close'].iloc[valley2] - data['Close'].iloc[valley1]) / (valley2 - valley1)
#         if abs(neckline_slope) > 0.001:  # Filter steep necklines
#             continue
            
#         # 7. Volume analysis
#         # Left shoulder advance volume
#         left_advance_vol = data['Volume'].iloc[valley1+1:H+1].mean()
#         # Right shoulder advance volume
#         right_advance_vol = data['Volume'].iloc[valley2+1:RS+1].mean()
#         # Head advance volume
#         head_advance_vol = data['Volume'].iloc[valley1+1:H+1].mean()
        
#         if not (head_advance_vol > left_advance_vol * volume_ratio and 
#                 right_advance_vol < head_advance_vol):
#             continue
            
#         # 8. Prior uptrend validation
#         lookback = (RS - LS) // 2
#         X = np.arange(max(0, LS-lookback), LS).reshape(-1, 1)
#         y = data['Close'].iloc[max(0, LS-lookback):LS]
#         if LinearRegression().fit(X, y).coef_[0] <= 0:
#             continue
            
#         # 9. Breakout confirmation
#         neckline_at_break = data['Close'].iloc[valley1] + neckline_slope * (RS - valley1)
#         breakout_confirmed = False
#         breakout_idx = None
        
#         for j in range(RS, min(RS + 20, len(data) - 2)):  # Check next 20 candles
#             if all(data['Close'].iloc[j+k] < neckline_at_break for k in range(3)):  # 3 consecutive closes
#                 breakout_confirmed = True
#                 breakout_idx = j + 2
#                 break
                
#         if not breakout_confirmed:
#             continue
            
#         # 10. Breakout volume check
#         if data['Volume'].iloc[breakout_idx] < data['Volume'].iloc[RS] * 0.8:  # Should have decent volume
#             continue
            
#         # Calculate pattern metrics
#         pattern_height = data['Close'].iloc[H] - neckline_at_break
#         target_price = neckline_at_break - pattern_height
        
#         patterns.append({
#             'left_shoulder': data.index[LS],
#             'head': data.index[H],
#             'right_shoulder': data.index[RS],
#             'neckline_points': (data.index[valley1], data.index[valley2]),
#             'neckline_price': neckline_at_break,
#             'breakout_point': data.index[breakout_idx],
#             'target_price': target_price,
#             'pattern_height': pattern_height,
#             'confidence': min(0.99, (1 - shoulder_diff) * (1 - time_diff))
#         })
    
#     return sorted(patterns, key=lambda x: -x['confidence'])  # Return sorted by confidence

def plot_head_and_shoulders(df, patterns):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "scatter"}]]
    )

    # Add price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Price', line=dict(color="#1E88E5", width=2),
        hoverinfo='x+y',
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    if 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA'],
            mode='lines', name="MA (50)", line=dict(color="#FB8C00", width=1.5),
            hoverinfo='x+y',
            hovertemplate='Date: %{x}<br>MA: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    for i, pattern in enumerate(patterns):
        LS, H, RS = pattern["left_shoulder"], pattern["head"], pattern["right_shoulder"]
        T1, T2 = pattern["neckline"]
        
        try:
            # Get all required price points
            ls_price = df.loc[LS, 'Close']
            h_price = df.loc[H, 'Close']
            rs_price = df.loc[RS, 'Close']
            t1_price = df.loc[T1, 'Close']
            t2_price = df.loc[T2, 'Close']
            
            # ========== NECKLINE CALCULATION ==========
            # Create horizontal neckline (average of trough prices)
            avg_neckline = (t1_price + t2_price) / 2
            
            # ========== FIND CONNECTION POINTS ==========
            # Left connection point (before left shoulder)
            left_conn_idx = LS
            for j in range(LS-1, max(0, LS-100), -1):  # Look back up to 100 candles
                if df.loc[j, 'Close'] <= avg_neckline * 1.02:  # 2% tolerance
                    left_conn_idx = j
                    break
            
            # Right connection point (after right shoulder)
            right_conn_idx = min(RS + 100, len(df)-1)  # Look ahead up to 100 candles
            for j in range(RS, right_conn_idx):
                if df.loc[j, 'Close'] <= avg_neckline * 1.02:  # 2% tolerance
                    right_conn_idx = j
                    break
            
            # ========== PATTERN MARKERS ==========
            # Left connection point (new)
            fig.add_trace(go.Scatter(
                x=[df.loc[left_conn_idx, 'Date']], 
                y=[df.loc[left_conn_idx, 'Close']],
                mode="markers",
                marker=dict(size=8, color="#673AB7", symbol="circle-x"),
                name=f"Start Point {i+1}",
                legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='Start Point<extra></extra>'
            ), row=1, col=1)
            
            # Left shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[LS, 'Date']], y=[ls_price],
                mode="markers+text", text=["LS"], textposition="top center",
                marker=dict(size=12, color="#FF5252", symbol="circle"),
                name=f"Left Shoulder {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='Left Shoulder<extra></extra>'
            ), row=1, col=1)
            
            # Head
            fig.add_trace(go.Scatter(
                x=[df.loc[H, 'Date']], y=[h_price],
                mode="markers+text", text=["H"], textposition="top center",
                marker=dict(size=14, color="#4CAF50", symbol="circle"),
                name=f"Head {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='Head<extra></extra>'
            ), row=1, col=1)
            
            # Right shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[RS, 'Date']], y=[rs_price],
                mode="markers+text", text=["RS"], textposition="top center",
                marker=dict(size=12, color="#FF5252", symbol="circle"),
                name=f"Right Shoulder {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='Right Shoulder<extra></extra>'
            ), row=1, col=1)
            
            # Right connection point (new)
            fig.add_trace(go.Scatter(
                x=[df.loc[right_conn_idx, 'Date']], 
                y=[df.loc[right_conn_idx, 'Close']],
                mode="markers",
                marker=dict(size=8, color="#673AB7", symbol="circle-x"),
                name=f"End Point {i+1}",
                legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='End Point<extra></extra>'
            ), row=1, col=1)

            # Neckline points (troughs)
            fig.add_trace(go.Scatter(
                x=[df.loc[T1, 'Date']], y=[t1_price],
                mode="markers", marker=dict(size=8, color="#673AB7", symbol="diamond"),
                name=f"Neck Point {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='Neck Point 1<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[df.loc[T2, 'Date']], y=[t2_price],
                mode="markers", marker=dict(size=8, color="#673AB7", symbol="diamond"),
                showlegend=False, legendgroup=f"pattern{i+1}",
                hoverinfo='x+y',
                hovertemplate='Neck Point 2<extra></extra>'
            ), row=1, col=1)

            # ========== NECKLINE ==========
            neckline_x = [df.loc[left_conn_idx, 'Date'], df.loc[right_conn_idx, 'Date']]
            neckline_y = [avg_neckline, avg_neckline]
            
            fig.add_trace(go.Scatter(
                x=neckline_x,
                y=neckline_y,
                mode="lines",
                line=dict(color="#673AB7", width=2, dash="dash"),
                name=f"Neckline {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # ========== FULL PATTERN OUTLINE ==========
            pattern_x = [
                df.loc[left_conn_idx, 'Date'],  # Start point
                df.loc[LS, 'Date'],            # Left shoulder
                df.loc[T1, 'Date'],            # Left trough
                df.loc[H, 'Date'],             # Head
                df.loc[T2, 'Date'],            # Right trough
                df.loc[RS, 'Date'],            # Right shoulder
                df.loc[right_conn_idx, 'Date']  # End point
            ]
            
            pattern_y = [
                df.loc[left_conn_idx, 'Close'],  # Start price
                ls_price,                       # Left shoulder
                t1_price,                      # Left trough
                h_price,                       # Head
                t2_price,                      # Right trough
                rs_price,                      # Right shoulder
                df.loc[right_conn_idx, 'Close'] # End price
            ]
            
            fig.add_trace(go.Scatter(
                x=pattern_x,
                y=pattern_y,
                mode="lines",
                line=dict(color="rgba(156, 39, 176, 0.5)", width=3),
                name=f"Pattern Outline {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # ========== TARGET PROJECTION ==========
            pattern_height = h_price - avg_neckline
            target_price = avg_neckline - pattern_height
            
            # Target line
            fig.add_trace(go.Scatter(
                x=[df.loc[RS, 'Date'], df.loc[right_conn_idx, 'Date']],
                y=[target_price, target_price],
                mode="lines",
                line=dict(color="#E91E63", width=1.5, dash="dot"),
                name=f"Target {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)
            
            # Target annotation
            fig.add_annotation(
                x=df.loc[right_conn_idx, 'Date'],
                y=target_price,
                text=f"Target: {target_price:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
                font=dict(size=10, color="#E91E63")
            )
            
            # Pattern height annotation
            fig.add_annotation(
                x=df.loc[H, 'Date'],
                y=avg_neckline + pattern_height/2,
                text=f"Height: {pattern_height:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=0,
                font=dict(size=10, color="#4CAF50")
            )
            
        except Exception as e:
            print(f"Error plotting pattern {i}: {str(e)}")
            continue

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'], 
            name="Volume", 
            marker=dict(
                color=np.where(df['Close'] >= df['Open'], '#26A69A', '#EF5350'),
                opacity=0.7
            ),
            hoverinfo='x+y',
            hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    # RSI chart (if available)
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['RSI'], 
                mode='lines', 
                name="RSI", 
                line=dict(color="#7B1FA2", width=1.5),
                hoverinfo='x+y',
                hovertemplate='Date: %{x}<br>RSI: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Overbought/oversold lines
        fig.add_hline(y=70, row=3, col=1, line=dict(color="red", width=1, dash="dash"))
        fig.add_hline(y=30, row=3, col=1, line=dict(color="green", width=1, dash="dash"))

    # ========== FINAL LAYOUT ==========
    fig.update_layout(
        title={
            'text': f"Head & Shoulders Patterns (Found: {len(patterns)})",
            'y':0.95,
            'x':0.5,
            'font': dict(size=20, color="#0D47A1")
        },
        height=900,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v",  # Vertical legend
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.25,          # Move legend outside plot
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        ),
        margin=dict(l=50, r=150, t=100, b=50),  # Extra right margin for legend
        xaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis2=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis3=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    
    # Axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    if 'RSI' in df.columns:
        fig.update_yaxes(title_text="RSI (14)", row=3, col=1)
    
    return fig

def detect_double_bottom(df, order=20, tolerance=0.05, min_pattern_length=10,
                        max_patterns=3, min_height_ratio=0.3, confirmation_bars=3):
    peaks, troughs = find_extrema(df, order=order)
    patterns = []
    
    if len(troughs) < 2:
        return patterns
    
    for i in range(len(troughs) - 1):
        trough1_idx = troughs[i]
        price1 = df['Close'].iloc[trough1_idx]
        
        for j in range(i + 1, len(troughs)):
            trough2_idx = troughs[j]
            
            # Basic validation
            if (trough1_idx >= trough2_idx or 
                trough2_idx - trough1_idx < min_pattern_length):
                continue
                
            price2 = df['Close'].iloc[trough2_idx]
            if abs(price1 - price2)/price1 > tolerance:
                continue
                
            # NEW: Peak validation
            between_peaks = [p for p in peaks if trough1_idx < p < trough2_idx]
            if not between_peaks:
                continue
                
            highest_peak = df['Close'].iloc[between_peaks].max()
            if highest_peak <= max(price1, price2) * 1.01:
                continue

                
            # Neckline validation
            between_slice = df['Close'].iloc[trough1_idx:trough2_idx+1]
            neckline_idx = between_slice.idxmax()
            neckline_price = df['Close'].iloc[neckline_idx]
            
            if (neckline_idx == trough1_idx or 
                neckline_idx == trough2_idx or
                neckline_price <= max(price1, price2) * 1.01):  # 1% buffer
                continue
                
            # Pattern height validation
            min_trough_price = min(price1, price2)
            if trough1_idx > 0:
                prev_high = df['Close'].iloc[:trough1_idx].max()
                if (neckline_price - min_trough_price) < min_height_ratio * (prev_high - min_trough_price):
                    continue
                    
            # Breakout confirmation
            breakout_idx, confirmation_idx = None, None
            for idx in range(trough2_idx, min(len(df), trough2_idx + 50)):
                if df['Close'].iloc[idx] > neckline_price:
                    breakout_idx = idx
                    if all(df['Close'].iloc[idx+k] > neckline_price 
                           for k in range(1, confirmation_bars+1)):
                        confirmation_idx = idx + confirmation_bars
                    break
                    
            if not breakout_idx:
                continue
                
            patterns.append({
                'trough1': trough1_idx,
                'trough2': trough2_idx,
                'neckline': neckline_idx,
                'neckline_price': neckline_price,
                'breakout': breakout_idx,
                'confirmation': confirmation_idx,
                'target': neckline_price + (neckline_price - min_trough_price),
                'pattern_height': neckline_price - min_trough_price,
                'trough_prices': (price1, price2)
            })
    
    # Filter overlapping patterns
    filtered_patterns = []
    last_end = -1
    for pattern in sorted(patterns, key=lambda x: x['trough1']):
        if pattern['trough1'] > last_end:
            filtered_patterns.append(pattern)
            last_end = pattern['trough2']
            if len(filtered_patterns) >= max_patterns:
                break
                
    return filtered_patterns

def plot_double_bottom(df, pattern_points):
    """
    Enhanced Double Bottom pattern visualization with clear W formation, neckline, 
    resistance, breakout points, and target levels.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        pattern_points (list): List of dictionaries containing pattern details.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    # Create a subplot with 2 rows (price and volume)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Chart with Double Bottom Patterns", "Volume")
    )
    
    # Add a simple line chart for the closing prices
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#1f77b4', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

    # Add moving average (optional)
    if 'MA' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['MA'], 
                mode='lines', 
                name="Moving Average (50)", 
                line=dict(color="#FB8C00", width=2)
            ),
            row=1, col=1
        )
    
    # Define colors for pattern visualization
    pattern_colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3']
    
    # Add pattern-specific visualization
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = pattern_colors[idx % len(pattern_colors)]
        
        # Extract pattern points
        trough1_idx = pattern['trough1']
        trough2_idx = pattern['trough2']
        neckline_idx = pattern['neckline']
        neckline_price = pattern['neckline_price']
        target_price = pattern['target']
        
        # Get dates for all points
        trough1_date = df['Date'].iloc[trough1_idx]
        trough2_date = df['Date'].iloc[trough2_idx]
        neckline_date = df['Date'].iloc[neckline_idx]
        
        # Add markers for troughs (bottoms) with improved visibility
        fig.add_trace(
            go.Scatter(
                x=[trough1_date], 
                y=[df['Close'].iloc[trough1_idx]],
                mode="markers+text", 
                text=["Bottom 1"], 
                textposition="bottom center",
                textfont=dict(size=12, color=color),
                marker=dict(size=12, color=color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Pattern {idx+1}: Bottom 1",
                legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[trough2_date], 
                y=[df['Close'].iloc[trough2_idx]],
                mode="markers+text", 
                text=["Bottom 2"], 
                textposition="bottom center",
                textfont=dict(size=12, color=color),
                marker=dict(size=12, color=color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Pattern {idx+1}: Bottom 2",
                legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Add neckline point
        fig.add_trace(
            go.Scatter(
                x=[neckline_date], 
                y=[neckline_price],
                mode="markers+text", 
                text=["Neckline"], 
                textposition="top center",
                textfont=dict(size=12, color=color),
                marker=dict(size=12, color=color, symbol="triangle-up", line=dict(width=2, color='white')),
                name=f"Pattern {idx+1}: Neckline",
                legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Add W formation visualization (connecting the points)
        w_formation_x = [trough1_date]
        w_formation_y = [df['Close'].iloc[trough1_idx]]
        
        # Add intermediate peaks for W formation
        if 'w_formation_peaks' in pattern and pattern['w_formation_peaks']:
            for peak_idx in pattern['w_formation_peaks']:
                w_formation_x.append(df['Date'].iloc[peak_idx])
                w_formation_y.append(df['Close'].iloc[peak_idx])
        
        w_formation_x.append(trough2_date)
        w_formation_y.append(df['Close'].iloc[trough2_idx])
        
        # Draw the W formation line
        fig.add_trace(
            go.Scatter(
                x=w_formation_x, 
                y=w_formation_y,
                mode="lines", 
                line=dict(color=color, width=2, dash='dot'),
                name=f"Pattern {idx+1}: W Formation",
                legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Draw horizontal neckline/resistance line
        # Extend the neckline from first trough to a bit after the second trough
        extension_length = int(0.3 * (trough2_idx - trough1_idx))
        end_idx = min(len(df) - 1, trough2_idx + extension_length)
        neckline_x = [trough1_date, df['Date'].iloc[end_idx]]
        neckline_y = [neckline_price, neckline_price]
        
        fig.add_trace(
            go.Scatter(
                x=neckline_x, 
                y=neckline_y,
                mode="lines", 
                line=dict(color=color, width=2),
                name=f"Pattern {idx+1}: Resistance Line",
                legendgroup=f"pattern{idx+1}"
            ),
            row=1, col=1
        )
        
        # Add breakout point if it exists
        if pattern.get('breakout') is not None:
            breakout_idx = pattern['breakout']
            breakout_date = df['Date'].iloc[breakout_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=[breakout_date], 
                    y=[df['Close'].iloc[breakout_idx]],
                    mode="markers+text", 
                    text=["Breakout"], 
                    textposition="top right",
                    textfont=dict(size=12, color=color),
                    marker=dict(size=12, color=color, symbol="star", line=dict(width=2, color='white')),
                    name=f"Pattern {idx+1}: Breakout",
                    legendgroup=f"pattern{idx+1}"
                ),
                row=1, col=1
            )
            
            # Add confirmation point if it exists
            if pattern.get('confirmation') is not None:
                confirm_idx = pattern['confirmation']
                confirm_date = df['Date'].iloc[confirm_idx]
                
                fig.add_trace(
                    go.Scatter(
                        x=[confirm_date], 
                        y=[df['Close'].iloc[confirm_idx]],
                        mode="markers+text", 
                        text=["Confirmation"], 
                        textposition="top right",
                        textfont=dict(size=12, color=color),
                        marker=dict(size=12, color=color, symbol="diamond", line=dict(width=2, color='white')),
                        name=f"Pattern {idx+1}: Confirmation",
                        legendgroup=f"pattern{idx+1}"
                    ),
                    row=1, col=1
                )
            
            # Add target price line (horizontal line at target price)
            # Only show target if breakout is confirmed
            target_x = [breakout_date, df['Date'].iloc[-1]]  # From breakout to end of chart
            target_y = [target_price, target_price]
            
            fig.add_trace(
                go.Scatter(
                    x=target_x, 
                    y=target_y,
                    mode="lines+text", 
                    text=["Target"], 
                    textposition="middle right",
                    textfont=dict(size=12, color=color),
                    line=dict(color=color, width=2, dash='dash'),
                    name=f"Pattern {idx+1}: Target",
                    legendgroup=f"pattern{idx+1}"
                ),
                row=1, col=1
            )
    
    # Add volume chart with color based on price movement
    colors = ['#26A69A' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF5350' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'], 
            name="Volume", 
            marker=dict(color=colors, opacity=0.8)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Double Bottom Pattern Detection",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",
            groupclick="toggleitem",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            tracegroupgap=5
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis to show proper price formatting
    fig.update_yaxes(
        title_text="Price",
        tickprefix="$",
        gridcolor='lightgray',
        row=1, col=1
    )
    
    # Update volume y-axis
    fig.update_yaxes(
        title_text="Volume",
        gridcolor='lightgray',
        row=2, col=1
    )
    
    return fig

def detect_cup_and_handle(df, order=15, cup_min_bars=20, handle_max_retrace=0.5):
    peaks, troughs = find_extrema(df, order=order)
    patterns = []
    
    if len(peaks) < 1 or len(troughs) < 1:
        return patterns
    
    for i in range(len(peaks) - 1):
        left_peak_idx = peaks[i]
        cup_troughs = [t for t in troughs if t > left_peak_idx]
        
        if not cup_troughs:
            continue
            
        cup_bottom_idx = cup_troughs[0]
        right_peaks = [p for p in peaks if p > cup_bottom_idx]
        
        if not right_peaks:
            continue
            
        right_peak_idx = right_peaks[0]
        
        # Validate cup formation
        if right_peak_idx - left_peak_idx < cup_min_bars:
            continue
            
        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]
        
        # Should be roughly equal peaks (within 5%)
        if abs(right_peak_price - left_peak_price) / left_peak_price > 0.05:
            continue
            
        # Find handle
        handle_troughs = [t for t in troughs if t > right_peak_idx]
        if not handle_troughs:
            continue
            
        handle_bottom_idx = handle_troughs[0]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]
        
        # Calculate handle end (first point above handle entry after bottom)
        handle_end_idx = None
        for j in range(handle_bottom_idx + 1, len(df)):
            if df['Close'].iloc[j] > right_peak_price * 0.98:  # 2% tolerance
                handle_end_idx = j
                break
                
        if not handle_end_idx:  # Never found a valid handle end
            continue
            
        # Validate handle retracement
        cup_height = ((left_peak_price + right_peak_price) / 2) - cup_bottom_price
        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            continue
            
        # Find breakout if any
        breakout_idx = None
        for j in range(handle_end_idx, len(df)):
            if df['Close'].iloc[j] > right_peak_price * 1.02:  # 2% above resistance
                breakout_idx = j
                break
                
        patterns.append({
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_start': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'handle_end': handle_end_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target': right_peak_price + cup_height,
            'cup_height': cup_height,
            'confidence': min(0.99, (1 - abs(left_peak_price-right_peak_price)/left_peak_price))
        })
    
    return patterns

def plot_cup_and_handle(df, pattern_points):
    """
    Enhanced Cup and Handle pattern visualization with clear resistance, breakout, and proper cup/handle formation.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        pattern_points (list): List of dictionaries containing pattern details
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Create a subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price Chart with Cup and Handle", "Volume", "RSI (14)")
    )
    
    # Add price line chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#26A69A', width=2)
        ),
        row=1, col=1
    )

    # Add moving average
    if 'MA' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['MA'], 
                mode='lines', 
                name="Moving Average (50)", 
                line=dict(color="#FB8C00", width=2)
            ),
            row=1, col=1
        )
    
    # Define colors for pattern visualization
    colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3']
    
    # Add pattern-specific visualization
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = colors[idx % len(colors)]
        
        # Extract pattern points
        left_peak_idx = pattern['left_peak']
        cup_bottom_idx = pattern['cup_bottom']
        right_peak_idx = pattern['right_peak']
        handle_bottom_idx = pattern['handle_bottom']
        resistance_level = pattern['resistance']
        
        # Add markers for key points
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[left_peak_idx]],
                y=[df['Close'].iloc[left_peak_idx]],
                mode="markers+text",
                text=["Left Cup Lip"],
                textposition="top right",
                textfont=dict(size=10),
                marker=dict(color="#3F51B5", size=12, symbol="circle"),
                name="Left Cup Lip"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[cup_bottom_idx]],
                y=[df['Close'].iloc[cup_bottom_idx]],
                mode="markers+text",
                text=["Cup Bottom"],
                textposition="bottom center",
                textfont=dict(size=10),
                marker=dict(color="#4CAF50", size=12, symbol="circle"),
                name="Cup Bottom"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[right_peak_idx]],
                y=[df['Close'].iloc[right_peak_idx]],
                mode="markers+text",
                text=["Right Cup Lip"],
                textposition="top left",
                textfont=dict(size=10),
                marker=dict(color="#3F51B5", size=12, symbol="circle"),
                name="Right Cup Lip"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[handle_bottom_idx]],
                y=[df['Close'].iloc[handle_bottom_idx]],
                mode="markers+text",
                text=["Handle Bottom"],
                textposition="bottom right",
                textfont=dict(size=10),
                marker=dict(color="#FF9800", size=12, symbol="circle"),
                name="Handle Bottom"
            ),
            row=1, col=1
        )
        
        # Create a smooth arc for the cup
        # Generate points for the cup arc
        num_points = 100  # More points for a smoother arc
        
        # Create x values (dates) for the arc
        left_date = df['Date'].iloc[left_peak_idx]
        right_date = df['Date'].iloc[right_peak_idx]
        bottom_date = df['Date'].iloc[cup_bottom_idx]
        
        # Calculate time deltas for interpolation
        total_seconds = (right_date - left_date).total_seconds()
        
        # Generate dates for the arc
        arc_dates = []
        for i in range(num_points):
            # Calculate position (0 to 1)
            t = i / (num_points - 1)
            # Calculate seconds from left peak
            seconds_offset = total_seconds * t
            # Calculate the date
            current_date = left_date + pd.Timedelta(seconds=seconds_offset)
            arc_dates.append(current_date)
        
        # Create y values (prices) for the arc
        left_price = df['Close'].iloc[left_peak_idx]
        right_price = df['Close'].iloc[right_peak_idx]
        bottom_price = df['Close'].iloc[cup_bottom_idx]
        
        # Calculate the midpoint between left and right peaks
        mid_price = (left_price + right_price) / 2
        
        # Calculate the depth of the cup
        cup_depth = mid_price - bottom_price
        
        # Generate smooth arc using a quadratic function
        arc_prices = []
        for i in range(num_points):
            # Normalized position (0 to 1)
            t = i / (num_points - 1)
            
            # Parabolic function for U shape: y = a*x^2 + b*x + c
            # Where x is normalized from -1 to 1 for symmetry
            x = 2 * t - 1  # Map t from [0,1] to [-1,1]
            
            # For a symmetric cup, use:
            if abs(left_price - right_price) < 0.05 * left_price:  # If peaks are within 5%
                # Symmetric parabola
                price = mid_price - cup_depth * (1 - x*x)
            else:
                # Asymmetric parabola - linear interpolation with quadratic dip
                if x <= 0:
                    # Left side
                    price = left_price + (mid_price - left_price) * (x + 1) - cup_depth * (1 - x*x)
                else:
                    # Right side
                    price = mid_price + (right_price - mid_price) * x - cup_depth * (1 - x*x)
            
            arc_prices.append(price)
        
        # Add the smooth cup arc
        fig.add_trace(
            go.Scatter(
                x=arc_dates,
                y=arc_prices,
                mode="lines",
                name="Cup Formation",
                line=dict(color="#9C27B0", width=3)
            ),
            row=1, col=1
        )
        
        # Add handle visualization
        handle_indices = list(range(right_peak_idx, handle_bottom_idx + 1))
        if handle_bottom_idx < len(df) - 1:
            # Find where handle ends (either at breakout or at end of data)
            if pattern.get('breakout') is not None:
                handle_end_idx = pattern['breakout']
            else:
                # Find where price recovers to at least 50% of handle depth
                handle_depth = df['Close'].iloc[right_peak_idx] - df['Close'].iloc[handle_bottom_idx]
                recovery_level = df['Close'].iloc[handle_bottom_idx] + (handle_depth * 0.5)
                
                post_handle_indices = df.index[df.index > handle_bottom_idx]
                recovery_indices = [i for i in post_handle_indices if df['Close'].iloc[i] >= recovery_level]
                
                if recovery_indices:
                    handle_end_idx = recovery_indices[0]
                else:
                    handle_end_idx = len(df) - 1
            
            handle_indices.extend(range(handle_bottom_idx + 1, handle_end_idx + 1))
        
        # Add handle line
        handle_dates = df['Date'].iloc[handle_indices].tolist()
        handle_prices = df['Close'].iloc[handle_indices].tolist()
        
        fig.add_trace(
            go.Scatter(
                x=handle_dates,
                y=handle_prices,
                mode="lines",
                name="Handle",
                line=dict(color="#FF9800", width=3)
            ),
            row=1, col=1
        )
        
        # Add resistance level
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[left_peak_idx],
            x1=df['Date'].iloc[-1],
            y0=resistance_level,
            y1=resistance_level,
            line=dict(color="#FF5722", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Add resistance annotation
        fig.add_annotation(
            x=df['Date'].iloc[right_peak_idx],
            y=resistance_level,
            text="Resistance",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF5722",
            ax=0,
            ay=-30,
            font=dict(size=10, color="#FF5722")
        )
        
        # Add breakout point and target if available
        if pattern.get('breakout') is not None:
            breakout_idx = pattern['breakout']
            
            # Add breakout marker
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[breakout_idx]],
                    y=[df['Close'].iloc[breakout_idx]],
                    mode="markers+text",
                    text=["Breakout"],
                    textposition="top right",
                    marker=dict(size=12, color="#4CAF50", symbol="triangle-up"),
                    name="Breakout"
                ),
                row=1, col=1
            )
            
            # Add target price line
            target_price = pattern['target']
            
            fig.add_shape(
                type="line",
                x0=df['Date'].iloc[breakout_idx],
                x1=df['Date'].iloc[-1],
                y0=target_price,
                y1=target_price,
                line=dict(color="#4CAF50", width=2, dash="dot"),
                row=1, col=1
            )
            
            # Add target annotation
            fig.add_annotation(
                x=df['Date'].iloc[-1],
                y=target_price,
                text=f"Target: {target_price:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#4CAF50",
                ax=-40,
                ay=0,
                font=dict(size=10, color="#4CAF50")
            )
            
            # Add measured move visualization
            cup_height = pattern['cup_height']
            
            # Add annotation for cup height (measured move)
            fig.add_annotation(
                x=df['Date'].iloc[breakout_idx],
                y=(resistance_level + target_price) / 2,
                text=f"Measured Move: {cup_height:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#4CAF50",
                ax=40,
                ay=0,
                font=dict(size=10, color="#4CAF50")
            )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'], 
            name="Volume", 
            marker=dict(
                color=np.where(df['Close'] >= df['Open'], '#26A69A', '#EF5350'),
                line=dict(color='rgba(0,0,0,0)', width=0)
            ),
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Add RSI chart
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['RSI'], 
                mode='lines', 
                name="RSI (14)", 
                line=dict(color="#7B1FA2", width=2)
            ),
            row=3, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line", line=dict(dash="dash", color="red", width=2),
            x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=70, y1=70,
            row=3, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", color="green", width=2),
            x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=30, y1=30,
            row=3, col=1
        )
        
        # Add annotations for overbought/oversold
        fig.add_annotation(
            x=df['Date'].iloc[0], y=70,
            text="Overbought (70)",
            showarrow=False,
            xanchor="left",
            font=dict(color="red"),
            row=3, col=1
        )
        fig.add_annotation(
            x=df['Date'].iloc[0], y=30,
            text="Oversold (30)",
            showarrow=False,
            xanchor="left",
            font=dict(color="green"),
            row=3, col=1
        )
    
    # Add pattern explanation
    fig.add_annotation(
        x=df['Date'].iloc[0],
        y=df['Close'].max(),
        text="Cup and Handle: Bullish continuation pattern with target equal to cup depth projected above breakout",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="#0D47A1"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#0D47A1",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Cup and Handle Pattern Detection",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.4,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        hovermode="x unified"
    )
    
    return fig

def plot_pattern(df, pattern_points, pattern_name):
    """
    Route to the appropriate pattern-specific plotting function.
    
    Args:
        df: DataFrame with OHLCV data
        pattern_points: Detected patterns
        pattern_name: One of ['Double Bottom', 'Cup and Handle', 'Head and Shoulders']
    
    Returns:
        go.Figure: The plotted chart
    """
    plotters = {
        'Double Bottom': plot_double_bottom,
        'Cup and Handle': plot_cup_and_handle,
        'Head and Shoulders': plot_head_and_shoulders
    }
    
    if pattern_name not in plotters:
        raise ValueError(f"Unsupported pattern type: {pattern_name}. Available: {list(plotters.keys())}")
    
    return plotters[pattern_name](df, pattern_points)

def evaluate_pattern_detection(df, patterns):
    total_patterns = 0
    correct_predictions = 0
    false_positives = 0
    look_forward_window = 10

    for pattern_type, pattern_list in patterns.items():
        total_patterns += len(pattern_list)
        for pattern in pattern_list:
            if pattern_type == "Head and Shoulders":
                last_point_idx = max(pattern['left_shoulder'], pattern['head'], pattern['right_shoulder'])
            elif pattern_type == "Double Top":
                last_point_idx = max(pattern['peak1'], pattern['peak2'])
            elif pattern_type == "Double Bottom":
                last_point_idx = max(pattern['trough1'], pattern['trough2'])
            elif pattern_type == "Cup and Handle":
                last_point_idx = pattern['handle_end']
            else:
                continue 

            if last_point_idx + look_forward_window < len(df):
                if pattern_type in ["Double Bottom", "Cup and Handle"]:
                    if df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        false_positives += 1
                elif pattern_type in ["Head and Shoulders", "Double Top"]:
                    if df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        false_positives += 1

    if total_patterns > 0:
        accuracy = correct_predictions / total_patterns
        precision = correct_predictions / (correct_predictions + false_positives) if (correct_predictions + false_positives) > 0 else 0
    else:
        accuracy = 0.0
        precision = 0.0

    return accuracy, precision

def is_trading_day(date):
    """Check if the given date is a trading day (Monday to Friday)."""
    return date.weekday() < 5 

def get_nearest_trading_day(date):
    """Adjust the date to the nearest previous trading day if it's a weekend or holiday."""
    while not is_trading_day(date):
        date -= datetime.timedelta(days=1)
    return date

def forecast_future_prices(df, forecast_days=30):
    """Forecast future prices using linear regression."""
    # Prepare data for linear regression
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Close'].values
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future prices
    future_X = np.array(range(len(df), len(df) + forecast_days)).reshape(-1, 1)
    future_prices = model.predict(future_X)
    
    # Create a DataFrame for the future data
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days + 1, freq='B')[1:]
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Close': future_prices
    })
    
    # Combine historical and future data
    combined_df = pd.concat([df, future_df], ignore_index=True)
    
    return combined_df

def main():
    # App header with title
    st.markdown("# Stock Pattern Scanner(Yahoo Finance)")
    st.markdown("Identify technical chart patterns with precision")

    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <style>
            .block-container {
                margin-bottom: 0.5rem; /* Adjust this value to change the spacing */
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("# Scanner Settings")
        
        # Date selection
        st.markdown("### 📅 Date Selection")
        date_option = st.radio(
            "Select Date Option",
            options=["Date Range"],
            index=0,
            help="Choose between a date range or a specific trading day"
        )

        start_date = None
        end_date = None
        single_date = None

        if date_option == "Date Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.date(2023, 1, 1),
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date(2100, 12, 31)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.date(2024, 1, 1),
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date(2100, 12, 31)
                )
            if end_date < start_date:
                st.error("End Date must be after Start Date.")
                return
            # Add to main() before scanning
            if (end_date - start_date).days < 30:
                st.error("Date range must be at least 30 days")
                st.stop()
                
            if (end_date - start_date).days > 365 * 3:
                st.warning("Large date range may impact performance. Consider narrowing your range.")

        # Forecast days input
        forecast_days = st.number_input(
            "Forecast Days",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days to forecast future prices"
        )

        # Scan button
        scan_button = st.button("Scan Stocks", use_container_width=True)
        
        if date_option == "Date Range":
            st.info(f"Selected: **{start_date}** to **{end_date}** with **{forecast_days}** forecast days")
            
    # Main content
    if scan_button:
        cancel_button = st.sidebar.button("Cancel Scan")
        if cancel_button:
            st.session_state.scan_cancelled = True
            st.warning("Scan cancelled by user")
            st.stop()
        try:
            with open("stock_symbols.txt", "r") as f:
                stock_symbols = [line.strip() for line in f]
        except FileNotFoundError:
            st.error("stock_symbols.txt not found. Please create the file with stock symbols, one per line.")
            return
        except Exception as e: 
            st.error(f"An error occurred while reading the stock symbols file: {e}")
            return
            
        with st.container():
            st.markdown("## 🔍 Scanning Stocks")
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_container = st.empty()
            
            stock_data = []
            
            for i, symbol in enumerate(stock_symbols):
                if st.session_state.scan_cancelled:
                    break

                status_container.info(f"Processing {symbol}... ({i+1}/{len(stock_symbols)})")
                
                try:
                    df = fetch_stock_data(symbol, start_date, end_date, forecast_days)
                    if df is None or df.empty:
                        continue
                    
                    patterns = {
                        "Head and Shoulders": detect_head_and_shoulders(df),
                        # "Double Top": detect_double_top(df),
                        "Double Bottom": detect_double_bottom(df),
                        "Cup and Handle": detect_cup_and_handle(df),
                    }
                    
                    accuracy, precision = evaluate_pattern_detection(df, patterns)
                    
                    stock_info = yf.Ticker(symbol).info
                    current_price = stock_info.get('currentPrice', None)
                    volume = stock_info.get('volume', None)
                    percent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

                    stock_data.append({
                        "Symbol": symbol, 
                        "Patterns": patterns, 
                        "Data": df,
                        "Current Price": current_price,
                        "Volume": volume,
                        "Percent Change": percent_change,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "MA": df['MA'].iloc[-1] if 'MA' in df.columns else None,
                        "RSI": df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
                    })
                
                except Exception as e:
                    st.error(f"Error processing {symbol}: {e}")
                    continue
                
                progress_bar.progress((i + 1) / len(stock_symbols))
            
            st.session_state.stock_data = stock_data
            st.session_state.selected_stock = None
            st.session_state.selected_pattern = None
            
            progress_container.empty()
            status_container.success("Scan completed successfully!")
    
    # Display results
    if st.session_state.stock_data:
        st.markdown("## 📊 Scan Results")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📋 Stock List", "📈 Pattern Visualization"])
        
        with tab1:
            # Prepare data for the table
            table_data = []
            for stock in st.session_state.stock_data:
                # Count patterns
                pattern_counts = {
                    pattern: len(stock["Patterns"][pattern]) 
                    for pattern in stock["Patterns"]
                }
                
                # Create row
                row = {
                    "Symbol": stock["Symbol"],
                    "Current Price": stock['Current Price'] if stock['Current Price'] else None,
                    "Volume": stock['Volume'] if stock['Volume'] else None,
                    "% Change": stock['Percent Change'],
                    "MA (50)": stock['MA'],
                    "RSI (14)": stock['RSI'],
                    "Accuracy": stock['Accuracy'],
                    "Precision": stock['Precision'],
                    "Head and Shoulders": pattern_counts.get("Head and Shoulders", 0),
                    "Double Top": pattern_counts.get("Double Top", 0),
                    "Double Bottom": pattern_counts.get("Double Bottom", 0),
                    "Cup and Handle": pattern_counts.get("Cup and Handle", 0),
                }
                table_data.append(row)
            
            # Create DataFrame
            df_table = pd.DataFrame(table_data)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0, step=1.0)
            with col2:
                min_volume = st.number_input("Min Volume", min_value=0, value=0, step=1000)
            with col3:
                pattern_filter = st.selectbox("Filter by Pattern", 
                                             ["All Patterns", "Head and Shoulders", "Double Top", "Double Bottom", "Cup and Handle"])
            
            # Apply filters
            filtered_df = df_table.copy()
            if min_price > 0:
                filtered_df = filtered_df[filtered_df["Current Price"] >= min_price]
            if min_volume > 0:
                filtered_df = filtered_df[filtered_df["Volume"] >= min_volume]
            if pattern_filter != "All Patterns":
                filtered_df = filtered_df[filtered_df[pattern_filter] > 0]
            
            # Format the table
            formatted_df = filtered_df.copy()
            formatted_df["Current Price"] = formatted_df["Current Price"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Volume"] = formatted_df["Volume"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")
            formatted_df["% Change"] = formatted_df["% Change"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            formatted_df["MA (50)"] = formatted_df["MA (50)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["RSI (14)"] = formatted_df["RSI (14)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            formatted_df["Precision"] = formatted_df["Precision"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            # Display the table with custom styling
            st.dataframe(
                formatted_df,
                height=500,
                use_container_width=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol"),
                    "Current Price": st.column_config.TextColumn("Price"),
                    "Volume": st.column_config.TextColumn("Volume"),
                    "% Change": st.column_config.TextColumn("Change (%)"),
                    "MA (50)": st.column_config.TextColumn("MA (50)"),
                    "RSI (14)": st.column_config.TextColumn("RSI (14)"),
                    "Accuracy": st.column_config.TextColumn("Accuracy"),
                    "Precision": st.column_config.TextColumn("Precision"),
                    "Head and Shoulders": st.column_config.NumberColumn("H&S", format="%d"),
                    "Double Top": st.column_config.NumberColumn("Double Top", format="%d"),
                    "Double Bottom": st.column_config.NumberColumn("Double Bottom", format="%d"),
                    "Cup and Handle": st.column_config.NumberColumn("Cup & Handle", format="%d"),
                }
            )
            
            # Show summary statistics
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks", len(filtered_df))
            with col2:
                total_patterns = filtered_df[["Head and Shoulders", "Double Top", "Double Bottom", "Cup and Handle"]].sum().sum()
                st.metric("Total Patterns", int(total_patterns))
            with col3:
                avg_accuracy = filtered_df["Accuracy"].mean()
                st.metric("Avg. Accuracy", f"{avg_accuracy:.2f}")
            with col4:
                avg_precision = filtered_df["Precision"].mean()
                st.metric("Avg. Precision", f"{avg_precision:.2f}")
        
        with tab2:
            st.markdown("### 🔍 Pattern Visualization")
            
            # Stock selection
            col1, col2 = st.columns(2)
            with col1:
                selected_stock = st.selectbox(
                    "Select Stock",
                    options=[stock["Symbol"] for stock in st.session_state.stock_data],
                    key='stock_select'
                )
            
            if selected_stock != st.session_state.selected_stock:
                st.session_state.selected_stock = selected_stock
                st.session_state.selected_pattern = None
            
            selected_data = next((item for item in st.session_state.stock_data 
                                if item["Symbol"] == st.session_state.selected_stock), None)
            
            if selected_data:
                # Pattern selection
                pattern_options = [p for p, v in selected_data["Patterns"].items() if v]
                
                if pattern_options:
                    with col2:
                        selected_pattern = st.selectbox(
                            "Select Pattern",
                            options=pattern_options,
                            key='pattern_select'
                        )
                    
                    if selected_pattern != st.session_state.selected_pattern:
                        st.session_state.selected_pattern = selected_pattern
                    
                    if st.session_state.selected_pattern:
                        pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                        
                        # Display stock info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${selected_data['Current Price']:.2f}" if selected_data['Current Price'] else "N/A")
                        with col2:
                            st.metric("Volume", f"{int(selected_data['Volume']):,}" if selected_data['Volume'] else "N/A")
                        with col3:
                            st.metric("% Change", f"{selected_data['Percent Change']:.2f}%")
                        with col4:
                            st.metric("RSI (14)", f"{selected_data['RSI']:.2f}" if selected_data['RSI'] else "N/A")
                        
                        # Plot the pattern
                        if not isinstance(pattern_points, list):
                            pattern_points = [pattern_points]
                        
                        st.plotly_chart(
                            plot_pattern(
                                selected_data["Data"],
                                pattern_points,
                                st.session_state.selected_pattern
                            ),
                            use_container_width=True
                        )
                        
                        # Pattern explanation
                        with st.expander("📚 Pattern Explanation"):
                            if selected_pattern == "Head and Shoulders":
                                st.markdown("""
                                **Head and Shoulders Pattern**
                                
                                A bearish reversal pattern that signals a potential trend change from bullish to bearish. It consists of:
                                - Left shoulder: A peak followed by a decline
                                - Head: A higher peak followed by another decline
                                - Right shoulder: A lower peak similar to the left shoulder
                                
                                **Trading Implications**: When the price breaks below the neckline (support level), it often indicates a strong sell signal.
                                """)
                            elif selected_pattern == "Double Top":
                                st.markdown("""
                                **Double Top Pattern**
                                
                                A bearish reversal pattern that forms after an extended upward trend. It consists of:
                                - Two peaks at approximately the same price level
                                - A valley (trough) between the peaks
                                
                                **Trading Implications**: When the price falls below the valley between the two tops, it signals a potential downtrend.
                                """)
                            elif selected_pattern == "Double Bottom":
                                st.markdown("""
                                **Double Bottom Pattern**
                                
                                A bullish reversal pattern that forms after an extended downward trend. It consists of:
                                - Two troughs at approximately the same price level
                                - A peak between the troughs
                                
                                **Trading Implications**: When the price rises above the peak between the two bottoms, it signals a potential uptrend.
                                """)
                            elif selected_pattern == "Cup and Handle":
                                st.markdown("""
                                **Cup and Handle Pattern**
                                
                                A bullish continuation pattern that signals a pause in an uptrend before continuing higher. It consists of:
                                - Cup: A rounded bottom formation (U-shape)
                                - Handle: A slight downward drift forming a consolidation
                                
                                **Trading Implications**: The pattern completes when the price breaks above the resistance level formed by the cup's rim.
                                """)
                else:
                    st.info(f"No patterns detected for {selected_stock}.")
            else:
                st.error("Selected stock data not found.")

if __name__ == "__main__":
    main()
