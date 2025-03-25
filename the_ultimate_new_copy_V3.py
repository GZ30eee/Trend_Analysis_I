import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from tqdm import tqdm
from plotly.subplots import make_subplots
import os
import datetime
from datetime import timedelta
import plotly.express as px
import logging
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(
    page_title="Stock Pattern Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: left;
        margin-bottom: 1rem;
        margin-top: 0; /* Add this line to remove the top margin */
        padding-top: 0; /* Add this line to remove any top padding */
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0D47A1;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #546E7A;
        margin-top: 0.3rem;
    }
    .pattern-positive {
        color: #2E7D32;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .pattern-negative {
        color: #C62828;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .sidebar .sidebar-content {
        background-color: #f1f8fe;
    }
    .dataframe {
        font-size: 0.8rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'selected_pattern' not in st.session_state:
    st.session_state.selected_pattern = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

def is_trading_day(date):
    # Check if it's a weekend
    if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    # Here you could add more checks for holidays if needed
    return True

def get_nearest_trading_day(date):
    # Try the next day first
    test_date = date
    for _ in range(7):  # Try up to a week forward
        test_date += timedelta(days=1)
        if is_trading_day(test_date):
            return test_date
    
    # If no trading day found forward, try backward
    test_date = date
    for _ in range(7):  # Try up to a week backward
        test_date -= timedelta(days=1)
        if is_trading_day(test_date):
            return test_date
    
    # If still no trading day found, return the original date
    return date

def validate_stock_data(df):
    # Check for missing values
    if df.isnull().values.any():
        st.warning("The dataset contains missing values. Please clean the data before proceeding.")
        logging.warning("The dataset contains missing values.")
        return False
    
    # Check for duplicate rows
    if df.duplicated().any():
        st.warning("The dataset contains duplicate rows. Please clean the data before proceeding.")
        logging.warning("The dataset contains duplicate rows.")
        return False
    
    # Check for negative values in price and volume columns
    if (df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']] < 0).any().any():
        st.warning("The dataset contains negative values in price or volume columns. Please clean the data before proceeding.")
        logging.warning("The dataset contains negative values in price or volume columns.")
        return False
    
    return True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_stock_data_from_excel(file_path):
    try:
        with st.spinner("Reading Excel file..."):
            df = pd.read_excel(file_path)
            # Ensure the required columns are present
            required_columns = ['TIMESTAMP', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            if not all(column in df.columns for column in required_columns):
                st.error(f"The Excel file must contain the following columns: {required_columns}")
                logging.error(f"Missing required columns in the Excel file: {required_columns}")
                return None
            
            # Convert TIMESTAMP to datetime
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            
            # Remove commas from numeric columns and convert to numeric
            for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                df[col] = df[col].replace({',': ''}, regex=True).astype(float)
            
            # Strip leading/trailing spaces from SYMBOL
            df['SYMBOL'] = df['SYMBOL'].str.strip()
            
            return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        logging.error(f"Error reading Excel file: {e}")
        return None

def fetch_stock_data(symbol, start_date, end_date, df):
    try:
        # Convert start_date and end_date to datetime64[ns] for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data for the selected symbol and date range
        df_filtered = df[(df['SYMBOL'] == symbol) & (df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date)]
        
        if df_filtered.empty:
            logging.warning(f"No data found for {symbol} within the date range {start_date} to {end_date}.")
            return None
        
        # Reset index to ensure we have sequential integer indices
        df_filtered = df_filtered.reset_index(drop=True)
        
        # Rename columns to match the expected format
        df_filtered = df_filtered.rename(columns={
            'TIMESTAMP': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        # Forecast future prices
        # df_filtered = forecast_future_prices(df_filtered, forecast_days)
        
        # Calculate Moving Average and RSI
        df_filtered = calculate_moving_average(df_filtered)
        df_filtered = calculate_rsi(df_filtered)
        df_filtered = calculate_moving_average_two(df_filtered)
        
        return df_filtered
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_moving_average(df, window=50):
    df['MA'] = df['Close'].rolling(window=window, min_periods=1).mean()
    return df

def calculate_moving_average_two(df, window=200):
    df['MA2'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def find_extrema(df, order=20):
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    return peaks, troughs

def find_peaks(data):
    """Find all peaks in the close price data with additional smoothing."""
    peaks = []
    for i in range(2, len(data) - 2):  # Extended window for better peak detection
        if (data['Close'].iloc[i] > data['Close'].iloc[i-1] and 
            data['Close'].iloc[i] > data['Close'].iloc[i+1] and
            data['Close'].iloc[i] > data['Close'].iloc[i-2] and  # Additional checks
            data['Close'].iloc[i] > data['Close'].iloc[i+2]):
            peaks.append(i)
    return peaks

def find_valleys(data):
    """Find all valleys in the close price data with additional smoothing."""
    valleys = []
    for i in range(2, len(data) - 2):  # Extended window for better valley detection
        if (data['Close'].iloc[i] < data['Close'].iloc[i-1] and 
            data['Close'].iloc[i] < data['Close'].iloc[i+1] and
            data['Close'].iloc[i] < data['Close'].iloc[i-2] and  # Additional checks
            data['Close'].iloc[i] < data['Close'].iloc[i+2]):
            valleys.append(i)
    return valleys

def detect_head_and_shoulders(data, 
                              is_inverse=False,
                              tolerance=0.15,              # Loosened from 0.08
                              min_pattern_length=5,        # Loosened from 8
                              volume_ratio_head=1.1,       # Loosened from 1.2
                              volume_ratio_breakout=1.05,  # Loosened from 1.1
                              time_symmetry_threshold=0.8, # Loosened from 0.5
                              neckline_slope_threshold=0.01, # Loosened from 0.005
                              min_trend_strength=0.0001,   # Loosened from 0.0005
                              breakout_lookahead=15,
                              breakout_confirmation_bars=1): # Loosened from 2
    peaks = find_peaks(data) if not is_inverse else find_valleys(data)
    valleys = find_valleys(data) if not is_inverse else find_peaks(data)
    patterns = []
    
    if len(peaks) < 3:
        return patterns
    
    for i in range(1, len(peaks) - 1):
        LS, H, RS = peaks[i-1], peaks[i], peaks[i+1]
        
        # Basic structure check (unchanged)
        if not is_inverse:
            if not (data['Close'].iloc[LS] < data['Close'].iloc[H] > data['Close'].iloc[RS]):
                continue
        else:
            if not (data['Close'].iloc[LS] > data['Close'].iloc[H] < data['Close'].iloc[RS]):
                continue
        
        # Shoulder symmetry (looser tolerance)
        shoulder_diff = abs(data['Close'].iloc[LS] - data['Close'].iloc[RS]) / max(data['Close'].iloc[LS], data['Close'].iloc[RS])
        if shoulder_diff > tolerance:
            continue
        
        # Time symmetry (looser threshold)
        left_time = H - LS
        right_time = RS - H
        time_diff = abs(left_time - right_time) / max(left_time, right_time)
        if time_diff > time_symmetry_threshold:
            continue
        
        # Minimum pattern duration (looser)
        if (RS - LS) < min_pattern_length:
            continue
        
        # Find neckline points
        valley1 = data['Close'].iloc[LS:H].idxmin() if not is_inverse else data['Close'].iloc[LS:H].idxmax()
        valley2 = data['Close'].iloc[H:RS].idxmin() if not is_inverse else data['Close'].iloc[H:RS].idxmax()
        T1 = data.index.get_loc(valley1)
        T2 = data.index.get_loc(valley2)
        
        if pd.isna(valley1) or pd.isna(valley2):
            continue
        
        # Neckline slope (looser threshold)
        neckline_slope = (data['Close'].iloc[T2] - data['Close'].iloc[T1]) / (T2 - T1)
        if abs(neckline_slope) > neckline_slope_threshold:
            continue
        
        # Volume analysis (looser ratios, optional)
        if 'Volume' in data.columns:
            head_vol = data['Volume'].iloc[H]
            left_vol = data['Volume'].iloc[LS]
            right_vol = data['Volume'].iloc[RS]
            avg_prior_vol = data['Volume'].iloc[max(0, LS-10):LS].mean()
            vol_head_ok = head_vol > avg_prior_vol * volume_ratio_head or head_vol > right_vol  # OR instead of AND
        else:
            vol_head_ok = True
        
        # Prior trend validation (looser, optional)
        trend_lookback = min(30, LS)
        if trend_lookback >= 5:  # Only check if enough data
            X = np.arange(LS - trend_lookback, LS).reshape(-1, 1)
            y = data['Close'].iloc[LS - trend_lookback:LS].values
            trend_coef = LinearRegression().fit(X, y).coef_[0]
            if (not is_inverse and trend_coef < min_trend_strength) or (is_inverse and trend_coef > -min_trend_strength):
                continue
        else:
            trend_coef = 0  # Skip trend check if too little data
        
        # Breakout detection (looser, optional)
        neckline_at_rs = data['Close'].iloc[T1] + neckline_slope * (RS - T1)
        breakout_confirmed = False
        breakout_idx = None
        throwback_low_idx = None
        pullback_high_idx = None
        
        for j in range(RS, min(RS + breakout_lookahead, len(data) - breakout_confirmation_bars)):
            prices = data['Close'].iloc[j:j + breakout_confirmation_bars]
            vols = data['Volume'].iloc[j:j + breakout_confirmation_bars] if 'Volume' in data.columns else [1]
            neckline_at_j = data['Close'].iloc[T1] + neckline_slope * (j - T1)
            
            if (not is_inverse and all(p < neckline_at_j for p in prices)) or \
               (is_inverse and all(p > neckline_at_j for p in prices)):
                if 'Volume' in data.columns:
                    avg_breakout_vol = vols.mean()
                    if avg_breakout_vol < avg_prior_vol * volume_ratio_breakout:
                        continue
                breakout_idx = j
                breakout_confirmed = True
                break
        
        # V-formation and Pullback High (unchanged, optional)
        if breakout_confirmed:
            v_search_range = min(breakout_idx + breakout_lookahead, len(data))
            v_prices = data['Close'].iloc[breakout_idx:v_search_range]
            throwback_low_idx = v_prices.idxmin() if not is_inverse else v_prices.idxmax()
            throwback_low_idx = data.index.get_loc(throwback_low_idx)
            pullback_prices = data['Close'].iloc[throwback_low_idx:v_search_range]
            pullback_high_idx = pullback_prices.idxmax() if not is_inverse else pullback_prices.idxmin()
            pullback_high_idx = data.index.get_loc(pullback_high_idx)
            neckline_at_pullback = data['Close'].iloc[T1] + neckline_slope * (pullback_high_idx - T1)
            v_confirmed = (not is_inverse and pullback_prices.max() >= neckline_at_pullback * 0.98) or \
                          (is_inverse and pullback_prices.min() <= neckline_at_pullback * 1.02)
        else:
            v_confirmed = False
        
        # Pattern metrics
        neckline_at_break = data['Close'].iloc[T1] + neckline_slope * (breakout_idx - T1) if breakout_idx else neckline_at_rs
        pattern_height = abs(data['Close'].iloc[H] - neckline_at_break)
        target_price = (neckline_at_break - pattern_height) if not is_inverse else (neckline_at_break + pattern_height)
        neckline_end_idx = pullback_high_idx if v_confirmed and pullback_high_idx else (breakout_idx if breakout_confirmed else RS)
        neckline_at_end = data['Close'].iloc[T1] + neckline_slope * (neckline_end_idx - T1)
        
        # Loosened confidence calculation
        confidence = min(0.99,
                        (1 - (shoulder_diff / tolerance)) * 0.25 +  # Reduced weight
                        (1 - (time_diff / time_symmetry_threshold)) * 0.15 +  # Reduced weight
                        (min(1, abs(trend_coef) * 1000)) * 0.2 +  # Looser scaling
                        (0.2 if vol_head_ok else 0.15) +  # Smaller penalty
                        (0.2 if breakout_confirmed else 0.05) +  # Smaller penalty for no breakout
                        (0.15 if v_confirmed else 0))  # Smaller bonus
        
        pattern_data = {
            'left_shoulder': data.index[LS],
            'head': data.index[H],
            'right_shoulder': data.index[RS],
            'neckline_points': (data.index[T1], data.index[T2]),
            'breakout_point': data.index[breakout_idx] if breakout_idx else None,
            'throwback_low': data.index[throwback_low_idx] if throwback_low_idx else None,
            'pullback_high': data.index[pullback_high_idx] if pullback_high_idx else None,
            'neckline_end': data.index[neckline_end_idx],
            'neckline_price': neckline_at_end,
            'target_price': target_price,
            'pattern_height': pattern_height,
            'confidence': confidence,
            'status': 'confirmed' if breakout_confirmed else 'forming',
            'type': 'standard' if not is_inverse else 'inverse'
        }
        patterns.append(pattern_data)
    
    return sorted(patterns, key=lambda x: -x['confidence'])

def plot_head_and_shoulders(df, patterns, stock_name=""):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.2, 0.15],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "scatter"}]]
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Price', line=dict(color="#1E88E5", width=2),
        hoverinfo='x+y', hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Moving average (optional)
    if 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA'],
            mode='lines', name="MA (50)", line=dict(color="#FFB300", width=1.5),
            hoverinfo='x+y', hovertemplate='Date: %{x}<br>MA: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    pattern_colors = ['#FF5252', '#4CAF50', '#673AB7', '#FBC02D', '#0288D1']

    for i, pattern in enumerate(patterns):
        try:
            # Extract key points
            LS = pattern["left_shoulder"]
            H = pattern["head"]
            RS = pattern["right_shoulder"]
            T1, T2 = pattern["neckline_points"]
            breakout_point = pattern.get("breakout_point")
            throwback_low = pattern.get("throwback_low")
            pullback_high = pattern.get("pullback_high")
            neckline_end = pattern["neckline_end"]

            # Convert to indices
            ls_idx = df.index.get_loc(LS)
            h_idx = df.index.get_loc(H)
            rs_idx = df.index.get_loc(RS)
            t1_idx = df.index.get_loc(T1)
            t2_idx = df.index.get_loc(T2)
            breakout_idx = df.index.get_loc(breakout_point) if breakout_point else None
            throwback_low_idx = df.index.get_loc(throwback_low) if throwback_low else None
            pullback_high_idx = df.index.get_loc(pullback_high) if pullback_high else None
            neckline_end_idx = df.index.get_loc(neckline_end)

            # Validate sequence
            if not (ls_idx < t1_idx < h_idx < t2_idx < rs_idx):
                print(f"Pattern {i}: Invalid sequence - LS: {ls_idx}, T1: {t1_idx}, H: {h_idx}, T2: {t2_idx}, RS: {rs_idx}")
                continue

            # Get prices
            ls_price = df.loc[LS, 'Close']
            h_price = df.loc[H, 'Close']
            rs_price = df.loc[RS, 'Close']
            t1_price = df.loc[T1, 'Close']
            t2_price = df.loc[T2, 'Close']
            neckline_end_price = pattern["neckline_price"]
            target_price = pattern["target_price"]
            pattern_color = pattern_colors[i % len(pattern_colors)]

            # Calculate neckline slope
            neckline_slope = (t2_price - t1_price) / (t2_idx - t1_idx) if (t2_idx - t1_idx) != 0 else 0

            # Extend neckline leftward
            left_start_idx = max(0, ls_idx - int((rs_idx - ls_idx) * 0.2))  # 20% before LS
            left_neckline_price = t1_price + neckline_slope * (left_start_idx - t1_idx)

            # Find intersection point after RS where price touches neckline
            intersection_idx = rs_idx
            for j in range(rs_idx, min(rs_idx + 30, len(df))):  # Look ahead 30 days max
                neckline_at_j = t1_price + neckline_slope * (j - t1_idx)
                price_at_j = df.iloc[j]['Close']
                if (not pattern["type"] == "inverse" and price_at_j <= neckline_at_j * 1.02) or \
                   (pattern["type"] == "inverse" and price_at_j >= neckline_at_j * 0.98):
                    intersection_idx = j
                    break
            intersection_price = df.iloc[intersection_idx]['Close']

            # ========== PATTERN MARKERS ==========
            # Left shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[LS, 'Date']], y=[ls_price],
                mode="markers+text", text=["LS"], textposition="top center",
                marker=dict(size=12, color=pattern_color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Left Shoulder {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Left Shoulder: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Head
            fig.add_trace(go.Scatter(
                x=[df.loc[H, 'Date']], y=[h_price],
                mode="markers+text", text=["H"], textposition="top center",
                marker=dict(size=14, color=pattern_color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Head {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Head: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Right shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[RS, 'Date']], y=[rs_price],
                mode="markers+text", text=["RS"], textposition="top center",
                marker=dict(size=12, color=pattern_color, symbol="circle", line=dict(width=2, color='white')),
                name=f"Right Shoulder {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Right Shoulder: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Neckline points
            fig.add_trace(go.Scatter(
                x=[df.loc[T1, 'Date'], df.loc[T2, 'Date']],
                y=[t1_price, t2_price],
                mode="markers", marker=dict(size=8, color=pattern_color, symbol="diamond"),
                name=f"Neckline Points {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='x+y', hovertemplate='Neckline Point: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Breakout point
            if breakout_idx:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[breakout_idx]['Date']], y=[df.iloc[breakout_idx]['Close']],
                    mode="markers+text", text=["Break"], textposition="bottom center",
                    marker=dict(size=10, color="#FF9800", symbol="triangle-down"),
                    name=f"Breakout {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='x+y', hovertemplate='Breakout: %{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Throwback Low
            if throwback_low_idx:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[throwback_low_idx]['Date']], y=[df.iloc[throwback_low_idx]['Close']],
                    mode="markers+text", text=["Throw"], textposition="bottom center",
                    marker=dict(size=10, color="#F44336", symbol="star"),
                    name=f"Throwback Low {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='x+y', hovertemplate='Throwback Low: %{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Pullback High
            if pullback_high_idx:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[pullback_high_idx]['Date']], y=[df.iloc[pullback_high_idx]['Close']],
                    mode="markers+text", text=["Pull"], textposition="top center",
                    marker=dict(size=10, color="#9C27B0", symbol="triangle-up"),
                    name=f"Pullback High {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='x+y', hovertemplate='Pullback High: %{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # ========== NECKLINE ==========
            neckline_x = [df.iloc[left_start_idx]['Date'], df.iloc[neckline_end_idx]['Date']]
            neckline_y = [left_neckline_price, neckline_end_price]
            fig.add_trace(go.Scatter(
                x=neckline_x, y=neckline_y,
                mode="lines", line=dict(color=pattern_color, width=2, dash="dash"),
                name=f"Neckline {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # ========== FULL PATTERN OUTLINE ==========
            # Connect to intersection point after RS
            pattern_x = [
                df.iloc[left_start_idx]['Date'],
                df.loc[LS, 'Date'],
                df.loc[T1, 'Date'],
                df.loc[H, 'Date'],
                df.loc[T2, 'Date'],
                df.loc[RS, 'Date'],
                df.iloc[intersection_idx]['Date']
            ]
            pattern_y = [
                left_neckline_price,
                ls_price,
                t1_price,
                h_price,
                t2_price,
                rs_price,
                intersection_price
            ]
            fig.add_trace(go.Scatter(
                x=pattern_x, y=pattern_y,
                mode="lines", line=dict(color=pattern_color, width=3, dash='solid'),
                opacity=0.6, name=f"Pattern Outline {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # ========== V-FORMATION ==========
            if breakout_idx and throwback_low_idx and pullback_high_idx:
                v_x = [
                    df.iloc[breakout_idx]['Date'],
                    df.iloc[throwback_low_idx]['Date'],
                    df.iloc[pullback_high_idx]['Date']
                ]
                v_y = [
                    df.iloc[breakout_idx]['Close'],
                    df.iloc[throwback_low_idx]['Close'],
                    df.iloc[pullback_high_idx]['Close']
                ]
                fig.add_trace(go.Scatter(
                    x=v_x, y=v_y,
                    mode="lines", line=dict(color="#F44336", width=2, dash="dot"),
                    name=f"V-Formation {i+1}", legendgroup=f"pattern{i+1}",
                    hoverinfo='none'
                ), row=1, col=1)

            # ========== TARGET PROJECTION ==========
            target_start_idx = breakout_idx if breakout_idx else rs_idx
            fig.add_trace(go.Scatter(
                x=[df.iloc[target_start_idx]['Date'], df.iloc[-1]['Date']],
                y=[target_price, target_price],
                mode="lines+text", text=["", f"Target: {target_price:.2f}"],
                textposition="middle right",
                line=dict(color="#E91E63", width=1.5, dash="dot"),
                name=f"Target {i+1}", legendgroup=f"pattern{i+1}",
                hoverinfo='none'
            ), row=1, col=1)

            # Annotations
            fig.add_annotation(
                x=df.loc[H, 'Date'],
                y=h_price + (pattern["pattern_height"] / 2 if not pattern["type"] == "inverse" else -pattern["pattern_height"] / 2),
                text=f"H: {pattern['pattern_height']:.2f}",
                showarrow=True, arrowhead=1, ax=0, ay=-30 if not pattern["type"] == "inverse" else 30,
                font=dict(size=10, color=pattern_color)
            )

        except Exception as e:
            print(f"Error plotting pattern {i}: {str(e)}")
            continue

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume", 
            marker=dict(color=np.where(df['Close'] >= df['Open'], '#26A69A', '#EF5350'), opacity=0.7),
            hoverinfo='x+y', hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    # RSI chart
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['RSI'],
                mode='lines', name="RSI", line=dict(color="#7B1FA2", width=1.5),
                hoverinfo='x+y', hovertemplate='Date: %{x}<br>RSI: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        fig.add_hline(y=70, row=3, col=1, line=dict(color="red", width=1, dash="dash"))
        fig.add_hline(y=30, row=3, col=1, line=dict(color="green", width=1, dash="dash"))

    # Final layout
    fig.update_layout(
        title={
            'text': f"Head and Shoulders Detection for {stock_name}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=900, template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v", groupclick="toggleitem",
            yanchor="top", y=1, xanchor="right", x=1.15,
            font=dict(size=10), bgcolor='rgba(255,255,255,0.8)', bordercolor='#CCCCCC', borderwidth=1
        ),
        margin=dict(l=50, r=150, t=100, b=50),
        plot_bgcolor='rgba(245,245,245,1)',
        paper_bgcolor='rgba(255,255,255,1)'
    )

    fig.update_yaxes(title_text="Price", gridcolor="lightgray", row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="lightgray", row=2, col=1)
    if 'RSI' in df.columns:
        fig.update_yaxes(title_text="RSI (14)", range=[0, 100], gridcolor="lightgray", row=3, col=1)

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

def plot_double_bottom(df, pattern_points, stock_name=""):
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
            'text': f"Double Bottom Detection for {stock_name}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800, template="plotly_white",
        legend=dict(
            orientation="v", groupclick="toggleitem",
            yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=10), tracegroupgap=5
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Price", tickprefix="$", gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor='lightgray', row=2, col=1)

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
        
        if right_peak_idx - left_peak_idx < cup_min_bars:
            continue
            
        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]
        
        if abs(right_peak_price - left_peak_price) / left_peak_price > 0.05:
            continue
            
        cup_height = ((left_peak_price + right_peak_price) / 2) - cup_bottom_price
        
        handle_troughs = [t for t in troughs if t > right_peak_idx]
        if not handle_troughs:
            continue
            
        handle_bottom_idx = handle_troughs[0]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]
        
        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            continue
            
        if handle_bottom_idx + 1 >= len(df):
            continue
            
        post_handle_data = df.iloc[handle_bottom_idx:]
        breakout_indices = post_handle_data[post_handle_data['Close'] > right_peak_price].index
        
        if len(breakout_indices) == 0:
            breakout_idx = None
        else:
            breakout_idx = breakout_indices[0]
        
        target_price = right_peak_price + cup_height
        
        patterns.append({
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target': target_price,
            'cup_height': cup_height
        })
    
    return patterns

def plot_cup_and_handle(df, pattern_points, stock_name=""):
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
            'text': f"Cup and Handle Detection for {stock_name}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800, template="plotly_white",
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="right", x=1.4,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=150, t=100, b=40),
        hovermode="x unified"
    )

    return fig

def plot_pattern(df, pattern_points, pattern_name, stock_name=""):
    if pattern_name == "Head and Shoulders":
        return plot_head_and_shoulders(df, pattern_points, stock_name)
    elif pattern_name == "Double Bottom":
        return plot_double_bottom(df, pattern_points, stock_name)
    elif pattern_name == "Cup and Handle":
        return plot_cup_and_handle(df, pattern_points, stock_name)
    else:
        st.error(f"Unsupported pattern type: {pattern_name}")
        return None

def evaluate_pattern_detection(df, patterns, look_forward_window=10):
    """
    Evaluate the performance of detected patterns and calculate metrics per pattern type.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        patterns (dict): Dictionary of pattern types and their detected instances.
        look_forward_window (int): Number of bars to look forward for evaluation.
    
    Returns:
        dict: Metrics (Accuracy, Precision, Recall, F1) for each pattern type.
    """
    metrics = {}
    
    for pattern_type, pattern_list in patterns.items():
        TP = 0  # True Positives: Correctly predicted direction
        FP = 0  # False Positives: Incorrectly predicted direction
        FN = 0  # False Negatives: Missed patterns (approximated)
        TN = 0  # True Negatives: Correctly identified no pattern (approximated)

        if not pattern_list:
            # No patterns detected for this type
            metrics[pattern_type] = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "Total Patterns": 0,
                "Correct Predictions": 0
            }
            continue

        total_patterns = len(pattern_list)

        for pattern in pattern_list:
            # Determine the last point of the pattern
            if pattern_type == "Head and Shoulders":
                last_point_idx = max(int(pattern['left_shoulder']), int(pattern['head']), int(pattern['right_shoulder']))
            elif pattern_type == "Double Bottom":
                last_point_idx = max(int(pattern['trough1']), int(pattern['trough2']))
            elif pattern_type == "Cup and Handle":
                last_point_idx = int(pattern['handle_bottom'])
            else:
                continue

            # Check if enough data exists after the pattern
            if last_point_idx + look_forward_window >= len(df):
                FN += 1  # Not enough data to evaluate, treat as missed
                continue

            last_price = df['Close'].iloc[last_point_idx]
            future_price = df['Close'].iloc[last_point_idx + look_forward_window]

            # Evaluate based on pattern type
            if pattern_type == "Head and Shoulders":  # Bearish
                if future_price < last_price:
                    TP += 1  # Correct bearish prediction
                else:
                    FP += 1  # Incorrect bearish prediction
            elif pattern_type in ["Double Bottom", "Cup and Handle"]:  # Bullish
                if future_price > last_price:
                    TP += 1  # Correct bullish prediction
                else:
                    FP += 1  # Incorrect bullish prediction

        # Approximate FN and TN for simplicity (assuming non-pattern periods are TN)
        # For a more robust approach, you'd need labeled ground truth data
        total_periods = len(df) - look_forward_window
        non_pattern_periods = total_periods - total_patterns
        FN = max(0, total_patterns - (TP + FP))  # Remaining undetected patterns
        TN = max(0, non_pattern_periods - FP)  # Rough estimate of correct negatives

        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[pattern_type] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Total Patterns": total_patterns,
            "Correct Predictions": TP
        }

    return metrics

def create_stock_dashboard(selected_data):
    
    # Create pattern summary
    st.write("**Pattern Detection Summary**")
    
    pattern_cols = st.columns(3)
    patterns = ["Head and Shoulders", "Double Bottom", "Cup and Handle"]

    
    for i, pattern in enumerate(patterns):
        with pattern_cols[i]:
            has_pattern = len(selected_data["Patterns"][pattern]) > 0
            st.write(f"{pattern}: {'✅' if has_pattern else '❌'}")
            
    # Create columns for metrics
    st.markdown('**Key Metrics**')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"${selected_data['Current Price']:.2f}")
    
    with col5:
        percent_change = selected_data["Percent Change"]
        delta_color = "normal"  # Use 'normal' for default behavior
        st.metric("Change", f"{percent_change:.2f}%", delta_color=delta_color)
    
    with col4:
        rsi_value = selected_data["Data"]["RSI"].iloc[-1] if "RSI" in selected_data["Data"].columns else 0
        # RSI doesn't directly support custom colors in Streamlit metrics
        st.metric("RSI (50)", f"{rsi_value:.2f}")
    
    with col2:
        ma_value_50 = selected_data["Data"]["MA"].iloc[-1] if "MA" in selected_data["Data"].columns else 0
        st.metric("MA (50)", f"{ma_value_50:.2f}")
        
    with col3:
        ma_value_200 = selected_data["Data"]["MA2"].iloc[-1] if "MA2" in selected_data["Data"].columns else 0
        st.metric("MA (200)", f"{ma_value_200:.2f}")

def main():
    # Initialize session state
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "stock_data" not in st.session_state:
        st.session_state.stock_data = None
    if "selected_pattern" not in st.session_state:
        st.session_state.selected_pattern = None

    # Header
    st.markdown('<div class="main-header">📈 Advanced Stock Pattern Scanner (Static)</div>', unsafe_allow_html=True)

    # Sidebar setup
    st.sidebar.markdown('<div style="text-align: center; font-weight: bold; font-size: 1.5rem; margin-bottom: 1rem;">Scanner Settings</div>', unsafe_allow_html=True)

    # Fetch Excel files
    st.sidebar.markdown("### 📁 Data Source")
    folder_path = "excel files"  # Define the folder name
    try:
        excel_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
    except FileNotFoundError:
        st.error(f"The folder '{folder_path}' was not found. Please create it and add Excel files.")
        st.stop()
    except PermissionError:
        st.error(f"Permission denied to access the folder '{folder_path}'.")
        st.stop()

    excel_files_display = [os.path.splitext(f)[0] for f in excel_files]

    if not excel_files:
        st.error(f"No Excel files found in the '{folder_path}' folder. Please add Excel files.")
        st.stop()

    selected_index = st.sidebar.selectbox("Select Excel File", range(len(excel_files_display)), 
                                         format_func=lambda x: excel_files_display[x], 
                                         key="file_select")
    selected_file = os.path.join(folder_path, excel_files[selected_index])  # Update to include folder path

    # Load data if file changes
    if selected_file != st.session_state.selected_file:
        st.session_state.selected_file = selected_file
        with st.spinner("Loading data..."):
            st.session_state.df = read_stock_data_from_excel(selected_file)  # Pass full path
        st.session_state.stock_data = None
        st.session_state.selected_pattern = None
    if st.session_state.df is not None:
        min_date = st.session_state.df['TIMESTAMP'].min()
        max_date = st.session_state.df['TIMESTAMP'].max()
        
        st.sidebar.markdown(f"### 📅 Date Range")
        st.sidebar.markdown(f"File contains data from **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}**")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="start_date")
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="end_date")
        
        if end_date < start_date:
            st.sidebar.error("End Date must be after Start Date.")
            st.stop()

        st.sidebar.markdown("### 🔍 Scan Stocks")
        scan_button = st.sidebar.button("🔍 Scan Stocks", use_container_width=True, key="scan_button")

        if scan_button:
            stock_data = []
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = st.empty()
            
            stock_symbols = st.session_state.df['SYMBOL'].unique()

            for i, symbol in enumerate(stock_symbols):
                try:
                    status_text.text(f"Processing {symbol} ({i+1}/{len(stock_symbols)})")
                    df_filtered = fetch_stock_data(symbol, start_date, end_date, st.session_state.df)
                    if df_filtered is None or df_filtered.empty:
                        continue

                    patterns = {
                        "Head and Shoulders": detect_head_and_shoulders(df_filtered),
                        "Double Bottom": detect_double_bottom(df_filtered),
                        "Cup and Handle": detect_cup_and_handle(df_filtered)
                    }

                    # Get per-pattern metrics
                    pattern_metrics = evaluate_pattern_detection(df_filtered, patterns)

                    if any(len(p) > 0 for p in patterns.values()):
                        stock_data.append({
                            "Symbol": symbol,
                            "Patterns": patterns,
                            "Pattern Metrics": pattern_metrics,  # Store metrics per pattern
                            "Data": df_filtered,
                            "Current Price": df_filtered['Close'].iloc[-1],
                            "Volume": df_filtered['Volume'].iloc[-1],
                            "Percent Change": ((df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[0]) / df_filtered['Close'].iloc[0]) * 100,
                            "MA": df_filtered['MA'].iloc[-1] if 'MA' in df_filtered.columns else None,
                            "RSI": df_filtered['RSI'].iloc[-1] if 'RSI' in df_filtered.columns else None,
                        })

                except Exception as e:
                    st.error(f"Error processing {symbol}: {str(e)}")
                    continue
                
                progress_bar.progress((i + 1) / len(stock_symbols))

            st.session_state.stock_data = stock_data
            file_display_name = os.path.splitext(selected_file)[0]
            if stock_data:
                st.success(f"✅ Scan completed for **{file_display_name}** successfully!")
            else:
                st.warning(f"⚠️ No patterns found in **{file_display_name}** for the selected criteria.")
            st.session_state.selected_pattern = None

        if st.session_state.stock_data:
            selected_data = st.session_state.stock_data[0]  # Assuming one stock for simplicity

            st.markdown(f"### Analyzing Stock: {selected_data['Symbol']}")
            create_stock_dashboard(selected_data)

            pattern_options = [p for p, v in selected_data["Patterns"].items() if v]
            if pattern_options:
                st.markdown("### Pattern Visualization")
                
                pattern_container = st.empty()
                with pattern_container:
                    selected_pattern = st.selectbox(
                        "Select Pattern to Visualize",
                        options=pattern_options,
                        key=f"pattern_select_{selected_data['Symbol']}"
                    )

                if selected_pattern != st.session_state.selected_pattern:
                    st.session_state.selected_pattern = selected_pattern
                    st.session_state.chart_container = st.empty()

                if st.session_state.selected_pattern:
                    pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                    stock_name = selected_data["Symbol"]

                    if "chart_container" not in st.session_state:
                        st.session_state.chart_container = st.empty()

                    with st.session_state.chart_container:
                        if st.session_state.selected_pattern == "Head and Shoulders":
                            fig = plot_head_and_shoulders(selected_data["Data"], pattern_points, stock_name=stock_name)
                        else:
                            fig = plot_pattern(selected_data["Data"], pattern_points, st.session_state.selected_pattern, stock_name=stock_name)
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.selected_pattern}_{selected_data['Symbol']}")

                    # Display metrics below the chart
                    st.markdown(f"### Metrics for {selected_pattern} Pattern")
                    metrics = selected_data["Pattern Metrics"][selected_pattern]
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
                    with metric_cols[1]:
                        st.metric("Precision", f"{metrics['Precision']:.2f}")
                    with metric_cols[2]:
                        st.metric("Recall", f"{metrics['Recall']:.2f}")
                    with metric_cols[3]:
                        st.metric("F1 Score", f"{metrics['F1']:.2f}")

            else:
                st.info("No patterns detected for this stock and date range.")

            # Overall accuracy metrics (optional, kept for reference)
            st.markdown("### Overall Pattern Detection Accuracy")
            acc_cols = st.columns(3)
            with acc_cols[0]:
                accuracy = sum(m["Accuracy"] * m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) / sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) if sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) > 0 else 0
                st.metric("Accuracy Score", f"{accuracy:.2f}")
            with acc_cols[1]:
                precision = sum(m["Precision"] * m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) / sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) if sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) > 0 else 0
                st.metric("Precision Score", f"{precision:.2f}")
            with acc_cols[2]:
                volume = selected_data.get("Volume", 0)
                st.metric("Trading Volume", f"{volume:,.0f}")

if __name__ == "__main__":
    main()
