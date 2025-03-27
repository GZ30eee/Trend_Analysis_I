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

def detect_head_and_shoulders(data, min_peaks=3, min_troughs=2, sma_window=50, peak_order=5):
    """
    Detect Head and Shoulders patterns with improved logic.
    
    Args:
        data (pd.DataFrame): Stock data with 'Date' and 'Close'.
        min_peaks (int): Minimum number of peaks required.
        min_troughs (int): Minimum number of troughs required.
        sma_window (int): Window for SMA trend analysis.
        peak_order (int): Order for peak/trough detection.
        
    Returns:
        list: Detected H&S patterns with details.
    """
    # Input validation
    if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
        raise ValueError("Input data must be a DataFrame with 'Close' column")
    
    if len(data) < sma_window:
        print("Not enough data for trend analysis")
        return []
    
    # Step 1: Trend analysis using SMA
    data = data.copy()
    sma = data['Close'].rolling(window=sma_window).mean()
    sma_slope = (sma.iloc[-1] - sma.iloc[0]) / len(sma)
    
    # Require significant upward trend
    if sma_slope <= 0:
        print("No prior uptrend detected")
        return []
    
    # Step 2: Detect peaks and troughs
    peaks = argrelextrema(data['Close'].values, np.greater, order=peak_order)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=peak_order)[0]
    
    if len(peaks) < min_peaks or len(troughs) < min_troughs:
        print(f"Not enough peaks ({len(peaks)}) or troughs ({len(troughs)}) for H&S")
        return []
    
    patterns = []
    
    # Step 3: Iterate through potential patterns
    for i in range(len(peaks) - 2):
        ls_idx = peaks[i]  # Left shoulder
        h_idx = peaks[i+1]  # Head
        rs_idx = peaks[i+2]  # Right shoulder
        
        # Validate peak progression
        if not (data['Close'].iloc[ls_idx] < data['Close'].iloc[h_idx] > data['Close'].iloc[rs_idx]):
            continue
            
        # Find troughs between peaks
        ls_troughs = [t for t in troughs if ls_idx < t < h_idx]
        rs_troughs = [t for t in troughs if h_idx < t < rs_idx]
        
        if not ls_troughs or not rs_troughs:
            continue
            
        # Take the highest trough between LS and H, and between H and RS
        t1_idx = ls_troughs[np.argmax(data['Close'].iloc[ls_troughs])]
        t2_idx = rs_troughs[np.argmax(data['Close'].iloc[rs_troughs])]
        
        # Neckline validation
        neckline_slope = (data['Close'].iloc[t2_idx] - data['Close'].iloc[t1_idx]) / (t2_idx - t1_idx)
        
        # Right shoulder should be similar height to left shoulder
        rs_price = data['Close'].iloc[rs_idx]
        ls_price = data['Close'].iloc[ls_idx]
        if abs(rs_price - ls_price) > 0.1 * ls_price:  # Within 10% of LS price
            continue
            
        # Find breakout point
        breakout_idx = None
        for j in range(rs_idx, min(rs_idx + 20, len(data))):
            neckline_price = data['Close'].iloc[t1_idx] + neckline_slope * (j - t1_idx)
            if data['Close'].iloc[j] < neckline_price * 0.98:  # 2% below neckline
                breakout_idx = j
                break
                
        # Calculate pattern metrics
        pattern_height = data['Close'].iloc[h_idx] - max(data['Close'].iloc[t1_idx], data['Close'].iloc[t2_idx])
        target_price = (data['Close'].iloc[t1_idx] + neckline_slope * (rs_idx - t1_idx)) - pattern_height
        
        patterns.append({
            'left_shoulder': ls_idx,
            'head': h_idx,
            'right_shoulder': rs_idx,
            'neckline_trough1': t1_idx,
            'neckline_trough2': t2_idx,
            'breakout': breakout_idx,
            'neckline_slope': neckline_slope,
            'target_price': target_price,
            'pattern_height': pattern_height,
            'confidence': min(1.0, pattern_height / data['Close'].iloc[h_idx])  # Simple confidence metric
        })
    
    print(f"Total H&S patterns detected: {len(patterns)}")
    return patterns

def plot_head_and_shoulders(df, patterns, stock_name=""):
    """
    Enhanced plotting function for Head and Shoulders patterns.
    
    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        patterns (list): List of detected H&S patterns.
        stock_name (str): Name of the stock for title.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    if not patterns:
        print("No patterns to plot")
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'], 
        mode='lines', 
        name='Price', 
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    # Color cycle for multiple patterns
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, pattern in enumerate(patterns):
        color = colors[i % len(colors)]
        
        # Get key points
        ls_idx = pattern['left_shoulder']
        h_idx = pattern['head']
        rs_idx = pattern['right_shoulder']
        t1_idx = pattern['neckline_trough1']
        t2_idx = pattern['neckline_trough2']
        breakout_idx = pattern['breakout']
        
        # Neckline calculation
        neckline_x = [df['Date'].iloc[t1_idx], df['Date'].iloc[t2_idx]]
        neckline_y = [df['Close'].iloc[t1_idx], df['Close'].iloc[t2_idx]]
        
        # Extend neckline for plotting
        extended_neckline_x = [
            df['Date'].iloc[max(0, t1_idx - 5)],
            df['Date'].iloc[min(len(df)-1, t2_idx + 5)]
        ]
        extended_neckline_y = [
            df['Close'].iloc[t1_idx] - 5 * pattern['neckline_slope'],
            df['Close'].iloc[t2_idx] + 5 * pattern['neckline_slope']
        ]
        
        # Plot neckline
        fig.add_trace(go.Scatter(
            x=extended_neckline_x,
            y=extended_neckline_y,
            mode='lines',
            line=dict(color=color, dash='dash', width=1.5),
            name=f'Neckline {i+1}'
        ), row=1, col=1)
        
        # Plot key points
        for point, label in zip([ls_idx, h_idx, rs_idx], ['LS', 'H', 'RS']):
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[point]],
                y=[df['Close'].iloc[point]],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=label,
                textposition='top center',
                showlegend=False
            ), row=1, col=1)
        
        # Plot breakout if exists
        if breakout_idx:
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[breakout_idx]],
                y=[df['Close'].iloc[breakout_idx]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='x'),
                name=f'Breakout {i+1}',
            ), row=1, col=1)
            
            # Plot target line
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[breakout_idx], df['Date'].iloc[-1]],
                y=[pattern['target_price'], pattern['target_price']],
                mode='lines',
                line=dict(color=color, dash='dot', width=1),
                name=f'Target {i+1}'
            ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker=dict(color='#7f7f7f', opacity=0.6)
    ), row=2, col=1)
    
    # Layout
    # Layout
    fig.update_layout(
        title=f'Head & Shoulders Patterns: {stock_name}',
        height=800,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1
        )
    )
    
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    return fig

def detect_double_bottom(df, order=5, tolerance=0.05, min_pattern_length=10, max_patterns=3):
    """
    Detect Double Bottom patterns with a prior downtrend, similar bottoms, and breakout confirmation.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', and 'Volume'.
        order (int): Window size for peak/trough detection.
        tolerance (float): Max % difference between bottom prices (increased to 5%).
        min_pattern_length (int): Min days between bottoms.
        max_patterns (int): Max number of patterns to return.

    Returns:
        list: Detected Double Bottom patterns.
    """
    # Detect troughs and peaks
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    patterns = []

    if len(troughs) < 2 or len(peaks) < 1:
        print(f"Not enough troughs ({len(troughs)}) or peaks ({len(peaks)}) for Double Bottom")
        return patterns

    print(f"Troughs: {len(troughs)}, Peaks: {len(peaks)}, Data length: {len(df)}")

    for i in range(len(troughs) - 1):
        # Step 1: Identify First Bottom
        trough1_idx = troughs[i]
        price1 = df['Close'].iloc[trough1_idx]

        # Validate prior downtrend (simple slope check over 20 days)
        downtrend_lookback = 20
        if trough1_idx < downtrend_lookback:
            continue
        prior_data = df['Close'].iloc[trough1_idx - downtrend_lookback:trough1_idx]
        if prior_data.iloc[-1] >= prior_data.iloc[0]:  # Not a downtrend
            print(f"Trough {trough1_idx}: No prior downtrend")
            continue

        # Step 2: Find Temporary High (Neckline) after First Bottom
        between_peaks = [p for p in peaks if trough1_idx < p < troughs[i + 1]]
        if not between_peaks:
            continue
        neckline_idx = max(between_peaks, key=lambda p: df['Close'].iloc[p])
        neckline_price = df['Close'].iloc[neckline_idx]

        # Step 3: Identify Second Bottom
        trough2_idx = troughs[i + 1]
        if trough2_idx - trough1_idx < min_pattern_length:
            print(f"Troughs {trough1_idx}-{trough2_idx}: Too close ({trough2_idx - trough1_idx} days)")
            continue

        price2 = df['Close'].iloc[trough2_idx]

        # Check if bottoms are at similar levels (within 5% tolerance)
        if abs(price1 - price2) / min(price1, price2) > tolerance:
            print(f"Troughs {trough1_idx}-{trough2_idx}: Bottoms not similar ({price1:.2f} vs {price2:.2f})")
            continue

        # Step 4: Confirm Breakout
        breakout_idx = None
        for idx in range(trough2_idx, min(len(df), trough2_idx + 30)):  # Look forward 30 days
            if df['Close'].iloc[idx] > neckline_price:
                breakout_idx = idx
                break

        # Calculate pattern metrics
        pattern_height = neckline_price - min(price1, price2)
        target_price = neckline_price + pattern_height

        # Confidence based on breakout and similarity
        confidence = 0.5  # Base confidence
        if breakout_idx:
            confidence += 0.3  # Boost for breakout
        confidence += (1 - abs(price1 - price2) / min(price1, price2) / tolerance) * 0.2  # Similarity bonus

        # Step 5: Store pattern (no strict confidence filter here)
        patterns.append({
            'trough1': trough1_idx,
            'trough2': trough2_idx,
            'neckline': neckline_idx,
            'neckline_price': neckline_price,
            'breakout': breakout_idx,
            'target': target_price,
            'pattern_height': pattern_height,
            'trough_prices': (price1, price2),
            'confidence': min(0.99, confidence),
            'status': 'confirmed' if breakout_idx else 'forming'
        })
        print(f"Pattern detected: T1 {df['Date'].iloc[trough1_idx]}, T2 {df['Date'].iloc[trough2_idx]}, Neckline {df['Date'].iloc[neckline_idx]}, Confidence: {confidence:.2f}")

    # Sort by confidence and limit to max_patterns
    patterns = sorted(patterns, key=lambda x: -x['confidence'])[:max_patterns]
    print(f"Total Double Bottom patterns detected: {len(patterns)}")
    return patterns

def plot_double_bottom(df, pattern_points, stock_name=""):
    """
    Plot Double Bottom patterns with clear W formation, neckline, breakout, and target.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        pattern_points (list): List of detected Double Bottom patterns.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Price", line=dict(color='#1E88E5')),
                  row=1, col=1)

    for idx, pattern in enumerate(pattern_points):
        color = '#E91E63'  # Consistent color for simplicity

        # Key points
        trough1_idx = pattern['trough1']
        trough2_idx = pattern['trough2']
        neckline_idx = pattern['neckline']
        trough1_date = df['Date'].iloc[trough1_idx]
        trough2_date = df['Date'].iloc[trough2_idx]
        neckline_date = df['Date'].iloc[neckline_idx]
        neckline_price = pattern['neckline_price']
        breakout_date = df['Date'].iloc[pattern['breakout']] if pattern['breakout'] else None

        # Find the peak between the two troughs
        start_idx = trough1_idx
        end_idx = trough2_idx
        segment = df.iloc[start_idx:end_idx+1]
        peak_idx = segment['Close'].idxmax()
        peak_date = df['Date'].iloc[peak_idx]
        peak_price = df['Close'].iloc[peak_idx]

        # Plot the W formation (connecting lines)
        fig.add_trace(go.Scatter(
            x=[trough1_date, peak_date, trough2_date],
            y=[pattern['trough_prices'][0], peak_price, pattern['trough_prices'][1]],
            mode='lines',
            line=dict(color=color, width=2),
            name=f'W Formation ({idx+1})',
            showlegend=False
        ), row=1, col=1)

        # Plot troughs
        fig.add_trace(go.Scatter(x=[trough1_date], y=[pattern['trough_prices'][0]], mode="markers+text",
                                text=["B1"], textposition="bottom center", marker=dict(size=12, color=color),
                                name=f"Bottom 1 ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[trough2_date], y=[pattern['trough_prices'][1]], mode="markers+text",
                                text=["B2"], textposition="bottom center", marker=dict(size=12, color=color),
                                name=f"Bottom 2 ({idx+1})"), row=1, col=1)

        # Plot neckline (extended)
        neckline_start_date = df['Date'].iloc[max(0, trough1_idx - 5)]  # Extend left
        neckline_end_date = df['Date'].iloc[min(len(df)-1, trough2_idx + 5)]  # Extend right
        fig.add_trace(go.Scatter(
            x=[neckline_start_date, neckline_end_date], 
            y=[neckline_price, neckline_price], 
            mode="lines",
            line=dict(color=color, dash='dash'), 
            name=f"Neckline ({idx+1})"
        ), row=1, col=1)

        # Plot breakout and target
        if breakout_date:
            fig.add_trace(go.Scatter(x=[breakout_date], y=[df['Close'].iloc[pattern['breakout']]],
                                    mode="markers+text", text=["Breakout"], textposition="top center",
                                    marker=dict(size=12, color=color), name=f"Breakout ({idx+1})"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[breakout_date, df['Date'].iloc[-1]], y=[pattern['target']] * 2,
                                    mode="lines", line=dict(color=color, dash='dot'), name=f"Target ({idx+1})"),
                          row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker=dict(color='#26A69A', opacity=0.7)),
                  row=2, col=1)

    fig.update_layout(
        title=f"Double Bottom Patterns for {stock_name}", 
        height=800, 
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1
        )
    )
    return fig

def detect_cup_and_handle(df, order=10, cup_min_bars=20, handle_max_retrace=0.5):
    """
    Detect Cup and Handle patterns with relaxed constraints for better detection.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        order (int): Window size for peak/trough detection.
        cup_min_bars (int): Minimum bars for cup duration (reduced to 20).
        handle_max_retrace (float): Max handle retracement (increased to 0.5).

    Returns:
        list: Detected Cup and Handle patterns.
    """
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    patterns = []

    if len(peaks) < 2 or len(troughs) < 1:
        print(f"Not enough peaks ({len(peaks)}) or troughs ({len(troughs)}) for Cup and Handle")
        return patterns

    print(f"Peaks: {len(peaks)}, Troughs: {len(troughs)}, Data length: {len(df)}")

    for i in range(len(peaks) - 1):
        # Step 1: Relaxed Uptrend Precondition (30 days prior to left rim)
        left_peak_idx = peaks[i]
        uptrend_lookback = 30
        if left_peak_idx < uptrend_lookback:
            continue
        prior_data = df['Close'].iloc[left_peak_idx - uptrend_lookback:left_peak_idx]
        if prior_data.iloc[-1] <= prior_data.iloc[0] * 1.05:  # Allow 5% flatness
            print(f"Peak {left_peak_idx}: No prior uptrend")
            continue

        # Step 2: Detect Cup Formation
        cup_troughs = [t for t in troughs if t > left_peak_idx]
        if not cup_troughs:
            continue
        cup_bottom_idx = cup_troughs[0]
        right_peaks = [p for p in peaks if p > cup_bottom_idx]
        if not right_peaks:
            continue
        right_peak_idx = right_peaks[0]

        if right_peak_idx - left_peak_idx < cup_min_bars:
            print(f"Peaks {left_peak_idx}-{right_peak_idx}: Cup too short ({right_peak_idx - left_peak_idx} bars)")
            continue

        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]

        # Validate cup: Rims within 10% and depth 20%-60% of uptrend move
        if abs(right_peak_price - left_peak_price) / left_peak_price > 0.10:
            print(f"Peaks {left_peak_idx}-{right_peak_idx}: Rims not similar ({left_peak_price:.2f} vs {right_peak_price:.2f})")
            continue

        uptrend_move = left_peak_price - prior_data.min()
        cup_height = left_peak_price - cup_bottom_price
        cup_depth_ratio = cup_height / uptrend_move
        if not (0.2 <= cup_depth_ratio <= 0.6):
            print(f"Peaks {left_peak_idx}-{right_peak_idx}: Cup depth invalid ({cup_depth_ratio:.2%})")
            continue

        # Step 3: Detect Handle Formation
        handle_troughs = [t for t in troughs if t > right_peak_idx]
        if not handle_troughs:
            continue
        handle_bottom_idx = handle_troughs[0]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]

        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            print(f"Handle {right_peak_idx}-{handle_bottom_idx}: Retrace too deep ({handle_retrace:.2%})")
            continue

        # Find handle end
        handle_end_idx = None
        for j in range(handle_bottom_idx + 1, len(df)):
            if df['Close'].iloc[j] >= right_peak_price * 0.98:
                handle_end_idx = j
                break
        if not handle_end_idx:
            continue

        # Step 4: Confirm Breakout
        breakout_idx = None
        for j in range(handle_end_idx, len(df)):
            if df['Close'].iloc[j] > right_peak_price * 1.02:
                breakout_idx = j
                break

        # Step 5: Calculate Metrics
        target_price = right_peak_price + cup_height
        confidence = 0.6
        if breakout_idx:
            confidence += 0.3
        confidence += (1 - abs(left_peak_price - right_peak_price) / left_peak_price / 0.10) * 0.1

        patterns.append({
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'handle_end': handle_end_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target': target_price,
            'cup_height': cup_height,
            'confidence': min(0.99, confidence),
            'status': 'confirmed' if breakout_idx else 'forming'
        })
        print(f"Pattern detected: Left {df['Date'].iloc[left_peak_idx]}, Bottom {df['Date'].iloc[cup_bottom_idx]}, Right {df['Date'].iloc[right_peak_idx]}, Handle {df['Date'].iloc[handle_bottom_idx]}, Confidence: {confidence:.2f}")

    print(f"Total Cup and Handle patterns detected: {len(patterns)}")
    return patterns

def plot_cup_and_handle(df, pattern_points, stock_name=""):
    """
    Plot Cup and Handle patterns with a curved cup line, distinct handle color, resistance, and breakout.

    Args:
        df (pd.DataFrame): Stock data with 'Date', 'Close', 'Volume'.
        pattern_points (list): List of detected Cup and Handle patterns.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Price", line=dict(color='#1E88E5')),
                  row=1, col=1)

    for idx, pattern in enumerate(pattern_points):
        cup_color = '#9C27B0'  # Purple for the cup
        handle_color = '#FF9800'  # Orange for the handle

        # Extract points
        left_peak_date = df['Date'].iloc[pattern['left_peak']]
        cup_bottom_date = df['Date'].iloc[pattern['cup_bottom']]
        right_peak_date = df['Date'].iloc[pattern['right_peak']]
        handle_bottom_date = df['Date'].iloc[pattern['handle_bottom']]
        handle_end_date = df['Date'].iloc[pattern['handle_end']]
        breakout_date = df['Date'].iloc[pattern['breakout']] if pattern['breakout'] else None

        # Plot key points
        fig.add_trace(go.Scatter(x=[left_peak_date], y=[df['Close'].iloc[pattern['left_peak']]],
                                mode="markers+text", text=["Left Rim"], textposition="top right",
                                marker=dict(size=12, color=cup_color), name=f"Left Rim ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[cup_bottom_date], y=[df['Close'].iloc[pattern['cup_bottom']]],
                                mode="markers+text", text=["Bottom"], textposition="bottom center",
                                marker=dict(size=12, color=cup_color), name=f"Bottom ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[right_peak_date], y=[df['Close'].iloc[pattern['right_peak']]],
                                mode="markers+text", text=["Right Rim"], textposition="top left",
                                marker=dict(size=12, color=cup_color), name=f"Right Rim ({idx+1})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[handle_bottom_date], y=[df['Close'].iloc[pattern['handle_bottom']]],
                                mode="markers+text", text=["Handle Low"], textposition="bottom right",
                                marker=dict(size=12, color=handle_color), name=f"Handle Low ({idx+1})"), row=1, col=1)

        # Create a smooth curve for the cup (left rim -> bottom -> right rim)
        num_points = 50  # Number of points for the curve
        cup_dates = [left_peak_date, cup_bottom_date, right_peak_date]
        cup_prices = [df['Close'].iloc[pattern['left_peak']], df['Close'].iloc[pattern['cup_bottom']],
                      df['Close'].iloc[pattern['right_peak']]]

        # Convert dates to numeric for interpolation
        cup_dates_numeric = [(d - cup_dates[0]).days for d in cup_dates]
        t = np.linspace(0, 1, num_points)
        t_orig = [0, 0.5, 1]  # Normalized positions of left rim, bottom, right rim

        # Quadratic interpolation for smooth curve
        from scipy.interpolate import interp1d
        interp_func = interp1d(t_orig, cup_dates_numeric, kind='quadratic')
        interp_dates_numeric = interp_func(t)
        interp_prices = interp1d(t_orig, cup_prices, kind='quadratic')(t)

        # Convert numeric dates back to datetime
        interp_dates = [cup_dates[0] + pd.Timedelta(days=d) for d in interp_dates_numeric]

        fig.add_trace(go.Scatter(x=interp_dates, y=interp_prices, mode="lines",
                                line=dict(color=cup_color, width=2, dash='dot'), name=f"Cup Curve ({idx+1})"),
                      row=1, col=1)

        # Plot handle with a different color
        handle_x = df['Date'].iloc[pattern['right_peak']:pattern['handle_end'] + 1]
        handle_y = df['Close'].iloc[pattern['right_peak']:pattern['handle_end'] + 1]
        fig.add_trace(go.Scatter(x=handle_x, y=handle_y, mode="lines",
                                line=dict(color=handle_color, width=2), name=f"Handle ({idx+1})"), row=1, col=1)

        # Plot resistance line
        fig.add_trace(go.Scatter(x=[left_peak_date, df['Date'].iloc[-1]], y=[pattern['resistance']] * 2,
                                mode="lines", line=dict(color=cup_color, dash='dash'), name=f"Resistance ({idx+1})"),
                      row=1, col=1)

        # Plot breakout and target
        if breakout_date:
            fig.add_trace(go.Scatter(x=[breakout_date], y=[df['Close'].iloc[pattern['breakout']]],
                                    mode="markers+text", text=["Breakout"], textposition="top center",
                                    marker=dict(size=12, color=handle_color), name=f"Breakout ({idx+1})"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[breakout_date, df['Date'].iloc[-1]], y=[pattern['target']] * 2,
                                    mode="lines", line=dict(color=handle_color, dash='dot'), name=f"Target ({idx+1})"),
                          row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker=dict(color='#26A69A', opacity=0.7)),
                  row=2, col=1)

    fig.update_layout(
        title=f"Cup and Handle Patterns for {stock_name}",
        height=800,
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.1)
    )
    return fig

import streamlit as st
def plot_pattern(df, pattern_points, pattern_name, stock_name=""):
    if pattern_name == "Head and Shoulders":
        return plot_head_and_shoulders(df, pattern_points, stock_name)
    elif pattern_name == "Double Bottom":
        return plot_double_bottom(df, pattern_points, stock_name)
    elif pattern_name == "Cup and Handle":
        return plot_cup_and_handle(df, pattern_points, stock_name)
    else:
        st.error(f"Unsupported pattern type: {pattern_name}")
        return go.Figure()
    
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
    st.markdown('<div class="main-header">📈 Advanced Stock Pattern Scanner (Hitorical Data)</div>', unsafe_allow_html=True)

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
