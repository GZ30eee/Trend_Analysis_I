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

def fetch_stock_data(symbol, start_date, end_date, df, forecast_days=30):
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
        df_filtered = forecast_future_prices(df_filtered, forecast_days)
        
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

import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression  # For trend validation

# BASIC
# def detect_head_and_shoulders(df):
#     prices = df['Close']
#     peaks = argrelextrema(prices.values, np.greater, order=10)[0]
#     patterns = []

#     for i in range(len(peaks) - 2):
#         LS, H, RS = peaks[i], peaks[i + 1], peaks[i + 2]

#         if prices.iloc[H] > prices.iloc[LS] and prices.iloc[H] > prices.iloc[RS]:
#             shoulder_diff = abs(prices.iloc[LS] - prices.iloc[RS]) / max(prices.iloc[LS], prices.iloc[RS])
#             if shoulder_diff <= 0.05:  
#                 T1 = prices.iloc[LS:H + 1].idxmin()  
#                 T2 = prices.iloc[H:RS + 1].idxmin() 
#                 patterns.append({
#                     "left_shoulder": LS,
#                     "head": H,
#                     "right_shoulder": RS,
#                     "neckline": (T1, T2)
#                 })

#     return patterns

# =============== SIR1
# def detect_head_and_shoulders(data, tolerance=0.03, min_pattern_length=20):
#     peaks = [i for i in range(1, len(data)-1) 
#               if data['Close'][i] > data['Close'][i-1] 
#               and data['Close'][i] > data['Close'][i+1]]
    
#     valleys = [i for i in range(1, len(data)-1) 
#                if data['Close'][i] < data['Close'][i-1] 
#                and data['Close'][i] < data['Close'][i+1]]
    
#     patterns = []
    
#     for i in range(len(peaks)-2):
#         LS, H, RS = peaks[i], peaks[i+1], peaks[i+2]
        
#         # 1. Price structure validation
#         if not (data['Close'][LS] < data['Close'][H] > data['Close'][RS]):
#             continue
            
#         # 2. Shoulder symmetry (price)
#         if abs(data['Close'][LS] - data['Close'][RS]) / max(data['Close'][LS], data['Close'][RS]) > tolerance:
#             continue
            
#         # 3. Time symmetry
#         if abs((H-LS)-(RS-H)) / max(H-LS, RS-H) > 0.3:
#             continue
            
#         # 4. Neckline points
#         valley1 = min([v for v in valleys if LS < v < H], key=lambda x: data['Close'][x], default=None)
#         valley2 = min([v for v in valleys if H < v < RS], key=lambda x: data['Close'][x], default=None)
#         if not valley1 or not valley2:
#             continue
            
#         # 5. Neckline slope
#         slope = (data['Close'][valley2] - data['Close'][valley1]) / (valley2 - valley1)
#         if abs(slope) > 0.001:
#             continue
            
#         # 6. Volume confirmation
#         if not (data['Volume'][H] > 1.2*data['Volume'][LS] and data['Volume'][RS] < data['Volume'][H]):
#             continue
            
#         # 7. Prior uptrend
#         lookback = (RS-LS)//2
#         X = np.arange(max(0,LS-lookback), LS).reshape(-1,1)
#         y = data['Close'][max(0,LS-lookback):LS]
#         if LinearRegression().fit(X,y).coef_[0] <= 0:
#             continue
            
#         # 8. Breakout confirmation
#         neckline = data['Close'][valley1] + slope*(RS-valley1)
#         breakout = next((j+2 for j in range(RS, min(RS+20, len(data)-2)) 
#                     if all(data['Close'][j+k] < neckline for k in range(3))), None)
#         if not breakout:
#             continue
            
#         patterns.append({
#             'left_shoulder': LS,
#             'head': H,
#             'right_shoulder': RS,
#             'neckline': (valley1, valley2),
#             'breakout': breakout,
#             'target': neckline - (data['Close'][H] - neckline),
#             'confidence': min(0.99,  # Placeholder for ML confidence score
#                             (1 - abs(data['Close'][LS]-data['Close'][RS])/data['Close'][LS]) * 
#                             (1 - abs((H-LS)-(RS-H))/max(H-LS, RS-H)))
#         })
    
#     return sorted(patterns, key=lambda x: -x['confidence'])[:3]  # Return top 3 most confident patterns

# ================SIR2
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

def detect_head_and_shoulders(data, tolerance=0.03, min_pattern_length=20, volume_ratio=1.2):
    """
    Enhanced Head & Shoulders detection with:
    - Volume analysis
    - Trend confirmation
    - Neckline validation
    - Breakout confirmation
    """
    peaks = find_peaks(data)
    valleys = find_valleys(data)
    patterns = []
    
    for i in range(len(peaks) - 2):
        LS, H, RS = peaks[i], peaks[i+1], peaks[i+2]
        
        # 1. Basic structure validation
        if not (data['Close'].iloc[LS] < data['Close'].iloc[H] > data['Close'].iloc[RS]):
            continue
            
        # 2. Shoulder symmetry (price)
        shoulder_diff = abs(data['Close'].iloc[LS] - data['Close'].iloc[RS]) / max(data['Close'].iloc[LS], data['Close'].iloc[RS])
        if shoulder_diff > tolerance:
            continue
            
        # 3. Time symmetry
        time_diff = abs((H - LS) - (RS - H)) / max(H - LS, RS - H)
        if time_diff > 0.3:  # Allow 30% time difference
            continue
            
        # 4. Minimum pattern duration
        if (RS - LS) < min_pattern_length:
            continue
            
        # 5. Neckline points
        valley1 = min([v for v in valleys if LS < v < H], key=lambda x: data['Close'].iloc[x], default=None)
        valley2 = min([v for v in valleys if H < v < RS], key=lambda x: data['Close'].iloc[x], default=None)
        if not valley1 or not valley2:
            continue
            
        # 6. Neckline slope validation
        neckline_slope = (data['Close'].iloc[valley2] - data['Close'].iloc[valley1]) / (valley2 - valley1)
        if abs(neckline_slope) > 0.001:  # Filter steep necklines
            continue
            
        # 7. Volume analysis
        # Left shoulder advance volume
        left_advance_vol = data['Volume'].iloc[valley1+1:H+1].mean()
        # Right shoulder advance volume
        right_advance_vol = data['Volume'].iloc[valley2+1:RS+1].mean()
        # Head advance volume
        head_advance_vol = data['Volume'].iloc[valley1+1:H+1].mean()
        
        if not (head_advance_vol > left_advance_vol * volume_ratio and 
                right_advance_vol < head_advance_vol):
            continue
            
        # 8. Prior uptrend validation
        lookback = (RS - LS) // 2
        X = np.arange(max(0, LS-lookback), LS).reshape(-1, 1)
        y = data['Close'].iloc[max(0, LS-lookback):LS]
        if LinearRegression().fit(X, y).coef_[0] <= 0:
            continue
            
        # 9. Breakout confirmation
        neckline_at_break = data['Close'].iloc[valley1] + neckline_slope * (RS - valley1)
        breakout_confirmed = False
        breakout_idx = None
        
        for j in range(RS, min(RS + 20, len(data) - 2)):  # Check next 20 candles
            if all(data['Close'].iloc[j+k] < neckline_at_break for k in range(3)):  # 3 consecutive closes
                breakout_confirmed = True
                breakout_idx = j + 2
                break
                
        if not breakout_confirmed:
            continue
            
        # 10. Breakout volume check
        if data['Volume'].iloc[breakout_idx] < data['Volume'].iloc[RS] * 0.8:  # Should have decent volume
            continue
            
        # Calculate pattern metrics
        pattern_height = data['Close'].iloc[H] - neckline_at_break
        target_price = neckline_at_break - pattern_height
        
        patterns.append({
            'left_shoulder': data.index[LS],
            'head': data.index[H],
            'right_shoulder': data.index[RS],
            'neckline_points': (data.index[valley1], data.index[valley2]),
            'neckline_price': neckline_at_break,
            'breakout_point': data.index[breakout_idx],
            'target_price': target_price,
            'pattern_height': pattern_height,
            'confidence': min(0.99, (1 - shoulder_diff) * (1 - time_diff))
        })
    
    return sorted(patterns, key=lambda x: -x['confidence'])  # Return sorted by confidence

# ================ONE
# def detect_head_and_shoulders(df, order=14, shoulder_tol=0.03, min_pattern_length=30, volume_threshold=1.2):
#     if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
#         raise ValueError("DataFrame must contain OHLCV columns")
    
#     prices = df['Close'].values
#     volumes = df['Volume'].values
#     patterns = []
    
#     peaks = argrelextrema(prices, np.greater, order=order)[0]
    
#     for i in range(len(peaks) - 2):
#         LS, H, RS = peaks[i], peaks[i+1], peaks[i+2]
        
#         # --- PRICE STRUCTURE VALIDATION ---
#         # 1. Head must be higher than shoulders
#         if not (prices[LS] < prices[H] > prices[RS]):
#             continue
            
#         # 2. Shoulder symmetry (price)
#         shoulder_diff = abs(prices[LS] - prices[RS]) / max(prices[LS], prices[RS])
#         if shoulder_diff > shoulder_tol:
#             continue
            
#         # 3. Time symmetry (shoulders equidistant from head)
#         time_diff = abs((H - LS) - (RS - H)) / max(H - LS, RS - H)
#         if time_diff > 0.3:  # 30% max asymmetry
#             continue
            
#         # 4. Minimum pattern duration
#         if (RS - LS) < min_pattern_length:
#             continue
            
#         # --- NECKLINE VALIDATION ---
#         # Find lowest points between LS-H and H-RS
#         left_trough = LS + np.argmin(prices[LS:H])
#         right_trough = H + np.argmin(prices[H:RS])
        
#         # Neckline slope (must be near-horizontal)
#         neckline_slope = (prices[right_trough] - prices[left_trough]) / (right_trough - left_trough)
#         if abs(neckline_slope) > 0.0005:  # ~0.05% per candle max slope
#             continue
            
#         # --- VOLUME VALIDATION ---
#         # Head volume must be > shoulders, declining into RS
#         ls_vol = volumes[LS]
#         h_vol = volumes[H]
#         rs_vol = volumes[RS]
#         if not (h_vol >= volume_threshold * ls_vol and rs_vol < h_vol):
#             continue
            
#         # --- TREND CONTEXT ---
#         # Prior uptrend (50% of pattern length before LS)
#         lookback = (RS - LS) // 2
#         trend_start = max(0, LS - lookback)
#         X = np.arange(trend_start, LS).reshape(-1, 1)
#         y = prices[trend_start:LS]
#         model = LinearRegression().fit(X, y)
#         if model.coef_[0] <= 0:  # Not an uptrend
#             continue
            
#         # --- BREAKOUT CONFIRMATION ---
#         neckline_at_break = prices[left_trough] + neckline_slope * (RS - left_trough)
#         breakout_window = prices[RS:min(RS + 20, len(prices))]  # 20-candle breakout window
        
#         # Require 3 consecutive closes below neckline
#         confirmed = False
#         for j in range(len(breakout_window) - 2):
#             if all(breakout_window[j:j+3] < neckline_at_break):
#                 confirmed = True
#                 breakout_idx = RS + j + 2
#                 break
                
#         if not confirmed:
#             continue
            
#         # --- PATTERN METRICS ---
#         pattern_height = prices[H] - neckline_at_break
#         target_price = neckline_at_break - pattern_height
        
#         patterns.append({
#             "left_shoulder": LS,
#             "head": H,
#             "right_shoulder": RS,
#             "neckline": (left_trough, right_trough),
#             "neckline_slope": neckline_slope,
#             "breakout": breakout_idx,
#             "target": target_price,
#             "pattern_height": pattern_height,
#             "volume_ratio": (ls_vol, h_vol, rs_vol),
#             "prior_trend_slope": model.coef_[0]
#         })
    
#     return patterns

# ================TWO
# def detect_head_and_shoulders(df, order=10, shoulder_tolerance=0.05, min_pattern_length=20):
#     """
#     Enhanced Head & Shoulders pattern detection with:
#     - Volume validation
#     - Neckline slope analysis
#     - Breakout confirmation
#     - Time symmetry checks
#     - Price target calculation
    
#     Args:
#         df: DataFrame with OHLCV data
#         order: Minimum distance between peaks (in candles)
#         shoulder_tolerance: Max price difference between shoulders
#         min_pattern_length: Minimum pattern duration in candles
        
#     Returns:
#         List of valid H&S patterns with complete metrics
#     """
#     prices = df['Close']
#     volumes = df['Volume']
#     peaks = argrelextrema(prices.values, np.greater, order=order)[0]
#     patterns = []
    
#     for i in range(len(peaks) - 2):
#         ls_idx, head_idx, rs_idx = peaks[i], peaks[i + 1], peaks[i + 2]
        
#         # Basic structure validation
#         if not (prices.iloc[ls_idx] < prices.iloc[head_idx] > prices.iloc[rs_idx]):
#             continue
            
#         # Shoulder symmetry validation
#         shoulder_diff = abs(prices.iloc[ls_idx] - prices.iloc[rs_idx]) / max(prices.iloc[ls_idx], prices.iloc[rs_idx])
#         if shoulder_diff > shoulder_tolerance:
#             continue
            
#         # Time symmetry validation (shoulders should be roughly equidistant from head)
#         ls_to_head = head_idx - ls_idx
#         rs_to_head = rs_idx - head_idx
#         time_diff = abs(ls_to_head - rs_to_head) / max(ls_to_head, rs_to_head)
#         if time_diff > 0.4:  # Allow 40% time difference
#             continue
            
#         # Neckline detection (connects lowest points between LS-H and H-RS)
#         trough1 = prices.iloc[ls_idx:head_idx].idxmin()
#         trough2 = prices.iloc[head_idx:rs_idx].idxmin()
        
#         # Neckline slope validation (should be horizontal or slightly ascending/descending)
#         neckline_slope = (prices.iloc[trough2] - prices.iloc[trough1]) / (trough2 - trough1)
#         if abs(neckline_slope) > 0.002:  # Filter steep necklines
#             continue
            
#         # Volume pattern validation (should decrease into right shoulder)
#         ls_vol = volumes.iloc[ls_idx]
#         head_vol = volumes.iloc[head_idx]
#         rs_vol = volumes.iloc[rs_idx]
#         if not (head_vol > ls_vol and rs_vol < head_vol):
#             continue
            
#         # Breakout confirmation (price closes below neckline)
#         neckline_value = prices.iloc[trough1] + neckline_slope * (rs_idx - trough1)
#         breakout_idx = None
        
#         for j in range(rs_idx, min(len(df), rs_idx + 20)):  # Check next 20 candles
#             if prices.iloc[j] < neckline_value:
#                 breakout_idx = j
#                 break
                
#         if not breakout_idx:
#             continue
            
#         # Calculate pattern height and target
#         pattern_height = prices.iloc[head_idx] - neckline_value
#         target_price = neckline_value - pattern_height
        
#         patterns.append({
#             "left_shoulder": ls_idx,
#             "head": head_idx,
#             "right_shoulder": rs_idx,
#             "neckline": (trough1, trough2),
#             "neckline_slope": neckline_slope,
#             "breakout": breakout_idx,
#             "target": target_price,
#             "pattern_height": pattern_height,
#             "volume_ratio": (ls_vol, head_vol, rs_vol)
#         })
    
#     return patterns

# ----------------OLD PLOTTING
# def plot_head_and_shoulders(df, patterns):
#     fig = make_subplots(
#         rows=3, cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.03,
#         row_heights=[0.6, 0.2, 0.2],
#         specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "scatter"}]]
#     )

#     # Add price line
#     fig.add_trace(go.Scatter(
#         x=df['Date'], y=df['Close'],
#         mode='lines', name='Price', line=dict(color="#1E88E5", width=2),
#         hoverinfo='x+y',
#         hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
#     ), row=1, col=1)

#     if 'MA' in df.columns:
#         fig.add_trace(go.Scatter(
#             x=df['Date'], y=df['MA'],
#             mode='lines', name="MA (50)", line=dict(color="#FB8C00", width=1.5),
#             hoverinfo='x+y',
#             hovertemplate='Date: %{x}<br>MA: %{y:.2f}<extra></extra>'
#         ), row=1, col=1)

#     for i, pattern in enumerate(patterns):
#         LS, H, RS = pattern["left_shoulder"], pattern["head"], pattern["right_shoulder"]
#         T1, T2 = pattern["neckline"]
        
#         try:
#             # Get all required price points
#             ls_price = df.loc[LS, 'Close']
#             h_price = df.loc[H, 'Close']
#             rs_price = df.loc[RS, 'Close']
#             t1_price = df.loc[T1, 'Close']
#             t2_price = df.loc[T2, 'Close']
            
#             # ========== NEW IMPROVEMENTS ==========
#             # 1. Create a HORIZONTAL neckline (using average of trough prices)
#             avg_neckline = (t1_price + t2_price) / 2
            
#             # 2. Find the START POINT (left of left shoulder)
#             # Look left until we find a point that touches the neckline level
#             start_idx = LS
#             for j in range(LS-1, max(0, LS-50), -1):  # Look back up to 50 candles
#                 if df.loc[j, 'Close'] <= avg_neckline * 1.02:  # 2% tolerance
#                     start_idx = j
#                     break
            
#             # 3. Find the END POINT (right of right shoulder)
#             # Look right until breakout or end of data
#             end_idx = min(RS + 50, len(df)-1)  # Look ahead up to 50 candles
            
#             # ========== VISUAL ELEMENTS ==========
#             # Pattern markers (left shoulder, head, right shoulder)
#             fig.add_trace(go.Scatter(
#                 x=[df.loc[LS, 'Date']], y=[ls_price],
#                 mode="markers+text", text=["LS"], textposition="top center",
#                 marker=dict(size=12, color="#FF5252", symbol="circle"),
#                 name=f"LS {i+1}", legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)
            
#             fig.add_trace(go.Scatter(
#                 x=[df.loc[H, 'Date']], y=[h_price],
#                 mode="markers+text", text=["H"], textposition="top center",
#                 marker=dict(size=14, color="#4CAF50", symbol="circle"),
#                 name=f"H {i+1}", legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)
            
#             fig.add_trace(go.Scatter(
#                 x=[df.loc[RS, 'Date']], y=[rs_price],
#                 mode="markers+text", text=["RS"], textposition="top center",
#                 marker=dict(size=12, color="#FF5252", symbol="circle"),
#                 name=f"RS {i+1}", legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)

#             # Trough markers (neckline points)
#             fig.add_trace(go.Scatter(
#                 x=[df.loc[T1, 'Date']], y=[t1_price],
#                 mode="markers", marker=dict(size=8, color="#673AB7", symbol="diamond"),
#                 name=f"Trough {i+1}", legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)
            
#             fig.add_trace(go.Scatter(
#                 x=[df.loc[T2, 'Date']], y=[t2_price],
#                 mode="markers", marker=dict(size=8, color="#673AB7", symbol="diamond"),
#                 showlegend=False, legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)

#             # ========== NEW: HORIZONTAL NECKLINE ==========
#             neckline_x = [df.loc[start_idx, 'Date'], df.loc[end_idx, 'Date']]
#             neckline_y = [avg_neckline, avg_neckline]
            
#             fig.add_trace(go.Scatter(
#                 x=neckline_x,
#                 y=neckline_y,
#                 mode="lines",
#                 line=dict(color="#673AB7", width=2, dash="dash"),
#                 name=f"Neckline {i+1}", legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)

#             # ========== NEW: CONNECTING LINES ==========
#             # Full pattern outline (connects all key points)
#             pattern_x = [
#                 df.loc[start_idx, 'Date'],  # Start point
#                 df.loc[LS, 'Date'],        # Left shoulder
#                 df.loc[T1, 'Date'],        # Left trough
#                 df.loc[H, 'Date'],         # Head
#                 df.loc[T2, 'Date'],        # Right trough
#                 df.loc[RS, 'Date'],        # Right shoulder
#                 df.loc[end_idx, 'Date']    # End point
#             ]
            
#             pattern_y = [
#                 df.loc[start_idx, 'Close'],  # Start price
#                 ls_price,                   # Left shoulder
#                 t1_price,                  # Left trough
#                 h_price,                   # Head
#                 t2_price,                  # Right trough
#                 rs_price,                  # Right shoulder
#                 df.loc[end_idx, 'Close']    # End price
#             ]
            
#             fig.add_trace(go.Scatter(
#                 x=pattern_x,
#                 y=pattern_y,
#                 mode="lines",
#                 line=dict(color="rgba(156, 39, 176, 0.5)", width=3),
#                 name=f"Pattern {i+1}", legendgroup=f"pattern{i+1}",
#                 hoverinfo='skip'
#             ), row=1, col=1)

#             # ========== TARGET PROJECTION ==========
#             pattern_height = h_price - avg_neckline
#             target_price = avg_neckline - pattern_height
            
#             # Target line (horizontal)
#             fig.add_trace(go.Scatter(
#                 x=[df.loc[RS, 'Date'], df.loc[end_idx, 'Date']],
#                 y=[target_price, target_price],
#                 mode="lines",
#                 line=dict(color="#E91E63", width=1.5, dash="dot"),
#                 name=f"Target {i+1}", legendgroup=f"pattern{i+1}"
#             ), row=1, col=1)
            
#             # Target annotation
#             fig.add_annotation(
#                 x=df.loc[end_idx, 'Date'],
#                 y=target_price,
#                 text=f"Target: {target_price:.2f}",
#                 showarrow=True,
#                 arrowhead=1,
#                 ax=0,
#                 ay=-30,
#                 font=dict(size=10, color="#E91E63")
#             )
            
#             # Measured move annotation
#             fig.add_annotation(
#                 x=df.loc[H, 'Date'],
#                 y=avg_neckline + pattern_height/2,
#                 text=f"Height: {pattern_height:.2f}",
#                 showarrow=True,
#                 arrowhead=1,
#                 ax=0,
#                 ay=0,
#                 font=dict(size=10, color="#4CAF50")
#             )
            
#         except Exception as e:
#             print(f"Error plotting pattern {i}: {str(e)}")
#             continue

#     # Volume chart
#     fig.add_trace(
#         go.Bar(
#             x=df['Date'], 
#             y=df['Volume'], 
#             name="Volume", 
#             marker=dict(
#                 color=np.where(df['Close'] >= df['Open'], '#26A69A', '#EF5350'),
#                 opacity=0.7
#             ),
#             hoverinfo='x+y',
#             hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
#         ),
#         row=2, col=1
#     )

#     # RSI chart (if available)
#     if 'RSI' in df.columns:
#         fig.add_trace(
#             go.Scatter(
#                 x=df['Date'], 
#                 y=df['RSI'], 
#                 mode='lines', 
#                 name="RSI", 
#                 line=dict(color="#7B1FA2", width=1.5),
#                 hoverinfo='x+y',
#                 hovertemplate='Date: %{x}<br>RSI: %{y:.2f}<extra></extra>'
#             ),
#             row=3, col=1
#         )
        
#         # Overbought/oversold lines
#         fig.add_hline(y=70, row=3, col=1, line=dict(color="red", width=1, dash="dash"))
#         fig.add_hline(y=30, row=3, col=1, line=dict(color="green", width=1, dash="dash"))

#     # Layout improvements
#     fig.update_layout(
#         title={
#             'text': f"Head & Shoulders Patterns (Found: {len(patterns)})",
#             'y':0.95,
#             'x':0.5,
#             'font': dict(size=20, color="#0D47A1")
#         },
#         height=900,
#         template="plotly_white",
#         hovermode="x unified",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         margin=dict(l=50, r=50, t=100, b=50)
#     )
    
#     # Axis titles
#     fig.update_yaxes(title_text="Price", row=1, col=1)
#     fig.update_yaxes(title_text="Volume", row=2, col=1)
#     if 'RSI' in df.columns:
#         fig.update_yaxes(title_text="RSI (14)", row=3, col=1)
    
#     return fig

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

# ----------------OLD DB DETETCION FUNC.
# def detect_double_bottom(df, order=20, tolerance=0.05, min_pattern_length=10, max_patterns=3):
   
#     peaks, troughs = find_extrema(df, order=order)
#     patterns = []
    
#     if len(troughs) < 2:
#         return patterns
    
#     for i in range(len(troughs) - 1):
#         trough1_idx = troughs[i]
#         for j in range(i + 1, len(troughs)):
#             trough2_idx = troughs[j]
            
#             if trough1_idx >= trough2_idx:
#                 continue
                
#             if trough2_idx - trough1_idx < min_pattern_length:
#                 continue
                
#             price1 = df['Close'].iloc[trough1_idx]
#             price2 = df['Close'].iloc[trough2_idx]
            
#             price_diff_pct = abs(price1 - price2) / price1
#             if price_diff_pct > tolerance:
#                 continue
            
#             between_slice = df['Close'].iloc[trough1_idx:trough2_idx]
#             neckline_idx = between_slice.idxmax()
#             neckline_price = df['Close'].iloc[neckline_idx]
            
#             min_trough_price = min(price1, price2)
#             if (neckline_price - min_trough_price) < 0.3 * (df['Close'].iloc[:trough1_idx].max() - min_trough_price):
#                 continue
            
#             w_formation_peaks = []
#             for peak_idx in peaks:
#                 if trough1_idx < peak_idx < trough2_idx + int(0.2 * (trough2_idx - trough1_idx)):
#                     w_formation_peaks.append(peak_idx)
            
#             if not w_formation_peaks:
#                 continue
                
#             breakout_idx = None
#             confirmation_idx = None
#             for idx in range(trough2_idx, min(len(df), trough2_idx + 50)): 
#                 if df['Close'].iloc[idx] > neckline_price:
#                     breakout_idx = idx
                    
#                     if idx + 2 < len(df) and df['Close'].iloc[idx+1] > neckline_price and df['Close'].iloc[idx+2] > neckline_price:
#                         confirmation_idx = idx + 2
#                     break
            
#             pattern_height = neckline_price - min_trough_price
#             target_price = neckline_price + pattern_height
            
#             patterns.append({
#                 'trough1': trough1_idx,
#                 'trough2': trough2_idx,
#                 'neckline': neckline_idx,
#                 'neckline_price': neckline_price,
#                 'breakout': breakout_idx,
#                 'confirmation': confirmation_idx,
#                 'target': target_price,
#                 'pattern_height': pattern_height,
#                 'w_formation_peaks': w_formation_peaks
#             })
    
#     filtered_patterns = []
#     last_trough2_idx = -1 
    
#     for pattern in sorted(patterns, key=lambda x: x['trough1']):
#         if pattern['trough1'] > last_trough2_idx:
#             filtered_patterns.append(pattern)
#             last_trough2_idx = pattern['trough2']
        
#         if len(filtered_patterns) >= max_patterns:
#             break
    
#     return filtered_patterns

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
    Plot the detected pattern based on the pattern type.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with 'Close' and 'Date' columns.
        pattern_points (list): List of dictionaries containing pattern details.
        pattern_name (str): Name of the pattern to plot.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    if pattern_name == "Head and Shoulders":
        return plot_head_and_shoulders(df, pattern_points)
    elif pattern_name == "Double Bottom":
        return plot_double_bottom(df, pattern_points)
    elif pattern_name == "Cup and Handle":
        return plot_cup_and_handle(df, pattern_points)
    else:
        st.error(f"Unsupported pattern type: {pattern_name}")
        return None

def evaluate_pattern_detection(df, patterns):
    total_patterns = 0
    correct_predictions = 0
    false_positives = 0
    look_forward_window = 10

    for pattern_type, pattern_list in patterns.items():
        total_patterns += len(pattern_list)
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

            # Check if we have enough data after the pattern
            if last_point_idx + look_forward_window < len(df):
                # Evaluate based on pattern type
                if pattern_type in ["Double Bottom", "Cup and Handle"]:  # Bullish patterns
                    if df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        false_positives += 1
                elif pattern_type in ["Head and Shoulders"]:  # Bearish patterns
                    if df['Close'][last_point_idx + look_forward_window] < df['Close'][last_point_idx]:
                        correct_predictions += 1
                    elif df['Close'][last_point_idx + look_forward_window] > df['Close'][last_point_idx]:
                        false_positives += 1

    # Calculate metrics
    if total_patterns > 0:
        accuracy = correct_predictions / total_patterns
        precision = correct_predictions / (correct_predictions + false_positives) if (correct_predictions + false_positives) > 0 else 0
    else:
        accuracy = 0.0
        precision = 0.0

    return accuracy, precision, correct_predictions, total_patterns

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
    # Header with logo and title
    st.markdown('<div class="main-header">📈 Advanced Stock Pattern Scanner(Static)</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div style="text-align: center; font-weight: bold; font-size: 1.5rem; margin-bottom: 1rem;">Scanner Settings</div>', unsafe_allow_html=True)
    
    # Dropdown to select Excel file
    st.sidebar.markdown("### 📁 Data Source")
    excel_files = sorted([f for f in os.listdir() if f.endswith('.xlsx')])  # Sort alphabetically
    if not excel_files:
        st.error("No Excel files found in the directory. Please add Excel files.")
        st.stop()

    selected_file = st.sidebar.selectbox("Select Excel File", excel_files)

    if selected_file != st.session_state.selected_file:
        st.session_state.selected_file = selected_file
        with st.spinner("Loading data..."):
            st.session_state.df = read_stock_data_from_excel(selected_file)

    if st.session_state.df is not None:
        # Get the date range from the selected file
        min_date = st.session_state.df['TIMESTAMP'].min()
        max_date = st.session_state.df['TIMESTAMP'].max()
        
        st.sidebar.markdown(f"### 📅 Date Range")
        st.sidebar.markdown(f"File contains data from **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}**")

        # Date range selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        if end_date < start_date:
            st.sidebar.error("End Date must be after Start Date.")
            st.stop()

        # Forecast days input
        forecast_days = st.sidebar.number_input(
            "Forecast Days",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days to forecast future prices"
        )

        # Scan button with enhanced styling
        st.sidebar.markdown("### 🔍 Scan Stocks")
        scan_button = st.sidebar.button("🔍 Scan Stocks", use_container_width=True)

        if scan_button:
            stock_data = []
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = st.empty()
            
            # Get unique symbols from the DataFrame
            stock_symbols = st.session_state.df['SYMBOL'].unique()
            
            for i, symbol in enumerate(stock_symbols):
                try:
                    status_text.text(f"Processing {symbol} ({i+1}/{len(stock_symbols)})")
                    
                    df_filtered = fetch_stock_data(symbol, start_date, end_date, st.session_state.df, forecast_days)
                    if df_filtered is None or df_filtered.empty:
                        print(f"No data for {symbol} within the date range.")
                        continue
                    
                    print(f"Processing {symbol} with {len(df_filtered)} rows.")
                    
                    patterns = {}
                    patterns["Head and Shoulders"] = detect_head_and_shoulders(df_filtered)
                    patterns["Double Bottom"] = detect_double_bottom(df_filtered)
                    patterns["Cup and Handle"] = detect_cup_and_handle(df_filtered)
                    
                    print(f"Patterns detected: {patterns}")
                    
                    accuracy, precision, correct_predictions, total_patterns = evaluate_pattern_detection(df_filtered, patterns)
                    
                    # Only add stocks with detected patterns
                    has_patterns = any(len(p) > 0 for p in patterns.values())
                    if has_patterns:
                        stock_data.append({
                            "Symbol": symbol, 
                            "Patterns": patterns, 
                            "Data": df_filtered,
                            "Current Price": df_filtered['Close'].iloc[-1],
                            "Volume": df_filtered['Volume'].iloc[-1],
                            "Percent Change": ((df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[0]) / df_filtered['Close'].iloc[0]) * 100,
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Correct Predictions": correct_predictions,
                            "Total Patterns": total_patterns,
                            "MA": df_filtered['MA'].iloc[-1] if 'MA' in df_filtered.columns else None,
                            "RSI": df_filtered['RSI'].iloc[-1] if 'RSI' in df_filtered.columns else None,
                        })
                    
                except Exception as e:
                    st.error(f"Error processing {symbol}: {str(e)}")
                    continue
                
                progress_bar.progress((i + 1) / len(stock_symbols))
                progress_container.empty()
                status_text.empty()

            st.session_state.stock_data = stock_data
            st.session_state.selected_stock = None
            st.session_state.selected_pattern = None
            
            if len(stock_data) > 0:
                st.success(f"✅ Scan completed! Found patterns in {len(stock_data)} stocks.")
            else:
                st.warning("No patterns found in any stocks for the selected criteria.")

        # Display results if stock data exists
        if st.session_state.stock_data:
            # Get selected stock data (assuming only one stock is processed at a time in the Excel file)
            selected_data = st.session_state.stock_data[0]  # Directly fetch the first stock data
            
            # Display stock symbol as plain text
            print(f"Analyzing Stock: {selected_data['Symbol']}")
            
            # Create dashboard for the selected stock
            create_stock_dashboard(selected_data)  # Ensure this function outputs content properly

            # Display pattern selection and graph if patterns are available
            pattern_options = [p for p, v in selected_data["Patterns"].items() if v]
            if pattern_options:
                print("Pattern Visualization")
                
                selected_pattern = st.selectbox(
                    "Select Pattern to Visualize",
                    options=pattern_options,
                    key='pattern_select'
                )
                
                if selected_pattern != st.session_state.selected_pattern:
                    st.session_state.selected_pattern = selected_pattern
                
                if st.session_state.selected_pattern:
                    pattern_points = selected_data["Patterns"][st.session_state.selected_pattern]
                    
                    # Display appropriate chart based on pattern type
                    if st.session_state.selected_pattern == "Head and Shoulders":
                        fig = plot_head_and_shoulders(
                            selected_data["Data"],
                            pattern_points
                        )
                    elif st.session_state.selected_pattern == "Double Bottom":
                        fig = plot_pattern(
                            selected_data["Data"],
                            selected_data["Patterns"][st.session_state.selected_pattern],
                            st.session_state.selected_pattern
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{st.session_state.selected_pattern}")
                    else:
                        fig = plot_pattern(
                            selected_data["Data"],
                            pattern_points,
                            st.session_state.selected_pattern
                        )
                        
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
            else:
                print("No patterns detected for this stock and date range.")
                
            # Create accuracy metrics
            st.write("**Pattern Detection Accuracy**")
            
            acc_cols = st.columns(3)
            with acc_cols[0]:
                accuracy = selected_data.get("Accuracy", 0)
                st.metric("Accuracy Score", f"{accuracy:.2f}")
            
            with acc_cols[1]:
                precision = selected_data.get("Precision", 0)
                st.metric("Precision Score", f"{precision:.2f}")
            
            with acc_cols[2]:
                volume = selected_data.get("Volume", 0)
                st.metric("Trading Volume", f"{volume:,.0f}")

if __name__ == "__main__":
    main()
