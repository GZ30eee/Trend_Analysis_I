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

def detect_double_bottom(df, order=7, tolerance=0.1, min_pattern_length=20, 
                        max_patterns=10, min_retracement=0.3, max_retracement=0.7,
                        min_volume_ratio=0.6, lookback_period=90, debug=False):
    """Enhanced Double Bottom detection"""
    # Input validation
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        if debug: print("Invalid DataFrame")
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    # Data quality checks
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    if data['Close'].isna().any():
        if debug: print("NaN values in Close prices")
        return []
    
    # Volume indicators
    if 'Volume' in data.columns:
        data['Volume_5MA'] = data['Volume'].rolling(5).mean()
        data['Volume_20MA'] = data['Volume'].rolling(20).mean()
    
    # Find extrema
    troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    
    if len(troughs) < 2:
        if debug: print(f"Insufficient troughs ({len(troughs)})")
        return []

    patterns = []
    used_indices = set()

    # First pass: Find potential bottoms
    potential_bottoms = []
    for i in range(len(troughs) - 1):
        t1_idx, t2_idx = troughs[i], troughs[i+1]
        
        # Pattern structure validation
        if (t2_idx - t1_idx) < min_pattern_length:
            continue
            
        price1 = data['Close'].iloc[t1_idx]
        price2 = data['Close'].iloc[t2_idx]
        price_diff = abs(price1 - price2) / min(price1, price2)
        
        if price_diff > tolerance:
            continue
            
        potential_bottoms.append((t1_idx, t2_idx))

    # Second pass: Validate patterns
    for t1_idx, t2_idx in potential_bottoms:
        if t1_idx in used_indices or t2_idx in used_indices:
            continue
            
        # Neckline identification
        between_points = data.iloc[t1_idx:t2_idx+1]
        neckline_idx = between_points['Close'].idxmax()
        
        if neckline_idx in [t1_idx, t2_idx] or neckline_idx in used_indices:
            continue
            
        neckline_price = data['Close'].iloc[neckline_idx]
        price1 = data['Close'].iloc[t1_idx]
        price2 = data['Close'].iloc[t2_idx]
        
        # Retracement validation
        move_up = neckline_price - price1
        move_down = neckline_price - price2
        if move_up <= 0:
            continue
            
        retracement = move_down / move_up
        if not (min_retracement <= retracement <= max_retracement):
            continue
            
        # Volume validation
        vol_ok = True
        if 'Volume' in data.columns:
            vol1 = data['Volume_5MA'].iloc[t1_idx]
            vol2 = data['Volume_5MA'].iloc[t2_idx]
            if vol2 < vol1 * min_volume_ratio:
                vol_ok = False
        
        # Breakout detection
        breakout_idx = None
        breakout_strength = 0
        breakout_volume_confirmation = False
        
        for j in range(t2_idx, min(len(data), t2_idx + lookback_period)):
            if j in used_indices:
                break
                
            current_price = data['Close'].iloc[j]
            if current_price > neckline_price * 1.02:  # 2% breakout threshold
                breakout_strength += (current_price - neckline_price) / neckline_price
                if breakout_idx is None:
                    breakout_idx = j
                if 'Volume' in data.columns and data['Volume'].iloc[j] > data['Volume_20MA'].iloc[j]:
                    breakout_volume_confirmation = True
                if j > breakout_idx + 2 and breakout_strength > 0.05:
                    break
            elif breakout_idx and current_price < neckline_price:
                breakout_idx = None
                breakout_strength = 0
        
        if not breakout_idx:
            continue
        
        # Confidence calculation
        confidence = 0.5
        confidence += 0.2 * min(1.0, breakout_strength / 0.1)
        confidence += (1 - price_diff / tolerance) * 0.2
        confidence += 0.1 if vol_ok else 0
        confidence += 0.1 if breakout_volume_confirmation else 0
        
        # Final validation
        pattern_indices = set(range(t1_idx, t2_idx + 1)) | {neckline_idx, breakout_idx}
        if not (pattern_indices & used_indices):
            patterns.append({
                'trough1_idx': t1_idx,
                'trough2_idx': t2_idx,
                'neckline_idx': neckline_idx,
                'neckline_price': neckline_price,
                'breakout_idx': breakout_idx,
                'breakout_strength': breakout_strength,
                'target_price': neckline_price + (neckline_price - min(price1, price2)),
                'pattern_height': neckline_price - min(price1, price2),
                'trough_prices': (float(price1), float(price2)),
                'confidence': min(0.95, confidence),
                'status': 'confirmed',
                'retracement': retracement,
                'volume_ratio': vol2 / vol1 if 'Volume' in data.columns else 1.0,
                'breakout_volume_confirmed': breakout_volume_confirmation,
                'time_span': (data['Date'].iloc[t2_idx] - data['Date'].iloc[t1_idx]).days
            })
            used_indices.update(pattern_indices)
    
    return sorted(patterns, key=lambda x: -x['confidence'])[:max_patterns]

def plot_double_bottom(df, pattern_points, stock_name=""):
    """Plot confirmed Double Bottom patterns with scrollable pattern info."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name="Price",
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    valid_patterns = []
    used_indices = set()
    
    # Filter for confirmed patterns and ensure no overlap
    for pattern in sorted(pattern_points, key=lambda x: x.get('confidence', 0), reverse=True):
        if not all(k in pattern for k in ['trough1_idx', 'neckline_idx', 'trough_prices', 'neckline_price', 'breakout_idx']):
            continue  # Skip if missing required keys or no breakout (unconfirmed)
            
        t1_idx = pattern['trough1_idx']
        nl_idx = pattern['neckline_idx']
        breakout_idx = pattern['breakout_idx']
        t2_idx = pattern['trough2_idx']
        
        # Define the range of indices this pattern occupies
        pattern_indices = set(range(t1_idx, breakout_idx + 1)) | {nl_idx}
        if pattern_indices & used_indices:
            continue  # Skip if overlaps with any previously plotted pattern
        
        # Validate second bottom (B2) after neckline
        search_end = min(len(df) - 1, breakout_idx)
        b2_segment = df.iloc[nl_idx:search_end + 1]
        b2_idx = b2_segment['Close'].idxmin()
        b2_price = df['Close'].iloc[b2_idx]
        
        if b2_idx <= nl_idx or b2_price >= pattern['neckline_price']:
            continue
        
        # Add B2 to used indices
        pattern_indices.add(b2_idx)
        
        # Find prior peak (PP) before first bottom
        pp_segment = df.iloc[max(0, t1_idx - 30):t1_idx + 1]
        pp_idx = pp_segment['Close'].idxmax()
        pp_price = df['Close'].iloc[pp_idx]

        # Calculate risk/reward since we have a breakout
        risk = pattern['neckline_price'] - min(pattern['trough_prices'][0], b2_price)
        reward = (pattern['neckline_price'] + risk) - pattern['neckline_price']
        rr_ratio = reward / risk if risk > 0 else 0

        valid_patterns.append({
            **pattern,
            'pp_idx': pp_idx,
            'pp_price': pp_price,
            'b2_idx': b2_idx,
            'b2_price': b2_price,
            'rr_ratio': rr_ratio
        })
        used_indices.update(pattern_indices)

    pattern_stats = []
    
    for idx, pattern in enumerate(valid_patterns):
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        
        pp_idx = pattern['pp_idx']
        pp_price = pattern['pp_price']
        t1_idx = pattern['trough1_idx']
        t1_price = pattern['trough_prices'][0]
        nl_idx = pattern['neckline_idx']
        nl_price = pattern['neckline_price']
        b2_idx = pattern['b2_idx']
        b2_price = pattern['b2_price']
        breakout_idx = pattern['breakout_idx']
        breakout_strength = pattern['breakout_strength']
        rr_ratio = pattern['rr_ratio']
        
        w_x = [df['Date'].iloc[pp_idx], 
               df['Date'].iloc[t1_idx],
               df['Date'].iloc[nl_idx],
               df['Date'].iloc[b2_idx],
               df['Date'].iloc[breakout_idx]]
        w_y = [pp_price, t1_price, nl_price, b2_price, df['Close'].iloc[breakout_idx]]
        
        fig.add_trace(
            go.Scatter(
                x=w_x,
                y=w_y,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=10, color=color, line=dict(width=1, color='white')),
                name=f'Pattern {idx+1}',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        fig.add_annotation(
            x=df['Date'].iloc[breakout_idx],
            y=df['Close'].iloc[breakout_idx],
            text=f"BO<br>{breakout_strength:.1%}",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=10, color=color)
        )
        
        point_labels = [
            ('PP', pp_idx, pp_price, 'top center'),
            ('B1', t1_idx, t1_price, 'bottom center'),
            ('MP', nl_idx, nl_price, 'top center'),
            ('B2', b2_idx, b2_price, 'bottom center'),
            ('BO', breakout_idx, df['Close'].iloc[breakout_idx], 'top right')
        ]
        
        for label, p_idx, p_price, pos in point_labels:
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[p_idx]],
                    y=[p_price],
                    mode="markers+text",
                    text=[label],
                    textposition=pos,
                    marker=dict(
                        size=12 if label in ['PP', 'BO'] else 10,
                        color=color,
                        symbol='diamond' if label in ['PP', 'BO'] else 'circle',
                        line=dict(width=1, color='white')
                    ),
                    textfont=dict(size=10, color='white'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        neckline_x = [df['Date'].iloc[max(0, t1_idx - 10)],
                      df['Date'].iloc[min(len(df) - 1, breakout_idx + 20)]]
        fig.add_trace(
            go.Scatter(
                x=neckline_x,
                y=[nl_price, nl_price],
                mode="lines",
                line=dict(color=color, dash='dash', width=1.5),
                showlegend=False,
                opacity=0.7,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        duration = (df['Date'].iloc[b2_idx] - df['Date'].iloc[t1_idx]).days
        price_diff = abs(t1_price - b2_price) / t1_price if t1_price != 0 else 0
        
        pattern_stats.append([
            f"Pattern {idx+1}",
            f"{t1_price:.2f}",
            f"{b2_price:.2f}",
            f"{nl_price:.2f}",
            f"{price_diff:.2%}",
            f"{pattern['retracement']:.1%}",
            f"{pattern['volume_ratio']:.1f}x",
            f"{duration} days",
            "Yes",
            f"{df['Close'].iloc[breakout_idx]:.2f}",
            f"{pattern['target_price']:.2f}",
            f"{rr_ratio:.1f}",
            f"{pattern['confidence']*100:.1f}%",
            f"{breakout_strength:.1%}"
        ])
    
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume",
            marker=dict(color='#7f7f7f', opacity=0.4),
            hovertemplate='Volume: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )
    
    for pattern in valid_patterns:
        start_idx = max(0, pattern['trough1_idx'] - 10)
        end_idx = min(len(df) - 1, pattern['breakout_idx'] + 10)
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[start_idx], df['Date'].iloc[end_idx]],
                y=[df['Volume'].max() * 1.1] * 2,
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor=f'hsla({(valid_patterns.index(pattern) * 60) % 360}, 70%, 50%, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Pattern", "Bottom 1", "Bottom 2", "Neckline", "Price Diff",
                        "Retrace", "Vol Ratio", "Duration", "Confirmed", "Breakout",
                        "Target", "R/R", "Confidence", "BO Strength"],
                font=dict(size=10, color='white'),
                fill_color='#1f77b4',
                align='center'
            ),
            cells=dict(
                values=list(zip(*pattern_stats)),
                font=dict(size=9),
                align='center',
                fill_color=['rgba(245,245,245,0.8)', 'white'],
                height=25
            ),
            columnwidth=[0.7] + [0.6] * 13
        ),
        row=3, col=1
    )
    
    # Create scrollable pattern annotations
    num_patterns = len(valid_patterns)
    if num_patterns > 0:
        # Add each pattern annotation in a vertical stack
        for idx, pattern in enumerate(valid_patterns):
            color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
            breakout_idx = pattern['breakout_idx']
            rr_ratio = pattern['rr_ratio']
            
            annotation_text = [
                f"<b><span style='color:{color}'>● Pattern {idx+1}</span></b>",
                f"PP: {pattern['pp_price']:.2f} | B1: {pattern['trough_prices'][0]:.2f}",
                f"MP: {pattern['neckline_price']:.2f} | B2: {pattern['b2_price']:.2f}",
                f"BO: {df['Close'].iloc[breakout_idx]:.2f} (Strength: {pattern['breakout_strength']:.1%})",
                f"Target: {pattern['target_price']:.2f} | R/R: {rr_ratio:.1f}",
                f"Confidence: {pattern['confidence']*100:.1f}%"
            ]
            
            fig.add_annotation(
                x=1.30,
                y=0.98 - (idx * 0.12),
                xref="paper",
                yref="paper",
                text="<br>".join(annotation_text),
                showarrow=False,
                font=dict(size=9),
                align='left',
                bordercolor="#AAAAAA",
                borderwidth=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                yanchor="top"
            )

    fig.update_layout(
        title=dict(
            text=f"<b>Double Bottom Analysis for {stock_name}</b><br>",
            x=0.5,
            font=dict(size=20),
            xanchor='center'
        ),
        height=1200,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(r=250, t=120, b=20, l=50),  # Reduced right margin
        legend=dict(
            title=dict(text='<b>Pattern Legend</b>'),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.05,  # Moved legend closer
            bordercolor="#E1E1E1",
            borderwidth=1,
            font=dict(size=10)
        ),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", side="right"),
        xaxis_rangeslider_visible=False
    )

    return fig
def detect_head_and_shoulders(df, depth=3, min_pattern_separation=10, debug=False):
    try:
        if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
            if debug: print("Invalid DataFrame - missing 'Close' column")
            return []
        
        data = df.copy()
        if 'Date' not in data.columns:
            data['Date'] = data.index
        
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isna().any():
            if debug: print("Non-numeric values found in Close prices")
            return []

        # Lower depth for more extrema
        peaks = argrelextrema(data['Close'].values, np.greater, order=depth)[0]
        troughs = argrelextrema(data['Close'].values, np.less, order=depth)[0]
        
        if len(peaks) < 3:
            if debug: print(f"Not enough peaks ({len(peaks)}) to form H&S pattern")
            return []
        
        patterns = []
        used_indices = set()
        
        for i in range(len(peaks) - 2):
            if peaks[i] in used_indices or peaks[i+1] in used_indices or peaks[i+2] in used_indices:
                continue
            
            ls_idx, h_idx, rs_idx = peaks[i], peaks[i+1], peaks[i+2]
            
            try:
                ls_price = float(data['Close'].iloc[ls_idx])
                h_price = float(data['Close'].iloc[h_idx])
                rs_price = float(data['Close'].iloc[rs_idx])
                
                if not (h_price > ls_price and h_price > rs_price):
                    if debug: print(f"Invalid structure at {data['Date'].iloc[h_idx]}")
                    continue
                
                # Relax shoulder difference to 10%
                shoulder_diff = abs(ls_price - rs_price) / min(ls_price, rs_price)
                if shoulder_diff > 0.10:
                    if debug: print(f"Shoulders not balanced at {data['Date'].iloc[h_idx]}")
                    continue
                
                # Relax time symmetry to 50%
                ls_to_head_duration = h_idx - ls_idx
                rs_to_head_duration = rs_idx - h_idx
                duration_diff = abs(ls_to_head_duration - rs_to_head_duration) / min(ls_to_head_duration, rs_to_head_duration)
                if duration_diff > 0.5:
                    if debug: print(f"Shoulders not symmetric in time at {data['Date'].iloc[h_idx]}")
                    continue
                
                t1_candidates = [t for t in troughs if ls_idx < t < h_idx]
                t2_candidates = [t for t in troughs if h_idx < t < rs_idx]
                
                if not t1_candidates or not t2_candidates:
                    if debug: print(f"No troughs found between peaks at {data['Date'].iloc[h_idx]}")
                    continue
                
                t1_idx = t1_candidates[np.argmin(data['Close'].iloc[t1_candidates])]
                t2_idx = t2_candidates[np.argmin(data['Close'].iloc[t2_candidates])]
                
                # Relax neckline trough difference to 10%
                trough_diff = abs(data['Close'].iloc[t1_idx] - data['Close'].iloc[t2_idx]) / min(data['Close'].iloc[t1_idx], data['Close'].iloc[t2_idx])
                if trough_diff > 0.10:
                    if debug: print(f"Neckline troughs not balanced at {data['Date'].iloc[h_idx]}")
                    continue
                
                neckline_slope = (data['Close'].iloc[t2_idx] - data['Close'].iloc[t1_idx]) / (t2_idx - t1_idx)
                neckline_price_at_rs = data['Close'].iloc[t1_idx] + neckline_slope * (rs_idx - t1_idx)
                
                breakout_idx = None
                min_breakout_distance = 3
                # Extend look-ahead to 60 periods and lower drop threshold to 1%
                for j in range(rs_idx + min_breakout_distance, min(rs_idx + 60, len(data))):
                    neckline_at_j = data['Close'].iloc[t1_idx] + neckline_slope * (j - t1_idx)
                    if data['Close'].iloc[j] < neckline_at_j and (neckline_at_j - data['Close'].iloc[j]) / neckline_at_j > 0.01:
                        breakout_idx = j
                        break
                
                if breakout_idx is not None:
                    if any(abs(breakout_idx - p['breakout_idx']) < min_pattern_separation for p in patterns):
                        if debug: print(f"Breakout too close at {data['Date'].iloc[breakout_idx]}")
                        continue
                
                if breakout_idx is None:
                    if debug: print(f"No breakout below neckline after RS at {data['Date'].iloc[rs_idx]}")
                    continue
                
                valid_pattern = True
                for k in range(i + 3, len(peaks)):
                    next_peak_idx = peaks[k]
                    if next_peak_idx > rs_idx + 30:
                        break
                    if float(data['Close'].iloc[next_peak_idx]) > h_price:
                        valid_pattern = False
                        if debug: print(f"Pattern invalidated by higher peak at {data['Date'].iloc[next_peak_idx]}")
                        break
                
                if not valid_pattern:
                    continue
                
                pattern_height = h_price - max(data['Close'].iloc[t1_idx], data['Close'].iloc[t2_idx])
                target_price = neckline_price_at_rs - pattern_height
                duration_days = (data['Date'].iloc[rs_idx] - data['Date'].iloc[ls_idx]).days
                
                confidence = min(0.99, 0.6 + (0.3 * (1 - shoulder_diff / 0.10)))  # Adjusted base confidence
                
                pattern = {
                    'left_shoulder_idx': ls_idx,
                    'head_idx': h_idx,
                    'right_shoulder_idx': rs_idx,
                    'neckline_trough1_idx': t1_idx,
                    'neckline_trough2_idx': t2_idx,
                    'breakout_idx': breakout_idx,
                    'left_shoulder_price': ls_price,
                    'head_price': h_price,
                    'right_shoulder_price': rs_price,
                    'neckline_slope': neckline_slope,
                    'target_price': target_price,
                    'pattern_height': pattern_height,
                    'duration_days': duration_days,
                    'confidence': confidence
                }
                
                patterns.append(pattern)
                used_indices.update([ls_idx, h_idx, rs_idx])
                
            except Exception as e:
                if debug: print(f"Skipping pattern due to error: {e}")
                continue
        
        final_patterns = []
        sorted_patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)
        for pattern in sorted_patterns:
            if not any(
                (pattern['left_shoulder_idx'] <= existing['right_shoulder_idx'] and 
                 pattern['right_shoulder_idx'] >= existing['left_shoulder_idx'])
                for existing in final_patterns
            ):
                final_patterns.append(pattern)
        
        return final_patterns
        
    except Exception as e:
        if debug: print(f"Error in detect_head_and_shoulders: {e}")
        return []

def plot_head_and_shoulders(df, pattern_points, stock_name=""):
    """Plot Head and Shoulders patterns with detailed statistics and improved visualization"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create figure with 3 rows: price chart, volume, and statistics table
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    # Price line (row 1)
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            mode='lines', 
            name="Price",
            line=dict(color='#1E88E5', width=2),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    pattern_stats = []  # Store statistics for each pattern

    for idx, pattern in enumerate(pattern_points):
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        ls_idx = pattern['left_shoulder_idx']
        h_idx = pattern['head_idx']
        rs_idx = pattern['right_shoulder_idx']
        t1_idx = pattern['neckline_trough1_idx']
        t2_idx = pattern['neckline_trough2_idx']
        breakout_idx = pattern['breakout_idx']

        # Improved Neckline Positioning - Extending the shorter neckline to the left
        # Calculate neckline points starting from before the left shoulder
        neckline_start_price = df['Close'].iloc[t1_idx]
        neckline_end_price = df['Close'].iloc[t1_idx] + pattern['neckline_slope'] * (t2_idx - t1_idx)
        
        # Extend neckline to the left (e.g., 10 periods before the first trough) and to the breakout
        neckline_x = [
            df['Date'].iloc[max(0, t1_idx - 10)],  # Start 10 periods before the first trough
            df['Date'].iloc[breakout_idx]           # End at Breakout point
        ]
        
        # Calculate neckline y-values using the slope
        neckline_y = [
            neckline_start_price + pattern['neckline_slope'] * (max(0, t1_idx - 10) - t1_idx),  # Left extension
            neckline_start_price + pattern['neckline_slope'] * (breakout_idx - t1_idx)           # Breakout point
        ]
        
        # Ensure neckline stays below Right Shoulder price at rs_idx
        rs_price = pattern['right_shoulder_price']
        neckline_at_rs = neckline_start_price + pattern['neckline_slope'] * (rs_idx - t1_idx)
        if neckline_at_rs >= rs_price:
            adjustment_factor = 0.98  # Place neckline at 98% of RS price
            adjusted_neckline_slope = (rs_price * adjustment_factor - neckline_start_price) / (rs_idx - t1_idx)
            neckline_y = [
                neckline_start_price + adjusted_neckline_slope * (max(0, t1_idx - 10) - t1_idx),
                neckline_start_price + adjusted_neckline_slope * (breakout_idx - t1_idx)
            ]

        # Add extended neckline trace
        fig.add_trace(
            go.Scatter(
                x=neckline_x,
                y=neckline_y,
                mode="lines",
                line=dict(color=color, dash='dash', width=2),
                name=f"Neckline {idx+1}"
            ),
            row=1, col=1
        )

        # Calculate detailed pattern metrics
        left_trough_price = min(df['Close'].iloc[t1_idx], df['Close'].iloc[ls_idx])
        right_trough_price = min(df['Close'].iloc[t2_idx], df['Close'].iloc[rs_idx])
        
        left_shoulder_height = pattern['left_shoulder_price'] - left_trough_price
        head_height = pattern['head_price'] - max(df['Close'].iloc[t1_idx], df['Close'].iloc[t2_idx])
        right_shoulder_height = pattern['right_shoulder_price'] - right_trough_price
        
        left_duration = (df['Date'].iloc[h_idx] - df['Date'].iloc[ls_idx]).days
        right_duration = (df['Date'].iloc[rs_idx] - df['Date'].iloc[h_idx]).days
        total_duration = (df['Date'].iloc[rs_idx] - df['Date'].iloc[ls_idx]).days
        
        neckline_at_breakout = df['Close'].iloc[t1_idx] + pattern['neckline_slope'] * (breakout_idx - t1_idx)
        neckline_break = neckline_at_breakout - df['Close'].iloc[breakout_idx]
        break_percentage = (neckline_break / neckline_at_breakout) * 100
        
        # Calculate symmetry metrics
        price_symmetry = abs(pattern['left_shoulder_price'] - pattern['right_shoulder_price']) / min(pattern['left_shoulder_price'], pattern['right_shoulder_price'])
        time_symmetry = abs(left_duration - right_duration) / min(left_duration, right_duration)
        
        # Store statistics for table
        pattern_stats.append([
            f"Pattern {idx+1}",
            f"{pattern['left_shoulder_price']:.2f}",
            f"{left_shoulder_height:.2f}",
            f"{pattern['head_price']:.2f}",
            f"{head_height:.2f}",
            f"{pattern['right_shoulder_price']:.2f}",
            f"{right_shoulder_height:.2f}",
            f"{left_duration}/{right_duration} days",
            f"{price_symmetry*100:.1f}%/{time_symmetry*100:.1f}%",
            f"{neckline_break:.2f} ({break_percentage:.1f}%)",
            f"{pattern['target_price']:.2f}",
            f"{pattern['confidence']*100:.1f}%"
        ])

        # Plot Left Shoulder
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[ls_idx]], 
                y=[pattern['left_shoulder_price']],
                mode="markers+text",
                text=["LS"],
                textposition="bottom center",
                marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey'))
            ),
            row=1, col=1
        )

        # Plot Head
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[h_idx]], 
                y=[pattern['head_price']],
                mode="markers+text",
                text=["H"],
                textposition="top center",
                marker=dict(size=14, symbol="diamond", color=color, line=dict(width=2, color='DarkSlateGrey'))
            ),
            row=1, col=1
        )

        # Plot Right Shoulder
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[rs_idx]], 
                y=[pattern['right_shoulder_price']],
                mode="markers+text",
                text=["RS"],
                textposition="bottom center",
                marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey'))
            ),
            row=1, col=1
        )

        # Plot Breakout
        if breakout_idx:
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[breakout_idx]],
                    y=[df['Close'].iloc[breakout_idx]],
                    mode="markers+text",
                    text=["Breakout"],
                    textposition="top right",
                    marker=dict(size=10, symbol="x", color=color, line=dict(width=2))
                ),
                row=1, col=1
            )
            
            # Target line
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[breakout_idx], df['Date'].iloc[-1]],
                    y=[pattern['target_price']] * 2,
                    mode="lines",
                    line=dict(color=color, dash='dot', width=1.5)
                ),
                row=1, col=1
            )

    # Volume (row 2)
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'],
            name="Volume",
            marker=dict(color='#26A69A', opacity=0.7),
            hovertemplate='Volume: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )

    # Statistics table (row 3)
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Pattern", "LS Price", "LS Height", "Head Price", "Head Height", 
                    "RS Price", "RS Height", "Duration", "Symmetry (P/T)", 
                    "Neckline Break", "Target", "Confidence"
                ],
                font=dict(size=10, color='white'),
                fill_color='#1E88E5',
                align=['left', 'center', 'center', 'center', 'center', 
                      'center', 'center', 'center', 'center', 'center', 'center'],
                height=30
            ),
            cells=dict(
                values=list(zip(*pattern_stats)),
                font=dict(size=10),
                align=['left', 'center', 'center', 'center', 'center', 
                      'center', 'center', 'center', 'center', 'center', 'center'],
                height=25,
                fill_color=['rgba(245,245,245,0.8)', 'white']
            )
        ),
        row=3, col=1
    )

    # Update layout with enhanced legend
    fig.update_layout(
        title=dict(
            text=f"<b>Head & Shoulders Analysis for {stock_name}</b>",
            x=0.5,
            font=dict(size=20),
            xanchor='center'
        ),
        height=1100,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            title=dict(
                text='<b>Pattern Details</b>',
                font=dict(size=12)
            ),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.1,
            bordercolor="#E1E1E1",
            borderwidth=1,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            itemsizing='constant',
            itemwidth=40,
            traceorder='normal',
            itemclick='toggle',
            itemdoubleclick='toggleothers'
        ),
        hovermode='x unified',
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", side="right"),
        xaxis_rangeslider_visible=False,
        margin=dict(r=300, t=100, b=20, l=50)
    )

    # Add custom legend annotations with extra spacing
    for idx, pattern in enumerate(pattern_points):
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        fig.add_annotation(
            x=1.38,
            y=0.98 - (idx * 0.15),
            xref="paper",
            yref="paper",
            text=(
                f"<b><span style='color:{color}'>● Pattern {idx+1}</span></b><br>"
                f"LS: {pattern['left_shoulder_price']:.2f}<br>"
                f"H: {pattern['head_price']:.2f}<br>"
                f"RS: {pattern['right_shoulder_price']:.2f}<br>"
                f"Neckline Slope: {pattern['neckline_slope']:.4f}<br>"
                f"Breakout: {df['Close'].iloc[pattern['breakout_idx']]:.2f}<br>"
                f"Target: {pattern['target_price']:.2f}<br>"
                f"Confidence: {pattern['confidence']*100:.1f}%"
            ),
            showarrow=False,
            font=dict(size=10),
            align='left',
            bordercolor="#AAAAAA",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(255,255,255,0.8)"
        )

    return fig
def detect_cup_and_handle(df, order=10, cup_min_bars=20, handle_max_retrace=0.5, debug=False):
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        if debug: print("Invalid DataFrame")
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    if data['Close'].isna().any():
        if debug: print("NaN values in Close prices")
        return []
    
    peaks = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=order)[0]
    
    if len(peaks) < 2 or len(troughs) < 1:
        if debug: print(f"Insufficient peaks ({len(peaks)}) or troughs ({len(troughs)})")
        return []

    patterns = []
    used_indices = set()  # Track used indices to prevent overlap
    
    for i in range(len(peaks) - 1):
        left_peak_idx = peaks[i]
        
        # Skip if this peak is already part of another pattern
        if left_peak_idx in used_indices:
            continue
            
        # Uptrend validation
        uptrend_lookback = min(30, left_peak_idx)
        prior_data = data['Close'].iloc[left_peak_idx - uptrend_lookback:left_peak_idx]
        if prior_data.iloc[-1] <= prior_data.iloc[0] * 1.05:
            continue
            
        cup_troughs = [t for t in troughs if left_peak_idx < t < (left_peak_idx + 200)]
        if not cup_troughs:
            continue
            
        cup_bottom_idx = min(cup_troughs, key=lambda x: data['Close'].iloc[x])
        if cup_bottom_idx in used_indices:
            continue
            
        cup_bottom_price = data['Close'].iloc[cup_bottom_idx]
        
        right_peaks = [p for p in peaks if cup_bottom_idx < p < (cup_bottom_idx + 100)]
        if not right_peaks:
            continue
            
        right_peak_idx = right_peaks[0]
        if right_peak_idx in used_indices:
            continue
            
        right_peak_price = data['Close'].iloc[right_peak_idx]
        
        if right_peak_idx - left_peak_idx < cup_min_bars:
            continue
            
        if abs(right_peak_price - data['Close'].iloc[left_peak_idx]) / data['Close'].iloc[left_peak_idx] > 0.10:
            continue
            
        handle_troughs = [t for t in troughs if right_peak_idx < t < (right_peak_idx + 50)]
        if not handle_troughs:
            continue
            
        handle_bottom_idx = handle_troughs[0]
        if handle_bottom_idx in used_indices:
            continue
            
        handle_bottom_price = data['Close'].iloc[handle_bottom_idx]
        
        cup_height = data['Close'].iloc[left_peak_idx] - cup_bottom_price
        handle_retrace = (right_peak_price - handle_bottom_price) / cup_height
        if handle_retrace > handle_max_retrace:
            continue
            
        handle_end_idx = None
        for j in range(handle_bottom_idx + 1, min(handle_bottom_idx + 30, len(data))):
            if j in used_indices:
                break
            if data['Close'].iloc[j] >= right_peak_price * 0.98:
                handle_end_idx = j
                break
                
        if not handle_end_idx:
            continue
            
        breakout_idx = None
        for j in range(handle_end_idx, min(handle_end_idx + 30, len(data))):
            if j in used_indices:
                break
            if data['Close'].iloc[j] > right_peak_price * 1.02:
                breakout_idx = j
                break
                
        # Define pattern range
        pattern_start = left_peak_idx
        pattern_end = breakout_idx if breakout_idx else handle_end_idx
        pattern_range = set(range(pattern_start, pattern_end + 1))
        
        # Check for overlap with existing patterns
        if pattern_range & used_indices:
            if debug: print(f"Pattern at {data['Date'].iloc[left_peak_idx]} overlaps with existing pattern")
            continue
        
        confidence = 0.6
        if breakout_idx:
            confidence += 0.3
        confidence += (1 - abs(data['Close'].iloc[left_peak_idx] - right_peak_price) / data['Close'].iloc[left_peak_idx] / 0.10) * 0.1
        
        pattern = {
            'left_peak': left_peak_idx,
            'cup_bottom': cup_bottom_idx,
            'right_peak': right_peak_idx,
            'handle_bottom': handle_bottom_idx,
            'handle_end': handle_end_idx,
            'breakout': breakout_idx,
            'resistance': right_peak_price,
            'target_price': right_peak_price + cup_height,
            'cup_height': cup_height,
            'confidence': min(0.95, confidence),
            'status': 'confirmed' if breakout_idx else 'forming'
        }
        
        patterns.append(pattern)
        used_indices.update(pattern_range)
    
    return sorted(patterns, key=lambda x: -x['confidence'])

def plot_cup_and_handle(df, pattern_points, stock_name=""):
    """
    Enhanced Cup and Handle plot with statistics, pattern indexing, and improved visualization.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    from scipy.interpolate import interp1d
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name="Price",
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    pattern_stats = []
    
    for idx, pattern in enumerate(pattern_points):
        # Assign unique color to each pattern
        color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
        handle_color = f'hsl({(idx * 60 + 30) % 360}, 70%, 50%)'  # Slightly different hue for handle
        
        # Extract points
        left_peak_idx = pattern['left_peak']
        cup_bottom_idx = pattern['cup_bottom']
        right_peak_idx = pattern['right_peak']
        handle_bottom_idx = pattern['handle_bottom']
        handle_end_idx = pattern['handle_end']
        breakout_idx = pattern.get('breakout')
        
        left_peak_date = df['Date'].iloc[left_peak_idx]
        cup_bottom_date = df['Date'].iloc[cup_bottom_idx]
        right_peak_date = df['Date'].iloc[right_peak_idx]
        handle_bottom_date = df['Date'].iloc[handle_bottom_idx]
        handle_end_date = df['Date'].iloc[handle_end_idx]
        breakout_date = df['Date'].iloc[breakout_idx] if breakout_idx else None
        
        left_peak_price = df['Close'].iloc[left_peak_idx]
        cup_bottom_price = df['Close'].iloc[cup_bottom_idx]
        right_peak_price = df['Close'].iloc[right_peak_idx]
        handle_bottom_price = df['Close'].iloc[handle_bottom_idx]
        handle_end_price = df['Close'].iloc[handle_end_idx]
        breakout_price = df['Close'].iloc[breakout_idx] if breakout_idx else None
        
        # Create smooth cup curve
        num_points = 50
        cup_dates = [left_peak_date, cup_bottom_date, right_peak_date]
        cup_prices = [left_peak_price, cup_bottom_price, right_peak_price]
        
        cup_dates_numeric = [(d - cup_dates[0]).days for d in cup_dates]
        t = np.linspace(0, 1, num_points)
        t_orig = [0, 0.5, 1]
        
        interp_func = interp1d(t_orig, cup_dates_numeric, kind='quadratic')
        interp_dates_numeric = interp_func(t)
        interp_prices = interp1d(t_orig, cup_prices, kind='quadratic')(t)
        interp_dates = [cup_dates[0] + pd.Timedelta(days=d) for d in interp_dates_numeric]

        # Plot cup curve with dotted line
        fig.add_trace(
            go.Scatter(
                x=interp_dates, y=interp_prices,
                mode="lines",
                line=dict(color=color, width=3, dash='dot'),  # Added dash='dot'
                name=f'Pattern {idx+1} Cup',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Plot handle
        handle_x = df['Date'].iloc[right_peak_idx:handle_end_idx + 1]
        handle_y = df['Close'].iloc[right_peak_idx:handle_end_idx + 1]
        fig.add_trace(
            go.Scatter(
                x=handle_x, y=handle_y,
                mode="lines",
                line=dict(color=handle_color, width=3),
                name=f'Pattern {idx+1} Handle',
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot resistance line
        resistance_x = [left_peak_date, df['Date'].iloc[-1]]
        fig.add_trace(
            go.Scatter(
                x=resistance_x,
                y=[right_peak_price] * 2,
                mode="lines",
                line=dict(color=color, dash='dash', width=1.5),
                name=f'Resistance {idx+1}',
                opacity=0.7,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Plot breakout and target if exists
        if breakout_idx:
            # Breakout line
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[handle_end_idx], df['Date'].iloc[breakout_idx]],
                    y=[df['Close'].iloc[handle_end_idx], breakout_price],
                    mode="lines",
                    line=dict(color=handle_color, width=3),
                    name=f'Breakout {idx+1}',
                    hovertemplate='Breakout: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Breakout annotation
            fig.add_annotation(
                x=breakout_date,
                y=breakout_price,
                text=f"BO<br>{pattern.get('breakout_strength', 'N/A')}",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(size=10, color=handle_color)
            )

        # Plot key points with labels
        point_labels = [
            ('LR', left_peak_idx, left_peak_price, 'top right'),
            ('CB', cup_bottom_idx, cup_bottom_price, 'bottom center'),
            ('RR', right_peak_idx, right_peak_price, 'top left'),
            ('HL', handle_bottom_idx, handle_bottom_price, 'bottom right'),
            ('HE', handle_end_idx, handle_end_price, 'top center')
        ]
        
        if breakout_idx:
            point_labels.append(('BO', breakout_idx, breakout_price, 'top right'))
        
        for label, p_idx, p_price, pos in point_labels:
            fig.add_trace(
                go.Scatter(
                    x=[df['Date'].iloc[p_idx]],
                    y=[p_price],
                    mode="markers+text",
                    text=[label],
                    textposition=pos,
                    marker=dict(
                        size=12 if label in ['LR','RR','BO'] else 10,
                        color=color if label in ['LR','RR','CB'] else handle_color,
                        symbol='diamond' if label in ['LR','RR','BO'] else 'circle',
                        line=dict(width=1, color='white')
                    ),
                    textfont=dict(size=10, color='white'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # Calculate pattern statistics
        cup_duration = (right_peak_date - left_peak_date).days
        handle_duration = (handle_end_date - right_peak_date).days
        total_duration = cup_duration + handle_duration
        cup_depth_pct = (left_peak_price - cup_bottom_price) / left_peak_price * 100
        handle_retrace_pct = (right_peak_price - handle_bottom_price) / (right_peak_price - cup_bottom_price) * 100
        
        pattern_stats.append([
            f"Pattern {idx+1}",
            f"{left_peak_price:.2f}",
            f"{cup_bottom_price:.2f}",
            f"{right_peak_price:.2f}",
            f"{handle_bottom_price:.2f}",
            f"{cup_depth_pct:.1f}%",
            f"{handle_retrace_pct:.1f}%",
            f"{cup_duration} days",
            f"{handle_duration} days",
            f"{total_duration} days",
            "Yes" if breakout_idx else "No",
            f"{breakout_price:.2f}" if breakout_idx else "Pending",
            f"{pattern.get('target_price', 'N/A'):.2f}",
            f"{pattern.get('rr_ratio', 'N/A')}",
            f"{pattern['confidence']*100:.1f}%"
        ])

        # Add volume highlighting
        start_idx = max(0, left_peak_idx - 10)
        end_idx = min(len(df)-1, (breakout_idx if breakout_idx else handle_end_idx) + 10)
        
        fig.add_trace(
            go.Scatter(
                x=[df['Date'].iloc[start_idx], df['Date'].iloc[end_idx]],
                y=[df['Volume'].max() * 1.1] * 2,
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor=f'hsla({(idx * 60) % 360}, 70%, 50%, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Volume'],
            name="Volume",
            marker=dict(color='#7f7f7f', opacity=0.4),
            hovertemplate='Volume: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )

    # Pattern statistics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Pattern", "Left Rim", "Cup Bottom", "Right Rim", "Handle Low",
                       "Cup Depth", "Handle Retrace", "Cup Days", "Handle Days", "Total Days",
                       "Confirmed", "Breakout", "Target", "R/R", "Confidence"],
                font=dict(size=10, color='white'),
                fill_color='#1f77b4',
                align='center'
            ),
            cells=dict(
                values=list(zip(*pattern_stats)),
                font=dict(size=9),
                align='center',
                fill_color=['rgba(245,245,245,0.8)', 'white'],
                height=25
            ),
            columnwidth=[0.7]+[0.6]*13
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Cup and Handle Analysis for {stock_name}</b><br>",
            x=0.5,
            font=dict(size=20),
            xanchor='center'
        ),
        height=1200,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(r=350, t=120, b=20, l=50),
        legend=dict(
            title=dict(text='<b>Pattern Legend</b>'),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.07,
            bordercolor="#E1E1E1",
            borderwidth=1,
            font=dict(size=10)
        ),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", side="right"),
        xaxis_rangeslider_visible=False
    )

    # Add pattern annotations
    num_patterns = len(pattern_points)
    if num_patterns > 0:
        available_space = 0.9
        spacing = available_space / max(num_patterns, 1)
        start_y = 0.95
        
        for idx, pattern in enumerate(pattern_points):
            color = f'hsl({(idx * 60) % 360}, 70%, 50%)'
            handle_color = f'hsl({(idx * 60 + 30) % 360}, 70%, 50%)'
            breakout_idx = pattern.get('breakout')
            
            annotation_text = [
                f"<b><span style='color:{color}'>● Pattern {idx+1}</span></b>",
                f"Type: {'Confirmed' if breakout_idx else 'Potential'}",
                f"Left Rim: {df['Close'].iloc[pattern['left_peak']]:.2f}",
                f"Cup Bottom: {df['Close'].iloc[pattern['cup_bottom']]:.2f}",
                f"Right Rim: {df['Close'].iloc[pattern['right_peak']]:.2f}",
                f"Handle Low: {df['Close'].iloc[pattern['handle_bottom']]:.2f}",
                f"Breakout: {df['Close'].iloc[breakout_idx]:.2f}" if breakout_idx else "Breakout: Pending",
                f"Target: {pattern.get('target_price', 'N/A'):.2f}",
                f"Confidence: {pattern['confidence']*100:.1f}%"
            ]
            
            y_position = start_y - (idx * spacing)
            
            fig.add_annotation(
                x=1.38,
                y=0.98 - (idx * 0.15),
                xref="paper",
                yref="paper",
                text="<br>".join(annotation_text),
                showarrow=False,
                font=dict(size=10),
                align='left',
                bordercolor="#AAAAAA",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                yanchor="top"
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
    """
    metrics = {}
    
    for pattern_type, pattern_list in patterns.items():
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives

        if not pattern_list:
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
            # Determine the last point of the pattern based on pattern type
            if pattern_type == "Head and Shoulders":
                last_point_idx = pattern.get('right_shoulder_idx', len(df)-1)
            elif pattern_type == "Double Bottom":
                last_point_idx = max(pattern.get('trough1_idx', 0), pattern.get('trough2_idx', 0))
            elif pattern_type == "Cup and Handle":
                last_point_idx = pattern.get('handle_end', len(df)-1)
            else:
                continue

            # Check if enough data exists after the pattern
            if last_point_idx + look_forward_window >= len(df):
                FN += 1
                continue

            last_price = df['Close'].iloc[last_point_idx]
            future_price = df['Close'].iloc[last_point_idx + look_forward_window]

            # Evaluate based on pattern type
            if pattern_type == "Head and Shoulders":  # Bearish
                if future_price < last_price:
                    TP += 1
                else:
                    FP += 1
            elif pattern_type in ["Double Bottom", "Cup and Handle"]:  # Bullish
                if future_price > last_price:
                    TP += 1
                else:
                    FP += 1

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
    # st.markdown('**Key Metrics**')
    # col1, col2, col3, col4, col5 = st.columns(5)
    
    # with col1:
    #     st.metric("Current Price", f"${selected_data['Current Price']:.2f}")
    
    # with col5:
    #     percent_change = selected_data["Percent Change"]
    #     delta_color = "normal"  # Use 'normal' for default behavior
    #     st.metric("Change", f"{percent_change:.2f}%", delta_color=delta_color)
    
    # with col4:
    #     rsi_value = selected_data["Data"]["RSI"].iloc[-1] if "RSI" in selected_data["Data"].columns else 0
    #     # RSI doesn't directly support custom colors in Streamlit metrics
    #     st.metric("RSI (50)", f"{rsi_value:.2f}")
    
    # with col2:
    #     ma_value_50 = selected_data["Data"]["MA"].iloc[-1] if "MA" in selected_data["Data"].columns else 0
    #     st.metric("MA (50)", f"{ma_value_50:.2f}")
        
    # with col3:
    #     ma_value_200 = selected_data["Data"]["MA2"].iloc[-1] if "MA2" in selected_data["Data"].columns else 0
    #     st.metric("MA (200)", f"{ma_value_200:.2f}")

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
    st.header('📈 Advanced Stock Pattern Scanner (Hitorical Data)')

    # Sidebar setup
    st.sidebar.markdown('<div style="text-align: center; font-weight: bold; font-size: 1.5rem; margin-bottom: 1rem;">Scanner Settings</div>', unsafe_allow_html=True)

    # Fetch Excel files
    st.sidebar.markdown("### 📁 Data Source")
    folder_path = "excels"
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

    # Initialize random selection in session state
    if 'random_index' not in st.session_state:
        st.session_state.random_index = 0  # Default to first file

    # Create columns for file selector and random button
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        # Selectbox with the current selection (either manual or random)
        selected_index = st.selectbox(
            "Select Excel File", 
            range(len(excel_files_display)), 
            index=st.session_state.random_index,  # Set the current index
            format_func=lambda x: excel_files_display[x], 
            key="file_select"
        )

    with col2:
        # Use empty string with markdown and adjust vertical alignment
        st.markdown("<br>", unsafe_allow_html=True)  # Smaller spacer
        if st.button("🎲", help="Select a random stock file"):
            # Generate random index and store in session state
            st.session_state.random_index = np.random.randint(0, len(excel_files_display))
            st.session_state.auto_scan = True  # Flag to trigger auto-scan
            st.rerun()

    # Get the selected file (either from dropdown or random selection)
    selected_index = st.session_state.random_index if 'random_index' in st.session_state else selected_index
    selected_file = os.path.join(folder_path, excel_files[selected_index])

    # If we're coming from a random selection and auto-scan is enabled
    if 'auto_scan' in st.session_state and st.session_state.auto_scan:
        st.session_state.auto_scan = False  # Reset the flag
        scan_button = True  # Set this to trigger the scan automatically
    

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
                    # st.markdown(f"### Metrics for {selected_pattern} Pattern")
                    # metrics = selected_data["Pattern Metrics"][selected_pattern]
                    # metric_cols = st.columns(4)
                    # with metric_cols[0]:
                    #     st.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
                    # with metric_cols[1]:
                    #     st.metric("Precision", f"{metrics['Precision']:.2f}")
                    # with metric_cols[2]:
                    #     st.metric("Recall", f"{metrics['Recall']:.2f}")
                    # with metric_cols[3]:
                    #     st.metric("F1 Score", f"{metrics['F1']:.2f}")

            else:
                st.info("No patterns detected for this stock and date range.")

            # Overall accuracy metrics (optional, kept for reference)
            # st.markdown("### Overall Pattern Detection Accuracy")
            # acc_cols = st.columns(3)
            # with acc_cols[0]:
            #     accuracy = sum(m["Accuracy"] * m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) / sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) if sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) > 0 else 0
            #     st.metric("Accuracy Score", f"{accuracy:.2f}")
            # with acc_cols[1]:
            #     precision = sum(m["Precision"] * m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) / sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) if sum(m["Total Patterns"] for m in selected_data["Pattern Metrics"].values()) > 0 else 0
            #     st.metric("Precision Score", f"{precision:.2f}")
            # with acc_cols[2]:
            #     volume = selected_data.get("Volume", 0)
            #     st.metric("Trading Volume", f"{volume:,.0f}")

if __name__ == "__main__":
    main()
