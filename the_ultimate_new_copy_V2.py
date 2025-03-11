import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import datetime
from datetime import timedelta
import plotly.express as px

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

# Function to check if a date is a trading day
def is_trading_day(date):
    # Check if it's a weekend
    if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    # Here you could add more checks for holidays if needed
    return True

# Function to get the nearest trading day
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

# Function to read stock data from Excel
def read_stock_data_from_excel(file_path):
    try:
        with st.spinner("Reading Excel file..."):
            df = pd.read_excel(file_path)
            # Ensure the required columns are present
            required_columns = ['TIMESTAMP', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            if not all(column in df.columns for column in required_columns):
                st.error(f"The Excel file must contain the following columns: {required_columns}")
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
        return None

# Function to fetch stock data for a specific symbol and date range
import logging

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
        
        # Calculate Moving Average and RSI
        df_filtered = calculate_moving_average(df_filtered)
        df_filtered = calculate_rsi(df_filtered)
        df_filtered = calculate_moving_average_two(df_filtered)
        
        return df_filtered
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to calculate moving average
def calculate_moving_average(df, window=50):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_moving_average_two(df, window=200):
    df['MA2'] = df['Close'].rolling(window=window).mean()
    return df

# Function to calculate RSI
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Function to find extrema
def find_extrema(df, order=5):
    peaks = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    troughs = argrelextrema(df['Close'].values, np.less, order=order)[0]
    return peaks, troughs

# Function to detect head and shoulders pattern
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Improved Head and Shoulders pattern visualization function
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
        mode='lines', name='Stock Price', line=dict(color="#1E88E5", width=2)
    ), row=1, col=1)

    if 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA'],
            mode='lines', name="Moving Average (50)", line=dict(color="#FB8C00", width=2)
        ), row=1, col=1)

    for i, pattern in enumerate(patterns):
        LS, H, RS = pattern["left_shoulder"], pattern["head"], pattern["right_shoulder"]
        T1, T2 = pattern["neckline"]

        try:
            # Add markers for left shoulder, head, and right shoulder
            fig.add_trace(go.Scatter(
                x=[df.loc[int(LS), 'Date']], y=[df.loc[int(LS), 'Close']],
                mode="markers+text", text=["Left Shoulder"], textposition="top center",
                marker=dict(size=12, color="#FF5252", symbol="circle"), name=f"Left Shoulder {i + 1}"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[df.loc[int(H), 'Date']], y=[df.loc[int(H), 'Close']],
                mode="markers+text", text=["Head"], textposition="top center",
                marker=dict(size=14, color="#4CAF50", symbol="circle"), name=f"Head {i + 1}"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[df.loc[int(RS), 'Date']], y=[df.loc[int(RS), 'Close']],
                mode="markers+text", text=["Right Shoulder"], textposition="top center",
                marker=dict(size=12, color="#FF5252", symbol="circle"), name=f"Right Shoulder {i + 1}"
            ), row=1, col=1)

            # Add trough markers
            fig.add_trace(go.Scatter(
                x=[df.loc[int(T1), 'Date']], y=[df.loc[int(T1), 'Close']],
                mode="markers", marker=dict(size=10, color="#673AB7", symbol="diamond"),
                name=f"Left Trough {i + 1}"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[df.loc[int(T2), 'Date']], y=[df.loc[int(T2), 'Close']],
                mode="markers", marker=dict(size=10, color="#673AB7", symbol="diamond"),
                name=f"Right Trough {i + 1}"
            ), row=1, col=1)

            # Draw the neckline connecting the troughs
            neckline_x = [df.loc[int(T1), 'Date'], df.loc[int(T2), 'Date']]
            neckline_y = [df.loc[int(T1), 'Close'], df.loc[int(T2), 'Close']]
            
            # Calculate the slope of the neckline
            days_diff = (df.loc[int(T2), 'Date'] - df.loc[int(T1), 'Date']).days
            if days_diff == 0:
                neckline_slope = 0
            else:
                neckline_slope = (neckline_y[1] - neckline_y[0]) / days_diff
            
            # Extend the neckline to the right (for breakout and target projection)
            # Find the index after the right shoulder
            post_pattern_indices = df.index[df.index > int(RS)]
            if len(post_pattern_indices) > 0:
                # Extend by at least 20 days or to the end of data
                extension_days = min(20, len(post_pattern_indices))
                extended_idx = post_pattern_indices[extension_days-1]
                days_extension = (df.loc[extended_idx, 'Date'] - df.loc[int(T2), 'Date']).days
                extended_y = neckline_y[1] + neckline_slope * days_extension
                
                # Add the extended neckline
                extended_x = df.loc[extended_idx, 'Date']
                
                # Draw the complete neckline
                fig.add_trace(go.Scatter(
                    x=neckline_x + [extended_x],
                    y=neckline_y + [extended_y],
                    mode="lines", name=f"Neckline {i + 1}", 
                    line=dict(color="#673AB7", width=2, dash="dash")
                ), row=1, col=1)
                
                # Calculate profit target (measured move)
                head_height = df.loc[int(H), 'Close']
                
                # Calculate neckline value at head position
                head_date = df.loc[int(H), 'Date']
                days_to_head = (head_date - df.loc[int(T1), 'Date']).days
                neckline_at_head = neckline_y[0] + neckline_slope * days_to_head
                
                # Calculate the distance from head to neckline
                head_to_neckline = head_height - neckline_at_head
                
                # Calculate the profit target level (project the same distance below the neckline)
                profit_target_y = extended_y - head_to_neckline
                
                # Add profit target line and marker
                fig.add_trace(go.Scatter(
                    x=[extended_x],
                    y=[profit_target_y],
                    mode="markers+text",
                    text=["Profit Target"],
                    textposition="bottom right",
                    marker=dict(size=12, color="#E91E63", symbol="triangle-down"),
                    name=f"Profit Target {i + 1}"
                ), row=1, col=1)
                
                # Add a vertical line showing the measured move
                fig.add_trace(go.Scatter(
                    x=[extended_x, extended_x],
                    y=[extended_y, profit_target_y],
                    mode="lines",
                    line=dict(color="#E91E63", width=2, dash="dot"),
                    name=f"Measured Move {i + 1}"
                ), row=1, col=1)
                
                # Add annotation explaining the measured move
                fig.add_annotation(
                    x=extended_x,
                    y=(extended_y + profit_target_y) / 2,
                    text=f"Measured Move: {head_to_neckline:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#E91E63",
                    ax=30,
                    ay=0,
                    font=dict(size=10, color="#E91E63")
                )
                
                # Add breakout annotation
                fig.add_annotation(
                    x=df.loc[int(T2), 'Date'],
                    y=neckline_y[1],
                    text="Breakout Point",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#673AB7",
                    ax=0,
                    ay=30,
                    font=dict(size=10, color="#673AB7")
                )
            
            # Connect the pattern points to show the formation
            pattern_x = [df.loc[int(LS), 'Date'], df.loc[int(T1), 'Date'], 
                         df.loc[int(H), 'Date'], df.loc[int(T2), 'Date'], 
                         df.loc[int(RS), 'Date']]
            pattern_y = [df.loc[int(LS), 'Close'], df.loc[int(T1), 'Close'], 
                         df.loc[int(H), 'Close'], df.loc[int(T2), 'Close'], 
                         df.loc[int(RS), 'Close']]
            
            fig.add_trace(go.Scatter(
                x=pattern_x,
                y=pattern_y,
                mode="lines",
                line=dict(color="rgba(156, 39, 176, 0.7)", width=3),
                name=f"Pattern Formation {i + 1}"
            ), row=1, col=1)
            
        except KeyError as e:
            # Skip this pattern if any points are not in the dataframe
            print(f"KeyError in H&S pattern: {e}")
            continue
        except Exception as e:
            print(f"Error in H&S pattern: {e}")
            continue

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
        text="Head & Shoulders: Bearish reversal pattern with profit target equal to the distance from head to neckline",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="#0D47A1"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#0D47A1",
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        title={
            'text': "Head & Shoulders Pattern Detection",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color="#0D47A1")
        },
        xaxis_title="Date",
        xaxis=dict(visible=False, showticklabels=False, showgrid=False),
        xaxis2=dict(visible=False, showticklabels=False, showgrid=False),
        xaxis3=dict(title="Date"),
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="RSI",
        showlegend=True,
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

# Improved Cup and Handle pattern visualization
def plot_pattern(df, pattern_points, pattern_name):
    # Create a subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price Chart with Pattern", "Volume", "RSI (14)")
    )
    
    # Add price line chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#26A69A')
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
    colors = [
        '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', 
        '#03A9F4', '#00BCD4', '#009688', '#4CAF50', '#8BC34A', 
        '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722'
    ]
    
    # Add pattern-specific visualization
    if not isinstance(pattern_points, list):
        pattern_points = [pattern_points]
    
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = colors[idx % len(colors)]
        
        # Pattern-specific visualizations
        if pattern_name == "Cup and Handle" and "left_peak" in pattern and "min_idx" in pattern and "right_peak" in pattern:
            try:
                # Get cup points
                left_peak_idx = pattern['left_peak']
                cup_bottom_idx = pattern['min_idx']
                right_peak_idx = pattern['right_peak']
                
                # Add markers for key points
                fig.add_trace(
                    go.Scatter(
                        x=[df.loc[left_peak_idx, 'Date']],
                        y=[df.loc[left_peak_idx, 'Close']],
                        mode="markers+text",
                        text=["Left Cup Lip"],
                        textposition="top right",
                        textfont=dict(size=10),
                        marker=dict(color="#3F51B5", size=10, symbol="circle"),
                        name="Left Cup Lip"
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.loc[cup_bottom_idx, 'Date']],
                        y=[df.loc[cup_bottom_idx, 'Close']],
                        mode="markers+text",
                        text=["Cup Bottom"],
                        textposition="bottom center",
                        textfont=dict(size=10),
                        marker=dict(color="#4CAF50", size=10, symbol="circle"),
                        name="Cup Bottom"
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.loc[right_peak_idx, 'Date']],
                        y=[df.loc[right_peak_idx, 'Close']],
                        mode="markers+text",
                        text=["Right Cup Lip"],
                        textposition="top left",
                        textfont=dict(size=10),
                        marker=dict(color="#3F51B5", size=10, symbol="circle"),
                        name="Right Cup Lip"
                    ),
                    row=1, col=1
                )
                
                # Create a smooth arc for the cup - separate from the price line
                # Generate points for the cup arc
                num_points = 100  # More points for a smoother arc
                
                # Create x values (dates) for the arc
                left_date = df.loc[left_peak_idx, 'Date']
                right_date = df.loc[right_peak_idx, 'Date']
                bottom_date = df.loc[cup_bottom_idx, 'Date']
                
                # Calculate time deltas for interpolation
                total_days = (right_date - left_date).total_seconds()
                
                # Generate dates for the arc
                arc_dates = []
                for i in range(num_points):
                    # Calculate position (0 to 1)
                    t = i / (num_points - 1)
                    # Calculate days from left peak
                    days_offset = total_days * t
                    # Calculate the date
                    current_date = left_date + pd.Timedelta(seconds=days_offset)
                    arc_dates.append(current_date)
                
                # Create y values (prices) for the arc
                left_price = df.loc[left_peak_idx, 'Close']
                right_price = df.loc[right_peak_idx, 'Close']
                bottom_price = df.loc[cup_bottom_idx, 'Close']
                
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
                    
                    # Calculate price using parabola
                    # At x=-1 (left peak), y=left_price
                    # At x=0 (bottom), y=bottom_price
                    # At x=1 (right peak), y=right_price
                    
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
                
                # Add the smooth cup arc - separate from the price line
                fig.add_trace(
                    go.Scatter(
                        x=arc_dates,
                        y=arc_prices,
                        mode="lines",
                        name="Cup Arc",
                        line=dict(color="#9C27B0", width=3)
                    ),
                    row=1, col=1
                )
                
                # Handle visualization
                if "handle_start" in pattern and "handle_end" in pattern:
                    handle_start_idx = pattern['handle_start']
                    handle_end_idx = pattern['handle_end']
                    
                    # Add handle markers
                    fig.add_trace(
                        go.Scatter(
                            x=[df.loc[handle_start_idx, 'Date']],
                            y=[df.loc[handle_start_idx, 'Close']],
                            mode="markers+text",
                            text=["Handle Start"],
                            textposition="top right",
                            textfont=dict(size=10),
                            marker=dict(color="#FF9800", size=10, symbol="circle"),
                            name="Handle Start"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[df.loc[handle_end_idx, 'Date']],
                            y=[df.loc[handle_end_idx, 'Close']],
                            mode="markers+text",
                            text=["Handle End"],
                            textposition="top right",
                            textfont=dict(size=10),
                            marker=dict(color="#FF9800", size=10, symbol="circle"),
                            name="Handle End"
                        ),
                        row=1, col=1
                    )
                    
                    # Get handle data points
                    handle_indices = list(range(handle_start_idx, handle_end_idx + 1))
                    handle_dates = df.loc[handle_indices, 'Date'].tolist()
                    handle_prices = df.loc[handle_indices, 'Close'].tolist()
                    
                    # Add handle line
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
                
                # Add breakout level and target
                if "handle_end" in pattern:
                    # Calculate the cup height (for profit target)
                    cup_height = df.loc[right_peak_idx, 'Close'] - df.loc[cup_bottom_idx, 'Close']
                    
                    # Breakout level is typically the right cup lip
                    breakout_level = df.loc[right_peak_idx, 'Close']
                    
                    # Target is typically cup height added to breakout level
                    target_level = breakout_level + cup_height
                    
                    # Add breakout level line
                    fig.add_shape(
                        type="line",
                        x0=df.loc[right_peak_idx, 'Date'],
                        x1=df['Date'].iloc[-1],
                        y0=breakout_level,
                        y1=breakout_level,
                        line=dict(color="#FF5722", width=2, dash="dash"),
                        row=1, col=1
                    )
                    
                    # Add target level line
                    fig.add_shape(
                        type="line",
                        x0=df.loc[right_peak_idx, 'Date'],
                        x1=df['Date'].iloc[-1],
                        y0=target_level,
                        y1=target_level,
                        line=dict(color="#4CAF50", width=2, dash="dash"),
                        row=1, col=1
                    )
                    
                    # Add annotations
                    fig.add_annotation(
                        x=df['Date'].iloc[-1],
                        y=breakout_level,
                        text="Breakout Level",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#FF5722",
                        ax=-40,
                        ay=0,
                        font=dict(size=10, color="#FF5722")
                    )
                    
                    fig.add_annotation(
                        x=df['Date'].iloc[-1],
                        y=target_level,
                        text=f"Target (+{cup_height:.2f})",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#4CAF50",
                        ax=-40,
                        ay=0,
                        font=dict(size=10, color="#4CAF50")
                    )
                
            except KeyError as e:
                print(f"KeyError in Cup and Handle pattern: {e}")
                continue
            except Exception as e:
                print(f"Error in Cup and Handle pattern: {e}")
                continue
                
        elif pattern_name in ["Double Top", "Double Bottom"]:
            # Add markers for pattern points
            for key, point_idx in pattern.items():
                if isinstance(point_idx, (np.int64, int)):  
                    try:
                        fig.add_trace(
                            go.Scatter(
                                x=[df.loc[int(point_idx), 'Date']], 
                                y=[df.loc[int(point_idx), 'Close']],
                                mode="markers+text", 
                                text=[key.replace('_', ' ').title()], 
                                textposition="top center",
                                textfont=dict(size=10, color=color),
                                marker=dict(size=10, color=color, symbol="circle", line=dict(width=2, color='white')),
                                name=f"{key.replace('_', ' ').title()} {idx + 1}"
                            ),
                            row=1, col=1
                        )
                    except KeyError:
                        # Skip this point if it's not in the dataframe
                        continue
            
            # Connect pattern points
            x_values = [df.loc[pattern[key], 'Date'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            y_values = [df.loc[pattern[key], 'Close'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=y_values,
                    mode="lines", 
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{pattern_name} Pattern {idx + 1}"
                ),
                row=1, col=1
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
    if pattern_name == "Cup and Handle":
        fig.add_annotation(
            x=df['Date'].iloc[0],
            y=df['Close'].max(),
            text="Cup & Handle: Bullish continuation pattern with target equal to cup depth projected above breakout",
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
            'text': f"{pattern_name} Pattern Detection for {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}",
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
    
    # Update axes
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="RSI", row=3, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )
    
    return fig

# Function to detect double top pattern
def detect_double_top(df):
    peaks, _ = find_extrema(df, order=10)
    if len(peaks) < 2:
        return []

    patterns = []
    for i in range(len(peaks) - 1):
        if abs(df['Close'][peaks[i]] - df['Close'][peaks[i + 1]]) < df['Close'][peaks[i]] * 0.03:
            patterns.append({'peak1': peaks[i], 'peak2': peaks[i + 1]})
    return patterns

# Function to detect double bottom pattern
def detect_double_bottom(df):
    _, troughs = find_extrema(df, order=10)
    if len(troughs) < 2:
        return []

    patterns = []
    for i in range(len(troughs) - 1):
        if abs(df['Close'][troughs[i]] - df['Close'][troughs[i + 1]]) < df['Close'][troughs[i]] * 0.03:
            patterns.append({'trough1': troughs[i], 'trough2': troughs[i + 1]})
    return patterns

# Function to detect cup and handle pattern
def detect_cup_and_handle(df):
    _, troughs = find_extrema(df, order=20)
    if len(troughs) < 1:
        return []

    patterns = []
    min_idx = troughs[0] 
    left_peak = df.iloc[:min_idx]['Close'].idxmax()
    right_peak = df.iloc[min_idx:]['Close'].idxmax()

    if not left_peak or not right_peak:
        return []

    handle_start_idx = right_peak
    handle_end_idx = None
    
    handle_retracement_level = 0.5 
    
    handle_start_price = df['Close'][handle_start_idx]
    
    potential_handle_bottom_idx = df.iloc[handle_start_idx:]['Close'].idxmin()

    if potential_handle_bottom_idx is not None:
        handle_bottom_price = df['Close'][potential_handle_bottom_idx]
        
        handle_top_price = handle_start_price - (handle_start_price - handle_bottom_price) * handle_retracement_level

        for i in range(potential_handle_bottom_idx + 1, len(df)):
            if df['Close'][i] > handle_top_price:
                handle_end_idx = i
                break

    if handle_end_idx:
         patterns.append({
            "left_peak": left_peak,
            "min_idx": min_idx,
            "right_peak": right_peak,
            "handle_start": handle_start_idx,
            "handle_end": handle_end_idx
        })
    return patterns

# Function to plot patterns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_pattern(df, pattern_points, pattern_name):
    # Create a subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price Chart with Pattern", "Volume", "RSI (14)")
    )
    
    # Add price line chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#26A69A')
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
    colors = [
        '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', 
        '#03A9F4', '#00BCD4', '#009688', '#4CAF50', '#8BC34A', 
        '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722'
    ]
    
    # Add pattern-specific visualization
    if not isinstance(pattern_points, list):
        pattern_points = [pattern_points]
    
    for idx, pattern in enumerate(pattern_points):
        if not isinstance(pattern, dict):
            continue
        
        color = colors[idx % len(colors)]
        
        # Add markers for pattern points
        for key, point_idx in pattern.items():
            if isinstance(point_idx, (np.int64, int)):  
                try:
                    fig.add_trace(
                        go.Scatter(
                            x=[df.loc[int(point_idx), 'Date']], 
                            y=[df.loc[int(point_idx), 'Close']],
                            mode="markers+text", 
                            text=[key.replace('_', ' ').title()], 
                            textposition="top center",
                            textfont=dict(size=10, color=color),
                            marker=dict(size=10, color=color, symbol="circle", line=dict(width=2, color='white')),
                            name=f"{key.replace('_', ' ').title()} {idx + 1}"
                        ),
                        row=1, col=1
                    )
                except KeyError:
                    # Skip this point if it's not in the dataframe
                    continue
            elif isinstance(point_idx, tuple) and key == "neckline":
                # Handle neckline points
                try:
                    fig.add_trace(
                        go.Scatter(
                            x=[df.loc[int(point_idx[0]), 'Date'], df.loc[int(point_idx[1]), 'Date']],
                            y=[df.loc[int(point_idx[0]), 'Close'], df.loc[int(point_idx[1]), 'Close']],
                            mode="lines", 
                            name=f"Neckline {idx + 1}", 
                            line=dict(color=color, width=2, dash="dash")
                        ),
                        row=1, col=1
                    )
                except KeyError:
                    # Skip this point if it's not in the dataframe
                    continue
            
        # Pattern-specific visualizations
        if pattern_name == "Cup and Handle" and "handle_start" in pattern and "handle_end" in pattern:
            # Cup visualization
            cup_dates = df['Date'][pattern['left_peak']:pattern['right_peak']+1].tolist()
            cup_prices = df['Close'][pattern['left_peak']:pattern['right_peak']+1].tolist()
            
            # Handle visualization
            handle_dates = df['Date'][pattern['handle_start']:pattern['handle_end']+1].tolist()
            handle_prices = df['Close'][pattern['handle_start']:pattern['handle_end']+1].tolist()

            # Add cup line
            fig.add_trace(
                go.Scatter(
                    x=cup_dates, 
                    y=cup_prices, 
                    mode="lines", 
                    name="Cup", 
                    line=dict(color="#9C27B0", width=3)
                ),
                row=1, col=1
            )
            
            # Add handle line
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

            # Add cup lip markers
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[pattern['left_peak'], 'Date']], 
                    y=[df.loc[pattern['left_peak'], 'Close']],
                    mode="markers+text", 
                    text=["Left Cup Lip"], 
                    textposition="top right",
                    textfont=dict(size=10),
                    marker=dict(color="#3F51B5", size=10, symbol="circle"),
                    name="Left Cup Lip"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[pattern['right_peak'], 'Date']], 
                    y=[df.loc[pattern['right_peak'], 'Close']],
                    mode="markers+text", 
                    text=["Right Cup Lip"], 
                    textposition="top left",
                    textfont=dict(size=10),
                    marker=dict(color="#3F51B5", size=10, symbol="circle"),
                    name="Right Cup Lip"
                ),
                row=1, col=1
            )

            # Add cup base line
            min_cup_price = min(cup_prices)
            fig.add_trace(
                go.Scatter(
                    x=[cup_dates[0], cup_dates[-1]], 
                    y=[min_cup_price, min_cup_price],
                    mode="lines", 
                    name="Cup Base", 
                    line=dict(color="#4CAF50", width=2, dash="dot")
                ),
                row=1, col=1
            )
    
        # Connect pattern points for better visualization
        if pattern_name in ["Head and Shoulders", "Double Top", "Double Bottom"]:
            x_values = [df.loc[pattern[key], 'Date'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            y_values = [df.loc[pattern[key], 'Close'] for key in pattern if isinstance(pattern[key], (np.int64, int))]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=y_values,
                    mode="lines", 
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{pattern_name} Pattern {idx + 1}"
                ),
                row=1, col=1
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
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{pattern_name} Pattern Detection for {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}",
            'y': 0.95,  # Adjust the vertical position of the title
            'x': 0.5,   # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color="#0D47A1")
        },
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",  # Vertical orientation for the legend
            yanchor="top",    # Anchor the legend to the top
            y=1,              # Position the legend at the top vertically
            xanchor="right",  # Anchor the legend to the right
            x=1.4,            # Position the legend to the right of the chart
            font=dict(size=10)
        ),
        margin=dict(l=40, r=150, t=100, b=40),  # Adjust right margin to accommodate the legend
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="RSI", row=3, col=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )
    
    return fig


# Function to evaluate pattern detection
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
            elif pattern_type == "Double Top":
                last_point_idx = max(int(pattern['peak1']), int(pattern['peak2']))
            elif pattern_type == "Double Bottom":
                last_point_idx = max(int(pattern['trough1']), int(pattern['trough2']))
            elif pattern_type == "Cup and Handle":
                last_point_idx = int(pattern['handle_end'])
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
                elif pattern_type in ["Head and Shoulders", "Double Top"]:  # Bearish patterns
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


# Function to create a summary dashboard for a stock
def create_stock_dashboard(selected_data):
    
    # Create pattern summary
    st.write("**Pattern Detection Summary**")
    
    pattern_cols = st.columns(4)
    patterns = ["Head and Shoulders", "Double Top", "Double Bottom", "Cup and Handle"]
    
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
    
    
    
    


# Main application
def main():
    # Header with logo and title
    st.markdown('<div class="main-header">📈 Advanced Stock Pattern Scanner(Static)</div>', unsafe_allow_html=True)
    
    
    st.sidebar.markdown('<div style="text-align: center; font-weight: bold; font-size: 1.5rem; margin-bottom: 1rem;">Scanner Settings</div>', unsafe_allow_html=True)
    
    
    # Dropdown to select Excel file
    st.sidebar.markdown("### 📁 Data Source")
    excel_files = [f for f in os.listdir() if f.endswith('.xlsx')]
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


    
        
        # Scan button with enhanced styling
        st.sidebar.markdown("### 🔍 Scan Stocks")

        # Scan button with enhanced styling
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
                    
                    df_filtered = fetch_stock_data(symbol, start_date, end_date, st.session_state.df)
                    if df_filtered is None or df_filtered.empty:
                        print(f"No data for {symbol} within the date range.")
                        continue
                    
                    print(f"Processing {symbol} with {len(df_filtered)} rows.")
                    
                    patterns = {}
                    patterns["Head and Shoulders"] = detect_head_and_shoulders(df_filtered)
                    patterns["Double Top"] = detect_double_top(df_filtered)
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
                    # Optionally, log the exception for further analysis
                    # logging.error(f"Error processing {symbol}: {str(e)}")
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
                    else:
                        fig = plot_pattern(
                            selected_data["Data"],
                            pattern_points,
                            st.session_state.selected_pattern
                        )
                        
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)

                        
                    # Add pattern explanation
                    pattern_explanations = {
                        "Head and Shoulders": "A bearish reversal pattern with three peaks, where the middle peak (head) is higher than the two surrounding peaks (shoulders). Signals a potential downtrend.",
                        "Double Top": "A bearish reversal pattern with two peaks at approximately the same price level. Indicates resistance and potential downward movement.",
                        "Double Bottom": "A bullish reversal pattern with two troughs at approximately the same price level. Indicates support and potential upward movement.",
                        "Cup and Handle": "A bullish continuation pattern resembling a cup with a handle. The cup forms a 'U' shape, and the handle has a slight downward drift. Signals potential upward movement."
                    }
                    
                    print(f"About {st.session_state.selected_pattern} Pattern")
                    print(pattern_explanations.get(st.session_state.selected_pattern, ""))
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
