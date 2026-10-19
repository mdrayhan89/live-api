from fastapi import FastAPI, Query
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
from tvdatafeed import TvDatafeed, Interval

app = FastAPI(title="TradingView Custom Signal API")

# TradingView datafeed instance (Guest login)
tv = TvDatafeed()

# Timeframe Mapping
TIMEFRAME_MAP = {
    "M1": Interval.in_1_minute,
    "M5": Interval.in_5_minute,
    "M15": Interval.in_15_minute,
    "M30": Interval.in_30_minute,
    "H1": Interval.in_1_hour,
    "H4": Interval.in_4_hour,
    "D": Interval.in_daily,
}

def parse_pair(pair_str: str):
    """
    Format pair string into TradingView Symbol and Exchange
    Example: EUR_USD -> EURUSD (FX_IDC / OANDA)
    """
    clean_pair = pair_str.replace("_", "").replace("/", "").replace("-", "").upper()
    return clean_pair

# -------------------------------------------------------------
# ENDPOINT 1: Live Candles
# -------------------------------------------------------------
@app.get("/api/v2/qx.php")
def get_live_candles(
    pair: str = Query(..., description="e.g. EUR_USD"),
    timeframe: str = Query("M1", description="M1, M5, H1, etc."),
    count: int = Query(3000, ge=1, le=5000)
):
    symbol = parse_pair(pair)
    tf = TIMEFRAME_MAP.get(timeframe.upper(), Interval.in_1_minute)
    
    try:
        # Fetch candles from TradingView (Exchange = FX_IDC for Forex)
        df = tv.get_hist(symbol=symbol, exchange='FX_IDC', interval=tf, n_bars=count)
        if df is None or df.empty:
            df = tv.get_hist(symbol=symbol, exchange='OANDA', interval=tf, n_bars=count)
            
        if df is None or df.empty:
            return {"error": "No data found for the pair"}

        # Reset index to get datetime column
        df = df.reset_index()
        
        candles = []
        for _, row in df.iterrows():
            candles.append({
                "time": str(row['datetime']),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
            
        return {
            "status": "success",
            "pair": pair,
            "timeframe": timeframe,
            "count": len(candles),
            "candles": candles
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# ENDPOINT 2: Future Signal Analysis (Dynamic Accuracy Backtest)
# -------------------------------------------------------------
@app.get("/api/v2/future.php")
def get_future_signals(
    pair: str = Query(..., description="e.g. EUR_USD"),
    timeframe: str = Query("M1"),
    start_time: str = Query("09:00", description="HH:MM format in UTC+6"),
    end_time: str = Query("18:00", description="HH:MM format in UTC+6"),
    min_accuracy: float = Query(80.0, ge=0.0, le=100.0),
    days: int = Query(7, ge=1, le=30)
):
    symbol = parse_pair(pair)
    tf = TIMEFRAME_MAP.get(timeframe.upper(), Interval.in_1_minute)
    
    # Calculate required candle count based on days
    bars_needed = days * 1440  # Approximate daily candles for M1
    if bars_needed > 5000:
        bars_needed = 5000  # Cap at max fetchable limit
        
    try:
        df = tv.get_hist(symbol=symbol, exchange='FX_IDC', interval=tf, n_bars=bars_needed)
        if df is None or df.empty:
            df = tv.get_hist(symbol=symbol, exchange='OANDA', interval=tf, n_bars=bars_needed)

        if df is None or df.empty:
            return {"error": "No data available for signal backtest"}

        df = df.reset_index()
        
        # Convert datetime to UTC+6 (Bangladesh Time)
        df['datetime'] = pd.to_datetime(df['datetime']) + timedelta(hours=6)
        
        # Filter by date range (last N days)
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['datetime'] >= cutoff_date]

        # Parse start and end time filters
        sh, sm = map(int, start_time.split(":"))
        eh, em = map(int, end_time.split(":"))
        start_t = time(sh, sm)
        end_t = time(eh, em)

        # Basic Strategy Signal Generation (e.g., Simple Candle Direction/Engulfing pattern)
        # 1 = CALL (BUY), -1 = PUT (SELL)
        df['signal'] = 0
        df.loc[df['close'] > df['open'], 'signal'] = 1   # CALL
        df.loc[df['close'] < df['open'], 'signal'] = -1  # PUT

        # Next candle outcome check for backtest win/loss
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        
        # Win condition logic
        df['win'] = False
        df.loc[(df['signal'] == 1) & (df['next_close'] > df['next_open']), 'win'] = True
        df.loc[(df['signal'] == -1) & (df['next_close'] < df['next_open']), 'win'] = True

        # Group signals by time slot (HH:MM) to backtest accuracy for specific times
        df['time_str'] = df['datetime'].dt.strftime('%H:%M')
        
        time_groups = df.groupby('time_str')
        matching_signals = []

        for t_str, group in time_groups:
            t_obj = datetime.strptime(t_str, "%H:%M").time()
            
            # Check if time falls between start_time and end_time
            if start_t <= t_obj <= end_t:
                total_trades = len(group)
                wins = group['win'].sum()
                
                if total_trades > 0:
                    accuracy = (wins / total_trades) * 100
                    
                    # DYNAMIC ACCURACY FILTER (0 to 100 check)
                    if accuracy >= min_accuracy:
                        # Determine dominant signal direction for this time slot
                        last_signal = group['signal'].iloc[-1]
                        sig_type = "CALL" if last_signal == 1 else "PUT"
                        
                        matching_signals.append({
                            "time": t_str,
                            "signal": sig_type,
                            "accuracy": round(accuracy, 2),
                            "total_backtest_trades": int(total_trades),
                            "wins": int(wins)
                        })

        return {
            "status": "success",
            "pair": pair,
            "timeframe": timeframe,
            "backtest_days": days,
            "min_accuracy_applied": min_accuracy,
            "time_window": f"{start_time} - {end_time}",
            "total_signals_found": len(matching_signals),
            "signals": matching_signals
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
