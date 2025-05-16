import json
import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import ta
from utils import load_config, save_signals, load_signals
from telegram import send_message, send_chart_image

# پیکربندی را بارگذاری کنید
coins = load_config('config/coins.json')
strategy = load_config('config/strategy.json')

# بارگذاری سیگنال‌های فعلی
signals = load_signals()

def fetch_ohlcv_data(symbol, timeframe='30m', limit=100):
    """
    دریافت داده‌های OHLCV از اندپوینت عمومی بایننس
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': f"{symbol}USDT",
        'interval': timeframe,
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                      'close_time', 'quote_asset_volume', 'number_of_trades',
                                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # تبدیل ستون‌ها به نوع عددی
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def calculate_indicators(df):
    """
    محاسبه شاخص‌های تکنیکال
    """
    # میانگین متحرک نمایی
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema100'] = ta.trend.ema_indicator(df['close'], window=100)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # حجم نسبی
    df['volume_sma20'] = ta.volume.volume_sma_indicator(df['close'], df['volume'], window=20)
    
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    
    # OBV (On-Balance Volume)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    return df

def generate_chart(df, symbol):
    """
    ایجاد تصویر چارت برای ارسال به تلگرام
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{symbol} Chart', 'Indicators'))
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['timestamp'].tail(60),
        open=df['open'].tail(60),
        high=df['high'].tail(60),
        low=df['low'].tail(60),
        close=df['close'].tail(60),
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df['timestamp'].tail(60), y=df['ema20'].tail(60), 
                             line=dict(color='blue', width=1), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'].tail(60), y=df['ema50'].tail(60), 
                             line=dict(color='orange', width=1), name='EMA50'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['timestamp'].tail(60), y=df['rsi'].tail(60), 
                            line=dict(color='purple', width=1), name='RSI'), row=2, col=1)
    
    # Add horizontal lines at RSI 30 and 70
    fig.add_shape(type="line", x0=df['timestamp'].tail(60).iloc[0], x1=df['timestamp'].tail(60).iloc[-1],
                  y0=30, y1=30, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=df['timestamp'].tail(60).iloc[0], x1=df['timestamp'].tail(60).iloc[-1],
                  y0=70, y1=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    
    # Layout
    fig.update_layout(
        title_text=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        width=1200,
        showlegend=True
    )
    
    # Y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    # Save chart
    chart_path = f'data/{symbol}_chart.png'
    fig.write_image(chart_path)
    return chart_path

def check_buy_signal(df, strategy_params):
    """
    بررسی شرایط سیگنال خرید
    """
    last_row = df.iloc[-2]  # استفاده از کندل قبلی برای جلوگیری از سیگنال‌های کاذب در کندل فعلی
    
    # شرایط مختلف استراتژی
    ema_cross_up = (last_row['ema5'] > last_row['ema20']) and (df.iloc[-3]['ema5'] <= df.iloc[-3]['ema20'])
    rsi_oversold_recovery = (last_row['rsi'] > 30) and (df.iloc[-3]['rsi'] < 30)
    macd_cross_up = (last_row['macd'] > last_row['macd_signal']) and (df.iloc[-3]['macd'] <= df.iloc[-3]['macd_signal'])
    
    # بررسی الگوی کندل
    bullish_engulfing = (
        last_row['close'] > last_row['open'] and
        df.iloc[-3]['close'] < df.iloc[-3]['open'] and
        last_row['close'] > df.iloc[-3]['open'] and
        last_row['open'] < df.iloc[-3]['close']
    )
    
    # حمایت و مقاومت - بررسی اگر قیمت به حمایت نزدیک است
    near_support = last_row['close'] <= last_row['bb_lower'] * 1.01
    
    # معیارهای حجم
    volume_surge = last_row['volume'] > last_row['volume_sma20'] * 1.5
    
    # استراتژی ترکیبی - باید چندین شرط برقرار باشد
    min_conditions_met = 0
    
    if ema_cross_up: min_conditions_met += 1
    if rsi_oversold_recovery: min_conditions_met += 1
    if macd_cross_up: min_conditions_met += 1
    if bullish_engulfing: min_conditions_met += 1
    if near_support: min_conditions_met += 1
    if volume_surge: min_conditions_met += 1
    
    # فیلتر روند - بررسی اینکه آیا روند بلندمدت صعودی است
    uptrend_condition = last_row['ema50'] > last_row['ema100']
    
    # شروط نهایی برای سیگنال خرید
    min_required = strategy_params.get('min_conditions_required', 3)
    
    return (min_conditions_met >= min_required) and uptrend_condition

def check_sell_signal(df, strategy_params):
    """
    بررسی شرایط سیگنال فروش
    """
    last_row = df.iloc[-2]  # استفاده از کندل قبلی
    
    # شرایط مختلف استراتژی
    ema_cross_down = (last_row['ema5'] < last_row['ema20']) and (df.iloc[-3]['ema5'] >= df.iloc[-3]['ema20'])
    rsi_overbought_drop = (last_row['rsi'] < 70) and (df.iloc[-3]['rsi'] > 70)
    macd_cross_down = (last_row['macd'] < last_row['macd_signal']) and (df.iloc[-3]['macd'] >= df.iloc[-3]['macd_signal'])
    
    # بررسی الگوی کندل
    bearish_engulfing = (
        last_row['close'] < last_row['open'] and
        df.iloc[-3]['close'] > df.iloc[-3]['open'] and
        last_row['close'] < df.iloc[-3]['open'] and
        last_row['open'] > df.iloc[-3]['close']
    )
    
    # حمایت و مقاومت - بررسی اگر قیمت به مقاومت نزدیک است
    near_resistance = last_row['close'] >= last_row['bb_upper'] * 0.99
    
    # معیارهای حجم
    volume_surge = last_row['volume'] > last_row['volume_sma20'] * 1.5
    
    # استراتژی ترکیبی - باید چندین شرط برقرار باشد
    min_conditions_met = 0
    
    if ema_cross_down: min_conditions_met += 1
    if rsi_overbought_drop: min_conditions_met += 1
    if macd_cross_down: min_conditions_met += 1
    if bearish_engulfing: min_conditions_met += 1
    if near_resistance: min_conditions_met += 1
    if volume_surge: min_conditions_met += 1
    
    # فیلتر روند - بررسی اینکه آیا روند بلندمدت نزولی است
    downtrend_condition = last_row['ema50'] < last_row['ema100']
    
    # شروط نهایی برای سیگنال فروش
    min_required = strategy_params.get('min_conditions_required', 3)
    
    return (min_conditions_met >= min_required) and downtrend_condition

def calculate_targets_and_stoploss(df, direction):
    """
    محاسبه هدف‌های قیمتی و حد ضرر
    """
    last_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    if direction == 'buy':
        stop_loss = last_price - (2 * atr)
        target1 = last_price + (1.5 * atr)
        target2 = last_price + (3 * atr)
        target3 = last_price + (5 * atr)
    else:  # sell
        stop_loss = last_price + (2 * atr)
        target1 = last_price - (1.5 * atr)
        target2 = last_price - (3 * atr)
        target3 = last_price - (5 * atr)
    
    return {
        'entry': round(last_price, 8),
        'stop_loss': round(stop_loss, 8),
        'target1': round(target1, 8),
        'target2': round(target2, 8),
        'target3': round(target3, 8)
    }

def analyze_coins():
    """
    تحلیل لیست ارزهای دیجیتال
    """
    new_signals = []
    
    for coin in coins:
        symbol = coin['symbol']
        print(f"Analyzing {symbol}...")
        
        # دریافت داده‌ها و محاسبه شاخص‌ها
        df = fetch_ohlcv_data(symbol)
        if df.empty:
            print(f"Failed to get data for {symbol}, skipping...")
            continue
            
        df = calculate_indicators(df)
        
        # بررسی وجود سیگنال فعال برای این ارز
        active_signal = next((s for s in signals if s['symbol'] == symbol and s['status'] == 'active'), None)
        
        if not active_signal:
            # بررسی سیگنال‌های جدید
            if check_buy_signal(df, strategy):
                targets = calculate_targets_and_stoploss(df, 'buy')
                
                signal = {
                    'symbol': symbol,
                    'direction': 'buy',
                    'entry_price': targets['entry'],
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': targets['stop_loss'],
                    'target1': targets['target1'],
                    'target2': targets['target2'],
                    'target3': targets['target3'],
                    'status': 'active',
                    'targets_hit': []
                }
                
                # ارسال سیگنال به تلگرام
                message = f"🔔 سیگنال خرید #{symbol}\n\n" \
                          f"💰 قیمت ورود: {targets['entry']}\n" \
                          f"🛑 حد ضرر: {targets['stop_loss']} ({round((targets['stop_loss'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n" \
                          f"🎯 هدف اول: {targets['target1']} ({round((targets['target1'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n" \
                          f"🎯 هدف دوم: {targets['target2']} ({round((targets['target2'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n" \
                          f"🎯 هدف سوم: {targets['target3']} ({round((targets['target3'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n\n" \
                          f"⏰ زمان سیگنال: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                chart_path = generate_chart(df, symbol)
                send_message(message)
                send_chart_image(chart_path, f"نمودار تکنیکال {symbol}")
                
                new_signals.append(signal)
                
            elif check_sell_signal(df, strategy):
                targets = calculate_targets_and_stoploss(df, 'sell')
                
                signal = {
                    'symbol': symbol,
                    'direction': 'sell',
                    'entry_price': targets['entry'],
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': targets['stop_loss'],
                    'target1': targets['target1'],
                    'target2': targets['target2'],
                    'target3': targets['target3'],
                    'status': 'active',
                    'targets_hit': []
                }
                
                # ارسال سیگنال به تلگرام
                message = f"🔔 سیگنال فروش #{symbol}\n\n" \
                          f"💰 قیمت ورود: {targets['entry']}\n" \
                          f"🛑 حد ضرر: {targets['stop_loss']} ({round((targets['stop_loss'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n" \
                          f"🎯 هدف اول: {targets['target1']} ({round((targets['target1'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n" \
                          f"🎯 هدف دوم: {targets['target2']} ({round((targets['target2'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n" \
                          f"🎯 هدف سوم: {targets['target3']} ({round((targets['target3'] - targets['entry']) / targets['entry'] * 100, 2)}%)\n\n" \
                          f"⏰ زمان سیگنال: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                chart_path = generate_chart(df, symbol)
                send_message(message)
                send_chart_image(chart_path, f"نمودار تکنیکال {symbol}")
                
                new_signals.append(signal)
        
    # بروزرسانی سیگنال‌ها
    updated_signals = signals + new_signals
    save_signals(updated_signals)
    print(f"Analysis complete. Found {len(new_signals)} new signals.")

if __name__ == "__main__":
    analyze_coins()
