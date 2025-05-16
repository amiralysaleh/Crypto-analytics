import json
import os
from datetime import datetime

def load_config(file_path):
    """
    بارگذاری فایل پیکربندی
    """
    if not os.path.exists(file_path):
        if 'coins.json' in file_path:
            return default_coins()
        elif 'strategy.json' in file_path:
            return default_strategy()
        return {}
    
    with open(file_path, 'r') as f:
        return json.load(f)

def save_signals(signals):
    """
    ذخیره سیگنال‌ها در فایل
    """
    # اطمینان از وجود پوشه داده‌ها
    os.makedirs('data', exist_ok=True)
    
    with open('data/signals.json', 'w') as f:
        json.dump(signals, f, indent=2)

def load_signals():
    """
    بارگذاری سیگنال‌ها از فایل
    """
    if not os.path.exists('data/signals.json'):
        return []
    
    with open('data/signals.json', 'r') as f:
        return json.load(f)

def default_coins():
    """
    لیست پیش‌فرض ارزهای دیجیتال
    """
    return [
        {"symbol": "BTC", "name": "Bitcoin"},
        {"symbol": "ETH", "name": "Ethereum"},
        {"symbol": "BNB", "name": "Binance Coin"},
        {"symbol": "ADA", "name": "Cardano"},
        {"symbol": "SOL", "name": "Solana"},
        {"symbol": "XRP", "name": "Ripple"},
        {"symbol": "DOT", "name": "Polkadot"},
        {"symbol": "DOGE", "name": "Dogecoin"},
        {"symbol": "AVAX", "name": "Avalanche"},
        {"symbol": "LINK", "name": "Chainlink"},
        {"symbol": "ATOM", "name": "Cosmos"},
        {"symbol": "UNI", "name": "Uniswap"},
        {"symbol": "MATIC", "name": "Polygon"},
        {"symbol": "LTC", "name": "Litecoin"},
        {"symbol": "ALGO", "name": "Algorand"}
    ]

def default_strategy():
    """
    تنظیمات پیش‌فرض استراتژی
    """
    return {
        "min_conditions_required": 3,
        "ema_crossover_enabled": True,
        "rsi_enabled": True,
        "macd_enabled": True,
        "bollinger_bands_enabled": True,
        "volume_analysis_enabled": True,
        "price_action_enabled": True,
        "risk_reward_ratio": 2.0,
        "partial_take_profit": True
    }
