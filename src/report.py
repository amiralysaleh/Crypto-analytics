import json
import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import ta
from utils import load_config, save_signals, load_signals
from telegram import send_message

def fetch_current_price(symbol):
    """
    دریافت قیمت فعلی ارز از اندپوینت عمومی بایننس
    """
    url = f"https://api.binance.com/api/v3/ticker/price"
    params = {'symbol': f"{symbol}USDT"}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    return float(data['price'])

def update_signal_status(signal):
    """
    بروزرسانی وضعیت سیگنال
    """
    if signal['status'] != 'active':
        return signal
    
    symbol = signal['symbol']
    current_price = fetch_current_price(symbol)
    
    # وضعیت فعلی سیگنال
    updated_signal = signal.copy()
    
    # بررسی برخورد به تارگت‌ها یا حد ضرر
    if signal['direction'] == 'buy':
        # بررسی حد ضرر
        if current_price <= signal['stop_loss']:
            updated_signal['status'] = 'stoploss'
            updated_signal['exit_price'] = current_price
            updated_signal['exit_time'] = datetime.now().isoformat()
        
        # بررسی تارگت‌ها
        else:
            if current_price >= signal['target1'] and 'target1' not in signal['targets_hit']:
                updated_signal['targets_hit'].append('target1')
            
            if current_price >= signal['target2'] and 'target2' not in signal['targets_hit']:
                updated_signal['targets_hit'].append('target2')
            
            if current_price >= signal['target3'] and 'target3' not in signal['targets_hit']:
                updated_signal['targets_hit'].append('target3')
                updated_signal['status'] = 'completed'
                updated_signal['exit_price'] = current_price
                updated_signal['exit_time'] = datetime.now().isoformat()
    
    else:  # sell signal
        # بررسی حد ضرر
        if current_price >= signal['stop_loss']:
            updated_signal['status'] = 'stoploss'
            updated_signal['exit_price'] = current_price
            updated_signal['exit_time'] = datetime.now().isoformat()
        
        # بررسی تارگت‌ها
        else:
            if current_price <= signal['target1'] and 'target1' not in signal['targets_hit']:
                updated_signal['targets_hit'].append('target1')
            
            if current_price <= signal['target2'] and 'target2' not in signal['targets_hit']:
                updated_signal['targets_hit'].append('target2')
            
            if current_price <= signal['target3'] and 'target3' not in signal['targets_hit']:
                updated_signal['targets_hit'].append('target3')
                updated_signal['status'] = 'completed'
                updated_signal['exit_price'] = current_price
                updated_signal['exit_time'] = datetime.now().isoformat()
    
    return updated_signal

def generate_status_report():
    """
    تولید گزارش وضعیت سیگنال‌ها
    """
    signals = load_signals()
    
    # آپدیت وضعیت تمام سیگنال‌های فعال
    updated_signals = []
    for signal in signals:
        updated_signal = update_signal_status(signal)
        updated_signals.append(updated_signal)
    
    # ذخیره سیگنال‌های بروزرسانی شده
    save_signals(updated_signals)
    
    # تفکیک سیگنال‌ها براساس وضعیت
    active_signals = [s for s in updated_signals if s['status'] == 'active']
    completed_signals = [s for s in updated_signals if s['status'] == 'completed' and s.get('reported', False) != True]
    stoploss_signals = [s for s in updated_signals if s['status'] == 'stoploss' and s.get('reported', False) != True]
    
    # تولید گزارش
    report = "📊 گزارش وضعیت سیگنال‌ها 📊\n\n"
    
    # سیگنال‌های فعال
    if active_signals:
        report += "🟢 سیگنال‌های فعال:\n"
        for signal in active_signals:
            current_price = fetch_current_price(signal['symbol'])
            
            if signal['direction'] == 'buy':
                profit_percent = (current_price - signal['entry_price']) / signal['entry_price'] * 100
            else:
                profit_percent = (signal['entry_price'] - current_price) / signal['entry_price'] * 100
            
            target_status = ""
            for i, target in enumerate(['target1', 'target2', 'target3']):
                if target in signal['targets_hit']:
                    target_status += f"✅ هدف {i+1} "
                else:
                    target_status += f"⬜ هدف {i+1} "
            
            report += f"#{signal['symbol']} ({signal['direction']}): {round(profit_percent, 2)}% | {target_status}\n"
        
        report += "\n"
    
    # سیگنال‌های موفق
    if completed_signals:
        report += "✅ سیگنال‌های به هدف رسیده:\n"
        for signal in completed_signals:
            if signal['direction'] == 'buy':
                profit_percent = (signal['exit_price'] - signal['entry_price']) / signal['entry_price'] * 100
            else:
                profit_percent = (signal['entry_price'] - signal['exit_price']) / signal['entry_price'] * 100
            
            report += f"#{signal['symbol']} ({signal['direction']}): +{round(profit_percent, 2)}% | هدف کامل\n"
            
            # علامت‌گذاری به عنوان گزارش شده
            signal['reported'] = True
        
        report += "\n"
    
    # سیگنال‌های حد ضرر
    if stoploss_signals:
        report += "🔴 سیگنال‌های به حد ضرر رسیده:\n"
        for signal in stoploss_signals:
            if signal['direction'] == 'buy':
                loss_percent = (signal['exit_price'] - signal['entry_price']) / signal['entry_price'] * 100
            else:
                loss_percent = (signal['entry_price'] - signal['exit_price']) / signal['entry_price'] * 100
            
            report += f"#{signal['symbol']} ({signal['direction']}): {round(loss_percent, 2)}% | حد ضرر\n"
            
            # علامت‌گذاری به عنوان گزارش شده
            signal['reported'] = True
        
        report += "\n"
    
    # آمار کلی
    total_signals = len([s for s in updated_signals if s.get('reported', False) == True or s['status'] == 'active'])
    successful_signals = len([s for s in updated_signals if s['status'] == 'completed' and len(s['targets_hit']) > 0])
    stoploss_signals_count = len([s for s in updated_signals if s['status'] == 'stoploss'])
    
    if total_signals > 0:
        success_rate = (successful_signals / total_signals) * 100
        report += f"📊 آمار کلی: از {total_signals} سیگنال، {successful_signals} موفق ({round(success_rate, 1)}%) و {stoploss_signals_count} حد ضرر\n"
    
    # ارسال گزارش به تلگرام
    if active_signals or completed_signals or stoploss_signals:
        send_message(report)
        
        # بروزرسانی وضعیت گزارش سیگنال‌ها
        save_signals(updated_signals)
    else:
        print("No signals to report.")

if __name__ == "__main__":
    generate_status_report()
