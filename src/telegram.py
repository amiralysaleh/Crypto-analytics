import os
import requests

def send_message(message):
    """
    ارسال پیام به کانال تلگرام
    """
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Error: Telegram bot token or chat ID not set.")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")
        return False
    
    return True

def send_chart_image(image_path, caption=""):
    """
    ارسال تصویر چارت به کانال تلگرام
    """
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Error: Telegram bot token or chat ID not set.")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    with open(image_path, 'rb') as image_file:
        files = {'photo': image_file}
        data = {
            "chat_id": chat_id,
            "caption": caption
        }
        
        response = requests.post(url, data=data, files=files)
    
    if response.status_code != 200:
        print(f"Failed to send image: {response.text}")
        return False
    
    return True
