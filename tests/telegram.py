import requests

# Your bot token from BotFather
bot_token = "7870837007:AAE-uxE50x1_NNCWJM6OlwMcJQWAtOUOjmY"

# Chat ID (your ID or group ID)
chat_id = "1687055799"

# Message to send
message = "⚡ Alert: Suspicious activity detected on CCTV!"

# API URL
url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

# Payload (message and chat details)
payload = {
    "chat_id": chat_id,
    "text": message
}

# Send message request
try:
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Telegram alert sent successfully!")
    else:
        print(f"❌ Failed to send alert. Status code: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")
