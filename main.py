
import os
import requests
import speech_recognition as sr
from openai import OpenAI

# -------------------------------------------------
# Configuration
# -------------------------------------------------
HF_TOKEN = "hftoken"  # Replace with your actual Hugging Face token

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

EXIT_WORDS = {
    "bye", "goodbye", "exit", "quit", "stop", "end", "close", "see you",
    "farewell", "later", "ciao", "adios", "disconnect", "finish", "shutdown"
}

LOCATION_KEYWORDS = ["near me", "around me", "nearby", "close to me"]

recognizer = sr.Recognizer()

def listen():
    """Capture voice input and convert to text."""
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You (voice): {text}")
            return text
        except sr.UnknownValueError:
            print("⚠️ Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"⚠️ Speech recognition error: {e}")
            return ""

def get_location_via_ip():
    """Fetch current location (lat, lon) using IP geolocation."""
    try:
        # Use some free API, e.g., ipapi.co
        resp = requests.get("https://ipapi.co/json/")  # or any other IP-geolocation service
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("latitude") or data.get("lat")
            lon = data.get("longitude") or data.get("lon")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception as e:
        print(f"⚠️ Could not fetch location: {e}")
    return None, None

def preprocess_input(user_input: str) -> str:
    """If input has location keywords, append lat/lon."""
    lower = user_input.lower()
    for kw in LOCATION_KEYWORDS:
        if kw in lower:
            lat, lon = get_location_via_ip()
            if lat is not None and lon is not None:
                # Example: "food near me" → "food near me (12.34 N, 56.78 E)"
                return f"{user_input} ({lat:.2f} N, {lon:.2f} E)"
            else:
                # fallback: just send original
                return user_input
    return user_input

def main():
    print("🤖 Chatbot with Location Injection is running.")
    messages = []

    while True:
        choice = input("Press Enter for voice input, or type your message: ").strip()

        if choice == "":
            user_input = listen()
        else:
            user_input = choice

        if not user_input:
            continue

        lower = user_input.lower()

        if any(word in lower for word in EXIT_WORDS):
            print("🤖 Chatbot: Goodbye! 👋")
            break

        modified = preprocess_input(user_input)

        messages.append({"role": "user", "content": modified})

        try:
            completion = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct:together",
                messages=messages,
            )
            reply = completion.choices[0].message.content
            print(f"🤖 Chatbot: {reply}")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"⚠️ Error: {e}")
            break

if __name__ == "__main__":
    main()
