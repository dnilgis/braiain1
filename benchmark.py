import os
import time
import json
import requests
from datetime import datetime

# --- CONFIGURATION (Change these names if new models come out) ---
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
GEMINI_MODEL = "gemini-1.5-flash"
PROMPT = "Explain the concept of quantum entanglement to a college student in exactly 550 words."

def test_openai(api_key):
    if not api_key: return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 100
    }
    start = time.time()
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        duration = round(time.time() - start, 4)
        # Calculate cost (Approximate based on input/output)
        return {"provider": "OpenAI", "model": OPENAI_MODEL, "time": duration, "status": "Online"}
    except Exception as e:
        print(f"OpenAI API Failure: {e}")
        # Return a high time to push it to the bottom of the list
        return {"provider": "OpenAI", "model": OPENAI_MODEL, "time": 99.9999, "status": "API FAILURE"}
        
def test_anthropic(api_key):
    if not api_key: return None
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    data = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": PROMPT}]
    }
    start = time.time()
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        response.raise_for_status()
        duration = round(time.time() - start, 2)
        return {"provider": "Anthropic", "model": ANTHROPIC_MODEL, "time": duration, "status": "Online"}
    except Exception as e:
        print(f"Anthropic API Failure: {e}")
        return {"provider": "Anthropic", "model": ANTHROPIC_MODEL, "time": 99.9999, "status": "API FAILURE"}
        
def test_gemini(api_key):
    if not api_key: return None
    # Google uses a URL parameter for the key
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = {"contents": [{"parts": [{"text": PROMPT}]}]}
    start = time.time()
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        response.raise_for_status()
        duration = round(time.time() - start, 2)
        return {"provider": "Google", "model": GEMINI_MODEL, "time": duration, "status": "Online"}
    except Exception as e:
        print(f"Gemini API Failure: {e}")
        return {"provider": "Google", "model": GEMINI_MODEL, "time": 99.9999, "status": "API FAILURE"}

def update_json():
    results = []
    
    # Get keys from environment variables (set in Phase 4)
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    # Run Tests
    print("Testing OpenAI...")
    res = test_openai(openai_key)
    if res: results.append(res)
    
    print("Testing Anthropic...")
    res = test_anthropic(anthropic_key)
    if res: results.append(res)
    
    print("Testing Google...")
    res = test_gemini(gemini_key)
    if res: results.append(res)

    # Sort by speed (fastest first)
    results.sort(key=lambda x: x['time'] if x['status'] == 'Online' else 999)

    final_data = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "results": results
    }
    
    with open("data.json", "w") as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    update_json()
