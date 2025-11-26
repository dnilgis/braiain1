import os
import time 
import json
import requests
from datetime import datetime

# --- CONFIGURATION ---
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620" 
GEMINI_MODEL = "gemini-2.5-flash"

PROMPT = "Write a complete, three-paragraph summary of the history of the internet, ending with a prediction for 2030." 
MAX_TOKENS = 300 

# --- UTILITY ---
def get_preview(text, max_chars=150):
    """Truncates text and sanitizes it for safe display on web."""
    if not text:
        return ""
    # Simple sanitization (removing newlines/tabs, replacing quotes)
    clean_text = text.replace('\n', ' ').replace('\t', ' ').strip()
    
    if len(clean_text) > max_chars:
        return clean_text[:max_chars] + "..."
    return clean_text

# --- API TEST FUNCTIONS ---

def test_openai(api_key):
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    response_text = ""
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status() 
        duration = round(time.monotonic() - start, 4)
        
        # --- NEW: Extracting the response text ---
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        return {
            "provider": "OpenAI", 
            "model": OPENAI_MODEL, 
            "time": duration, 
            "status": "Online",
            "response_preview": get_preview(response_text) # NEW
        }
    except Exception as e:
        print(f"OpenAI API Failure: {e}")
        return {
            "provider": "OpenAI", 
            "model": OPENAI_MODEL, 
            "time": 99.9999, 
            "status": "API FAILURE",
            "response_preview": f"Error: {e}" # NEW
        }

def test_anthropic(api_key):
    if not api_key: return None
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT}]
            }
        ]
    }
    
    start = time.monotonic()
    response_text = ""
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        # --- NEW: Extracting the response text ---
        response_data = response.json()
        response_text = response_data.get('content', [{}])[0].get('text', '')
        
        return {
            "provider": "Anthropic", 
            "model": ANTHROPIC_MODEL, 
            "time": duration, 
            "status": "Online",
            "response_preview": get_preview(response_text) # NEW
        }
    except Exception as e:
        print(f"Anthropic API Failure: {e}")
        return {
            "provider": "Anthropic", 
            "model": ANTHROPIC_MODEL, 
            "time": 99.9999, 
            "status": "API FAILURE",
            "response_preview": f"Error: {e}" # NEW
        }

def test_gemini(api_key):
    if not api_key: return None
    
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = {"contents": [{"parts": [{"text": PROMPT}]}]}
    
    start = time.monotonic()
    response_text = ""
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        # --- NEW: Extracting the response text ---
        response_data = response.json()
        response_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        
        return {
            "provider": "Google", 
            "model": GEMINI_MODEL, 
            "time": duration, 
            "status": "Online",
            "response_preview": get_preview(response_text) # NEW
        }
    except Exception as e:
        print(f"Gemini API Failure: {e}")
        return {
            "provider": "Google", 
            "model": GEMINI_MODEL, 
            "time": 99.9999, 
            "status": "API FAILURE",
            "response_preview": f"Error: {e}" # NEW
        }

# --- MAIN LOGIC ---

def update_json():
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')

    results = []

    print("Testing OpenAI...")
    res = test_openai(openai_key)
    if res: results.append(res)
    
    print("Testing Anthropic...")
    res = test_anthropic(anthropic_key)
    if res: results.append(res)
    
    print("Testing Google...")
    res = test_gemini(gemini_key)
    if res: results.append(res)

    if results:
        results.sort(key=lambda x: x['time'])

    final_data = {
        "last_updated": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        "results": results
    }

    with open('data.json', 'w') as f:
        json.dump(final_data, f, indent=4)
        print("\n--- SUCCESSFULLY WROTE data.json ---")

if __name__ == "__main__":
    update_json()
