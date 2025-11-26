import os
import time 
import json
import requests
from datetime import datetime

# --- CONFIGURATION ---
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
GEMINI_MODEL = "gemini-2.0-flash-exp"

PROMPT = "Write a complete, three-paragraph summary of the history of the internet, ending with a prediction for 2030." 
MAX_TOKENS = 300 
TIMEOUT = 30
MAX_RETRIES = 2

# PRICING (per million tokens - approximate as of late 2024)
# Format: {"input": price_per_1M_input_tokens, "output": price_per_1M_output_tokens}
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00}  # Free during preview
}

# Approximate input tokens for our prompt (rough estimate)
PROMPT_TOKENS = 30

# --- UTILITY ---
def get_preview(text, max_chars=150):
    """Truncates text and sanitizes it for safe display on web."""
    if not text:
        return ""
    clean_text = text.replace('\n', ' ').replace('\t', ' ').strip()
    
    if len(clean_text) > max_chars:
        return clean_text[:max_chars] + "..."
    return clean_text

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate approximate cost for a request."""
    if model_name not in PRICING:
        return None
    
    pricing = PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return round(total_cost, 6)

def make_request_with_retry(request_func, max_retries=MAX_RETRIES):
    """Retry logic for API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return request_func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries}...")

def load_history():
    """Load existing history data."""
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            return data.get('history', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def update_history(history, new_entry):
    """Update history, keeping last 30 days."""
    history.append(new_entry)
    
    # Keep only last 30 entries
    if len(history) > 30:
        history = history[-30:]
    
    return history

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
        def make_request():
            return requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status() 
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Extract token usage
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MAX_TOKENS)
        
        # Calculate metrics
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(OPENAI_MODEL, input_tokens, output_tokens)
        
        return {
            "provider": "OpenAI", 
            "model": OPENAI_MODEL, 
            "time": duration, 
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"OpenAI API Failure: {e}")
        return {
            "provider": "OpenAI", 
            "model": OPENAI_MODEL, 
            "time": duration, 
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
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
        def make_request():
            return requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('content', [{}])[0].get('text', '')
        
        # Extract token usage
        usage = response_data.get('usage', {})
        input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('output_tokens', MAX_TOKENS)
        
        # Calculate metrics
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(ANTHROPIC_MODEL, input_tokens, output_tokens)
        
        return {
            "provider": "Anthropic", 
            "model": ANTHROPIC_MODEL, 
            "time": duration, 
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Anthropic API Failure: {e}")
        return {
            "provider": "Anthropic", 
            "model": ANTHROPIC_MODEL, 
            "time": duration, 
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def test_gemini(api_key):
    if not api_key: return None
    
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS
        }
    }
    
    start = time.monotonic()
    response_text = ""
    try:
        def make_request():
            return requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        
        # Gemini's token usage is in usageMetadata
        usage = response_data.get('usageMetadata', {})
        input_tokens = usage.get('promptTokenCount', PROMPT_TOKENS)
        output_tokens = usage.get('candidatesTokenCount', MAX_TOKENS)
        
        # Calculate metrics
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(GEMINI_MODEL, input_tokens, output_tokens)
        
        return {
            "provider": "Google", 
            "model": GEMINI_MODEL, 
            "time": duration, 
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Gemini API Failure: {e}")
        return {
            "provider": "Google", 
            "model": GEMINI_MODEL, 
            "time": duration, 
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
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

    # Load existing history
    history = load_history()
    
    # Create new history entry
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    history_entry = {
        "timestamp": timestamp,
        "results": {}
    }
    
    for result in results:
        provider = result['provider']
        history_entry["results"][provider] = {
            "time": result['time'],
            "tps": result['tokens_per_second'],
            "status": result['status'],
            "cost": result['cost_per_request']
        }
    
    # Update history
    history = update_history(history, history_entry)

    final_data = {
        "last_updated": timestamp,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "results": results,
        "history": history
    }

    with open('data.json', 'w') as f:
        json.dump(final_data, f, indent=4)
        print("\n--- SUCCESSFULLY WROTE data.json ---")

if __name__ == "__main__":
    update_json()
