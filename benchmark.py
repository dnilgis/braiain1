import os
import time 
import json
import requests
from datetime import datetime

# --- CONFIGURATION ---
MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-2.0-flash-exp",
    "groq": "llama-3.1-70b-versatile",  # Free & ultra-fast
    "mistral": "mistral-large-latest",
    "cohere": "command-r-plus",
    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
}

PROMPT = "Write a complete, three-paragraph summary of the history of the internet, ending with a prediction for 2030." 
MAX_TOKENS = 300 
TIMEOUT = 30
MAX_RETRIES = 2

# PRICING (per million tokens - approximate as of late 2024)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Free preview
    "llama-3.1-70b-versatile": {"input": 0.00, "output": 0.00},  # Groq is free
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "command-r-plus": {"input": 3.00, "output": 15.00},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.18, "output": 0.18}
}

PROMPT_TOKENS = 30

# --- UTILITY ---
def get_preview(text, max_chars=150):
    if not text:
        return ""
    clean_text = text.replace('\n', ' ').replace('\t', ' ').strip()
    if len(clean_text) > max_chars:
        return clean_text[:max_chars] + "..."
    return clean_text

def calculate_cost(model_name, input_tokens, output_tokens):
    if model_name not in PRICING:
        return None
    pricing = PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)

def make_request_with_retry(request_func, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return request_func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries}...")

def load_history():
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            return data.get('history', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def update_history(history, new_entry):
    history.append(new_entry)
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
        "model": MODELS["openai"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, json=data, timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(MODELS["openai"], input_tokens, output_tokens)
        
        return {
            "provider": "OpenAI",
            "model": MODELS["openai"],
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
            "model": MODELS["openai"],
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
        "model": MODELS["anthropic"],
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=data, timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('content', [{}])[0].get('text', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('output_tokens', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(MODELS["anthropic"], input_tokens, output_tokens)
        
        return {
            "provider": "Anthropic",
            "model": MODELS["anthropic"],
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
            "model": MODELS["anthropic"],
            "time": duration,
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def test_google(api_key):
    if not api_key: return None
    
    url = f"https://generativelanguage.googleapis.com/v1/models/{MODELS['google']}:generateContent?key={api_key}"
    data = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS}
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=TIMEOUT)
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        usage = response_data.get('usageMetadata', {})
        input_tokens = usage.get('promptTokenCount', PROMPT_TOKENS)
        output_tokens = usage.get('candidatesTokenCount', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(MODELS["google"], input_tokens, output_tokens)
        
        return {
            "provider": "Google",
            "model": MODELS["google"],
            "time": duration,
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Google API Failure: {e}")
        return {
            "provider": "Google",
            "model": MODELS["google"],
            "time": duration,
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def test_groq(api_key):
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODELS["groq"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=data, timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(MODELS["groq"], input_tokens, output_tokens)
        
        return {
            "provider": "Groq",
            "model": MODELS["groq"],
            "time": duration,
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Groq API Failure: {e}")
        return {
            "provider": "Groq",
            "model": MODELS["groq"],
            "time": duration,
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def test_mistral(api_key):
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODELS["mistral"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers, json=data, timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(MODELS["mistral"], input_tokens, output_tokens)
        
        return {
            "provider": "Mistral AI",
            "model": MODELS["mistral"],
            "time": duration,
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Mistral API Failure: {e}")
        return {
            "provider": "Mistral AI",
            "model": MODELS["mistral"],
            "time": duration,
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def test_cohere(api_key):
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODELS["cohere"],
        "message": PROMPT,
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.cohere.com/v1/chat",
                headers=headers, json=data, timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('text', '')
        # Cohere uses different token field names
        usage = response_data.get('meta', {}).get('billed_units', {})
        input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('output_tokens', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(MODELS["cohere"], input_tokens, output_tokens)
        
        return {
            "provider": "Cohere",
            "model": MODELS["cohere"],
            "time": duration,
            "status": "Online",
            "response_preview": get_preview(response_text),
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Cohere API Failure: {e}")
        return {
            "provider": "Cohere",
            "model": MODELS["cohere"],
            "time": duration,
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def test_together(api_key):
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODELS["together"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.together.xyz/v1/chat/completions",
