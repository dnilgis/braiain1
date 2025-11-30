import os
import time 
import json
import requests
from datetime import datetime, timezone

# --- CONFIGURATION ---
# Model names to try in order (fallback system)
MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620"
    ],
    "google": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro"
    ],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    "mistral": ["mistral-large-latest", "mistral-medium-latest"],
    "cohere": ["command-r-plus", "command-r"],
    "together": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"]
}

PROMPT = """Write a complete, three-paragraph summary of the history of the internet, ending with a prediction for 2030.

CRITICAL REQUIREMENTS - YOU WILL BE REJECTED IF YOU VIOLATE THESE:
1. Your response MUST be between 1000-1200 characters (COUNT AS YOU WRITE!)
2. You MUST end with a complete sentence - no partial thoughts
3. If you go over 1200 characters, you FAILED
4. If you write under 1000 characters, you FAILED
5. Plan your response to finish between characters 1000-1200

This is MANDATORY. Responses outside 1000-1200 characters will be automatically rejected.""" 
MAX_TOKENS = 280  # ~1120 chars at 4 chars/token (leaves room for variation)
MAX_CHARACTERS = 1200  # Approximately 4 chars per token
MIN_CHARACTERS = 1000  # Minimum to ensure substance 
TIMEOUT = 30
MAX_RETRIES = 2

PRICING = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    # Google (all free)
    "gemini-1.5-pro": {"input": 0.00, "output": 0.00},
    "gemini-1.5-flash": {"input": 0.00, "output": 0.00},
    "gemini-pro": {"input": 0.00, "output": 0.00},
    "gemini-pro": {"input": 0.00, "output": 0.00},
    # Groq (all free)
    "llama-3.1-70b-versatile": {"input": 0.00, "output": 0.00},
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},
    "mixtral-8x7b-32768": {"input": 0.00, "output": 0.00},
    # Others
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-medium-latest": {"input": 2.70, "output": 8.10},
    "command-r-plus": {"input": 3.00, "output": 15.00},
    "command-r": {"input": 0.50, "output": 1.50},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.18, "output": 0.18}
}

PROMPT_TOKENS = 30

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

def validate_character_count(char_count, model_name):
    """Validate character count and print appropriate message"""
    if char_count < MIN_CHARACTERS:
        print(f"  ⚠️  Response too short: {char_count} chars (minimum: {MIN_CHARACTERS})")
        return False
    elif char_count > MAX_CHARACTERS:
        print(f"  ⚠️  Response too long: {char_count} chars (maximum: {MAX_CHARACTERS})")
        return False
    else:
        print(f"  ✓ Character count within range: {char_count} chars")
        return True

def make_api_request_with_validation(api_call_func, max_attempts=3):
    """Make API request and retry if character count is out of range"""
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"  🔄 Retry attempt {attempt}/{max_attempts} due to character count...")
        
        result = api_call_func()
        
        # If it's an error result, return immediately (don't retry on API errors)
        if result and result.get('status') == 'API FAILURE':
            return result
        
        # Check character count
        if result and 'character_count' in result:
            char_count = result['character_count']
            if MIN_CHARACTERS <= char_count <= MAX_CHARACTERS:
                return result
            else:
                print(f"  ❌ Attempt {attempt} failed validation (got {char_count} chars)")
                if attempt < max_attempts:
                    print(f"  ⏳ Waiting 2 seconds before retry...")
                    time.sleep(2)
        else:
            # No character_count in result, return as-is
            return result
    
    print(f"  ❌ All {max_attempts} attempts failed character validation")
    return result  # Return last attempt even if validation failed

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

def test_openai(api_key):
    if not api_key: 
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try each model in order until one works
    for model in MODELS["openai"]:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS
        }
        
        start = time.monotonic()
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
            char_count = len(response_text)
            usage = response_data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
            output_tokens = usage.get('completion_tokens', MAX_TOKENS)
            tps = round(output_tokens / duration, 2) if duration > 0 else 0
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            print(f"  ✓ Successfully used model: {model}")
            validate_character_count(char_count, model)
            
            return {
                "provider": "OpenAI",
                "model": "GPT-4o Mini",
                "time": duration,
                "status": "Online",
                "response_preview": get_preview(response_text),
                "full_response": response_text,
                "tokens_per_second": tps,
                "output_tokens": output_tokens,
                "character_count": char_count,
                "cost_per_request": cost
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ✗ Model {model} not found, trying next...")
                continue
            else:
                duration = round(time.monotonic() - start, 4)
                print(f"OpenAI API Failure: {e}")
                return {
                    "provider": "OpenAI",
                    "model": "GPT-4o Mini",
                    "time": duration,
                    "status": "API FAILURE",
                    "response_preview": get_preview(str(e), 100),
                    "full_response": str(e),
                    "tokens_per_second": 0,
                    "output_tokens": 0,
                    "cost_per_request": None
                }
        except Exception as e:
            duration = round(time.monotonic() - start, 4)
            print(f"OpenAI API Failure: {e}")
            return {
                "provider": "OpenAI",
                "model": "GPT-4o Mini",
                "time": duration,
                "status": "API FAILURE",
                "response_preview": get_preview(str(e), 100),
                "full_response": str(e),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "cost_per_request": None
            }
    
    # All models failed
    return {
        "provider": "OpenAI",
        "model": "GPT-4o Mini",
        "time": 99.9999,
        "status": "API FAILURE",
        "response_preview": "All model versions failed",
        "full_response": "All model versions failed",
        "tokens_per_second": 0,
        "output_tokens": 0,
        "cost_per_request": None
    }

def test_anthropic(api_key):
    if not api_key: 
        return None
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Try each model in order until one works
    for model in MODELS["anthropic"]:
        data = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
        }
        
        start = time.monotonic()
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
            char_count = len(response_text)
            usage = response_data.get('usage', {})
            input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
            output_tokens = usage.get('output_tokens', MAX_TOKENS)
            tps = round(output_tokens / duration, 2) if duration > 0 else 0
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            print(f"  ✓ Successfully used model: {model}")
            validate_character_count(char_count, model)
            
            return {
                "provider": "Anthropic",
                "model": "Claude 3.5 Sonnet",
                "time": duration,
                "status": "Online",
                "response_preview": get_preview(response_text),
                "full_response": response_text,
                "tokens_per_second": tps,
                "output_tokens": output_tokens,
                "character_count": char_count,
                "cost_per_request": cost
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ✗ Model {model} not found, trying next...")
                continue  # Try next model
            else:
                # Other error, stop trying
                duration = round(time.monotonic() - start, 4)
                print(f"Anthropic API Failure: {e}")
                return {
                    "provider": "Anthropic",
                    "model": "Claude 3.5 Sonnet",
                    "time": duration,
                    "status": "API FAILURE",
                    "response_preview": get_preview(str(e), 100),
                    "full_response": str(e),
                    "tokens_per_second": 0,
                    "output_tokens": 0,
                    "cost_per_request": None
                }
        except Exception as e:
            duration = round(time.monotonic() - start, 4)
            print(f"Anthropic API Failure: {e}")
            return {
                "provider": "Anthropic",
                "model": "Claude 3.5 Sonnet",
                "time": duration,
                "status": "API FAILURE",
                "response_preview": get_preview(str(e), 100),
                "full_response": str(e),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "cost_per_request": None
            }
    
    # All models failed
    print(f"  ✗ All Anthropic models failed")
    return {
        "provider": "Anthropic",
        "model": "Claude 3.5 Sonnet",
        "time": 99.9999,
        "status": "API FAILURE",
        "response_preview": "All model versions failed",
        "full_response": "All model versions failed",
        "tokens_per_second": 0,
        "output_tokens": 0,
        "cost_per_request": None
    }

def test_google(api_key):
    if not api_key: 
        return None
    
    # Try each model with both API versions
    api_versions = ["v1", "v1beta"]
    
    for model_name in MODELS["google"]:
        for api_version in api_versions:
            print(f"  Trying {api_version}/{model_name}...")
            
            url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
            data = {
                "contents": [{"parts": [{"text": PROMPT}]}],
                "generationConfig": {"maxOutputTokens": MAX_TOKENS}
            }
            
            start = time.monotonic()
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
                char_count = len(response_text)
                usage = response_data.get('usageMetadata', {})
                input_tokens = usage.get('promptTokenCount', PROMPT_TOKENS)
                output_tokens = usage.get('candidatesTokenCount', MAX_TOKENS)
                tps = round(output_tokens / duration, 2) if duration > 0 else 0
                cost = 0.0  # Google is free
                
                print(f"  ✓ Successfully used model: {api_version}/{model_name}")
                validate_character_count(char_count, model_name)
                
                return {
                    "provider": "Google",
                    "model": "Gemini 1.5 Pro",
                    "time": duration,
                    "status": "Online",
                    "response_preview": get_preview(response_text),
                    "full_response": response_text,
                    "tokens_per_second": tps,
                    "output_tokens": output_tokens,
                    "character_count": char_count,
                    "cost_per_request": cost
                }
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  ✗ {api_version}/{model_name} not found")
                    continue  # Try next api_version/model combination
                else:
                    print(f"  ✗ HTTP error for {api_version}/{model_name}: {e.response.status_code}")
                    continue  # Try next combination
            except Exception as e:
                print(f"  ✗ Error with {api_version}/{model_name}: {str(e)[:100]}")
                continue  # Try next combination
    
    # All models failed
    print(f"  ✗ All Google models failed")
    return {
        "provider": "Google",
        "model": "Gemini 1.5 Pro",
        "time": 99.9999,
        "status": "API FAILURE",
        "response_preview": "All model versions failed",
        "full_response": "All model versions failed",
        "tokens_per_second": 0,
        "output_tokens": 0,
        "cost_per_request": None
    }

def test_groq(api_key):
    if not api_key: 
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try each model in order until one works
    for model in MODELS["groq"]:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        start = time.monotonic()
        try:
            def make_request():
                return requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, 
                    json=data, 
                    timeout=TIMEOUT
                )
            
            response = make_request_with_retry(make_request)
            response.raise_for_status()
            duration = round(time.monotonic() - start, 4)
            
            response_data = response.json()
            response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            char_count = len(response_text)
            usage = response_data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
            output_tokens = usage.get('completion_tokens', MAX_TOKENS)
            tps = round(output_tokens / duration, 2) if duration > 0 else 0
            cost = 0.0  # Groq is free
            
            print(f"  ✓ Successfully used model: {model}")
            validate_character_count(char_count, model)
            
            return {
                "provider": "Groq",
                "model": "Llama 3.1 70B",
                "time": duration,
                "status": "Online",
                "response_preview": get_preview(response_text),
                "full_response": response_text,
                "tokens_per_second": tps,
                "output_tokens": output_tokens,
                "character_count": char_count,
                "cost_per_request": cost
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [400, 404]:
                print(f"  ✗ Model {model} not available, trying next...")
                continue
            else:
                duration = round(time.monotonic() - start, 4)
                print(f"Groq API Failure: {e}")
                return {
                    "provider": "Groq",
                    "model": "Llama 3.1 70B",
                    "time": duration,
                    "status": "API FAILURE",
                    "response_preview": get_preview(str(e), 100),
                    "full_response": str(e),
                    "tokens_per_second": 0,
                    "output_tokens": 0,
                    "cost_per_request": None
                }
        except Exception as e:
            duration = round(time.monotonic() - start, 4)
            print(f"Groq API Failure: {e}")
            return {
                "provider": "Groq",
                "model": "Llama 3.1 70B",
                "time": duration,
                "status": "API FAILURE",
                "response_preview": get_preview(str(e), 100),
                "full_response": str(e),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "cost_per_request": None
            }
    
    # All models failed
    return {
        "provider": "Groq",
        "model": "Llama 3.1 70B",
        "time": 99.9999,
        "status": "API FAILURE",
        "response_preview": "All model versions failed",
        "full_response": "All model versions failed",
        "tokens_per_second": 0,
        "output_tokens": 0,
        "cost_per_request": None
    }

def test_mistral(api_key):
    if not api_key: 
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # FIXED: Try each model in order
    for model in MODELS["mistral"]:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS
        }
        
        start = time.monotonic()
        try:
            def make_request():
                return requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers, 
                    json=data, 
                    timeout=TIMEOUT
                )
            
            response = make_request_with_retry(make_request)
            response.raise_for_status()
            duration = round(time.monotonic() - start, 4)
            
            response_data = response.json()
            response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            char_count = len(response_text)
            usage = response_data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
            output_tokens = usage.get('completion_tokens', MAX_TOKENS)
            tps = round(output_tokens / duration, 2) if duration > 0 else 0
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            print(f"  ✓ Successfully used model: {model}")
            validate_character_count(char_count, model)
            
            return {
                "provider": "Mistral AI",
                "model": "Mistral Large",
                "time": duration,
                "status": "Online",
                "response_preview": get_preview(response_text),
                "full_response": response_text,
                "tokens_per_second": tps,
                "output_tokens": output_tokens,
                "character_count": char_count,
                "cost_per_request": cost
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ✗ Model {model} not found, trying next...")
                continue
            else:
                duration = round(time.monotonic() - start, 4)
                print(f"Mistral API Failure: {e}")
                return {
                    "provider": "Mistral AI",
                    "model": "Mistral Large",
                    "time": duration,
                    "status": "API FAILURE",
                    "response_preview": get_preview(str(e), 100),
                    "full_response": str(e),
                    "tokens_per_second": 0,
                    "output_tokens": 0,
                    "cost_per_request": None
                }
        except Exception as e:
            duration = round(time.monotonic() - start, 4)
            print(f"Mistral API Failure: {e}")
            return {
                "provider": "Mistral AI",
                "model": "Mistral Large",
                "time": duration,
                "status": "API FAILURE",
                "response_preview": get_preview(str(e), 100),
                "full_response": str(e),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "cost_per_request": None
            }
    
    return {
        "provider": "Mistral AI",
        "model": "Mistral Large",
        "time": 99.9999,
        "status": "API FAILURE",
        "response_preview": "All model versions failed",
        "full_response": "All model versions failed",
        "tokens_per_second": 0,
        "output_tokens": 0,
        "cost_per_request": None
    }

def test_cohere(api_key):
    if not api_key: 
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # FIXED: Try each model in order
    for model in MODELS["cohere"]:
        data = {
            "model": model,
            "message": PROMPT,
            "max_tokens": MAX_TOKENS
        }
        
        start = time.monotonic()
        try:
            def make_request():
                return requests.post(
                    "https://api.cohere.com/v1/chat",
                    headers=headers, 
                    json=data, 
                    timeout=TIMEOUT
                )
            
            response = make_request_with_retry(make_request)
            response.raise_for_status()
            duration = round(time.monotonic() - start, 4)
            
            response_data = response.json()
            response_text = response_data.get('text', '')
            char_count = len(response_text)
            usage = response_data.get('meta', {}).get('billed_units', {})
            input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
            output_tokens = usage.get('output_tokens', MAX_TOKENS)
            tps = round(output_tokens / duration, 2) if duration > 0 else 0
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            print(f"  ✓ Successfully used model: {model}")
            validate_character_count(char_count, model)
            
            return {
                "provider": "Cohere",
                "model": "Command R+",
                "time": duration,
                "status": "Online",
                "response_preview": get_preview(response_text),
                "full_response": response_text,
                "tokens_per_second": tps,
                "output_tokens": output_tokens,
                "character_count": char_count,
                "cost_per_request": cost
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ✗ Model {model} not found, trying next...")
                continue
            else:
                duration = round(time.monotonic() - start, 4)
                print(f"Cohere API Failure: {e}")
                return {
                    "provider": "Cohere",
                    "model": "Command R+",
                    "time": duration,
                    "status": "API FAILURE",
                    "response_preview": get_preview(str(e), 100),
                    "full_response": str(e),
                    "tokens_per_second": 0,
                    "output_tokens": 0,
                    "cost_per_request": None
                }
        except Exception as e:
            duration = round(time.monotonic() - start, 4)
            print(f"Cohere API Failure: {e}")
            return {
                "provider": "Cohere",
                "model": "Command R+",
                "time": duration,
                "status": "API FAILURE",
                "response_preview": get_preview(str(e), 100),
                "full_response": str(e),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "cost_per_request": None
            }
    
    return {
        "provider": "Cohere",
        "model": "Command R+",
        "time": 99.9999,
        "status": "API FAILURE",
        "response_preview": "All model versions failed",
        "full_response": "All model versions failed",
        "tokens_per_second": 0,
        "output_tokens": 0,
        "cost_per_request": None
    }

def test_together(api_key):
    if not api_key: 
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # FIXED: Use the model string directly from MODELS
    model = MODELS["together"][0]
    data = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS
    }
    
    start = time.monotonic()
    try:
        def make_request():
            return requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers, 
                json=data, 
                timeout=TIMEOUT
            )
        
        response = make_request_with_retry(make_request)
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        char_count = len(response_text)
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MAX_TOKENS)
        tps = round(output_tokens / duration, 2) if duration > 0 else 0
        cost = calculate_cost(model, input_tokens, output_tokens)
        
        print(f"  ✓ Successfully used model: {model}")
        validate_character_count(char_count, model)
        
        return {
            "provider": "Together AI",
            "model": "Llama 3.1 70B",
            "time": duration,
            "status": "Online",
            "response_preview": get_preview(response_text),
            "full_response": response_text,
            "tokens_per_second": tps,
            "output_tokens": output_tokens,
            "character_count": char_count,
            "cost_per_request": cost
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Together AI API Failure: {e}")
        return {
            "provider": "Together AI",
            "model": "Llama 3.1 70B",
            "time": duration,
            "status": "API FAILURE",
            "response_preview": get_preview(str(e), 100),
            "full_response": str(e),
            "tokens_per_second": 0,
            "output_tokens": 0,
            "cost_per_request": None
        }

def update_json():
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    google_key = os.getenv('GEMINI_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    mistral_key = os.getenv('MISTRAL_API_KEY')
    cohere_key = os.getenv('COHERE_API_KEY')
    together_key = os.getenv('TOGETHER_API_KEY')

    results = []

    tests = [
        ("OpenAI", lambda: test_openai(openai_key)),
        ("Anthropic", lambda: test_anthropic(anthropic_key)),
        ("Google", lambda: test_google(google_key)),
        ("Groq", lambda: test_groq(groq_key)),
        ("Mistral AI", lambda: test_mistral(mistral_key)),
        ("Cohere", lambda: test_cohere(cohere_key)),
        ("Together AI", lambda: test_together(together_key))
    ]

    for name, test_func in tests:
        print(f"Testing {name}...")
        try:
            res = test_func()
            if res:
                results.append(res)
        except Exception as e:
            print(f"  Skipped {name}: {e}")
            continue

    if results:
        # Sort: Online providers first (by time), then failed providers (by name)
        results.sort(key=lambda x: (x['status'] != 'Online', x['time'] if x['status'] == 'Online' else 999, x['provider']))
    else:
        print("WARNING: No successful API tests. Creating empty data file.")
        results = []

    history = load_history()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
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
    try:
        update_json()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
