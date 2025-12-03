import os
import time 
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

# --- CONFIGURATION ---
MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620"
    ],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    "mistral": ["mistral-large-latest", "mistral-medium-latest"],
    "cohere": ["command-r-plus", "command-r"],
    "together": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"]
}

PROMPT = """Write a complete, three-paragraph summary of the history of the internet, ending with a prediction for 2030.

ABSOLUTE REQUIREMENTS - YOUR RESPONSE WILL BE REJECTED IF YOU VIOLATE THESE:

YOU MUST WRITE EXACTLY 1000-1200 CHARACTERS. COUNT EVERY SINGLE CHARACTER.

WRITE EXACTLY THREE PARAGRAPHS. STOP WHEN YOU REACH 1200 CHARACTERS.

TARGET LENGTH: 1100 CHARACTERS

END WITH A COMPLETE SENTENCE AND A PERIOD (.)

DO NOT STOP WRITING UNTIL YOU REACH AT LEAST 1000 CHARACTERS. THIS IS MANDATORY.

Your response must be substantive and detailed. DO NOT write short summaries."""

# Optimized token limits per provider
MODEL_MAX_TOKENS = {
    "groq": 650,
    "together": 250,
    "openai": 240,
    "mistral": 260,
    "anthropic": 300,
    "google": 300,
    "cohere": 300
}

# Groq-specific sampling parameters
GROQ_SAMPLING_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "frequency_penalty": 0.3
}

# Validation constants
MIN_CHARACTERS = 1000
MAX_CHARACTERS = 1200
TIMEOUT = 30
MAX_RETRIES = 2
VALIDATION_ATTEMPTS = 3

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "gemini-1.5-pro": {"input": 0.00, "output": 0.00},
    "gemini-1.5-flash": {"input": 0.00, "output": 0.00},
    "gemini-pro": {"input": 0.00, "output": 0.00},
    "llama-3.1-70b-versatile": {"input": 0.00, "output": 0.00},
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},
    "mixtral-8x7b-32768": {"input": 0.00, "output": 0.00},
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-medium-latest": {"input": 2.70, "output": 8.10},
    "command-r-plus": {"input": 3.00, "output": 15.00},
    "command-r": {"input": 0.50, "output": 1.50},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.18, "output": 0.18}
}

PROMPT_TOKENS = 30


@dataclass
class BenchmarkResult:
    provider: str
    model: str
    time: float
    status: str
    response_preview: str
    full_response: str
    tokens_per_second: float
    output_tokens: int
    character_count: int
    cost_per_request: Optional[float]


def get_preview(text: str) -> str:
    """Return text for preview (CSS handles overflow)"""
    if not text:
        return ""
    return text.replace('\n', ' ').replace('\t', ' ').strip()


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Calculate API request cost based on token usage"""
    if model_name not in PRICING:
        return None
    pricing = PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def validate_character_count(char_count: int) -> bool:
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


def make_request_with_retry(request_func: Callable, max_retries: int = MAX_RETRIES):
    """Retry HTTP requests with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return request_func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
            time.sleep(wait_time)


def create_error_result(provider: str, model_display: str, error: Exception, duration: float) -> Dict:
    """Create a standardized error result"""
    return {
        "provider": provider,
        "model": model_display,
        "time": duration,
        "status": "API FAILURE",
        "response_preview": get_preview(str(error)),
        "full_response": str(error),
        "tokens_per_second": 0,
        "output_tokens": 0,
        "character_count": 0,
        "cost_per_request": None
    }


def test_api_with_fallback(
    provider: str,
    model_display: str,
    models_list: List[str],
    make_request_func: Callable,
    parse_response_func: Callable,
    is_free: bool = False
) -> Dict:
    """Generic API testing with model fallback and validation retry"""
    
    for model in models_list:
        print(f"  Trying model: {model}")
        
        for attempt in range(1, VALIDATION_ATTEMPTS + 1):
            if attempt > 1:
                print(f"  🔄 Retry attempt {attempt}/{VALIDATION_ATTEMPTS} due to character count...")
                time.sleep(2)
            
            start = time.monotonic()
            try:
                response, duration = make_request_func(model)
                result = parse_response_func(response, model, duration, is_free)
                
                char_count = result.get('character_count', 0)
                if MIN_CHARACTERS <= char_count <= MAX_CHARACTERS:
                    print(f"  ✓ Successfully used model: {model}")
                    validate_character_count(char_count)
                    return result
                else:
                    print(f"  ❌ Attempt {attempt} failed validation (got {char_count} chars)")
                    if attempt < VALIDATION_ATTEMPTS:
                        continue
                    else:
                        print(f"  ⚠️  Using result despite validation failure")
                        return result
                        
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  ✗ Model {model} not found, trying next...")
                    break
                else:
                    duration = time.monotonic() - start
                    print(f"{provider} API Failure: {e}")
                    return create_error_result(provider, model_display, e, duration)
            except Exception as e:
                duration = time.monotonic() - start
                print(f"{provider} API Failure: {e}")
                return create_error_result(provider, model_display, e, duration)
    
    print(f"  ✗ All {provider} models failed")
    return create_error_result(provider, model_display, Exception("All model versions failed"), 99.9999)


def create_standard_result(provider: str, model_display: str, response_text: str, 
                          duration: float, output_tokens: int, input_tokens: int, 
                          model_name: str, is_free: bool = False) -> Dict:
    """Create standardized result dictionary"""
    char_count = len(response_text)
    tps = round(output_tokens / duration, 2) if duration > 0 else 0
    cost = 0.0 if is_free else calculate_cost(model_name, input_tokens, output_tokens)
    
    return {
        "provider": provider,
        "model": model_display,
        "time": duration,
        "status": "Online",
        "response_preview": get_preview(response_text),
        "full_response": response_text,
        "tokens_per_second": tps,
        "output_tokens": output_tokens,
        "character_count": char_count,
        "cost_per_request": cost
    }


def test_openai(api_key: str) -> Optional[Dict]:
    """Test OpenAI API"""
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    def make_request(model: str):
        data = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MODEL_MAX_TOKENS["openai"]
        }
        
        start = time.monotonic()
        response = make_request_with_retry(
            lambda: requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        )
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        return response, duration
    
    def parse_response(response, model: str, duration: float, is_free: bool):
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MODEL_MAX_TOKENS["openai"])
        
        return create_standard_result(
            "OpenAI", "GPT-4o Mini", response_text, duration,
            output_tokens, input_tokens, model, is_free
        )
    
    return test_api_with_fallback(
        "OpenAI", "GPT-4o Mini", MODELS["openai"],
        make_request, parse_response
    )


def test_anthropic(api_key: str) -> Optional[Dict]:
    """Test Anthropic API"""
    if not api_key:
        return None
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    def make_request(model: str):
        data = {
            "model": model,
            "max_tokens": MODEL_MAX_TOKENS["anthropic"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
        }
        
        start = time.monotonic()
        response = make_request_with_retry(
            lambda: requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        )
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        return response, duration
    
    def parse_response(response, model: str, duration: float, is_free: bool):
        response_data = response.json()
        response_text = response_data.get('content', [{}])[0].get('text', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('output_tokens', MODEL_MAX_TOKENS["anthropic"])
        
        return create_standard_result(
            "Anthropic", "Claude 3.5 Sonnet", response_text, duration,
            output_tokens, input_tokens, model, is_free
        )
    
    return test_api_with_fallback(
        "Anthropic", "Claude 3.5 Sonnet", MODELS["anthropic"],
        make_request, parse_response
    )


def test_groq(api_key: str) -> Optional[Dict]:
    """Test Groq API"""
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    def make_request(model: str):
        data = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MODEL_MAX_TOKENS["groq"],
            "stop": None,
            **GROQ_SAMPLING_PARAMS
        }
        
        start = time.monotonic()
        response = make_request_with_retry(
            lambda: requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        )
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        return response, duration
    
    def parse_response(response, model: str, duration: float, is_free: bool):
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MODEL_MAX_TOKENS["groq"])
        
        return create_standard_result(
            "Groq", "Llama 3.3 70B", response_text, duration,
            output_tokens, input_tokens, model, True
        )
    
    return test_api_with_fallback(
        "Groq", "Llama 3.3 70B", MODELS["groq"],
        make_request, parse_response, is_free=True
    )


def test_google(api_key: str) -> Optional[Dict]:
    """Test Google Gemini API"""
    if not api_key:
        return None
    
    for model_name in MODELS["google"]:
        for api_version in ["v1", "v1beta"]:
            print(f"  Trying {api_version}/{model_name}...")
            
            url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
            data = {
                "contents": [{"parts": [{"text": PROMPT}]}],
                "generationConfig": {"maxOutputTokens": MODEL_MAX_TOKENS["google"]}
            }
            
            start = time.monotonic()
            try:
                response = make_request_with_retry(
                    lambda: requests.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=data,
                        timeout=TIMEOUT
                    )
                )
                response.raise_for_status()
                duration = round(time.monotonic() - start, 4)
                
                response_data = response.json()
                response_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                char_count = len(response_text)
                usage = response_data.get('usageMetadata', {})
                input_tokens = usage.get('promptTokenCount', PROMPT_TOKENS)
                output_tokens = usage.get('candidatesTokenCount', MODEL_MAX_TOKENS["google"])
                
                if MIN_CHARACTERS <= char_count <= MAX_CHARACTERS:
                    print(f"  ✓ Successfully used model: {api_version}/{model_name}")
                    validate_character_count(char_count)
                    
                    return create_standard_result(
                        "Google", "Gemini 1.5 Pro", response_text, duration,
                        output_tokens, input_tokens, model_name, True
                    )
                else:
                    print(f"  ⚠️  {api_version}/{model_name} returned {char_count} chars")
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  ✗ {api_version}/{model_name} not found")
                    continue
                else:
                    print(f"  ✗ HTTP error for {api_version}/{model_name}: {e.response.status_code}")
                    continue
            except Exception as e:
                print(f"  ✗ Error with {api_version}/{model_name}: {str(e)[:100]}")
                continue
    
    print(f"  ✗ All Google models failed")
    return create_error_result("Google", "Gemini 1.5 Pro", Exception("All model versions failed"), 99.9999)


def test_mistral(api_key: str) -> Optional[Dict]:
    """Test Mistral AI API"""
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    def make_request(model: str):
        data = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MODEL_MAX_TOKENS["mistral"]
        }
        
        start = time.monotonic()
        response = make_request_with_retry(
            lambda: requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        )
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        return response, duration
    
    def parse_response(response, model: str, duration: float, is_free: bool):
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MODEL_MAX_TOKENS["mistral"])
        
        return create_standard_result(
            "Mistral AI", "Mistral Large", response_text, duration,
            output_tokens, input_tokens, model, is_free
        )
    
    return test_api_with_fallback(
        "Mistral AI", "Mistral Large", MODELS["mistral"],
        make_request, parse_response
    )


def test_cohere(api_key: str) -> Optional[Dict]:
    """Test Cohere API"""
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    def make_request(model: str):
        data = {
            "model": model,
            "message": PROMPT,
            "max_tokens": MODEL_MAX_TOKENS["cohere"]
        }
        
        start = time.monotonic()
        response = make_request_with_retry(
            lambda: requests.post(
                "https://api.cohere.com/v1/chat",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        )
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        return response, duration
    
    def parse_response(response, model: str, duration: float, is_free: bool):
        response_data = response.json()
        response_text = response_data.get('text', '')
        usage = response_data.get('meta', {}).get('billed_units', {})
        input_tokens = usage.get('input_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('output_tokens', MODEL_MAX_TOKENS["cohere"])
        
        return create_standard_result(
            "Cohere", "Command R+", response_text, duration,
            output_tokens, input_tokens, model, is_free
        )
    
    return test_api_with_fallback(
        "Cohere", "Command R+", MODELS["cohere"],
        make_request, parse_response
    )


def test_together(api_key: str) -> Optional[Dict]:
    """Test Together AI API"""
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    model = MODELS["together"][0]
    data = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MODEL_MAX_TOKENS["together"]
    }
    
    start = time.monotonic()
    try:
        response = make_request_with_retry(
            lambda: requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=TIMEOUT
            )
        )
        response.raise_for_status()
        duration = round(time.monotonic() - start, 4)
        
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', PROMPT_TOKENS)
        output_tokens = usage.get('completion_tokens', MODEL_MAX_TOKENS["together"])
        
        print(f"  ✓ Successfully used model: {model}")
        validate_character_count(len(response_text))
        
        return create_standard_result(
            "Together AI", "Llama 3.1 70B", response_text, duration,
            output_tokens, input_tokens, model, False
        )
    except Exception as e:
        duration = round(time.monotonic() - start, 4)
        print(f"Together AI API Failure: {e}")
        return create_error_result("Together AI", "Llama 3.1 70B", e, duration)


def load_history() -> List[Dict]:
    """Load historical benchmark data"""
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            return data.get('history', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def update_history(history: List[Dict], new_entry: Dict) -> List[Dict]:
    """Update history with new entry, keeping last 30"""
    history.append(new_entry)
    return history[-30:]


def update_json():
    """Main function to run all benchmarks and update data.json"""
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'google': os.getenv('GEMINI_API_KEY'),
        'groq': os.getenv('GROQ_API_KEY'),
        'mistral': os.getenv('MISTRAL_API_KEY'),
        'cohere': os.getenv('COHERE_API_KEY'),
        'together': os.getenv('TOGETHER_API_KEY')
    }

    results = []
    tests = [
        ("OpenAI", lambda: test_openai(api_keys['openai'])),
        ("Anthropic", lambda: test_anthropic(api_keys['anthropic'])),
        ("Google", lambda: test_google(api_keys['google'])),
        ("Groq", lambda: test_groq(api_keys['groq'])),
        ("Mistral AI", lambda: test_mistral(api_keys['mistral'])),
        ("Cohere", lambda: test_cohere(api_keys['cohere'])),
        ("Together AI", lambda: test_together(api_keys['together']))
    ]

    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            res = test_func()
            if res:
                results.append(res)
        except Exception as e:
            print(f"  Skipped {name}: {e}")
            continue

    # Sort: Online first, then by time, then by provider name
    if results:
        results.sort(key=lambda x: (
            x['status'] != 'Online',
            x['time'] if x['status'] == 'Online' else 999,
            x['provider']
        ))
    else:
        print("WARNING: No successful API tests. Creating empty data file.")

    # Update history
    history = load_history()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    history_entry = {
        "timestamp": timestamp,
        "results": {
            result['provider']: {
                "time": result['time'],
                "tps": result['tokens_per_second'],
                "status": result['status'],
                "cost": result['cost_per_request']
            }
            for result in results
        }
    }
    
    history = update_history(history, history_entry)

    # Write final data
    final_data = {
        "last_updated": timestamp,
        "prompt": PROMPT,
        "max_tokens": MODEL_MAX_TOKENS.get("openai", 300),
        "results": results,
        "history": history
    }

    with open('data.json', 'w') as f:
        json.dump(final_data, f, indent=4)
        print("\n✅ SUCCESSFULLY WROTE data.json")


if __name__ == "__main__":
    try:
        update_json()
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
