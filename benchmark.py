#!/usr/bin/env python3
"""
BRAIAN Speed Index - LLM Benchmarking Script
Tests multiple LLM providers with identical prompts to measure performance
"""

import os
import time
import json
import requests
from datetime import datetime

# Configuration
MIN_CHARACTERS = 1000
MAX_CHARACTERS = 1400
MAX_RETRIES = 3

# Multi-step analytical reasoning prompt
PROMPT = """Analyze the evolution of artificial intelligence from 2010 to 2025. Identify and explain the THREE most significant breakthroughs that fundamentally changed the field. For each breakthrough, describe its technical innovation and its broader impact on AI capabilities.

Compare and contrast deep learning architectures versus transformer architectures. Explain why transformers became dominant for language tasks despite deep learning's earlier success. Include specific technical reasons for this paradigm shift.

Based on current trends, predict the next major AI breakthrough likely to occur post-2025. Provide THREE specific technical reasons supporting your prediction and a realistic timeline."""

# Provider configurations - ALL 12 PROVIDERS
PROVIDERS = {
    "OpenAI": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "input_price": 0.150,
        "output_price": 0.600,
        "max_tokens": 1000
    },
    "Anthropic": {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-sonnet-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
        "input_price": 3.0,
        "output_price": 15.0,
        "max_tokens": 1000,
        "anthropic_version": "2023-06-01"
    },
    "Google": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "model": "gemini-1.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "input_price": 0.075,
        "output_price": 0.30,
        "max_tokens": 1000
    },
    "Groq": {
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "input_price": 0.0,
        "output_price": 0.0,
        "max_tokens": 1000
    },
    "Mistral AI": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "input_price": 2.0,
        "output_price": 6.0,
        "max_tokens": 1000
    },
    "Cohere": {
        "api_url": "https://api.cohere.ai/v1/chat",
        "model": "command-r-plus",
        "api_key_env": "COHERE_API_KEY",
        "input_price": 2.5,
        "output_price": 10.0,
        "max_tokens": 1000
    },
    "Together AI": {
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "api_key_env": "TOGETHER_API_KEY",
        "input_price": 0.88,
        "output_price": 0.88,
        "max_tokens": 1000
    },
    "DeepSeek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "input_price": 0.14,
        "output_price": 0.28,
        "max_tokens": 1000
    },
    "Fireworks": {
        "api_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "api_key_env": "FIREWORKS_API_KEY",
        "input_price": 0.90,
        "output_price": 0.90,
        "max_tokens": 1000
    },
    "Cerebras": {
        "api_url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-70b",
        "api_key_env": "CEREBRAS_API_KEY",
        "input_price": 0.60,
        "output_price": 0.60,
        "max_tokens": 1000
    }
}


def call_openai_compatible(provider_name, config, api_key):
    """Call OpenAI-compatible API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": config["max_tokens"],
        "temperature": 1.0
    }
    
    response = requests.post(config["api_url"], headers=headers, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0)
    }


def call_anthropic(config, api_key):
    """Call Anthropic API"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": config["anthropic_version"],
        "content-type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "max_tokens": config["max_tokens"],
        "messages": [{"role": "user", "content": PROMPT}]
    }
    
    response = requests.post(config["api_url"], headers=headers, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    content = result["content"][0]["text"]
    usage = result.get("usage", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }


def call_google(config, api_key):
    """Call Google Gemini API"""
    url = f"{config['api_url']}?key={api_key}"
    
    data = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {
            "maxOutputTokens": config["max_tokens"],
            "temperature": 1.0
        }
    }
    
    response = requests.post(url, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    content = result["candidates"][0]["content"]["parts"][0]["text"]
    usage = result.get("usageMetadata", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0)
    }


def call_cohere(config, api_key):
    """Call Cohere API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "message": PROMPT,
        "max_tokens": config["max_tokens"],
        "temperature": 1.0
    }
    
    response = requests.post(config["api_url"], headers=headers, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    content = result["text"]
    usage = result.get("meta", {}).get("tokens", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }


def benchmark_provider(provider_name, config):
    """Benchmark a single provider with retries"""
    api_key = os.environ.get(config["api_key_env"])
    
    if not api_key:
        return {
            "provider": provider_name,
            "model": config["model"],
            "status": "API FAILURE",
            "time": 99.9999,
            "tokens_per_second": 0,
            "output_tokens": 0,
            "character_count": 0,
            "cost_per_request": 0.0,
            "full_response": "",
            "response_preview": ""
        }
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES}...", flush=True)
            
            start_time = time.time()
            
            if provider_name == "Anthropic":
                result = call_anthropic(config, api_key)
            elif provider_name == "Google":
                result = call_google(config, api_key)
            elif provider_name == "Cohere":
                result = call_cohere(config, api_key)
            else:
                result = call_openai_compatible(provider_name, config, api_key)
            
            elapsed_time = time.time() - start_time
            
            content = result["content"].strip()
            char_count = len(content)
            
            if MIN_CHARACTERS <= char_count <= MAX_CHARACTERS:
                input_tokens = result["input_tokens"]
                output_tokens = result["output_tokens"]
                cost = (input_tokens * config["input_price"] / 1_000_000) + \
                       (output_tokens * config["output_price"] / 1_000_000)
                
                tps = output_tokens / elapsed_time if elapsed_time > 0 else 0
                preview = content[:200] + "..." if len(content) > 200 else content
                
                print(f"  ✓ Success: {elapsed_time:.2f}s, {output_tokens} tokens, {char_count} chars")
                
                return {
                    "provider": provider_name,
                    "model": config["model"],
                    "status": "Online",
                    "time": round(elapsed_time, 2),
                    "tokens_per_second": round(tps, 0),
                    "output_tokens": output_tokens,
                    "character_count": char_count,
                    "cost_per_request": round(cost, 5),
                    "full_response": content,
                    "response_preview": preview
                }
            else:
                print(f"  ✗ Invalid length: {char_count} chars")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    
    return {
        "provider": provider_name,
        "model": config["model"],
        "status": "API FAILURE",
        "time": 99.9999,
        "tokens_per_second": 0,
        "output_tokens": 0,
        "character_count": 0,
        "cost_per_request": 0.0,
        "full_response": "",
        "response_preview": ""
    }


def load_history():
    """Load existing history from data.json"""
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
            return data.get("history", [])
    except FileNotFoundError:
        return []


def save_results(results, history):
    """Save results to data.json"""
    history = history[-30:] if len(history) > 30 else history
    
    output = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "prompt": PROMPT,
        "max_tokens": 1000,
        "results": results,
        "history": history
    }
    
    with open("data.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to data.json")
    print(f"  Total providers: {len(results)}")
    print(f"  Online: {sum(1 for r in results if r['status'] == 'Online')}")


def main():
    print("=" * 60)
    print("BRAIAN SPEED INDEX - LLM Benchmark")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Providers: {len(PROVIDERS)}")
    print("=" * 60)
    
    results = []
    
    for provider_name, config in PROVIDERS.items():
        print(f"\n[{provider_name}] Testing {config['model']}...")
        result = benchmark_provider(provider_name, config)
        results.append(result)
        time.sleep(1)
    
    results.sort(key=lambda x: (x["status"] != "Online", x["time"]))
    
    history = load_history()
    history_entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "results": {
            r["provider"]: {
                "time": r["time"],
                "status": r["status"],
                "tokens_per_second": r["tokens_per_second"]
            }
            for r in results
        }
    }
    history.append(history_entry)
    
    save_results(results, history)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    online = [r for r in results if r["status"] == "Online"]
    if online:
        print("\nPerformance Rankings:")
        for i, r in enumerate(online, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            print(f"  {medal} {r['provider']:15} {r['time']:6.2f}s  {r['tokens_per_second']:4.0f} TPS")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
