#!/usr/bin/env python3
"""
BRAIAIN SPEED INDEX - Enhanced LLM Benchmarking Script v3.0
Major improvements:
- TTFT (Time to First Token) measurement
- Streaming support with token delivery metrics
- Parallel testing option for faster benchmarks
- Enhanced error handling and diagnostics
- Reliability scoring and uptime tracking
- Fixed API issues for all providers
- Rate limit detection and exponential backoff
"""

import os
import time
import json
import requests
import asyncio
import aiohttp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
import traceback

# Configuration
MIN_CHARACTERS = 800
MAX_RETRIES = 3
RETRY_DELAY = 2  # Initial delay in seconds
PARALLEL_TESTING = False  # Set to True to enable parallel testing
ENABLE_STREAMING = True  # Set to True to measure streaming performance
TIMEOUT = 90  # Increased timeout for slow providers

# Multi-step analytical reasoning prompt designed to challenge models
PROMPT = """You are tasked with analyzing the evolution of artificial intelligence from 2010 to 2025. Your response must be comprehensive and well-structured.

PART 1 - HISTORICAL ANALYSIS (2010-2025):
Identify and explain the THREE most transformative breakthroughs in AI during this period. For each breakthrough:
- Describe the core technical innovation
- Explain why it was a paradigm shift (not just incremental progress)
- Analyze its broader impact on AI capabilities and applications
- Provide specific examples of what became possible after this breakthrough

PART 2 - ARCHITECTURAL COMPARISON:
Compare and contrast deep learning architectures versus transformer architectures:
- Explain the fundamental architectural differences
- Discuss why transformers became dominant for language tasks despite deep learning's earlier success in vision
- Analyze the specific technical limitations that deep learning hit for NLP
- Explain the key innovations in transformers (attention mechanism, positional encoding, etc.) that solved these limitations
- Compare computational efficiency and scalability between the two approaches

PART 3 - FUTURE PREDICTION:
Based on current trends and technological trajectories, predict the next major AI breakthrough likely to occur post-2025:
- Provide THREE specific, defensible technical reasons supporting your prediction
- Explain what current limitations this breakthrough would address
- Propose a realistic timeline with justification
- Discuss potential obstacles that could delay or prevent this breakthrough

Your response should demonstrate deep technical understanding, logical reasoning, and the ability to synthesize complex information. Aim for 2500-3500 characters with clear structure and specific technical details."""

# Provider configurations - ENHANCED WITH DEBUG INFO
PROVIDERS = {
    "OpenAI": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "input_price": 0.150,
        "output_price": 0.600,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Anthropic": {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-sonnet-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
        "input_price": 3.0,
        "output_price": 15.0,
        "max_tokens": 1000,
        "anthropic_version": "2023-06-01",
        "supports_streaming": True
    },
    "Google": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "model": "gemini-1.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "input_price": 0.075,
        "output_price": 0.30,
        "max_tokens": 1000,
        "supports_streaming": False
    },
    "Groq": {
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "input_price": 0.0,
        "output_price": 0.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Mistral AI": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "input_price": 2.0,
        "output_price": 6.0,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Cohere": {
        "api_url": "https://api.cohere.com/v2/chat",
        "model": "command-r-plus",
        "api_key_env": "COHERE_API_KEY",
        "input_price": 2.5,
        "output_price": 10.0,
        "max_tokens": 1000,
        "supports_streaming": False
    },
    "Together AI": {
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "api_key_env": "TOGETHER_API_KEY",
        "input_price": 0.88,
        "output_price": 0.88,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "DeepSeek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "input_price": 0.14,
        "output_price": 0.28,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Fireworks": {
        "api_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "api_key_env": "FIREWORKS_API_KEY",
        "input_price": 0.90,
        "output_price": 0.90,
        "max_tokens": 1000,
        "supports_streaming": True
    },
    "Cerebras": {
        "api_url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-70b",
        "api_key_env": "CEREBRAS_API_KEY",
        "input_price": 0.60,
        "output_price": 0.60,
        "max_tokens": 1000,
        "supports_streaming": True
    }
}


def call_openai_compatible_streaming(provider_name: str, config: Dict, api_key: str) -> Dict[str, Any]:
    """Call OpenAI-compatible API with streaming support for TTFT measurement"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": config["max_tokens"],
        "temperature": 1.0,
        "stream": True
    }
    
    start_time = time.time()
    ttft = None
    content_chunks = []
    token_timestamps = []
    
    response = requests.post(
        config["api_url"], 
        headers=headers, 
        json=data, 
        timeout=TIMEOUT,
        stream=True
    )
    response.raise_for_status()
    
    for line in response.iter_lines():
        if not line:
            continue
            
        line = line.decode('utf-8')
        if not line.startswith('data: '):
            continue
            
        if line.strip() == 'data: [DONE]':
            break
            
        try:
            json_data = json.loads(line[6:])  # Remove 'data: ' prefix
            
            if 'choices' in json_data and len(json_data['choices']) > 0:
                delta = json_data['choices'][0].get('delta', {})
                content = delta.get('content', '')
                
                if content:
                    if ttft is None:
                        ttft = time.time() - start_time
                    content_chunks.append(content)
                    token_timestamps.append(time.time() - start_time)
        except json.JSONDecodeError:
            continue
    
    full_content = ''.join(content_chunks)
    total_time = time.time() - start_time
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(full_content) // 4
    
    return {
        "content": full_content,
        "input_tokens": len(PROMPT) // 4,
        "output_tokens": estimated_tokens,
        "ttft": ttft,
        "total_time": total_time,
        "streaming_smoothness": calculate_streaming_smoothness(token_timestamps) if len(token_timestamps) > 1 else None
    }


def call_openai_compatible(provider_name: str, config: Dict, api_key: str) -> Dict[str, Any]:
    """Call OpenAI-compatible API without streaming"""
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
    
    start_time = time.time()
    response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
    total_time = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "ttft": None,  # Not available without streaming
        "total_time": total_time,
        "streaming_smoothness": None
    }


def call_anthropic_streaming(config: Dict, api_key: str) -> Dict[str, Any]:
    """Call Anthropic API with streaming support"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": config["anthropic_version"],
        "content-type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "max_tokens": config["max_tokens"],
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True
    }
    
    start_time = time.time()
    ttft = None
    content_chunks = []
    token_timestamps = []
    
    response = requests.post(
        config["api_url"],
        headers=headers,
        json=data,
        timeout=TIMEOUT,
        stream=True
    )
    response.raise_for_status()
    
    for line in response.iter_lines():
        if not line:
            continue
            
        line = line.decode('utf-8')
        if not line.startswith('data: '):
            continue
            
        try:
            json_data = json.loads(line[6:])
            
            if json_data.get('type') == 'content_block_delta':
                delta = json_data.get('delta', {})
                text = delta.get('text', '')
                
                if text:
                    if ttft is None:
                        ttft = time.time() - start_time
                    content_chunks.append(text)
                    token_timestamps.append(time.time() - start_time)
        except json.JSONDecodeError:
            continue
    
    full_content = ''.join(content_chunks)
    total_time = time.time() - start_time
    
    estimated_tokens = len(full_content) // 4
    
    return {
        "content": full_content,
        "input_tokens": len(PROMPT) // 4,
        "output_tokens": estimated_tokens,
        "ttft": ttft,
        "total_time": total_time,
        "streaming_smoothness": calculate_streaming_smoothness(token_timestamps) if len(token_timestamps) > 1 else None
    }


def call_anthropic(config: Dict, api_key: str) -> Dict[str, Any]:
    """Call Anthropic API without streaming"""
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
    
    start_time = time.time()
    response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
    total_time = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    
    content = result["content"][0]["text"]
    usage = result.get("usage", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "ttft": None,
        "total_time": total_time,
        "streaming_smoothness": None
    }


def call_google(config: Dict, api_key: str) -> Dict[str, Any]:
    """Call Google Gemini API"""
    url = f"{config['api_url']}?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {
            "maxOutputTokens": config["max_tokens"],
            "temperature": 1.0
        }
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data, timeout=TIMEOUT)
    total_time = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    
    content = result["candidates"][0]["content"]["parts"][0]["text"]
    usage = result.get("usageMetadata", {})
    
    return {
        "content": content,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "ttft": None,
        "total_time": total_time,
        "streaming_smoothness": None
    }


def call_cohere(config: Dict, api_key: str) -> Dict[str, Any]:
    """Call Cohere API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": PROMPT}]
    }
    
    start_time = time.time()
    response = requests.post(config["api_url"], headers=headers, json=data, timeout=TIMEOUT)
    total_time = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    
    content = result["message"]["content"][0]["text"]
    usage = result.get("usage", {})
    tokens = usage.get("tokens", {})
    
    return {
        "content": content,
        "input_tokens": tokens.get("input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "ttft": None,
        "total_time": total_time,
        "streaming_smoothness": None
    }


def calculate_streaming_smoothness(timestamps: List[float]) -> float:
    """
    Calculate streaming smoothness score (0-1)
    Lower variance in token delivery times = smoother streaming
    """
    if len(timestamps) < 2:
        return None
    
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    avg_interval = sum(intervals) / len(intervals)
    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    
    # Normalize variance to 0-1 score (lower variance = higher score)
    # Using coefficient of variation
    if avg_interval == 0:
        return 1.0
    cv = (variance ** 0.5) / avg_interval
    smoothness = max(0, 1 - min(cv, 1))
    
    return round(smoothness, 3)


def benchmark_provider(provider_name: str, config: Dict) -> Dict[str, Any]:
    """Benchmark a single provider with enhanced error handling and TTFT measurement"""
    api_key = os.environ.get(config["api_key_env"])
    
    if not api_key:
        return create_failure_result(provider_name, config, "NO_API_KEY", "API key not found in environment")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES}...", flush=True)
            
            # Choose streaming or non-streaming based on config and feature flag
            use_streaming = ENABLE_STREAMING and config.get("supports_streaming", False)
            
            if provider_name == "Anthropic":
                result = call_anthropic_streaming(config, api_key) if use_streaming else call_anthropic(config, api_key)
            elif provider_name == "Google":
                result = call_google(config, api_key)
            elif provider_name == "Cohere":
                result = call_cohere(config, api_key)
            else:
                result = call_openai_compatible_streaming(provider_name, config, api_key) if use_streaming else call_openai_compatible(provider_name, config, api_key)
            
            content = result["content"].strip()
            char_count = len(content)
            
            # Validate response has minimum content
            if char_count >= MIN_CHARACTERS:
                input_tokens = result["input_tokens"]
                output_tokens = result["output_tokens"]
                total_time = result["total_time"]
                
                cost = (input_tokens * config["input_price"] / 1_000_000) + \
                       (output_tokens * config["output_price"] / 1_000_000)
                
                tps = output_tokens / total_time if total_time > 0 else 0
                preview = content[:200] + "..." if len(content) > 200 else content
                
                print(f"  ✓ Success: {total_time:.2f}s, {output_tokens} tokens, {char_count} chars")
                if result["ttft"]:
                    print(f"    TTFT: {result['ttft']:.3f}s")
                
                return {
                    "provider": provider_name,
                    "model": config["model"],
                    "status": "Online",
                    "time": round(total_time, 2),
                    "ttft": round(result["ttft"], 3) if result["ttft"] else None,
                    "tokens_per_second": round(tps, 0),
                    "streaming_smoothness": result["streaming_smoothness"],
                    "output_tokens": output_tokens,
                    "character_count": char_count,
                    "cost_per_request": round(cost, 5),
                    "full_response": content,
                    "response_preview": preview,
                    "error_info": None
                }
            else:
                print(f"  ✗ Invalid length: {char_count} chars (min: {MIN_CHARACTERS})")
                
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {TIMEOUT}s"
            print(f"  ✗ Timeout: {error_msg}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"
            error_body = e.response.text[:200] if e.response else "No response body"
            error_msg = f"HTTP {status_code}: {error_body}"
            print(f"  ✗ HTTP Error: {error_msg}")
            
            # Don't retry on 401/403 (auth errors)
            if status_code in [401, 403]:
                return create_failure_result(provider_name, config, "AUTH_ERROR", error_msg)
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  ✗ Error: {error_msg}")
            print(f"    Traceback: {traceback.format_exc()[:200]}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    
    return create_failure_result(provider_name, config, "MAX_RETRIES_EXCEEDED", "All retry attempts failed")


def create_failure_result(provider_name: str, config: Dict, error_type: str, error_msg: str) -> Dict[str, Any]:
    """Create a standardized failure result with error diagnostics"""
    return {
        "provider": provider_name,
        "model": config["model"],
        "status": "API FAILURE",
        "time": 99.9999,
        "ttft": None,
        "tokens_per_second": 0,
        "streaming_smoothness": None,
        "output_tokens": 0,
        "character_count": 0,
        "cost_per_request": 0.0,
        "full_response": "",
        "response_preview": "",
        "error_info": {
            "type": error_type,
            "message": error_msg,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    }


def benchmark_provider_parallel(provider_item: Tuple[str, Dict]) -> Dict[str, Any]:
    """Wrapper for parallel benchmarking"""
    provider_name, config = provider_item
    print(f"\n[{provider_name}] Testing {config['model']}...")
    return benchmark_provider(provider_name, config)


def load_history() -> List[Dict]:
    """Load existing history from data.json"""
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
            return data.get("history", [])
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print("⚠ Warning: data.json is corrupted, starting fresh history")
        return []


def calculate_reliability_scores(results: List[Dict], history: List[Dict]) -> Dict[str, float]:
    """Calculate reliability scores (uptime percentage) for each provider"""
    reliability = {}
    
    # Count successes and failures for each provider across history
    provider_stats = {}
    
    for entry in history[-30:]:  # Last 30 data points
        for provider, data in entry.get("results", {}).items():
            if provider not in provider_stats:
                provider_stats[provider] = {"online": 0, "total": 0}
            provider_stats[provider]["total"] += 1
            if data.get("status") == "Online":
                provider_stats[provider]["online"] += 1
    
    # Calculate percentages
    for provider, stats in provider_stats.items():
        if stats["total"] > 0:
            reliability[provider] = round((stats["online"] / stats["total"]) * 100, 1)
    
    return reliability


def save_results(results: List[Dict], history: List[Dict]):
    """Save results to data.json with enhanced metadata"""
    # Limit history to last 30 entries
    history = history[-30:] if len(history) > 30 else history
    
    # Calculate reliability scores
    reliability_scores = calculate_reliability_scores(results, history)
    
    output = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "prompt": PROMPT,
        "max_tokens": 1000,
        "results": results,
        "history": history,
        "reliability_scores": reliability_scores,
        "metadata": {
            "version": "3.0",
            "parallel_testing": PARALLEL_TESTING,
            "streaming_enabled": ENABLE_STREAMING,
            "timeout": TIMEOUT,
            "retries": MAX_RETRIES
        }
    }
    
    with open("data.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to data.json")
    print(f"  Total providers: {len(results)}")
    online_count = sum(1 for r in results if r['status'] == 'Online')
    print(f"  Online: {online_count}")
    print(f"  Offline: {len(results) - online_count}")


def main():
    print("=" * 70)
    print("BRAIAIN SPEED INDEX - Enhanced LLM Benchmark v3.0")
    print("=" * 70)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Providers: {len(PROVIDERS)}")
    print(f"Parallel Testing: {'ENABLED' if PARALLEL_TESTING else 'DISABLED'}")
    print(f"Streaming: {'ENABLED' if ENABLE_STREAMING else 'DISABLED'}")
    print("=" * 70)
    
    results = []
    
    if PARALLEL_TESTING:
        print("\n🚀 Running parallel benchmarks...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(benchmark_provider_parallel, item): item[0] 
                      for item in PROVIDERS.items()}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"✗ Parallel execution error: {e}")
    else:
        print("\n🔄 Running sequential benchmarks...")
        for provider_name, config in PROVIDERS.items():
            print(f"\n[{provider_name}] Testing {config['model']}...")
            result = benchmark_provider(provider_name, config)
            results.append(result)
            time.sleep(1)  # Small delay between sequential tests
    
    # Sort results
    results.sort(key=lambda x: (x["status"] != "Online", x["time"]))
    
    # Update history
    history = load_history()
    history_entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "results": {
            r["provider"]: {
                "time": r["time"],
                "status": r["status"],
                "tokens_per_second": r["tokens_per_second"],
                "ttft": r["ttft"],
                "streaming_smoothness": r["streaming_smoothness"]
            }
            for r in results
        }
    }
    history.append(history_entry)
    
    save_results(results, history)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    online = [r for r in results if r["status"] == "Online"]
    if online:
        print("\nPerformance Rankings:")
        for i, r in enumerate(online, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            ttft_str = f" | TTFT: {r['ttft']:.3f}s" if r['ttft'] else ""
            smooth_str = f" | Smooth: {r['streaming_smoothness']:.2f}" if r['streaming_smoothness'] else ""
            print(f"  {medal} {r['provider']:15} {r['time']:6.2f}s  {r['tokens_per_second']:4.0f} TPS{ttft_str}{smooth_str}")
    
    # Show failures with error info
    failures = [r for r in results if r["status"] != "Online"]
    if failures:
        print("\n⚠ Failed Providers:")
        for r in failures:
            error_info = r.get("error_info", {})
            error_type = error_info.get("type", "UNKNOWN")
            error_msg = error_info.get("message", "No details")[:80]
            print(f"  ✗ {r['provider']:15} {error_type}: {error_msg}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
