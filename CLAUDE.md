# CLAUDE.md - AI Assistant Guide for Braiain Benchmark

## Project Overview

**Braiain AI Speed Index** is a daily LLM benchmarking system that measures and compares the performance of major AI API providers. The project automatically tests multiple AI models, records their response times, tokens per second (TPS), and cost metrics, then visualizes the results in an interactive web dashboard.

### Purpose
- Benchmark AI API providers daily for latency and performance
- Track historical performance trends over time
- Provide transparent, real-time comparison data for developers choosing AI providers
- Monitor API availability and response quality

### Tech Stack
- **Backend**: Python 3.x with `requests` library
- **Frontend**: Vanilla HTML/CSS/JavaScript with Chart.js
- **Automation**: GitHub Actions (scheduled daily runs)
- **Data Storage**: JSON file (`data.json`)
- **Deployment**: Static site (GitHub Pages compatible)

## Repository Structure

```
braiain-benchmark/
├── benchmark.py           # Main benchmarking script (core logic)
├── data.json             # Generated results data (auto-updated)
├── index.html            # Main dashboard/leaderboard page
├── debug.html            # Diagnostic tool for data loading
├── test.html             # Simple data fetch tester
├── leaderboard.js        # Empty/placeholder JS file
├── README.md             # Minimal project description
├── .github/
│   └── workflows/
│       └── daily_test.yml # GitHub Actions workflow
└── CLAUDE.md             # This file
```

### File Purposes

| File | Purpose | Auto-Generated |
|------|---------|----------------|
| `benchmark.py` | Executes API tests, calculates metrics, updates JSON | No |
| `data.json` | Stores current results + 30-day history | Yes (by script) |
| `index.html` | Interactive dashboard with charts and cards | No |
| `debug.html` | Troubleshooting tool for data fetch issues | No |
| `test.html` | Simple test page for data loading | No |
| `leaderboard.js` | Currently empty placeholder | No |

## Key Components

### 1. Benchmark Script (`benchmark.py`)

**Core Functionality:**
- Tests 7 AI providers: OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Together AI
- Implements fallback system: tries multiple model versions per provider
- Measures: response time, tokens/second, cost per request
- Maintains 30-day rolling history
- Handles retries with exponential backoff (2s, 4s intervals)

**Configuration Constants (lines 7-62):**
```python
MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20241022", ...],
    # ... (fallback lists for each provider)
}

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
    # ... (pricing data for all models)
}

PROMPT = "Write a complete, three-paragraph summary of the history of the internet, ending with a prediction for 2030."
MAX_TOKENS = 300
TIMEOUT = 30
MAX_RETRIES = 2
```

**Provider Test Functions:**
- `test_openai()` - lines 104-200
- `test_anthropic()` - lines 202-301
- `test_google()` - lines 303-395
- `test_groq()` - lines 397-494
- `test_mistral()` - lines 496-591
- `test_cohere()` - lines 593-688
- `test_together()` - lines 690-755

**Key Helper Functions:**
- `make_request_with_retry()` (line 80): Implements retry logic with exponential backoff
- `calculate_cost()` (line 72): Computes cost based on token usage and pricing
- `get_preview()` (line 64): Truncates response text for previews (150 chars)
- `load_history()` (line 90): Reads existing data.json
- `update_history()` (line 98): Maintains 30-entry rolling window

### 2. Data Structure (`data.json`)

**Schema:**
```json
{
  "last_updated": "2025-11-29 01:08:01 UTC",
  "prompt": "...",
  "max_tokens": 300,
  "results": [
    {
      "provider": "Groq",
      "model": "Llama 3.1 70B",
      "time": 1.8906,
      "status": "Online" | "API FAILURE",
      "response_preview": "First 150 chars...",
      "full_response": "Complete response text",
      "tokens_per_second": 257.59,
      "output_tokens": 487,
      "cost_per_request": 0.0 | null
    }
  ],
  "history": [
    {
      "timestamp": "2025-11-29 01:08:01 UTC",
      "results": {
        "Groq": {
          "time": 1.8906,
          "tps": 257.59,
          "status": "Online",
          "cost": 0.0
        }
      }
    }
  ]
}
```

**Data Flow:**
1. `benchmark.py` runs → API tests execute
2. Results sorted: Online first (by time), then failed (by name)
3. New entry appended to history (30-entry max)
4. `data.json` written with updated results + history
5. `index.html` fetches and visualizes data

### 3. Frontend (`index.html`)

**Features:**
- Responsive card-based layout for each provider
- Interactive Chart.js visualizations (latency/TPS/cost trends)
- Real-time data loading with fetch API
- SEO-optimized with meta tags
- Graceful degradation for missing data

**Chart Types:**
- Latency over time (line chart)
- Tokens per second comparison (bar chart)
- Cost per request analysis (line chart)

## Development Workflows

### Daily Automated Benchmark

**Trigger:** GitHub Actions cron schedule (`0 0 * * *` - midnight UTC)

**Process:**
1. Workflow checks out repo (deep clone)
2. Sets up Python 3.x environment
3. Installs `requests` library
4. Runs `benchmark.py` with API keys from secrets
5. Auto-commits updated `data.json` to main branch
6. Dashboard updates automatically

**Manual Triggers:**
- Push to `main` branch
- Manual workflow dispatch via GitHub UI

### Local Development

**Running Benchmarks Locally:**
```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export GROQ_API_KEY="..."
export MISTRAL_API_KEY="..."
export COHERE_API_KEY="..."
export TOGETHER_API_KEY="..."

# Install dependencies
pip install requests

# Run benchmark
python benchmark.py
```

**Testing Frontend Locally:**
```bash
# Serve with Python (handles CORS)
python -m http.server 8000

# Or use any static server
# Open http://localhost:8000/index.html
```

**Debugging Data Issues:**
- Open `debug.html` to test data loading paths
- Open `test.html` for simple fetch diagnostics
- Check browser console for JSON parse errors

## Configuration

### Required Environment Variables

| Variable | Provider | Format | Required |
|----------|----------|--------|----------|
| `OPENAI_API_KEY` | OpenAI | `sk-...` | Optional* |
| `ANTHROPIC_API_KEY` | Anthropic | `sk-ant-...` | Optional* |
| `GEMINI_API_KEY` | Google | API key string | Optional* |
| `GROQ_API_KEY` | Groq | API key string | Optional* |
| `MISTRAL_API_KEY` | Mistral AI | API key string | Optional* |
| `COHERE_API_KEY` | Cohere | API key string | Optional* |
| `TOGETHER_API_KEY` | Together AI | API key string | Optional* |

*At least one API key required to generate meaningful data. Missing keys cause providers to be skipped.

### GitHub Secrets Configuration

**Required for automated runs:**
Set secrets in: `Repository → Settings → Secrets and variables → Actions`

Add all 7 API keys as repository secrets with exact names matching environment variables above.

## Code Conventions

### Python Style
- Functions named with `test_<provider>()` pattern
- All providers return standardized result dictionaries
- Error handling: try/except with graceful degradation
- Logging: `print()` statements for execution tracking
- Time measurement: `time.monotonic()` for accuracy

### API Request Pattern
Every provider test follows this structure:
```python
def test_provider(api_key):
    if not api_key:
        return None

    # Try each model in fallback list
    for model in MODELS["provider"]:
        start = time.monotonic()
        try:
            # Make request with retry wrapper
            response = make_request_with_retry(lambda: requests.post(...))
            response.raise_for_status()
            duration = round(time.monotonic() - start, 4)

            # Extract response and metrics
            # Return success dict

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                continue  # Try next model
            # Return failure dict
        except Exception as e:
            # Return failure dict

    # All models failed
    return failure_dict_with_sentinel_time
```

### Fallback System
- Each provider has ordered list of model versions
- On 404 error → try next model
- On other HTTP error → stop and report failure
- On timeout/network error → retry with backoff
- After all models fail → return sentinel values (`time: 99.9999`)

### Cost Calculation
- Pricing in USD per 1 million tokens
- Input and output tokens charged separately
- Free providers (Google, Groq) → `cost: 0.0`
- Failed requests → `cost: null`

## Data Structures & Types

### Result Object Schema
```python
{
    "provider": str,           # Display name (e.g., "OpenAI")
    "model": str,              # Display name (e.g., "GPT-4o Mini")
    "time": float,             # Response time in seconds
    "status": str,             # "Online" or "API FAILURE"
    "response_preview": str,   # First 150 chars of response
    "full_response": str,      # Complete API response text
    "tokens_per_second": float,# Output tokens / duration
    "output_tokens": int,      # Number of tokens generated
    "cost_per_request": float | None  # Cost in USD or None if failed
}
```

### History Entry Schema
```python
{
    "timestamp": str,          # ISO format UTC timestamp
    "results": {
        "Provider Name": {
            "time": float,
            "tps": float,
            "status": str,
            "cost": float | None
        }
    }
}
```

### Sorting Logic
Results array sorted by:
1. Status (Online first)
2. Time (ascending for Online providers)
3. Provider name (alphabetical for failed providers)

## Testing & Debugging

### Debug Tools

**`debug.html`** - Comprehensive diagnostic tool:
- Tests multiple data.json fetch paths
- Displays detailed error messages
- Shows raw JSON structure
- Interactive retry buttons

**`test.html`** - Simple fetch validator:
- Tests 3 common path variations
- Shows HTTP status codes
- Minimal, fast loading

### Common Issues

**1. API Key Missing**
- Symptom: Provider skipped entirely
- Solution: Check environment variable is set
- Code location: Each `test_*()` function checks `if not api_key: return None`

**2. Model Not Found (404)**
- Symptom: "Trying next model..." messages
- Behavior: Automatically falls back to next model
- Solution: Verify MODELS dict has current model versions

**3. Rate Limiting**
- Symptom: 429 HTTP errors
- Behavior: Retries with exponential backoff
- Solution: Reduce test frequency or check quota

**4. Timeout**
- Symptom: Request exceeds 30s
- Behavior: Retries up to MAX_RETRIES times
- Solution: Increase TIMEOUT constant or check network

**5. Data Not Loading in Browser**
- Use `debug.html` to diagnose fetch issues
- Check CORS if serving from different origin
- Verify `data.json` exists and is valid JSON

### Running Tests

**Full Benchmark Run:**
```bash
python benchmark.py
```
Expected output:
```
Testing OpenAI...
  ✓ Successfully used model: gpt-4o-mini
Testing Anthropic...
  ✗ Model claude-3-5-sonnet-20241022 not found, trying next...
  ✓ Successfully used model: claude-3-5-sonnet-20240620
...
--- SUCCESSFULLY WROTE data.json ---
```

**Validate Output:**
```bash
python -c "import json; print(json.load(open('data.json'))['last_updated'])"
```

## Working with this Repository (AI Assistant Guidelines)

### When Making Changes

**Modifying Benchmark Script:**
1. **Adding a new provider:**
   - Add entry to `MODELS` dict (line 9-27)
   - Add pricing to `PRICING` dict (line 34-60)
   - Create `test_<provider>()` function following existing pattern
   - Add to tests list in `update_json()` (line 768-776)
   - Add API key env var to workflow (`.github/workflows/daily_test.yml`)

2. **Updating model versions:**
   - Modify fallback list in `MODELS` dict
   - Update pricing if model costs changed
   - Test locally before committing

3. **Changing test parameters:**
   - Modify `PROMPT`, `MAX_TOKENS`, `TIMEOUT` constants
   - Consider impact on historical comparisons
   - Document breaking changes

**Modifying Frontend:**
1. **Adding visualizations:**
   - Use Chart.js library (already loaded)
   - Follow existing chart initialization pattern
   - Ensure responsive design (max-width: 1200px)

2. **Styling changes:**
   - All styles inline in `<style>` tags
   - Use existing CSS variables for consistency
   - Test on mobile viewports

### Git Workflow for AI Assistants

**Branches:**
- `main` - Production branch (auto-deploys)
- `claude/claude-md-*` - AI assistant feature branches

**Committing:**
```bash
# Stage changes
git add <files>

# Commit with descriptive message
git commit -m "Add X provider to benchmark suite"

# Push to feature branch
git push -u origin claude/claude-md-mikeh3ry1st0sa6n-01DmwLNB2Yzanv8XwiYbzLJb
```

**Important:**
- Always develop on the designated `claude/` branch
- Never force push to `main`
- Test locally before pushing
- `data.json` is auto-generated - avoid manual edits

### Code Quality Standards

**Python:**
- Follow existing function signature patterns
- Include error handling for all API calls
- Use `print()` for progress logging
- Round floats to 2-4 decimal places
- Prefer explicit over implicit

**JavaScript:**
- Use `async/await` for data fetching
- Handle missing data gracefully (check for null/undefined)
- Cache bust with `?t=${Date.now()}` for fresh data
- Prefer vanilla JS over frameworks

**HTML/CSS:**
- Maintain semantic HTML structure
- Keep accessibility in mind (ARIA labels, alt text)
- Use responsive units (%, em, rem)
- Inline critical CSS, external for large libraries

### Performance Considerations

**Benchmark Script:**
- Runs sequentially (not parallel) to avoid rate limits
- Each test has 30s timeout max
- 2 retries with exponential backoff
- Total runtime ~2-5 minutes for all providers

**Frontend:**
- Chart.js loaded from CDN (cached)
- Single data.json fetch per page load
- No external API calls from browser
- Minimal JavaScript execution

**Data Size:**
- `data.json` grows with history (30 entries max)
- Each entry ~200-500 bytes per provider
- Expected max size: ~100KB
- Prune history beyond 30 entries automatically

### Security Notes

**API Keys:**
- Never commit API keys to repository
- Use environment variables exclusively
- Store in GitHub Secrets for Actions
- Keys have read-only/generation permissions only

**Input Validation:**
- Prompt is static (no user input)
- JSON parsing wrapped in try/except
- HTTP errors caught and logged
- No code execution from external sources

**Dependencies:**
- Python: `requests` only (well-maintained)
- JavaScript: Chart.js from CDN (versioned)
- No other third-party dependencies

### Extending the Project

**Common Extensions:**

1. **Add metrics:**
   - Modify result dict schema
   - Update visualization in `index.html`
   - Adjust history storage format

2. **Change benchmark prompt:**
   - Update `PROMPT` constant
   - Consider creating multiple test prompts
   - Track prompt version in data.json

3. **Add provider-specific features:**
   - Extract model-specific capabilities
   - Test with different parameter sets
   - Compare prompt engineering effectiveness

4. **Export data:**
   - Add CSV export from data.json
   - Create downloadable reports
   - Integrate with analytics platforms

### Troubleshooting for AI Assistants

**Script fails with import error:**
```bash
pip install requests
```

**Workflow fails to commit:**
- Check `permissions: contents: write` in workflow
- Verify `fetch-depth: 0` in checkout step
- Ensure file_pattern matches actual output

**Data not updating on site:**
- Check if `data.json` was committed
- Verify GitHub Pages is enabled
- Clear browser cache

**All providers show "API FAILURE":**
- Verify secrets are set in GitHub
- Check secret names match env vars exactly
- Review workflow run logs for detailed errors

## Conclusion

This repository is designed for automated, low-maintenance operation. The core benchmark logic is stable and follows consistent patterns. When making changes:

1. **Preserve the fallback system** - It ensures resilience
2. **Maintain data schema compatibility** - History depends on it
3. **Test locally first** - Avoid breaking the daily automation
4. **Document breaking changes** - Update this CLAUDE.md
5. **Follow existing patterns** - Consistency aids future maintenance

For questions or clarifications, refer to:
- Inline code comments in `benchmark.py`
- HTML structure in `index.html`
- Workflow configuration in `.github/workflows/daily_test.yml`
- This CLAUDE.md document
