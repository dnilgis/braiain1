name: Automated LLM Benchmarking

on:
  schedule:
    # Run every 6 hours at :00 minutes (00:00, 06:00, 12:00, 18:00 UTC)
    - cron: '0 */6 * * *'
  
  # Allow manual triggering from Actions tab
  workflow_dispatch:
    inputs:
      debug:
        description: 'Enable debug logging'
        required: false
        default: 'false'

# Cancel in-progress runs of the same workflow
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  benchmark:
    name: Run API Benchmarks
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    permissions:
      contents: write  # Required to commit and push changes
    
    steps:
      - name: 📥 Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1  # Shallow clone for faster checkout
      
      - name: 🐍 Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'  # Cache pip dependencies
      
      - name: 📦 Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests
          pip list  # Show installed packages for debugging
      
      - name: 🔍 Verify benchmark.py exists
        run: |
          if [ ! -f "benchmark.py" ]; then
            echo "❌ ERROR: benchmark.py not found!"
            exit 1
          fi
          echo "✅ benchmark.py found"
          python -c "import requests; print('✅ requests library available')"
      
      - name: 🚀 Run benchmark tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}
          TOGETHER_API_KEY: ${{ secrets.TOGETHER_API_KEY }}
        run: |
          echo "🧪 Starting benchmark tests..."
          python benchmark.py
          
          # Verify data.json was created/updated
          if [ ! -f "data.json" ]; then
            echo "❌ ERROR: data.json was not created!"
            exit 1
          fi
          
          echo "✅ data.json successfully generated"
          
          # Show file size for monitoring
          ls -lh data.json
          
          # Show number of results (for verification)
          python -c "
          import json
          with open('data.json', 'r') as f:
              data = json.load(f)
              results = data.get('results', [])
              history = data.get('history', [])
              print(f'📊 Results: {len(results)} providers')
              print(f'📈 History: {len(history)} data points')
              online = sum(1 for r in results if r.get('status') == 'Online')
              print(f'✅ Online: {online}/{len(results)} providers')
          "
      
      - name: 📊 Generate benchmark summary
        if: always()
        run: |
          if [ -f "data.json" ]; then
            python -c "
          import json
          from datetime import datetime
          
          with open('data.json', 'r') as f:
              data = json.load(f)
          
          results = data.get('results', [])
          online = [r for r in results if r.get('status') == 'Online']
          errors = [r for r in results if r.get('status') != 'Online']
          
          print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
          print('📊 BENCHMARK RESULTS SUMMARY')
          print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
          print(f'⏰ Timestamp: {data.get(\"last_updated\", \"N/A\")}')
          print(f'✅ Successful: {len(online)}/{len(results)} providers')
          print(f'❌ Failed: {len(errors)}/{len(results)} providers')
          print()
          
          if online:
              print('🏆 SUCCESSFUL TESTS (sorted by speed):')
              online.sort(key=lambda x: x.get('time', 999))
              for i, r in enumerate(online, 1):
                  time_val = r.get('time', 0)
                  tps = r.get('tokens_per_second', 0)
                  chars = r.get('character_count', 0)
                  print(f'  {i}. {r[\"provider\"]:15} | {time_val:6.2f}s | {tps:5.0f} TPS | {chars:4} chars')
          
          if errors:
              print()
              print('⚠️  FAILED TESTS:')
              for r in errors:
                  print(f'  ❌ {r[\"provider\"]:15} | {r.get(\"status\", \"Unknown error\")}')
          
          print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
            "
          else
            echo "⚠️  data.json not found, skipping summary"
          fi
      
      - name: 📤 Commit and push changes
        id: commit
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "BRAIAN Benchmark Bot"
          
          # Check if data.json changed
          if git diff --quiet data.json; then
            echo "📝 No changes to data.json"
            echo "changed=false" >> $GITHUB_OUTPUT
          else
            git add data.json
            
            # Create commit with timestamp and summary
            TIMESTAMP=$(date -u '+%Y-%m-%d %H:%M UTC')
            
            # Get provider count from data.json
            PROVIDERS=$(python -c "import json; data = json.load(open('data.json')); print(len([r for r in data.get('results', []) if r.get('status') == 'Online']))")
            
            git commit -m "🤖 Auto-update: $TIMESTAMP | $PROVIDERS providers online"
            
            echo "✅ Changes committed"
            echo "changed=true" >> $GITHUB_OUTPUT
          fi
      
      - name: 🚢 Push to repository
        if: steps.commit.outputs.changed == 'true'
        uses: ad-m/github-push-action@master
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          branch: ${{ github.ref }}
      
      - name: 📢 Job summary
        if: always()
        run: |
          echo "## 🤖 Benchmark Automation Report" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Workflow:** ${{ github.workflow }}" >> $GITHUB_STEP_SUMMARY
          echo "**Trigger:** ${{ github.event_name }}" >> $GITHUB_STEP_SUMMARY
          echo "**Timestamp:** $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          
          if [ -f "data.json" ]; then
            python -c "
          import json
          
          with open('data.json', 'r') as f:
              data = json.load(f)
          
          results = data.get('results', [])
          online = [r for r in results if r.get('status') == 'Online']
          
          print('### 📊 Results')
          print('')
          print(f'- **Total Providers:** {len(results)}')
          print(f'- **Online:** {len(online)}')
          print(f'- **Offline:** {len(results) - len(online)}')
          print('')
          
          if online:
              print('### 🏆 Performance Rankings')
              print('')
              print('| Rank | Provider | Speed | TPS | Characters |')
              print('|------|----------|-------|-----|------------|')
              online.sort(key=lambda x: x.get('time', 999))
              for i, r in enumerate(online[:7], 1):
                  time_val = r.get('time', 0)
                  tps = r.get('tokens_per_second', 0)
                  chars = r.get('character_count', 0)
                  emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
                  print(f'| {emoji} #{i} | {r[\"provider\"]} | {time_val:.2f}s | {tps:.0f} | {chars} |')
            " >> $GITHUB_STEP_SUMMARY
          else
            echo "⚠️ No data.json file found" >> $GITHUB_STEP_SUMMARY
          fi
      
      - name: ❌ Handle failure
        if: failure()
        run: |
          echo "## ⚠️ Benchmark Failed" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "The benchmark workflow encountered an error." >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Possible causes:**" >> $GITHUB_STEP_SUMMARY
          echo "- Missing or invalid API keys" >> $GITHUB_STEP_SUMMARY
          echo "- API rate limits exceeded" >> $GITHUB_STEP_SUMMARY
          echo "- Network connectivity issues" >> $GITHUB_STEP_SUMMARY
          echo "- Provider API outages" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "Check the logs above for detailed error messages." >> $GITHUB_STEP_SUMMARY
