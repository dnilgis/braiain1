#!/usr/bin/env python3
"""
Generates the GitHub Actions Job Summary.
Aligned with Braiain Speed Index v3.8 (Composite Scoring)
"""
import json
import os
import sys

if not os.path.exists('data.json'):
    print("⚠️ data.json not found - benchmark may have failed")
    sys.exit(1)

with open('data.json') as f:
    data = json.load(f)

# Header matching the Doom HUD style
print(f"### 🧠 Braiain Speed Index (v3.8)\n")
print(f"**Protocol:** Cognitive Gauntlet (Heavy Context)\n")
print(f"**Last Sync:** {data.get('last_updated', 'Unknown')}\n")

online = [r for r in data['results'] if r['status'] == 'Online']
# Sort by Braiain Score (Composite)
online.sort(key=lambda x: -x.get('braiain_score', 0))

if online:
    print("### 🏆 Live Leaderboard\n")
    # Columns matching index.html
    print("| Rank | Provider | Braiain Score | Quality | Speed | TTFT | Cost |")
    print("|:---:|---|:---:|:---:|:---:|:---:|:---:|")
    
    for i, r in enumerate(online, 1):
        medal = "👑" if i == 1 else f"#{i}"
        
        # Metrics
        b_score = r.get('braiain_score', 0)
        q_score = r.get('quality_score', 0)
        speed = f"{r['time']:.2f}s"
        ttft = f"{r.get('ttft', 0):.3f}s" if r.get('ttft') else "N/A"
        cost = f"${r.get('cost_per_request', 0):.5f}" if r.get('cost_per_request') > 0 else "FREE"
        
        # Status Icon based on Composite Score
        if b_score >= 90: icon = "🟢"
        elif b_score >= 70: icon = "🟡"
        else: icon = "🔴"
        
        print(f"| {medal} | **{r['provider']}** | {icon} **{b_score}** | {q_score} | {speed} | {ttft} | {cost} |")

    print("\n> **Scoring:** Composite of Quality (50%), Speed (30%), and Responsiveness (20%).")

# List failures if any
offline = [r for r in data['results'] if r['status'] != 'Online']
if offline:
    print("\n### ❌ Offline Nodes")
    for r in offline:
        error = r.get('error_info', {}).get('message', 'Unknown Error')
        print(f"- **{r['provider']}**: {error}")

print(f"\n*Generated automatically by Braiain Benchmark*")
