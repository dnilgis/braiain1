 #!/usr/bin/env python3
"""
Generate GitHub Actions job summary from benchmark results
"""

import json
import os
import sys

if not os.path.exists('data.json'):
    print("⚠️ data.json not found - benchmark may have failed")
    sys.exit(1)

with open('data.json') as f:
    data = json.load(f)

print(f"### 🎯 Benchmark Results\n")
print(f"**Last Updated:** {data['last_updated']}\n")
print(f"**Total Providers:** {len(data['results'])}\n")

online = [r for r in data['results'] if r['status'] == 'Online']
offline = [r for r in data['results'] if r['status'] != 'Online']

if online:
    print("### ✅ Online Providers\n")
    print("| Rank | Provider | Model | Time | TPS | Cost |")
    print("|------|----------|-------|------|-----|------|")
    
    for i, r in enumerate(online, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        cost = "FREE" if r['cost_per_request'] == 0 else f"${r['cost_per_request']:.5f}"
        print(f"| {medal} | {r['provider']} | {r['model']} | {r['time']:.2f}s | {r['tokens_per_second']:.0f} | {cost} |")
    print("")

if offline:
    print("### ❌ Offline Providers\n")
    for r in offline:
        print(f"- {r['provider']} ({r['model']})")
    print("")

print(f"**History Entries:** {len(data.get('history', []))}")
