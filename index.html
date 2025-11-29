<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braiain AI Speed Index | Daily LLM Benchmarks</title>
    <meta name="description" content="Real-time AI model speed benchmarks. Compare OpenAI, Anthropic, Google Gemini, Groq, and more API latency performance updated daily.">
    
    <!-- SEO Meta Tags -->
    <meta name="keywords" content="AI benchmarks, LLM speed test, OpenAI GPT-4, Claude API, Gemini speed, Groq performance, AI API comparison">
    <meta name="author" content="Braiain AI Speed Index">
    <meta name="robots" content="index, follow">
    
    <!-- Google Analytics GA4 -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-1Y5SESHWEE"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-1Y5SESHWEE');
    </script>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #f5f5f7;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f9f9f9;
            --text-primary: #1c1c1c;
            --text-secondary: #555;
            --text-tertiary: #777;
            --border-color: #e0e0e0;
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
            --shadow-lg: 0 6px 16px rgba(0,0,0,0.15);
        }
        
        [data-theme="dark"] {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #252525;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --text-tertiary: #888;
            --border-color: #404040;
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
            --shadow-lg: 0 6px 16px rgba(0,0,0,0.5);
        }
        
        * { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: background-color 0.3s, color 0.3s;
        }
        
        header {
            background-color: #1c1c1c;
            color: white;
            text-align: center;
            padding: 30px 20px 20px;
            margin-bottom: 20px;
            position: relative;
        }
        
        .theme-toggle {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 1.2em;
            transition: all 0.3s;
        }
        
        .theme-toggle:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
        }
        
        h1 { font-size: 2.4em; margin: 0 0 8px 0; }
        .tagline { font-size: 1.1em; color: #c0c0c0; margin: 0; }
        
        .controls-bar {
            max-width: 1200px;
            margin: 20px auto;
            padding: 15px 20px;
            background: var(--bg-secondary);
            border-radius: 8px;
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }
        
        .sort-btn {
            padding: 10px 20px;
            border: 2px solid #2196f3;
            background: var(--bg-secondary);
            color: #2196f3;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
            font-size: 0.95em;
        }
        
        .sort-btn.active {
            background: #2196f3;
            color: white;
        }
        
        .sort-btn:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-sm);
        }
        
        .about-section { 
            max-width: 1200px; 
            margin: 20px auto; 
            padding: 0 20px; 
        }
        
        h2 { 
            color: var(--text-primary);
            font-size: 1.5em; 
            margin-top: 30px; 
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 8px; 
        }
        
        .data-source-indicator {
            max-width: 1200px;
            margin: 10px auto;
            padding: 10px 20px;
            text-align: center;
            font-size: 0.9em;
            border-radius: 4px;
        }
        
        .data-live {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .data-fallback {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .leaderboard-cards-wrapper {
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px; 
            justify-content: center;
            align-items: flex-start; 
        }

        .model-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            width: 100%;
            max-width: 360px;
            box-shadow: var(--shadow-md);
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
        }

        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }
        
        .rank-badge {
            background-color: #666;
            color: white;
            padding: 6px 12px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 1.2em;
            margin-right: 15px;
            flex-shrink: 0;
            min-width: 45px;
            text-align: center;
        }
        
        .rank-badge.rank-1 { 
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); 
            color: #1c1c1c;
            box-shadow: 0 2px 8px rgba(255, 215, 0, 0.4);
        }
        .rank-badge.rank-2 { 
            background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%); 
            color: #1c1c1c;
            box-shadow: 0 2px 8px rgba(192, 192, 192, 0.4);
        }
        .rank-badge.rank-3 { 
            background: linear-gradient(135deg, #cd7f32 0%, #e6a85c 100%); 
            color: #1c1c1c;
            box-shadow: 0 2px 8px rgba(205, 127, 50, 0.4);
        }
        
        .provider-model-wrapper {
            display: flex;
            flex-direction: column;
            line-height: 1.2;
            overflow: hidden; 
            flex: 1;
        }
        
        .card-provider { font-size: 1.3em; font-weight: 700; }
        .card-model { font-size: 0.9em; color: var(--text-secondary); word-break: break-word; }

        .stats-wrapper {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 15px;
            padding: 10px 0;
        }
        
        .stat-group { text-align: center; }
        
        .stat-value {
            font-size: 1.3em;
            font-weight: bold;
            display: flex;
            flex-direction: column;
            align-items: center;
            line-height: 1.1;
        }
        
        .stat-label { 
            font-size: 0.75em; 
            color: var(--text-tertiary);
            text-transform: uppercase; 
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .speed-fast, .tps-fast { color: #2e7d32; }
        .speed-medium, .tps-medium { color: #f57c00; }
        .speed-slow, .tps-slow { color: #c62828; }
        
        .status-online { color: #2e7d32; }
        .status-error { color: #c62828; }
        
        .cost-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 5px;
        }
        
        .cost-free { background-color: #e8f5e9; color: #2e7d32; }
        .cost-cheap { background-color: #fff3e0; color: #f57c00; }
        .cost-expensive { background-color: #ffebee; color: #c62828; }

        /* RESPONSE PREVIEW AT BOTTOM */
        .response-preview-wrapper {
            margin-top: 15px;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border-left: 3px solid #2196f3;
            font-size: 0.85em;
            line-height: 1.5;
            color: var(--text-secondary);
            max-height: 120px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
            transition: all 0.3s;
            text-align: left;
        }
        
        .response-preview-wrapper:hover {
            box-shadow: var(--shadow-sm);
            border-left-color: #1976d2;
        }
        
        .response-preview-wrapper::after {
            content: '👁️ Click to read full response';
            position: absolute;
            bottom: 5px;
            right: 10px;
            font-size: 0.7em;
            color: #2196f3;
            font-weight: 600;
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 3px;
        }
