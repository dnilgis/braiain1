// CRASH PROTECTION WRAPPER
try {
    const AFFILIATE_LINKS = {
        'OpenAI': 'https://link.openai.com/signup?ref=your-openai-affiliate-id', 
        'Google': 'https://cloud.google.com/gemini/signup?ref=your-google-affiliate-id', 
        'Anthropic': 'https://www.anthropic.com/signup?ref=your-anthropic-affiliate-id'
    };

    function getAffiliateLink(provider) {
        return AFFILIATE_LINKS[provider] || '#';
    }

    function formatResults(results) {
        let html = '';

        results.forEach((item, index) => {
            const modelSlug = (item.provider + '-' + item.model).toLowerCase().replace(/[^a-z0-9-]+/g, '-');
            const rank = index + 1;
            
            let statusClass = item.status === 'Online' ? 'status-online' : 'status-error';
            let speedClass = item.time < 1.0 ? 'speed-fast' : (item.time < 5.0 ? 'speed-medium' : '');

            const isWinner = (index === 0 && item.status === 'Online'); 
            
            // Safe check for previous_time
            let previousTime = 'Prev: N/A';
            if (item.previous_time !== undefined && item.previous_time !== null) {
                previousTime = `Prev: ${item.previous_time}s`;
            }

            let affiliateButtonHTML = '';
            if (item.status === 'Online') {
                const link = getAffiliateLink(item.provider);
                const buttonText = isWinner ? '🥇 RUN THIS MODEL NOW' : 'RUN THIS MODEL';
                if (link && link !== '#') {
                     affiliateButtonHTML = `<a href="${link}" target="_blank" class="affiliate-btn">${buttonText}</a>`;
                }
            } else {
                affiliateButtonHTML = `<a class="affiliate-btn" style="background-color: #c62828;">API FAILURE</a>`;
            }

            const responseText = item.full_response || item.response_preview || "No response text available.";

            html += `
                <div class="model-card" id="${modelSlug}">
                    <div class="card-header">
                        <span class="rank-badge">#${rank}</span>
                        <div class="provider-model-wrapper">
                            <div class="card-provider">${item.provider}</div>
                            <div class="card-model">${item.model}</div>
                        </div>
                    </div>
                    
                    <div class="stats-wrapper">
                        <div class="stat-group">
                            <div class="stat-label">Current Speed</div>
                            <div class="stat-value ${speedClass}">
                                ${item.time}s
                                <div class="last-speed">${previousTime}</div>
                            </div>
                        </div>
                        <div class="stat-group">
                            <div class="stat-label">Status</div>
                            <div class="stat-value ${statusClass}">${item.status}</div>
                        </div>
                    </div>
                    
                    ${affiliateButtonHTML}

                    <div class="response-content-wrapper">
                        <div class="full-response-text">${responseText}</div>
                    </div>
                </div>
            `;
        });

        return html;
    }

    async function fetchData() {
        const paths = ['data.json', '/data.json']; 
        const cacheBuster = `?t=${new Date().getTime()}`;
        
        for (const path of paths) {
            try {
                const response = await fetch(path + cacheBuster); 
                if (!response.ok) throw new Error(`Status ${response.status}`);
                const data = await response.json();
                
                const resultsContainer = document.getElementById('results-container');
                const lastUpdatedElement = document.getElementById('last-updated');

                if (resultsContainer && data.results) {
                    resultsContainer.innerHTML = formatResults(data.results);
                }
                if (lastUpdatedElement && data.last_updated) {
                    lastUpdatedElement.textContent = `Last Updated: ${data.last_updated}`;
                }
                return; // Success!

            } catch (error) {
                console.warn('Data load attempt failed:', error);
            }
        }
        document.getElementById('results-container').innerHTML = `<p class="error-message" style="text-align: center; color: #c62828; padding: 20px; font-weight: bold; background: #fff; width: 100%;">
            🔴 FATAL ERROR: Live data failed to load.
        </p>`;
    }

    fetchData(); 

} catch (criticalError) {
    // If JS crashes, this will display the error message instead of 'Loading...'
    document.body.innerHTML = `<h1 style="color:red; text-align:center; margin-top:50px;">CRITICAL SCRIPT ERROR</h1><p style="text-align:center;">${criticalError.message}</p>`;
}
