const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const modeSelect = document.getElementById('modeSelect');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');

searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

// Placeholder text adapts to selected mode
const PLACEHOLDERS = {
    image: "Enter your search query (e.g., 'cute cat', 'dog playing')",
    video: "Describe a scene (e.g., 'a woman applying lipstick', 'archery')",
};
modeSelect.addEventListener('change', () => {
    searchInput.placeholder = PLACEHOLDERS[modeSelect.value] || PLACEHOLDERS.image;
});

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        showError('Please enter a search query');
        return;
    }

    loading.classList.remove('hidden');
    error.classList.add('hidden');
    results.innerHTML = '';

    // Mode-driven routing
    const mode = modeSelect.value;   // 'image' | 'video'
    let endpoint, reqBody, searchType, refImageId = null;

    if (mode === 'video') {
        // Video mode = text-to-video via Qwen3-VL-Embedding (no Similar/Material)
        endpoint = '/api/search-video';
        reqBody = { query, top_k: 10 };
        searchType = 'video';
    } else {
        // Image mode keeps existing Similar/Material conventions
        const isSimilarSearch  = query.startsWith('Similar to:');
        const isMaterialSearch = query.startsWith('Material:');
        if (isSimilarSearch) {
            refImageId = query.replace('Similar to:', '').trim();
            endpoint = '/api/search-similar';
            reqBody = { image_id: refImageId, top_k: 10 };
            searchType = 'similar';
        } else if (isMaterialSearch) {
            refImageId = query.replace('Material:', '').trim();
            endpoint = '/api/search-similar-material';
            reqBody = { image_id: refImageId, top_k: 10 };
            searchType = 'material';
        } else {
            endpoint = '/api/search';
            reqBody = { query, top_k: 10 };
            searchType = 'text';
        }
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqBody)
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || 'Search failed');
        }
        const data = await response.json();
        displayResults(data.results, refImageId, searchType,
                        data.reference_image_url, data.latency_ms, data.cache);
    } catch (err) {
        showError('Search failed: ' + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

async function findSimilar(e) {
    const imageId = e.target.dataset.imageId;

    loading.classList.remove('hidden');
    error.classList.add('hidden');
    results.innerHTML = '';

    try {
        const response = await fetch('/api/search-similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId, top_k: 10 })
        });

        if (!response.ok) throw new Error('Similar search failed');

        const data = await response.json();
        searchInput.value = `Similar to: ${imageId}`;
        displayResults(data.results, imageId, 'similar', data.reference_image_url);
    } catch (err) {
        showError('Similar search failed: ' + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

async function findMaterial(e) {
    const imageId = e.target.dataset.imageId;

    loading.classList.remove('hidden');
    error.classList.add('hidden');
    results.innerHTML = '';

    try {
        const response = await fetch('/api/search-similar-material', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId, top_k: 10 })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || 'Material search failed');
        }

        const data = await response.json();
        searchInput.value = `Material: ${imageId}`;
        displayResults(data.results, imageId, 'material', data.reference_image_url);
    } catch (err) {
        showError('Material search failed: ' + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

function displayResults(items, referenceImageId = null, searchType = 'text',
                        referenceImageUrl = null, latencyMs = null, cacheStatus = null) {
    results.innerHTML = '';

    if (items.length === 0) {
        results.innerHTML = '<p style="color: #333; text-align: center; grid-column: 1 / -1;">No results found</p>';
        return;
    }

    // 1. Reference card (only for similar/material searches in image mode)
    if (referenceImageId) {
        const refImgUrl = referenceImageUrl || `/images/animals/${referenceImageId}`;
        const refCard = document.createElement('div');
        refCard.className = 'result-card reference-card';
        refCard.innerHTML = `
            <div class="reference-header">Reference Image</div>
            <img src="${refImgUrl}" alt="${referenceImageId}">
            <div class="result-info">
                <div class="image-id" style="color: #333; font-weight: bold; font-size: 0.95rem;">ID: ${referenceImageId}</div>
            </div>
        `;
        results.appendChild(refCard);
    }

    // 2. Result cards — branch on searchType for image vs video rendering
    items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'result-card';

        if (searchType === 'video') {
            // HTML5 <video controls>; preload="metadata" avoids downloading entire clip on page load
            card.innerHTML = `
                <video controls preload="metadata" src="${item.video_url}" class="result-video"></video>
                <div class="result-info">
                    <div class="rank">Rank #${item.rank}</div>
                    <div class="score">Score: ${item.score.toFixed(4)}</div>
                    <div class="image-id" title="${item.video_id}">${item.video_id.split('/').pop()}</div>
                </div>
            `;
        } else {
            card.innerHTML = `
                <img src="${item.image_url}" alt="${item.image_id}">
                <div class="result-info">
                    <div class="rank">Rank #${item.rank}</div>
                    <div class="score">Score: ${item.score.toFixed(4)}</div>
                    <div class="image-id">${item.image_id}</div>
                    <div class="action-buttons">
                        <button class="find-similar-btn" data-image-id="${item.image_id}">Find Similar</button>
                        <button class="find-material-btn" data-image-id="${item.image_id}">Find Material</button>
                    </div>
                </div>
            `;
        }

        results.appendChild(card);
    });

    // 3. Re-bind listeners on the freshly-rendered buttons
    document.querySelectorAll('.find-similar-btn').forEach(btn => {
        btn.addEventListener('click', findSimilar);
    });
    document.querySelectorAll('.find-material-btn').forEach(btn => {
        btn.addEventListener('click', findMaterial);
    });

    // 4. Optional latency/cache status banner
    if (latencyMs !== null) {
        const status = document.createElement('div');
        status.className = 'search-status';
        const cacheTxt = cacheStatus ? `${cacheStatus}` : '';
        status.textContent = `${items.length} results in ${latencyMs.toFixed(1)} ms (${cacheTxt})`;
        results.prepend(status);
    }
}

function showError(message) {
    error.textContent = message;
    error.classList.remove('hidden');
}
