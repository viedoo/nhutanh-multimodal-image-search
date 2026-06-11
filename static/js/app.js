const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');

searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
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
    
    // Tự động nhận diện convention:
    //   "Similar to: cat.xxxx.jpg"  → gọi /api/search-similar
    //   "Material: cat.xxxx.jpg"    → gọi /api/search-similar-material
    const isSimilarSearch  = query.startsWith('Similar to:');
    const isMaterialSearch = query.startsWith('Material:');
    let endpoint = '/api/search';
    let reqBody = { query: query, top_k: 10 };
    let refImageId = null;
    let searchType = 'text';

    if (isSimilarSearch) {
        refImageId = query.replace('Similar to:', '').trim();
        endpoint   = '/api/search-similar';
        reqBody    = { image_id: refImageId, top_k: 10 };
        searchType = 'similar';
    } else if (isMaterialSearch) {
        refImageId = query.replace('Material:', '').trim();
        endpoint   = '/api/search-similar-material';
        reqBody    = { image_id: refImageId, top_k: 10 };
        searchType = 'material';
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqBody)
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();
        // Truyền thêm refImageId và searchType vào để hiển thị card tham chiếu đúng loại
        displayResults(data.results, refImageId, searchType);
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
        
        // Truyền thêm imageId vào để render thẻ đầu tiên
        displayResults(data.results, imageId);
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

        // Truyền referenceImageId và badge loại 'material'
        displayResults(data.results, imageId, 'material');
    } catch (err) {
        showError('Material search failed: ' + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

function displayResults(items, referenceImageId = null, searchType = 'similar') {
    results.innerHTML = '';

    if (items.length === 0) {
        results.innerHTML = '<p style="color: #333; text-align: center; grid-column: 1 / -1;">No results found</p>';
        return;
    }

    // 1. Nếu có referenceImageId, render card này TRƯỚC TIÊN
    if (referenceImageId) {
        const refCard = document.createElement('div');
        refCard.className = 'result-card reference-card'; // Tái sử dụng form của result-card
        refCard.innerHTML = `
            <div class="reference-header">Reference Image</div>
            <img src="/images/dataset/test_set/cats/${referenceImageId}" alt="${referenceImageId}">
            <div class="result-info">
                <div class="image-id" style="color: #333; font-weight: bold; font-size: 0.95rem;">ID: ${referenceImageId}</div>
            </div>
        `;
        results.appendChild(refCard);
    }

    // 2. Render các kết quả bình thường tiếp theo
    items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const badgeLabel = searchType === 'material' ? 'Find Material' : 'Find Similar';
        const badgeClass = searchType === 'material' ? 'find-material-btn' : 'find-similar-btn';
        
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
        
        results.appendChild(card);
    });

    // Cập nhật lại event listener cho các nút mới
    document.querySelectorAll('.find-similar-btn').forEach(btn => {
        btn.addEventListener('click', findSimilar);
    });
    document.querySelectorAll('.find-material-btn').forEach(btn => {
        btn.addEventListener('click', findMaterial);
    });
}

function showError(message) {
    error.textContent = message;
    error.classList.remove('hidden');
}