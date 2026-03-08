/**
 * Pipeline API Client
 * Handles communication with the Flask backend and polling.
 */

const Pipeline = (() => {
    const API = {
        analyze: '/api/analyze',
        status: '/api/status',
        results: '/api/results',
        export: '/api/export',
    };

    let pollInterval = null;

    /**
     * Send GeoJSON polygon to trigger the pipeline.
     */
    async function analyze(geojson) {
        try {
            const resp = await fetch(API.analyze, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(geojson),
            });
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || 'Pipeline request failed');
            }

            // Start polling
            startPolling();
            return data;
        } catch (err) {
            console.error('[Pipeline] analyze error:', err);
            Dashboard.showError(err.message);
            throw err;
        }
    }

    /**
     * Poll pipeline status every 2 seconds.
     */
    function startPolling() {
        stopPolling();
        pollInterval = setInterval(async () => {
            try {
                const resp = await fetch(API.status);
                const data = await resp.json();

                Dashboard.updateStatus(data);

                if (data.status === 'complete') {
                    stopPolling();
                    await loadResults();
                } else if (data.status === 'error') {
                    stopPolling();
                    Dashboard.showError(data.error || 'Pipeline failed');
                }
            } catch (err) {
                console.error('[Pipeline] poll error:', err);
            }
        }, 2000);
    }

    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }

    /**
     * Load final results after pipeline completes.
     */
    async function loadResults() {
        try {
            const resp = await fetch(API.results);
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || 'Failed to load results');
            }

            // Render map + dashboard
            FieldMap.renderPayload(data.payload);
            Dashboard.renderResults(data);

            // Enable export button
            const btn = document.getElementById('btn-export');
            btn.disabled = false;
        } catch (err) {
            console.error('[Pipeline] results error:', err);
            Dashboard.showError(err.message);
        }
    }

    /**
     * Download shapefile export.
     */
    function downloadExport() {
        window.location.href = API.export;
    }

    return { analyze, startPolling, stopPolling, loadResults, downloadExport };
})();

// Wire up export button
document.getElementById('btn-export').addEventListener('click', Pipeline.downloadExport);
