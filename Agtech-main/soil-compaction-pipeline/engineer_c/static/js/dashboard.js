/**
 * Dashboard Module
 * Handles summary statistics, LLM insights, and pipeline status display.
 */

const Dashboard = (() => {
    const STAGE_PROGRESS = {
        'idle': 0,
        'ingesting': 25,
        'modeling': 55,
        'insights': 80,
        'complete': 100,
        'error': 100,
    };

    /**
     * Update pipeline status indicator.
     */
    function updateStatus(statusData) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');
        const detail = document.getElementById('status-detail');
        const barContainer = document.getElementById('progress-bar-container');
        const bar = document.getElementById('progress-bar');

        // Remove old classes
        indicator.className = 'status';

        const st = statusData.status;

        if (st === 'idle') {
            indicator.classList.add('idle');
            text.textContent = 'Draw a polygon to begin';
        } else if (st === 'complete') {
            indicator.classList.add('complete');
            text.textContent = 'Analysis Complete';
        } else if (st === 'error') {
            indicator.classList.add('error');
            text.textContent = 'Pipeline Error';
        } else {
            indicator.classList.add('running');
            text.textContent = stageLabel(st);
        }

        // Detail text
        detail.textContent = statusData.stage_detail || '';
        if (statusData.elapsed_seconds) {
            detail.textContent += ` (${statusData.elapsed_seconds}s)`;
        }

        // Progress bar
        if (st !== 'idle') {
            barContainer.classList.remove('hidden');
            bar.style.width = (STAGE_PROGRESS[st] || 0) + '%';
        }
    }

    function stageLabel(stage) {
        const labels = {
            'ingesting': 'Ingesting Data...',
            'modeling': 'Running ML Models...',
            'insights': 'Generating AI Analysis...',
        };
        return labels[stage] || stage;
    }

    /**
     * Show error message.
     */
    function showError(message) {
        updateStatus({
            status: 'error',
            stage_detail: message,
        });
    }

    /**
     * Render full results (summary + insights).
     */
    function renderResults(data) {
        renderSummary(data.summary);
        renderInsights(data.insights);
    }

    /**
     * Render summary statistics sidebar.
     */
    function renderSummary(summary) {
        const section = document.getElementById('summary-section');
        section.classList.remove('hidden');

        document.getElementById('stat-total').textContent =
            summary.total_pixels.toLocaleString();
        document.getElementById('stat-acreage').textContent =
            summary.total_acreage.toFixed(1) + ' ac';

        // Action distribution table
        const container = document.getElementById('action-table-container');
        if (summary.actions && summary.actions.length > 0) {
            let html = `
                <table class="action-table">
                    <thead>
                        <tr>
                            <th>Action</th>
                            <th>Pixels</th>
                            <th>Acres</th>
                            <th>%</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            summary.actions.forEach(a => {
                const cls = a.action.includes('Tillage') ? 'tillage'
                    : a.action.includes('Monitor') ? 'monitor' : 'none';

                html += `
                    <tr>
                        <td><span class="action-badge ${cls}">${truncate(a.action, 28)}</span></td>
                        <td>${a.count.toLocaleString()}</td>
                        <td>${a.acreage.toFixed(1)}</td>
                        <td>${a.percentage.toFixed(1)}%</td>
                    </tr>
                `;
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        }
    }

    /**
     * Render LLM insights panel.
     */
    function renderInsights(insights) {
        if (!insights || !insights.field_summary) return;

        const section = document.getElementById('insights-section');
        section.classList.remove('hidden');

        // Model badge
        const badge = document.getElementById('insights-model');
        const model = insights.metadata?.model || 'unknown';
        badge.textContent = `Powered by ${model}`;

        // Field summary
        document.getElementById('field-summary').textContent =
            insights.field_summary;

        // Recommendations
        const recList = document.getElementById('recommendations-list');
        const recs = insights.recommendations || [];

        if (Array.isArray(recs)) {
            recList.innerHTML = recs.map((rec, i) => {
                // Handle both string and object recommendations
                const text = typeof rec === 'string' ? rec
                    : rec.action_description
                        ? `<strong>${rec.action_description}</strong><br>${rec.rationale || ''}`
                        : rec.action || JSON.stringify(rec);
                const priority = rec.action_priority || rec.priority || (i + 1);

                return `
                    <div class="recommendation-item">
                        <span class="priority">#${priority}</span>
                        ${text}
                    </div>
                `;
            }).join('');
        }

        // Economic outlook
        document.getElementById('economic-outlook').textContent =
            insights.economic_outlook || '';

        // Confidence
        document.getElementById('confidence-note').textContent =
            insights.confidence_note || '';
    }

    function truncate(str, len) {
        return str.length > len ? str.slice(0, len) + '…' : str;
    }

    return { updateStatus, showError, renderResults };
})();
