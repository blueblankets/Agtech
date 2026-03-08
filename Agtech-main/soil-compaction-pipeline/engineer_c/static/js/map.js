/**
 * Leaflet Map Module
 * Handles map initialization, polygon drawing, and choropleth rendering.
 *
 * Key design:
 * - Uses Canvas renderer for performance with 100k+ pixels
 * - Dynamic min/max color scaling per-field (not fixed 0-60)
 * - 10m pixel squares with tooltips
 */

const FieldMap = (() => {
    let map;
    let drawLayer;
    let resultLayer;
    let currentField = 'pred_ripper_depth_cm';
    let payloadCache = null;  // store for recoloring
    let dataStats = {};       // min/max per field

    // Viridis-inspired color ramp (works well for continuous data)
    const VIRIDIS = [
        '#440154', '#482777', '#3f4a8a', '#31678e',
        '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'
    ];

    // Red-Yellow-Green diverging ramp (good for depth/stress)
    const RYG = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43', '#d73027'];

    // Action colors
    const ACTION_COLORS = {
        'Targeted Deep Tillage': '#ea4335',
        'Monitor - Not Economically Viable': '#fbbc04',
        'None': '#34a853',
        'INVALID_DATA': '#888888',
    };

    // Canvas renderer for performance
    const canvasRenderer = L.canvas({ padding: 0.5 });

    function init() {
        map = L.map('map', {
            center: [39.0, -89.0],
            zoom: 6,
            zoomControl: true,
            preferCanvas: true,
        });

        // Satellite tile layer (Esri World Imagery — free, no API key)
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri',
            maxZoom: 19,
        }).addTo(map);

        // Draw layer for user polygon
        drawLayer = new L.FeatureGroup();
        map.addLayer(drawLayer);

        // Result layer for choropleth
        resultLayer = new L.FeatureGroup();
        map.addLayer(resultLayer);

        // Leaflet.draw controls
        const drawControl = new L.Control.Draw({
            draw: {
                polygon: {
                    shapeOptions: {
                        color: '#00bfa5',
                        weight: 2,
                        fillOpacity: 0.1,
                    },
                    allowIntersection: false,
                },
                polyline: false,
                circle: false,
                circlemarker: false,
                marker: false,
                rectangle: true,
            },
            edit: {
                featureGroup: drawLayer,
            },
        });
        map.addControl(drawControl);

        // Handle draw complete
        map.on(L.Draw.Event.CREATED, (e) => {
            drawLayer.clearLayers();
            drawLayer.addLayer(e.layer);

            // Hide the draw prompt
            document.getElementById('draw-prompt').classList.add('hidden');

            // Extract GeoJSON and send to pipeline
            const geojson = {
                type: 'FeatureCollection',
                features: [{
                    type: 'Feature',
                    properties: {},
                    geometry: e.layer.toGeoJSON().geometry,
                }],
            };

            Pipeline.analyze(geojson);
        });

        // Layer selector
        document.getElementById('layer-select').addEventListener('change', (e) => {
            currentField = e.target.value;
            if (payloadCache) {
                renderPayload(payloadCache);
            }
        });
    }

    /**
     * Compute min/max stats for numeric fields in the payload.
     */
    function computeStats(payload) {
        const fields = ['pred_ripper_depth_cm', 'roi', 'mapie_lower_bound', 'mapie_upper_bound'];
        const stats = {};

        fields.forEach(f => {
            const vals = payload
                .map(r => r[f])
                .filter(v => v != null && !isNaN(v));

            if (vals.length > 0) {
                stats[f] = {
                    min: Math.min(...vals),
                    max: Math.max(...vals),
                    mean: vals.reduce((a, b) => a + b, 0) / vals.length,
                };
            } else {
                stats[f] = { min: 0, max: 1, mean: 0.5 };
            }
        });

        return stats;
    }

    /**
     * Convert payload to 10m pixel squares and render on map.
     */
    function renderPayload(payload) {
        resultLayer.clearLayers();
        payloadCache = payload;
        dataStats = computeStats(payload);

        const pixelSizeM = 10;
        const half = pixelSizeM / 2;

        console.log(`[Map] Rendering ${payload.length} pixels (field: ${currentField})`);
        console.log(`[Map] Stats:`, dataStats[currentField]);

        payload.forEach(record => {
            const lat = record.lat;
            const lon = record.lon;

            if (lat == null || lon == null) return;

            // Convert center to bounding box (±5m)
            const latOff = half / 111320;
            const lonOff = half / (111320 * Math.cos(lat * Math.PI / 180));

            const bounds = [
                [lat - latOff, lon - lonOff],
                [lat + latOff, lon + lonOff],
            ];

            const color = getColor(record, currentField);

            const rect = L.rectangle(bounds, {
                color: color,
                fillColor: color,
                fillOpacity: 0.8,
                weight: 0,
                renderer: canvasRenderer,  // Canvas for performance
            });

            // Tooltip (only bind on hover for performance)
            rect.on('mouseover', function () {
                if (!this._tooltipBound) {
                    this.bindTooltip(buildTooltip(record), {
                        className: 'pixel-tooltip',
                        sticky: true,
                    });
                    this._tooltipBound = true;
                    this.openTooltip();
                }
            });

            resultLayer.addLayer(rect);
        });

        // Fit map to results
        if (resultLayer.getLayers().length > 0) {
            map.fitBounds(resultLayer.getBounds(), { padding: [30, 30] });
        }

        // Show legend
        updateLegend(currentField);
    }

    /**
     * Get color for a pixel based on field and DYNAMIC min/max.
     */
    function getColor(record, field) {
        if (field === 'action') {
            const action = record.action || 'None';
            return ACTION_COLORS[action] || '#888';
        }

        const val = record[field];
        if (val == null || isNaN(val)) return '#333';

        const stats = dataStats[field];
        if (!stats) return '#888';

        // Normalize to 0-1 using actual data range
        const range = stats.max - stats.min;
        const t = range > 0 ? (val - stats.min) / range : 0.5;

        // Use RYG ramp (green=low depth → red=high depth)
        return sampleRamp(RYG, Math.max(0, Math.min(1, t)));
    }

    /**
     * Sample a color ramp at position t (0-1).
     */
    function sampleRamp(ramp, t) {
        const idx = t * (ramp.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, ramp.length - 1);
        const frac = idx - lo;
        return lerpColor(ramp[lo], ramp[hi], frac);
    }

    /**
     * Linear interpolation between two hex colors.
     */
    function lerpColor(a, b, t) {
        const ar = parseInt(a.slice(1, 3), 16), ag = parseInt(a.slice(3, 5), 16), ab = parseInt(a.slice(5, 7), 16);
        const br = parseInt(b.slice(1, 3), 16), bg = parseInt(b.slice(3, 5), 16), bb = parseInt(b.slice(5, 7), 16);
        const rr = Math.round(ar + (br - ar) * t);
        const rg = Math.round(ag + (bg - ag) * t);
        const rb = Math.round(ab + (bb - ab) * t);
        return `#${rr.toString(16).padStart(2, '0')}${rg.toString(16).padStart(2, '0')}${rb.toString(16).padStart(2, '0')}`;
    }

    /**
     * Build HTML tooltip for a pixel.
     */
    function buildTooltip(r) {
        const actionClass = (r.action || '').includes('Tillage') ? 'tillage'
            : (r.action || '').includes('Monitor') ? 'monitor' : 'none';

        return `
            <div>
                <span class="tt-label">Pixel</span> <span class="tt-value">${r.pixel_id}</span><br>
                <span class="tt-label">Action</span> <span class="tt-value action-badge ${actionClass}">${r.action}</span><br>
                <span class="tt-label">Ripper Depth</span> <span class="tt-value">${(r.pred_ripper_depth_cm || 0).toFixed(1)} cm</span><br>
                <span class="tt-label">Confidence</span> <span class="tt-value">${(r.mapie_lower_bound || 0).toFixed(1)} – ${(r.mapie_upper_bound || 0).toFixed(1)} cm</span><br>
                <span class="tt-label">ROI</span> <span class="tt-value">${(r.roi || 0).toFixed(3)}</span>
            </div>
        `;
    }

    /**
     * Update sidebar legend based on current field with DYNAMIC range.
     */
    function updateLegend(field) {
        const section = document.getElementById('legend-section');
        const title = document.getElementById('legend-title');
        const content = document.getElementById('legend-content');
        section.classList.remove('hidden');

        if (field === 'action') {
            title.textContent = 'Legend — Action';
            content.innerHTML = Object.entries(ACTION_COLORS)
                .map(([action, color]) => `
                    <div class="legend-row">
                        <div class="legend-swatch" style="background:${color}"></div>
                        <span>${action}</span>
                    </div>
                `).join('');
        } else {
            const stats = dataStats[field] || { min: 0, max: 1 };
            const label = field === 'roi' ? 'ROI' : 'Ripper Depth (cm)';
            title.textContent = `Legend — ${label}`;

            const ramp = RYG;
            const gradientStops = ramp.map((c, i) =>
                `${c} ${(i / (ramp.length - 1) * 100).toFixed(0)}%`
            ).join(', ');

            const mid = ((stats.min + stats.max) / 2).toFixed(1);

            content.innerHTML = `
                <div class="legend-gradient" style="background: linear-gradient(90deg, ${gradientStops})"></div>
                <div class="legend-labels">
                    <span>${stats.min.toFixed(1)}</span>
                    <span>${mid}</span>
                    <span>${stats.max.toFixed(1)}</span>
                </div>
            `;
        }
    }

    // Initialize on load
    document.addEventListener('DOMContentLoaded', init);

    return { renderPayload };
})();
