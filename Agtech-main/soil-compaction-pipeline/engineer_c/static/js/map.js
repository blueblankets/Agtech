/**
 * Leaflet Map Module
 * Handles map initialization, polygon drawing, and choropleth rendering.
 */

const FieldMap = (() => {
    let map;
    let drawLayer;
    let resultLayer;
    let currentField = 'pred_ripper_depth_cm';

    // Color ramp: green → yellow → red (for ripper depth)
    const DEPTH_COLORS = [
        { val: 0, color: '#1a9850' },
        { val: 15, color: '#91cf60' },
        { val: 25, color: '#d9ef8b' },
        { val: 35, color: '#fee08b' },
        { val: 45, color: '#fc8d59' },
        { val: 60, color: '#d73027' },
    ];

    // Color ramp for ROI
    const ROI_COLORS = [
        { val: 0, color: '#d73027' },
        { val: 0.5, color: '#fc8d59' },
        { val: 0.8, color: '#fee08b' },
        { val: 1.0, color: '#d9ef8b' },
        { val: 1.5, color: '#91cf60' },
        { val: 2.0, color: '#1a9850' },
    ];

    // Action colors
    const ACTION_COLORS = {
        'Targeted Deep Tillage': '#ea4335',
        'Monitor - Not Economically Viable': '#fbbc04',
        'None': '#34a853',
    };

    function init() {
        map = L.map('map', {
            center: [39.0, -89.0],
            zoom: 6,
            zoomControl: true,
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
            if (resultLayer.getLayers().length > 0) {
                recolorLayer();
            }
        });
    }

    /**
     * Convert payload to 10m pixel squares and render on map.
     */
    function renderPayload(payload) {
        resultLayer.clearLayers();

        const pixelSizeM = 10;
        const half = pixelSizeM / 2;

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
                fillOpacity: 0.75,
                weight: 0,
            });

            // Tooltip
            rect.bindTooltip(buildTooltip(record), {
                className: 'pixel-tooltip',
                sticky: true,
            });

            rect.pixelData = record; // store for recoloring
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
     * Recolor existing layers when switching fields.
     */
    function recolorLayer() {
        resultLayer.eachLayer(layer => {
            if (layer.pixelData) {
                const color = getColor(layer.pixelData, currentField);
                layer.setStyle({ color, fillColor: color });
            }
        });
        updateLegend(currentField);
    }

    /**
     * Get color for a pixel based on field and value.
     */
    function getColor(record, field) {
        if (field === 'action') {
            const action = record.action || 'None';
            return ACTION_COLORS[action] || '#888';
        }

        const val = record[field] || 0;
        const ramp = field === 'roi' ? ROI_COLORS : DEPTH_COLORS;

        // Interpolate through color ramp
        for (let i = 0; i < ramp.length - 1; i++) {
            if (val <= ramp[i + 1].val) {
                const t = (val - ramp[i].val) / (ramp[i + 1].val - ramp[i].val);
                return lerpColor(ramp[i].color, ramp[i + 1].color, Math.max(0, Math.min(1, t)));
            }
        }
        return ramp[ramp.length - 1].color;
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
     * Update sidebar legend based on current field.
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
            const ramp = field === 'roi' ? ROI_COLORS : DEPTH_COLORS;
            const label = field === 'roi' ? 'ROI' : 'Ripper Depth (cm)';
            title.textContent = `Legend — ${label}`;

            const gradientStops = ramp.map((s, i) =>
                `${s.color} ${(i / (ramp.length - 1) * 100).toFixed(0)}%`
            ).join(', ');

            content.innerHTML = `
                <div class="legend-gradient" style="background: linear-gradient(90deg, ${gradientStops})"></div>
                <div class="legend-labels">
                    <span>${ramp[0].val}</span>
                    <span>${ramp[Math.floor(ramp.length / 2)].val}</span>
                    <span>${ramp[ramp.length - 1].val}</span>
                </div>
            `;
        }
    }

    // Initialize on load
    document.addEventListener('DOMContentLoaded', init);

    return { renderPayload, recolorLayer };
})();
