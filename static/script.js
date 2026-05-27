// ────────────────────────────────────────────────────────────────────
//  PANEL SWITCHING
// ────────────────────────────────────────────────────────────────────
const panelSelect = document.getElementById('panel-select');
const allPanels   = document.querySelectorAll('.panel');

function showPanel(panelName) {
    panelSelect.value = panelName;
    allPanels.forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + panelName).classList.add('active');
    drawPoints();
}

panelSelect.addEventListener('change', () => showPanel(panelSelect.value));

// ────────────────────────────────────────────────────────────────────
//  STREAM
// ────────────────────────────────────────────────────────────────────
const feed          = document.getElementById('feed');
const streamDot     = document.getElementById('stream-dot');
const streamStatus  = document.getElementById('stream-status');
const placeholder   = document.getElementById('video-placeholder');
const btnConnect    = document.getElementById('btn-connect');
const btnPause      = document.getElementById('btn-pause');
const btnDisconnect = document.getElementById('btn-disconnect');
const btnResults    = document.getElementById('btn-results');
const cameraSelect  = document.getElementById('camera-select');
const morphologyModelSelect = document.getElementById('morphology-model-path');
const tracerModelSelect = document.getElementById('tracer-model-path');
const timelineWrap = document.getElementById('timeline-wrap');
const timelineSlider = document.getElementById('timeline-slider');
const timelineCurrent = document.getElementById('timeline-current');
const timelineDuration = document.getElementById('timeline-duration');
let isPaused = false;
let timelineTimer = null;
let isSeekingTimeline = false;

function formatTime(seconds) {
    const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
    const minutes = Math.floor(safeSeconds / 60);
    const remainingSeconds = safeSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
}

function setTimelineVisible(isVisible) {
    timelineWrap.classList.toggle('active', isVisible);
}

function updateTimelineFromStatus(data) {
    const isVideo = data.source_kind === 'video';
    const totalFrames = Number(data.total_frames) || 0;
    const fps = Number(data.fps) || 0;
    const currentFrame = Number(data.current_frame) || 0;
    const duration = Number(data.duration) || (fps > 0 ? totalFrames / fps : 0);

    setTimelineVisible(isVideo && totalFrames > 0);

    if (!isVideo || totalFrames <= 0) return;

    timelineSlider.max = Math.max(0, totalFrames - 1);

    if (!isSeekingTimeline) {
        timelineSlider.value = Math.max(0, Math.min(currentFrame, totalFrames - 1));
    }

    const shownFrame = Number(timelineSlider.value) || currentFrame;
    timelineCurrent.textContent = formatTime(fps > 0 ? shownFrame / fps : 0);
    timelineDuration.textContent = formatTime(duration);
}

async function refreshTimelineStatus() {
    try {
        const res = await fetch('/playback_status');
        const data = await res.json().catch(() => ({}));
        if (res.ok && data.ok) {
            updateTimelineFromStatus(data);
        }
    } catch {
        // The stream error handler already shows connection issues.
    }
}

function startTimelinePolling() {
    if (timelineTimer !== null) return;
    refreshTimelineStatus();
    timelineTimer = window.setInterval(refreshTimelineStatus, 500);
}

function stopTimelinePolling() {
    if (timelineTimer === null) return;
    window.clearInterval(timelineTimer);
    timelineTimer = null;
}

function connectStream() {
    streamStatus.textContent = 'connecting…';
    feed.onload = () => {
        streamDot.className = 'live';
        streamStatus.textContent = isPaused ? 'paused' : 'live';
        feed.style.visibility = 'visible';
        placeholder.style.display = 'none';
        btnConnect.style.display    = 'none';
        btnPause.style.display      = 'inline-block';
        btnDisconnect.style.display = 'inline-block';
        // naturalWidth/Height are now known — safe to size the canvas
        resizeCanvas();
        startTimelinePolling();
    };
    feed.onerror = () => {
        streamDot.className = 'error';
        streamStatus.textContent = 'error — is the server running?';
    };
    feed.src = '/video_feed?t=' + Date.now();
}

function setStreamStoppedUi(statusText = 'idle') {
    isPaused = false;
    btnPause.textContent = 'Pause';
    stopTimelinePolling();
    feed.src = '';
    feed.style.visibility = 'hidden';
    placeholder.style.display = '';
    streamDot.className = '';
    streamStatus.textContent = statusText;
    btnConnect.style.display    = 'inline-block';
    btnPause.style.display      = 'none';
    btnDisconnect.style.display = 'none';
    setTimelineVisible(false);
}

async function disconnectStream() {
    try {
        await fetch('/stop_stream', { method: 'POST' });
    } catch {
        // Closing the browser-side stream still prevents further rendering.
    }
    setStreamStoppedUi('idle');
}

async function setPaused(nextPaused) {
    try {
        const res = await fetch('/playback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_paused: nextPaused }),
        });
        if (!res.ok) return false;
        isPaused = nextPaused;
        btnPause.textContent = isPaused ? 'Resume' : 'Pause';
        streamStatus.textContent = isPaused ? 'paused' : 'live';
        return true;
    } catch {
        streamDot.className = 'error';
        streamStatus.textContent = 'error — could not update playback';
        return false;
    }
}

btnConnect.addEventListener('click', connectStream);
btnPause.addEventListener('click', () => setPaused(!isPaused));
btnDisconnect.addEventListener('click', disconnectStream);
btnResults.addEventListener('click', () => loadResults({ stop: true }));

timelineSlider.addEventListener('input', () => {
    isSeekingTimeline = true;
    refreshTimelineStatus();
});

timelineSlider.addEventListener('change', async () => {
    const frame = parseInt(timelineSlider.value) || 0;

    try {
        const res = await fetch('/seek', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame }),
        });

        if (!res.ok) {
            streamDot.className = 'error';
            streamStatus.textContent = 'seek failed';
        }
    } catch {
        streamDot.className = 'error';
        streamStatus.textContent = 'seek failed';
    } finally {
        isSeekingTimeline = false;
        refreshTimelineStatus();
    }
});

// ────────────────────────────────────────────────────────────────────
//  CANVAS + GROUND CONTROL POINTS
// ────────────────────────────────────────────────────────────────────
const viewport = document.getElementById('video-viewport');
const feedWrap = document.getElementById('feed-wrap');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');

// Points store only feed-native pixel coords { x, y, r }.
// Screen positions are always derived on the fly via feedToCanvas().
let points     = [];
let dragging   = null;
let dragOffset = { x: 0, y: 0 };
let draggingStivHandle = null;
let isDrawingCvRoi = false;
let cvRoiPreview = null;
let cvRoiPoints = [];
let orthoReady = false;

const stivLine = {
    startX: 5,
    startY: 50,
    endX: 95,
    endY: 50,
};

const POINT_RADIUS = 7;
const STIV_HANDLE_RADIUS = 8;
const ROI_POINT_RADIUS = 5;

// Computes the contain-scaled size of the feed inside #video-viewport.
// Replicates object-fit:contain math so #feed-wrap matches exactly.
function getScaledSize() {
    const vw = viewport.offsetWidth;
    const vh = viewport.offsetHeight;
    if (!feed.naturalWidth || !feed.naturalHeight) {
        return { width: 0, height: 0 };
    }
    const scale = Math.min(vw / feed.naturalWidth, vh / feed.naturalHeight);
    return {
        width:  Math.round(feed.naturalWidth  * scale),
        height: Math.round(feed.naturalHeight * scale),
    };
}

// Sizes #feed-wrap (and therefore #feed + #canvas inside it) to the
// contain-scaled feed dimensions. The centering is handled purely by CSS
// (position:absolute + top/left 50% + transform translate(-50%,-50%)).
// Setting canvas.width/height clears the bitmap — redraw immediately after.
function resizeCanvas() {
    const { width, height } = getScaledSize();
    feedWrap.style.width  = width  + 'px';
    feedWrap.style.height = height + 'px';
    canvas.width  = width;
    canvas.height = height;
    drawPoints();
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// feed-native px  →  canvas px
function feedToCanvas(fx, fy) {
    return [
        (fx / feed.naturalWidth)  * canvas.width,
        (fy / feed.naturalHeight) * canvas.height,
    ];
}

// canvas px  →  feed-native px  (clamped to feed bounds)
function canvasToFeed(cx, cy) {
    return [
        Math.round(Math.max(0, Math.min(cx / canvas.width,  1)) * feed.naturalWidth),
        Math.round(Math.max(0, Math.min(cy / canvas.height, 1)) * feed.naturalHeight),
    ];
}

function isCvPanelActive() {
    return panelSelect.value === 'cv';
}

function clampPercent(value) {
    return Math.max(0, Math.min(100, value));
}

function formatPercent(value) {
    const rounded = Math.round(value * 10) / 10;
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

function syncStivLineFromInputs() {
    const fields = [
        ['stiv-start-x', 'startX'],
        ['stiv-start-y', 'startY'],
        ['stiv-end-x', 'endX'],
        ['stiv-end-y', 'endY'],
    ];

    fields.forEach(([id, key]) => {
        const el = document.getElementById(id);
        const value = Number.parseFloat(el.value);
        if (Number.isFinite(value)) {
            stivLine[key] = clampPercent(value);
        }
    });
}

function setStivInputsFromLine() {
    const fields = {
        'stiv-start-x': stivLine.startX,
        'stiv-start-y': stivLine.startY,
        'stiv-end-x': stivLine.endX,
        'stiv-end-y': stivLine.endY,
    };

    Object.entries(fields).forEach(([id, value]) => {
        document.getElementById(id).value = formatPercent(value);
    });
}

function stivPercentToCanvas(xPct, yPct) {
    return [
        (clampPercent(xPct) / 100) * canvas.width,
        (clampPercent(yPct) / 100) * canvas.height,
    ];
}

function canvasToStivPercent(cx, cy) {
    return [
        clampPercent((cx / Math.max(canvas.width, 1)) * 100),
        clampPercent((cy / Math.max(canvas.height, 1)) * 100),
    ];
}

function canvasToRoiPercent(cx, cy) {
    const [x, y] = canvasToStivPercent(cx, cy);
    return { x, y };
}

function roiPercentToCanvas(point) {
    return stivPercentToCanvas(point.x, point.y);
}

function updateCvRoiCount() {
    const el = document.getElementById('cv-roi-count');
    if (!el) return;
    const suffix = isDrawingCvRoi ? ', drawing' : '';
    el.textContent = `${cvRoiPoints.length} point(s)${suffix}`;
}

function setCvRoiDrawing(nextDrawing) {
    isDrawingCvRoi = nextDrawing;
    cvRoiPreview = null;
    draggingStivHandle = null;

    const btn = document.getElementById('btn-draw-cv-roi');
    if (btn) {
        btn.textContent = isDrawingCvRoi ? 'Finish PIV ROI' : 'Draw PIV ROI';
        btn.classList.toggle('active', isDrawingCvRoi);
    }

    updateCvRoiCount();
    drawPoints();
}

function clearCvRoi(disableCheckbox = true) {
    cvRoiPoints = [];
    cvRoiPreview = null;
    setCvRoiDrawing(false);
    if (disableCheckbox) {
        document.getElementById('cv-roi-enabled').checked = false;
    }
    updateCvRoiCount();
    drawPoints();
}

function drawCvRoiOverlay() {
    if (!isCvPanelActive() || !canvas.width || !canvas.height) return;
    if (cvRoiPoints.length === 0 && !cvRoiPreview) return;

    const canvasPoints = cvRoiPoints.map(roiPercentToCanvas);
    const previewPoint = isDrawingCvRoi && cvRoiPreview ? roiPercentToCanvas(cvRoiPreview) : null;

    ctx.save();

    if (canvasPoints.length >= 3) {
        ctx.beginPath();
        canvasPoints.forEach(([x, y], index) => {
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(0, 210, 210, 0.18)';
        ctx.fill();
    }

    if (canvasPoints.length >= 1) {
        ctx.beginPath();
        canvasPoints.forEach(([x, y], index) => {
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        if (previewPoint) {
            ctx.lineTo(previewPoint[0], previewPoint[1]);
        } else if (canvasPoints.length >= 3) {
            ctx.closePath();
        }
        ctx.strokeStyle = '#00d2d2';
        ctx.lineWidth = 2;
        ctx.setLineDash(isDrawingCvRoi ? [6, 4] : []);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    canvasPoints.forEach(([x, y], index) => {
        ctx.beginPath();
        ctx.arc(x, y, ROI_POINT_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#00d2d2';
        ctx.strokeStyle = '#062a2a';
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#062a2a';
        ctx.font = 'bold 8px Courier New';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(index + 1, x, y + 0.4);
    });

    ctx.restore();
}

function drawStivOverlay() {
    if (!isCvPanelActive() || !canvas.width || !canvas.height) return;

    syncStivLineFromInputs();

    const [sx, sy] = stivPercentToCanvas(stivLine.startX, stivLine.startY);
    const [ex, ey] = stivPercentToCanvas(stivLine.endX, stivLine.endY);

    ctx.save();
    ctx.lineCap = 'round';

    ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();

    ctx.strokeStyle = '#ff4fd8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();

    [
        [sx, sy, 'S'],
        [ex, ey, 'E'],
    ].forEach(([x, y, label]) => {
        ctx.beginPath();
        ctx.arc(x, y, STIV_HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#ff4fd8';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#111';
        ctx.font = 'bold 10px Courier New';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x, y + 0.5);
    });

    ctx.restore();
}

function hitTestStivHandle(mx, my) {
    if (!isCvPanelActive()) return null;

    syncStivLineFromInputs();

    const handles = [
        ['start', ...stivPercentToCanvas(stivLine.startX, stivLine.startY)],
        ['end', ...stivPercentToCanvas(stivLine.endX, stivLine.endY)],
    ];

    let closest = null;
    let closestDistance = Infinity;

    handles.forEach(([name, x, y]) => {
        const dx = mx - x;
        const dy = my - y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= STIV_HANDLE_RADIUS + 6 && distance < closestDistance) {
            closest = name;
            closestDistance = distance;
        }
    });

    return closest;
}

function drawPoints() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!shouldHideVgcpMarkers()) {
        points.forEach((p, i) => {
            const [cx, cy]   = feedToCanvas(p.x, p.y);
            const isSelected = (dragging === i);
            const rValue = Number.parseFloat(p.r);
            const hasValidR = Number.isFinite(rValue) && rValue > 0;

            // Crosshair lines (drawn before the dot so the dot sits on top)
            ctx.strokeStyle = isSelected ? 'rgba(26,92,150,0.45)' : 'rgba(224,92,16,0.35)';
            ctx.lineWidth   = 0.8;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(cx, 0);           ctx.lineTo(cx, canvas.height);
            ctx.moveTo(0,  cy);          ctx.lineTo(canvas.width, cy);
            ctx.stroke();
            ctx.setLineDash([]);

            // Dot
            ctx.save();
            ctx.shadowColor = 'rgba(0,0,0,0.5)';
            ctx.shadowBlur  = 4;
            ctx.beginPath();
            ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle   = isSelected ? '#1a5c96' : '#e05c10';
            ctx.strokeStyle = '#fff';
            ctx.lineWidth   = 1.5;
            ctx.fill();
            ctx.stroke();
            ctx.restore();

            // Index label
            ctx.fillStyle    = '#fff';
            ctx.font         = 'bold 9px Courier New';
            ctx.textAlign    = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i + 1, cx, cy);

            const label = hasValidR ? `R: ${rValue.toFixed(2)} m` : 'R: not set';
            ctx.save();
            ctx.font = 'bold 12px Courier New';
            ctx.textBaseline = 'middle';

            const labelWidth = ctx.measureText(label).width;
            let lx = cx + POINT_RADIUS + 8;
            let ly = cy - POINT_RADIUS - 8;
            let align = 'left';

            if (lx + labelWidth > canvas.width - 4) {
                lx = cx - POINT_RADIUS - 8;
                align = 'right';
            }

            if (ly < 12) {
                ly = cy + POINT_RADIUS + 12;
            }

            ctx.textAlign = align;
            ctx.lineWidth = 4;
            ctx.strokeStyle = 'rgba(0,0,0,0.85)';
            ctx.strokeText(label, lx, ly);
            ctx.fillStyle = hasValidR ? '#fff' : '#ffe66d';
            ctx.fillText(label, lx, ly);
            ctx.restore();
        });
    }
    drawCvRoiOverlay();
    drawStivOverlay();
}

function hitTest(mx, my) {
    for (let i = points.length - 1; i >= 0; i--) {
        const [cx, cy] = feedToCanvas(points[i].x, points[i].y);
        const dx = mx - cx;
        const dy = my - cy;
        if (Math.sqrt(dx * dx + dy * dy) <= POINT_RADIUS + 4) return i;
    }
    return -1;
}

canvas.addEventListener('mousedown', e => {
    // canvas.getBoundingClientRect() is correct here because the canvas is
    // already sized and positioned to overlay the feed exactly.
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (isCvPanelActive()) {
        if (!canvas.width || !canvas.height) return;

        if (isDrawingCvRoi) {
            cvRoiPoints.push(canvasToRoiPercent(mx, my));
            document.getElementById('cv-roi-enabled').checked = true;
            updateCvRoiCount();
            drawPoints();
            canvas.style.cursor = 'crosshair';
            return;
        }

        const hit = hitTestStivHandle(mx, my);
        if (hit) {
            draggingStivHandle = hit;
        } else {
            const [xPct, yPct] = canvasToStivPercent(mx, my);
            stivLine.startX = xPct;
            stivLine.startY = yPct;
            stivLine.endX = xPct;
            stivLine.endY = yPct;
            draggingStivHandle = 'end';
        }

        setStivInputsFromLine();
        drawPoints();
        canvas.style.cursor = 'grabbing';
        return;
    }

    if (shouldHideVgcpMarkers()) {
        return;
    }

    const hit = hitTest(mx, my);
    if (hit >= 0) {
        dragging = hit;
        const [cx, cy] = feedToCanvas(points[hit].x, points[hit].y);
        dragOffset = { x: mx - cx, y: my - cy };
        canvas.style.cursor = 'grabbing';
    } else {
        // Only place a point if the feed is actually live
        if (!feed.naturalWidth) return;
        const [fx, fy] = canvasToFeed(mx, my);
        points.push({ x: fx, y: fy, r: '' });
        drawPoints();
        renderTable();
    }
});

canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (isCvPanelActive() && isDrawingCvRoi) {
        cvRoiPreview = canvasToRoiPercent(mx, my);
        drawPoints();
        canvas.style.cursor = 'crosshair';
        return;
    }

    if (draggingStivHandle !== null) {
        const [xPct, yPct] = canvasToStivPercent(mx, my);
        if (draggingStivHandle === 'start') {
            stivLine.startX = xPct;
            stivLine.startY = yPct;
        } else {
            stivLine.endX = xPct;
            stivLine.endY = yPct;
        }
        setStivInputsFromLine();
        drawPoints();
        canvas.style.cursor = 'grabbing';
        return;
    }

    if (isCvPanelActive()) {
        canvas.style.cursor = hitTestStivHandle(mx, my) ? 'grab' : 'crosshair';
        return;
    }

    if (shouldHideVgcpMarkers()) {
        canvas.style.cursor = 'default';
        return;
    }

    if (dragging !== null) {
        // Clamp drag so the point can't leave the feed area
        const cx = Math.max(0, Math.min(mx - dragOffset.x, canvas.width));
        const cy = Math.max(0, Math.min(my - dragOffset.y, canvas.height));
        const [fx, fy] = canvasToFeed(cx, cy);
        points[dragging].x = fx;
        points[dragging].y = fy;
        drawPoints();
        renderTable();
    } else {
        canvas.style.cursor = hitTest(mx, my) >= 0 ? 'grab' : 'crosshair';
    }
});

canvas.addEventListener('mouseup', () => {
    dragging = null;
    draggingStivHandle = null;
    canvas.style.cursor = 'crosshair';
});

canvas.addEventListener('mouseleave', () => {
    // If mouse leaves canvas while dragging, drop the point where it is
    dragging = null;
    draggingStivHandle = null;
    if (isDrawingCvRoi) {
        cvRoiPreview = null;
        drawPoints();
    }
});

canvas.addEventListener('dblclick', e => {
    if (!isCvPanelActive() || !isDrawingCvRoi) return;
    e.preventDefault();
    setCvRoiDrawing(false);
});

document.addEventListener('keydown', e => {
    if (!isCvPanelActive()) return;
    const targetTag = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : '';
    if (targetTag === 'input' || targetTag === 'textarea' || targetTag === 'select' || e.target.isContentEditable) return;
    if (e.key !== 'Delete' && e.key !== 'Backspace') return;
    if (cvRoiPoints.length === 0 && !isDrawingCvRoi) return;

    e.preventDefault();
    deleteCvRoi();
});

// ────────────────────────────────────────────────────────────────────
//  VGCP TABLE
// ────────────────────────────────────────────────────────────────────
const tbody   = document.getElementById('vgcp-tbody');
const ptCount = document.getElementById('pt-count');
const calibrationReadout = document.getElementById('calibration-readout');
const calibrationStatus = document.getElementById('calibration-status');
const calibrationMpp = document.getElementById('calibration-mpp');
const calibrationPpm = document.getElementById('calibration-ppm');
const vgcpOrthoNote = document.getElementById('vgcp-ortho-note');
const resultsHeadline = document.getElementById('results-headline');
const resultsDuration = document.getElementById('results-duration');
const resultsFrames = document.getElementById('results-frames');
const resultsCsv = document.getElementById('results-csv');
const resultsTbody = document.getElementById('results-tbody');
const resultsFloatMethod = document.getElementById('results-float-method');

function formatCalibrationNumber(value, digits) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(digits) : '--';
}

function shouldHideVgcpMarkers() {
    const orthoToggle = document.getElementById('video-ortho');
    return orthoReady && Boolean(orthoToggle && orthoToggle.checked);
}

function updateVgcpOrthoVisibility() {
    const hidden = shouldHideVgcpMarkers();
    vgcpOrthoNote.classList.toggle('active', hidden);
    if (hidden) {
        dragging = null;
    }
    drawPoints();
}

function updateCalibrationReadout(data) {
    const ready = Boolean(data && data.ortho_ready);
    const status = data && data.ortho_status ? data.ortho_status : 'Not calibrated';
    const metersPerPixel = data ? data.meters_per_pixel : null;
    const pixelsPerMeter = data ? data.pixels_per_meter : null;
    orthoReady = ready;

    calibrationReadout.classList.toggle('ready', ready);
    calibrationStatus.textContent = ready ? `Ready - ${status}` : status;
    calibrationMpp.textContent = `m/px: ${formatCalibrationNumber(metersPerPixel, 6)}`;
    calibrationPpm.textContent = `px/m: ${formatCalibrationNumber(pixelsPerMeter, 2)}`;
    updateVgcpOrthoVisibility();
}

async function refreshCalibrationStatus() {
    try {
        const res = await fetch('/calibration_status');
        const data = await res.json().catch(() => ({}));
        if (res.ok && data.ok) {
            updateCalibrationReadout(data);
        }
    } catch {
        updateCalibrationReadout({ ortho_ready: false, ortho_status: 'Calibration status unavailable' });
    }
}

function formatResultVelocity(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(3)} m/s` : '--';
}

function formatResultError(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric >= 0 ? '+' : ''}${numeric.toFixed(3)} m/s` : '--';
}

function formatResultPercent(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(2)}%` : '--';
}

function formatResultDuration(seconds) {
    const numeric = Number(seconds);
    if (!Number.isFinite(numeric) || numeric <= 0) return '--';
    return numeric >= 60 ? formatTime(numeric) : `${numeric.toFixed(1)}s`;
}

function optionalNumberValue(el) {
    const raw = el.value.trim();
    if (raw === '') return null;
    const value = Number.parseFloat(raw);
    return Number.isFinite(value) ? value : null;
}

function buildResultsParams() {
    return {
        float_method_value: optionalNumberValue(resultsFloatMethod),
    };
}

function resultsRequest(stop) {
    const params = buildResultsParams();
    const cleanParams = Object.fromEntries(
        Object.entries(params).filter(([, value]) => value !== null)
    );

    if (stop) {
        return {
            url: '/results_summary',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cleanParams),
            },
        };
    }

    const query = new URLSearchParams(cleanParams).toString();
    return {
        url: query ? `/results_summary?${query}` : '/results_summary',
        options: { method: 'GET' },
    };
}

function resultMetric(data, key) {
    return data && data.metrics && data.metrics[key] ? data.metrics[key] : { count: 0 };
}

function updateResultCard(key, metric) {
    document.getElementById(`result-${key}-median`).textContent = formatResultVelocity(metric.median);
    const errorText = Number.isFinite(Number(metric.percent_error))
        ? ` | ${formatResultPercent(metric.percent_error)} error`
        : '';
    document.getElementById(`result-${key}-count`).textContent = `${Number(metric.count) || 0} samples${errorText}`;
}

function updateTracerCountCard(metric) {
    const latest = Number(metric.latest);
    document.getElementById('result-tracer-count-median').textContent = Number.isFinite(latest)
        ? String(Math.round(latest))
        : '--';
    document.getElementById('result-tracer-count-samples').textContent = `${Number(metric.count) || 0} samples`;
}

function resultTableRow(label, metric) {
    return `
        <tr>
            <td>${label}</td>
            <td>${formatResultVelocity(metric.median)}</td>
            <td>${formatResultVelocity(metric.mean)}</td>
            <td>${formatResultVelocity(metric.latest)}</td>
            <td>${formatResultError(metric.error)}</td>
            <td>${formatResultPercent(metric.percent_error)}</td>
            <td>${Number(metric.count) || 0}</td>
        </tr>
    `;
}

function updateResultsDisplay(data) {
    const rowCount = Number(data && data.row_count) || 0;
    const totalRowCount = Number(data && data.total_row_count) || rowCount;
    const duration = formatResultDuration(data && data.duration_seconds);
    const frameStart = data && data.frame_start !== null && data.frame_start !== undefined ? data.frame_start : '--';
    const frameEnd = data && data.frame_end !== null && data.frame_end !== undefined ? data.frame_end : '--';
    const csvPath = data && data.csv_path ? data.csv_path : '--';

    resultsHeadline.textContent = rowCount > 0
        ? `${rowCount} selected / ${totalRowCount} processed frame records`
        : 'No results yet';
    resultsDuration.textContent = `Duration: ${duration}`;
    resultsFrames.textContent = `Frames: ${frameStart} - ${frameEnd}`;
    resultsCsv.textContent = `CSV: ${csvPath}`;

    const metrics = {
        surface: resultMetric(data, 'surface'),
        tracer: resultMetric(data, 'tracer'),
        tracerCount: resultMetric(data, 'total_trash_tracers'),
        piv: resultMetric(data, 'piv'),
        stiv: resultMetric(data, 'stiv'),
    };

    ['surface', 'tracer', 'piv', 'stiv'].forEach((key) => updateResultCard(key, metrics[key]));
    updateTracerCountCard(metrics.tracerCount);

    if (rowCount === 0) {
        resultsTbody.innerHTML = '<tr class="empty-row"><td colspan="7">No velocity samples yet.</td></tr>';
        return;
    }

    resultsTbody.innerHTML = [
        resultTableRow('Surface', metrics.surface),
        resultTableRow('Tracer', metrics.tracer),
        resultTableRow('PIV', metrics.piv),
        resultTableRow('STIV', metrics.stiv),
    ].join('');
}

async function loadResults({ stop = false } = {}) {
    const feedbackEl = document.getElementById('fb-results');
    showPanel('results');
    feedbackEl.className = 'post-feedback';
    feedbackEl.textContent = stop ? 'Stopping stream…' : 'Loading results…';

    try {
        const request = resultsRequest(stop);
        const res = await fetch(request.url, request.options);
        const data = await res.json().catch(() => ({}));

        if (!res.ok || !data.ok) {
            feedbackEl.className = 'post-feedback err';
            feedbackEl.textContent = `✗ ${data.error || `Server returned ${res.status}`}`;
            return;
        }

        updateResultsDisplay(data);

        if (stop) {
            setStreamStoppedUi('results ready');
        }

        feedbackEl.className = 'post-feedback ok';
        feedbackEl.textContent = `✓ Results updated (${rowCountText(data.row_count)})`;
    } catch {
        if (stop) {
            setStreamStoppedUi('results unavailable');
        }
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Could not load results';
    }
}

function rowCountText(rowCount) {
    const count = Number(rowCount) || 0;
    return `${count} frame record${count === 1 ? '' : 's'}`;
}

async function resetResults() {
    const feedbackEl = document.getElementById('fb-results');
    feedbackEl.className = 'post-feedback';
    feedbackEl.textContent = 'Resetting…';

    try {
        const res = await fetch('/validation_reset', { method: 'POST' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.ok) {
            feedbackEl.className = 'post-feedback err';
            feedbackEl.textContent = `✗ ${data.error || `Server returned ${res.status}`}`;
            return;
        }
        updateResultsDisplay({ ok: true, row_count: 0, metrics: {} });
        feedbackEl.className = 'post-feedback ok';
        feedbackEl.textContent = '✓ Results reset';
    } catch {
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Could not reset results';
    }
}

function updatePointCount() {
    const validRCount = points.filter(p => {
        const rValue = Number.parseFloat(p.r);
        return Number.isFinite(rValue) && rValue > 0;
    }).length;

    ptCount.textContent = `${points.length} point(s), ${validRCount} R set`;
}

function renderTable() {
    updatePointCount();
    tbody.innerHTML = '';

    if (points.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No points placed yet. Click on the video to add.</td></tr>';
        return;
    }

    points.forEach((p, i) => {
        const tr = document.createElement('tr');

        const tdIdx = document.createElement('td');
        tdIdx.innerHTML = `<span class="pt-badge">${i + 1}</span>`;

        const tdX = document.createElement('td');
        const inX = document.createElement('input');
        inX.type  = 'number';
        inX.value = p.x;
        inX.addEventListener('change', () => {
            points[i].x = Math.max(0, Math.min(parseInt(inX.value) || 0, feed.naturalWidth));
            inX.value = points[i].x;
            drawPoints();
        });
        tdX.appendChild(inX);

        const tdY = document.createElement('td');
        const inY = document.createElement('input');
        inY.type  = 'number';
        inY.value = p.y;
        inY.addEventListener('change', () => {
            points[i].y = Math.max(0, Math.min(parseInt(inY.value) || 0, feed.naturalHeight));
            inY.value = points[i].y;
            drawPoints();
        });
        tdY.appendChild(inY);

        const tdR = document.createElement('td');
        const inR = document.createElement('input');
        inR.type        = 'number';
        inR.step        = 'any';
        inR.min         = '0.000001';
        inR.value       = p.r;
        inR.placeholder = 'meters';
        inR.addEventListener('input', () => {
            points[i].r = inR.value;
            updatePointCount();
            drawPoints();
        });
        tdR.appendChild(inR);

        const tdDel = document.createElement('td');
        const btnDel = document.createElement('button');
        btnDel.className   = 'del-btn';
        btnDel.textContent = '✕';
        btnDel.title       = 'Remove point';
        btnDel.addEventListener('click', () => {
            points.splice(i, 1);
            drawPoints();
            renderTable();
        });
        tdDel.appendChild(btnDel);

        tr.append(tdIdx, tdX, tdY, tdR, tdDel);
        tbody.appendChild(tr);
    });
}

document.getElementById('btn-clear-all').addEventListener('click', () => {
    if (points.length === 0) return;
    if (confirm('Clear all ground control points?')) {
        points = [];
        drawPoints();
        renderTable();
    }
});

// ────────────────────────────────────────────────────────────────────
//  POST HELPERS
// ────────────────────────────────────────────────────────────────────
async function postJSON(path, body, feedbackEl) {
    feedbackEl.className = 'post-feedback';
    feedbackEl.textContent = 'Sending…';
    try {
        const res = await fetch(path, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
        });
        feedbackEl.className   = res.ok ? 'post-feedback ok' : 'post-feedback err';
        feedbackEl.textContent = res.ok ? '✓ Posted successfully' : `✗ Server returned ${res.status}`;
        return res.ok;
    } catch {
        feedbackEl.className   = 'post-feedback err';
        feedbackEl.textContent = '✗ Could not reach server';
        return false;
    }
}

async function postJSONData(path, body, feedbackEl) {
    feedbackEl.className = 'post-feedback';
    feedbackEl.textContent = 'Sending…';
    try {
        const res = await fetch(path, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        feedbackEl.className = res.ok ? 'post-feedback ok' : 'post-feedback err';
        feedbackEl.textContent = res.ok ? '✓ Posted successfully' : `✗ ${data.error || `Server returned ${res.status}`}`;
        return { ok: res.ok, data };
    } catch {
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Could not reach server';
        return { ok: false, data: {} };
    }
}

function numberValue(id, fallback) {
    const value = Number.parseFloat(document.getElementById(id).value);

    return Number.isFinite(value) ? value : fallback;
}

function buildComputerVisionPayload(roiOverride = null) {
    const roiEnabled = roiOverride
        ? roiOverride.enabled
        : document.getElementById('cv-roi-enabled').checked;
    const roiPoints = roiOverride ? roiOverride.points : cvRoiPoints;

    return {
        enable_piv: document.getElementById('enable-piv').checked,
        piv_interval: parseInt(document.getElementById('piv-interval').value) || 15,
        piv_max_size: parseInt(document.getElementById('piv-max-size').value) || 384,
        cv_roi_enabled: roiEnabled,
        cv_roi_points: roiPoints.map(p => ({ x: p.x / 100, y: p.y / 100 })),
        enable_stiv: document.getElementById('enable-stiv').checked,
        stiv_history: parseInt(document.getElementById('stiv-history').value) || 48,
        stiv_start_x: numberValue('stiv-start-x', 5) / 100,
        stiv_start_y: numberValue('stiv-start-y', 50) / 100,
        stiv_end_x: numberValue('stiv-end-x', 95) / 100,
        stiv_end_y: numberValue('stiv-end-y', 50) / 100,
    };
}

async function deleteCvRoi() {
    const feedbackEl = document.getElementById('fb-cv');
    clearCvRoi();

    const ok = await postJSON('/yolo_params', buildComputerVisionPayload({
        enabled: false,
        points: [],
    }), feedbackEl);

    if (!ok) return;

    feedbackEl.className = 'post-feedback ok';
    feedbackEl.textContent = '✓ ROI deleted';
    await disconnectStream();
    connectStream();
}

function populateModelSelect(select, models, selectedPath) {
    select.innerHTML = '';
    if (!models.length) {
        select.innerHTML = '<option value="">No models found</option>';
        return;
    }
    models.forEach(path => {
        const option = document.createElement('option');
        option.value = path;
        option.textContent = path.replace(/^\.?\//, '');
        option.selected = path === selectedPath;
        select.appendChild(option);
    });
}

async function loadModelOptions() {
    const feedbackEl = document.getElementById('fb-yolo');
    try {
        const res = await fetch('/model_options');
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.ok) {
            throw new Error(data.error || `Server returned ${res.status}`);
        }
        const models = Array.isArray(data.models) ? data.models : [];
        populateModelSelect(morphologyModelSelect, models, data.morphology_model_path);
        populateModelSelect(tracerModelSelect, models, data.tracer_model_path);
    } catch (error) {
        morphologyModelSelect.innerHTML = '<option value="">Could not load models</option>';
        tracerModelSelect.innerHTML = '<option value="">Could not load models</option>';
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = `✗ ${error.message || 'Could not load models'}`;
    }
}

['stiv-start-x', 'stiv-start-y', 'stiv-end-x', 'stiv-end-y'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => {
        syncStivLineFromInputs();
        drawPoints();
    });
});

document.getElementById('btn-draw-cv-roi').addEventListener('click', () => {
    if (isDrawingCvRoi) {
        setCvRoiDrawing(false);
        return;
    }
    cvRoiPoints = [];
    document.getElementById('cv-roi-enabled').checked = true;
    setCvRoiDrawing(true);
});

document.getElementById('btn-clear-cv-roi').addEventListener('click', deleteCvRoi);
document.getElementById('cv-roi-enabled').addEventListener('change', e => {
    if (!e.target.checked) {
        deleteCvRoi();
        return;
    }
    drawPoints();
});
updateCvRoiCount();

async function uploadVideo() {
    const input = document.getElementById('video-file');
    const feedbackEl = document.getElementById('fb-video');

    if (!input.files.length) {
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Choose a video first';
        return;
    }

    const formData = new FormData();
    formData.append('video', input.files[0]);

    feedbackEl.className = 'post-feedback';
    feedbackEl.textContent = 'Uploading…';

    try {
        const res = await fetch('/upload_video', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json().catch(() => ({}));

        feedbackEl.className = res.ok ? 'post-feedback ok' : 'post-feedback err';
        feedbackEl.textContent = res.ok ? '✓ Video uploaded' : `✗ ${data.error || 'Upload failed'}`;

        if (!res.ok) return;

        updateCalibrationReadout(data);
        updateResultsDisplay({ ok: true, row_count: 0, metrics: {} });
        await disconnectStream();
        clearCvRoi();
        points = [];
        dragging = null;
        drawPoints();
        renderTable();
        connectStream();
    } catch {
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Could not reach server';
    }
}

async function loadCameraDevices() {
    const feedbackEl = document.getElementById('fb-video');

    cameraSelect.disabled = true;
    cameraSelect.innerHTML = '<option value="">Scanning cameras...</option>';

    try {
        const res = await fetch('/camera_devices');
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(data.error || `Server returned ${res.status}`);
        }

        cameraSelect.innerHTML = '';

        if (!data.devices || data.devices.length === 0) {
            cameraSelect.innerHTML = '<option value="">No cameras found</option>';
            feedbackEl.className = 'post-feedback err';
            feedbackEl.textContent = '✗ No available cameras found';
            return;
        }

        data.devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.index;
            option.textContent = device.label;
            cameraSelect.appendChild(option);
        });

        feedbackEl.className = 'post-feedback ok';
        feedbackEl.textContent = `✓ Found ${data.devices.length} camera(s)`;
    } catch (error) {
        cameraSelect.innerHTML = '<option value="">Could not load cameras</option>';
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = `✗ ${error.message || 'Could not load cameras'}`;
    } finally {
        cameraSelect.disabled = false;
    }
}

async function useCameraSource() {
    const feedbackEl = document.getElementById('fb-video');
    const cameraIndex = parseInt(cameraSelect.value);

    if (!Number.isInteger(cameraIndex)) {
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Choose a camera first';
        return;
    }

    const result = await postJSONData('/set_camera_source', {
        camera_index: cameraIndex,
    }, feedbackEl);

    if (!result.ok) return;

    feedbackEl.className = 'post-feedback ok';
    feedbackEl.textContent = `✓ Camera ${result.data.camera_index} selected`;
    updateCalibrationReadout(result.data);
    updateResultsDisplay({ ok: true, row_count: 0, metrics: {} });
    document.getElementById('video-ortho').checked = false;
    updateVgcpOrthoVisibility();
    clearCvRoi();
    points = [];
    dragging = null;
    drawPoints();
    renderTable();
    await disconnectStream();
    connectStream();
}

async function resetOrthorectification() {
    const feedbackEl = document.getElementById('fb-video');
    const result = await postJSONData('/reset_orthorectification', {}, feedbackEl);

    if (!result.ok) return;

    feedbackEl.className = 'post-feedback ok';
    feedbackEl.textContent = '✓ Raw video reset; orthorectification cleared';
    updateCalibrationReadout(result.data);
    document.getElementById('video-ortho').checked = false;
    updateVgcpOrthoVisibility();
    clearCvRoi();
    points = [];
    dragging = null;
    drawPoints();
    renderTable();
    await disconnectStream();
    connectStream();
}

document.getElementById('btn-upload-video').addEventListener('click', uploadVideo);
document.getElementById('btn-use-camera').addEventListener('click', useCameraSource);
document.getElementById('btn-refresh-cameras').addEventListener('click', loadCameraDevices);
document.getElementById('btn-reset-ortho').addEventListener('click', resetOrthorectification);
document.getElementById('btn-stop-results').addEventListener('click', () => loadResults({ stop: true }));
document.getElementById('btn-refresh-results').addEventListener('click', () => loadResults());
document.getElementById('btn-reset-results').addEventListener('click', resetResults);
loadCameraDevices();
loadModelOptions();
refreshCalibrationStatus();
updateResultsDisplay({ ok: true, row_count: 0, metrics: {} });

document.getElementById('btn-post-vgcp').addEventListener('click', async () => {
    const feedbackEl = document.getElementById('fb-vgcp');
    const result = await postJSONData('/vgcp', { points: points.map(p => ({ x: p.x, y: p.y, r: p.r })) }, feedbackEl);
    const ok = result.ok;
    if (ok) {
        updateCalibrationReadout(result.data);
        feedbackEl.textContent = result.data.ortho_ready
            ? `✓ Orthorectification ready (${formatCalibrationNumber(result.data.meters_per_pixel, 6)} m/px)`
            : `✗ ${result.data.ortho_status || 'Orthorectification not ready'}`;
        feedbackEl.className = result.data.ortho_ready ? 'post-feedback ok' : 'post-feedback err';
        if (result.data.ortho_ready) {
            document.getElementById('video-ortho').checked = true;
            updateVgcpOrthoVisibility();
        }
    }
    if (!ok) return;
    await disconnectStream();
    connectStream();
});

document.getElementById('btn-post-cam').addEventListener('click', async () => {
    const ok = await postJSON('/cam_settings', {
        cam_fov:       parseFloat(document.getElementById('cam-fov').value)       || 72.4,
        meters_per_pixel: parseFloat(document.getElementById('raw-meters-per-pixel').value) || 0.02,
        is_ortho:      document.getElementById('video-ortho').checked,
    }, document.getElementById('fb-cam'));
    if (!ok) return;
    refreshCalibrationStatus();
    updateVgcpOrthoVisibility();
    await disconnectStream();
    connectStream();
});

document.getElementById('btn-post-yolo').addEventListener('click', async () => {
    const ok = await postJSON('/yolo_params', {
        morphology_model_path: morphologyModelSelect.value,
        tracer_model_path: tracerModelSelect.value,
        threshold: parseFloat(document.getElementById('yolo-threshold').value) || 0.25,
        morphology_threshold: parseFloat(document.getElementById('morphology-threshold').value) || 0.35,
        tracer_threshold: parseFloat(document.getElementById('yolo-threshold').value) || 0.25,
        tracer_target_kind: document.getElementById('tracer-target-kind').value || 'all',
        target_sz: parseInt(document.getElementById('target-size').value) || 640,
        morphology_imgsz: parseInt(document.getElementById('morphology-size').value) || 640,
        morphology_mask_erode: parseInt(document.getElementById('morphology-mask-erode').value) || 0,
        tracer_imgsz: parseInt(document.getElementById('tracer-size').value) || 1024,
        stream_max_width: parseInt(document.getElementById('stream-max-width').value) || 640,
        target_stream_fps: parseFloat(document.getElementById('target-stream-fps').value) || 25,
        detect_interval: parseInt(document.getElementById('detect-interval').value) || 8,
        morphology_interval: parseInt(document.getElementById('morphology-interval').value) || 36,
        min_tracked_points: parseInt(document.getElementById('min-tracked-points').value) || 2,
    }, document.getElementById('fb-yolo'));
    if (!ok) return;
    await disconnectStream();
    connectStream();
});

document.getElementById('btn-post-cv').addEventListener('click', async () => {
    const feedbackEl = document.getElementById('fb-cv');
    const roiEnabled = document.getElementById('cv-roi-enabled').checked;

    if (isDrawingCvRoi) {
        setCvRoiDrawing(false);
    }

    if (roiEnabled && cvRoiPoints.length < 3) {
        feedbackEl.className = 'post-feedback err';
        feedbackEl.textContent = '✗ Draw at least 3 ROI points';
        return;
    }

    const ok = await postJSON('/yolo_params', buildComputerVisionPayload(), feedbackEl);
    if (!ok) return;
    await disconnectStream();
    connectStream();
});
