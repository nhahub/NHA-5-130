/**
 * Aerial Image Change Detection Studio
 * Frontend Application Controller
 */

// State
let currentModel = 'ensemble';
let currentThreshold = 0.50;
let customImageA = null;
let customImageB = null;
let currentViewMode = 'grid';
let lastMaskDataUri = null;
let lastOverlayDataUri = null;
let lastResponseData = null;
let isDraggingSwipe = false;

// 1. Tab Navigation
function switchTab(el, tabId) {
  try {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    if (el) {
      el.classList.add('active');
    }
    const target = document.getElementById('tab-' + tabId);
    if (target) {
      target.classList.add('active');
    }
  } catch (err) {
    console.error('Error switching tab:', err);
  }
}

// 2. Model Selection
function selectModel(modelName, el) {
  currentModel = modelName;
  document.querySelectorAll('.radio-card').forEach(c => c.classList.remove('selected'));
  if (el) {
    el.classList.add('selected');
    const radio = el.querySelector('input[type="radio"]');
    if (radio) radio.checked = true;
  }
}

// 3. Threshold Slider
function onThresholdChange(val) {
  currentThreshold = parseFloat(val);
  const display = document.getElementById('threshold-val');
  if (display) {
    display.innerText = currentThreshold.toFixed(2);
  }
}

// 4. Sample Mode Toggle (Preset vs Upload)
function onSampleModeChange() {
  const modeSel = document.getElementById('sample-mode');
  const mode = modeSel ? modeSel.value : 'preset';
  const presetControls = document.getElementById('preset-controls');
  const uploadControls = document.getElementById('upload-controls');

  if (mode === 'preset') {
    if (presetControls) presetControls.style.display = 'block';
    if (uploadControls) uploadControls.style.display = 'none';
    onPresetSelect();
  } else {
    if (presetControls) presetControls.style.display = 'none';
    if (uploadControls) uploadControls.style.display = 'block';
  }
}

// 5. Preset Selection
function onPresetSelect() {
  const sel = document.getElementById('preset-select');
  if (!sel) return;
  const stem = sel.value;
  if (!stem) return;

  const badge = document.getElementById('sample-title-badge');
  if (badge) {
    badge.innerText = 'SCENE: ' + stem;
  }

  const srcA = '/api/sample?id=' + encodeURIComponent(stem) + '&type=a';
  const srcB = '/api/sample?id=' + encodeURIComponent(stem) + '&type=b';
  const srcMask = '/api/sample?id=' + encodeURIComponent(stem) + '&type=mask';

  const boxA = document.getElementById('box-a');
  const boxB = document.getElementById('box-b');
  const boxGT = document.getElementById('box-gt');

  if (boxA) boxA.innerHTML = '<img src="' + srcA + '" alt="Time 1">';
  if (boxB) boxB.innerHTML = '<img src="' + srcB + '" alt="Time 2">';
  if (boxGT) boxGT.innerHTML = '<img src="' + srcMask + '" alt="Ground Truth">';

  const swipeA = document.getElementById('swipe-img-a');
  const swipeB = document.getElementById('swipe-img-b');
  if (swipeA) swipeA.src = srcA;
  if (swipeB) swipeB.src = srcB;
  setSwipePos(50);

  // Reset outputs
  const boxOverlay = document.getElementById('box-overlay');
  const boxMask = document.getElementById('box-mask');
  const boxError = document.getElementById('box-error');
  const geoCard = document.getElementById('geo-analytics-card');

  if (boxOverlay) boxOverlay.innerHTML = '<div class="placeholder-text">Click "Run Change Detection"</div>';
  if (boxMask) boxMask.innerHTML = '<div class="placeholder-text">Click "Run Change Detection"</div>';
  if (boxError) boxError.innerHTML = '<div class="placeholder-text">Click "Run Change Detection"</div>';
  if (geoCard) geoCard.style.display = 'none';

  clearMetrics();
}

// 6. Custom Upload
function onFileUpload(type) {
  const input = document.getElementById(type === 'a' ? 'file-a' : 'file-b');
  if (!input || !input.files || !input.files[0]) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    const dataUrl = e.target.result;
    if (type === 'a') {
      customImageA = dataUrl;
      const boxA = document.getElementById('box-a');
      if (boxA) boxA.innerHTML = '<img src="' + dataUrl + '" alt="Image A">';
      const swipeA = document.getElementById('swipe-img-a');
      if (swipeA) swipeA.src = dataUrl;
    } else {
      customImageB = dataUrl;
      const boxB = document.getElementById('box-b');
      if (boxB) boxB.innerHTML = '<img src="' + dataUrl + '" alt="Image B">';
      const swipeB = document.getElementById('swipe-img-b');
      if (swipeB) swipeB.src = dataUrl;
    }
  };
  reader.readAsDataURL(input.files[0]);
}

// 7. View Switch (6-Up Grid vs Swipe)
function switchView(mode) {
  currentViewMode = mode;
  const btnGrid = document.getElementById('btn-view-grid');
  const btnSwipe = document.getElementById('btn-view-swipe');
  const gridView = document.getElementById('grid-view-wrapper');
  const swipeView = document.getElementById('swipe-view-wrapper');

  if (mode === 'swipe') {
    if (btnGrid) btnGrid.classList.remove('active');
    if (btnSwipe) btnSwipe.classList.add('active');
    if (gridView) gridView.style.display = 'none';
    if (swipeView) swipeView.style.display = 'block';
    setSwipePos(50);
  } else {
    if (btnSwipe) btnSwipe.classList.remove('active');
    if (btnGrid) btnGrid.classList.add('active');
    if (swipeView) swipeView.style.display = 'none';
    if (gridView) gridView.style.display = 'grid';
  }
}

// 8. Swipe Divider Controls
function setSwipePos(pct) {
  pct = Math.max(0, Math.min(100, pct));
  const b = document.getElementById('swipe-img-b');
  const line = document.getElementById('swipe-divider-line');
  const handle = document.getElementById('swipe-handle');
  if (b) b.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
  if (line) line.style.left = pct + '%';
  if (handle) handle.style.left = 'calc(' + pct + '% - 18px)';
}

function initSwipeListeners() {
  const container = document.getElementById('swipe-view-wrapper');
  if (!container) return;

  const onMove = (e) => {
    if (!isDraggingSwipe) return;
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const pct = ((clientX - rect.left) / rect.width) * 100.0;
    setSwipePos(pct);
  };

  container.addEventListener('mousedown', (e) => { isDraggingSwipe = true; onMove(e); });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', () => { isDraggingSwipe = false; });

  container.addEventListener('touchstart', (e) => { isDraggingSwipe = true; onMove(e); }, { passive: true });
  window.addEventListener('touchmove', onMove, { passive: true });
  window.addEventListener('touchend', () => { isDraggingSwipe = false; });
}

// 9. Inference Execution
async function runDetection() {
  const btn = document.getElementById('btn-run');
  const spinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  const modeSel = document.getElementById('sample-mode');
  const mode = modeSel ? modeSel.value : 'preset';

  const ttaCheckbox = document.getElementById('tta-checkbox');
  const postCheckbox = document.getElementById('postprocess-checkbox');
  const tta = ttaCheckbox ? ttaCheckbox.checked : false;
  const postproc = postCheckbox ? postCheckbox.checked : true;

  if (btn) btn.disabled = true;
  if (spinner) spinner.style.display = 'inline-block';
  if (btnText) btnText.innerText = 'Detecting Changes...';

  const payload = {
    model: currentModel,
    threshold: currentThreshold,
    tta: tta,
    postprocess: postproc
  };

  if (mode === 'preset') {
    const sel = document.getElementById('preset-select');
    if (!sel || !sel.value) {
      alert('Please wait for samples to load or select a test scene.');
      resetRunBtn();
      return;
    }
    payload.sample_id = sel.value;
  } else {
    if (!customImageA || !customImageB) {
      alert('Please upload both Image A and Image B.');
      resetRunBtn();
      return;
    }
    payload.image_a = customImageA;
    payload.image_b = customImageB;
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) {
      alert('Inference error: ' + data.error);
      resetRunBtn();
      return;
    }

    lastMaskDataUri = data.mask_png;
    lastOverlayDataUri = data.overlay_png;
    lastResponseData = data;

    const boxOverlay = document.getElementById('box-overlay');
    const boxMask = document.getElementById('box-mask');
    const boxError = document.getElementById('box-error');

    if (boxOverlay) boxOverlay.innerHTML = '<img src="' + data.overlay_png + '" alt="Overlay">';
    if (boxMask) boxMask.innerHTML = '<img src="' + data.mask_png + '" alt="Mask">';

    const swipeB = document.getElementById('swipe-img-b');
    if (swipeB) swipeB.src = data.overlay_png;

    if (boxError) {
      if (data.error_map_png) {
        boxError.innerHTML = '<img src="' + data.error_map_png + '" alt="Error Map">';
      } else {
        boxError.innerHTML = '<div class="placeholder-text">Ground truth not available for custom uploads</div>';
      }
    }

    // Update metrics
    if (data.metrics) {
      const m = data.metrics;
      setMetric('m-iou', m.iou != null ? m.iou.toFixed(4) : 'N/A');
      setMetric('m-f1', m.f1 != null ? m.f1.toFixed(4) : 'N/A');
      setMetric('m-prec', m.precision != null ? m.precision.toFixed(4) : 'N/A');
      setMetric('m-rec', m.recall != null ? m.recall.toFixed(4) : 'N/A');
      setMetric('m-area', m.changed_percent != null ? m.changed_percent.toFixed(2) + '%' : '-');
    }
    setMetric('m-time', data.inference_time_ms + ' ms');

    // Update geospatial stats
    if (data.geo_stats) {
      const g = data.geo_stats;
      const card = document.getElementById('geo-analytics-card');
      if (card) card.style.display = 'block';

      setText('geo-area', (g.est_ground_area_m2 || 0).toLocaleString() + ' m² (' + (g.est_ground_area_ha || 0) + ' ha)');
      setText('geo-clusters', (g.cluster_count || 0) + ' structures');
      setText('geo-avg-size', (g.avg_footprint_m2 || 0).toLocaleString() + ' m²');
      setText('geo-max-size', (g.max_footprint_m2 || 0).toLocaleString() + ' m²');

      const badge = document.getElementById('geo-density-badge');
      if (badge) {
        badge.innerText = g.density_label || 'Calculated';
        badge.style.color = g.density_color || '#00E5FF';
        badge.style.border = '1px solid ' + (g.density_color || '#00E5FF');
      }
    }

  } catch (err) {
    console.error('Inference request failed:', err);
    alert('Inference failed: ' + err.message);
  } finally {
    resetRunBtn();
  }
}

function resetRunBtn() {
  const btn = document.getElementById('btn-run');
  const spinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  if (btn) btn.disabled = false;
  if (spinner) spinner.style.display = 'none';
  if (btnText) btnText.innerText = '🚀 Run Change Detection';
}

function setMetric(id, val) {
  const el = document.getElementById(id);
  if (el) el.innerText = val;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.innerText = val;
}

function clearMetrics() {
  ['m-iou', 'm-f1', 'm-prec', 'm-rec', 'm-area', 'm-time'].forEach(id => {
    setMetric(id, '-');
  });
}

// 10. Downloads & Copy
function downloadMask() {
  if (!lastMaskDataUri) {
    alert('Please run change detection first to generate a mask.');
    return;
  }
  const sel = document.getElementById('preset-select');
  const name = (sel ? sel.value : 'custom') || 'custom';
  const a = document.createElement('a');
  a.href = lastMaskDataUri;
  a.download = 'change_mask_' + name + '.png';
  a.click();
}

function downloadOverlay() {
  if (!lastOverlayDataUri) {
    alert('Please run change detection first to generate an overlay.');
    return;
  }
  const sel = document.getElementById('preset-select');
  const name = (sel ? sel.value : 'custom') || 'custom';
  const a = document.createElement('a');
  a.href = lastOverlayDataUri;
  a.download = 'change_overlay_' + name + '.png';
  a.click();
}

function copyStats() {
  if (!lastResponseData) {
    alert('Please run change detection first.');
    return;
  }
  const sel = document.getElementById('preset-select');
  const scene = (sel ? sel.value : 'custom') || 'custom';
  const m = lastResponseData.metrics || {};
  const g = lastResponseData.geo_stats || {};

  const lines = [
    '=== LEVIR-CD+ Aerial Change Detection Report ===',
    'Scene Identifier    : ' + scene,
    'Model Engine        : ' + currentModel.toUpperCase(),
    'Decision Threshold  : ' + currentThreshold.toFixed(2),
    'Sample IoU          : ' + (m.iou != null ? m.iou.toFixed(4) : 'N/A'),
    'Sample F1-Score     : ' + (m.f1 != null ? m.f1.toFixed(4) : 'N/A'),
    'Precision / Recall  : ' + (m.precision != null ? m.precision.toFixed(4) : 'N/A') + ' / ' + (m.recall != null ? m.recall.toFixed(4) : 'N/A'),
    'Changed Ground Area : ' + (m.changed_percent ? m.changed_percent.toFixed(2) : '-') + '% (' + (g.est_ground_area_m2 || '-') + ' m², ' + (g.est_ground_area_ha || '-') + ' ha)',
    'Building Footprints : ' + (g.cluster_count || '-') + ' structures detected',
    'Mean Footprint Size : ' + (g.avg_footprint_m2 || '-') + ' m²',
    'Max Complex Size    : ' + (g.max_footprint_m2 || '-') + ' m²',
    'Activity Density    : ' + (g.density_label || '-'),
    'Inference Latency   : ' + lastResponseData.inference_time_ms + ' ms'
  ];
  const txt = lines.join('\n');

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(txt).then(() => {
      alert('Inspection report copied to clipboard!');
    }).catch(() => {
      prompt('Copy inspection report:', txt);
    });
  } else {
    prompt('Copy inspection report:', txt);
  }
}

// 11. Initial Boot
function initApp() {
  initSwipeListeners();
  setSwipePos(50);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
