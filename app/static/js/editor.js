/* ============================================================
   Image Text Editor — Frontend Engine
   Handles: upload, canvas masking, inpainting API calls,
            text overlay placement, save/download.
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────
const S = {
  sessionId:    null,
  currentStep:  1,
  imageW:       0,
  imageH:       0,

  // Canvas transform
  scale:        1.0,
  offsetX:      0,
  offsetY:      0,

  // Mask tool
  tool:         'brush',     // 'brush' | 'eraser'
  maskOpacity:  0.5,
  maskVisible:  true,
  isDrawing:    false,
  lastX:        0,
  lastY:        0,
  maskHistory:  [],          // undo stack (ImageData)
  maskRedo:     [],          // redo stack

  // Inpainting
  engine:       'iopaint',   // 'iopaint' | 'comfyui'
  inpaintDone:  false,

  // Text placement
  textX:        50,
  textY:        50,
  textAlign:    'left',
  textBold:     false,
  textItalic:   false,
  placingText:  false,

  // Current display image URL (for comparison)
  urls: {
    source:     null,
    inpainted:  null,
    withText:   null,
    current:    null,
  },

  // Save
  downloadUrl:  null,

  // Sampled original-text style (from Step 3 inpaint or /api/sample-text)
  sample: null,        // { color, font_size, bold, italic, best_family, matches, ... }
  sampleUrl: null,     // URL of the original text crop image
};

// ── Canvas refs ─────────────────────────────────────────────
let cvBase, ctxBase, cvMask, ctxMask, cvText, ctxText;

// ── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  cvBase = document.getElementById('cvBase');
  cvMask = document.getElementById('cvMask');
  cvText = document.getElementById('cvText');
  ctxBase = cvBase.getContext('2d');
  ctxMask = cvMask.getContext('2d');
  ctxText = cvText.getContext('2d');

  setupDropZone();
  setupMaskCanvas();
  checkHealth();

  // file inputs (header + panel)
  document.getElementById('fileInput').addEventListener('change', e => handleFileSelect(e.target.files[0]));
  document.getElementById('fileInput2').addEventListener('change', e => handleFileSelect(e.target.files[0]));

  // Model select → show/hide prompt fields
  document.getElementById('modelSelect').addEventListener('change', () => {
    const sdMode = document.getElementById('modelSelect').value === 'sd-inpainting';
    document.getElementById('promptField').style.display    = sdMode ? 'block' : 'none';
    document.getElementById('negPromptField').style.display = sdMode ? 'block' : 'none';
  });
});

// ── Health check ────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    setDot('dotIopaint', d.iopaint === 'ok');
    setDot('dotComfy',   d.comfyui === 'ok');
  } catch {
    setDot('dotIopaint', false);
    setDot('dotComfy',   false);
  }
}

function setDot(id, ok) {
  const el = document.getElementById(id);
  el.classList.toggle('ok',  ok);
  el.classList.toggle('err', !ok);
}

// ── Step navigation ──────────────────────────────────────────
function goStep(n) {
  // Don't allow jumping ahead of current progress
  if (n > 1 && !S.sessionId)      { toast('Upload an image first.', 'error'); return; }
  if (n > 3 && !S.inpaintDone)    { /* allow skipping inpaint */ }

  // Hide all panels
  document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
  document.getElementById(`panel${n}`).classList.remove('hidden');

  // Update step nav
  document.querySelectorAll('.step').forEach(s => {
    const sn = +s.dataset.step;
    s.classList.toggle('active', sn === n);
    s.classList.toggle('done',   sn < n);
  });

  S.currentStep = n;

  // Step-specific setup
  if (n === 2) activateMaskTool();
  if (n === 4) activateTextTool();
  if (n !== 4) deactivateTextTool();
  if (n === 5) setupSavePanel();
}

// ── Drop zone / file upload ──────────────────────────────────
function setupDropZone() {
  const dz = document.getElementById('dropZone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelect(f);
  });
  dz.addEventListener('click', () => document.getElementById('fileInput').click());
}

async function handleFileSelect(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { toast('Please select an image file.', 'error'); return; }

  showLoading('Uploading…');
  const fd = new FormData();
  fd.append('image', file);

  try {
    const r  = await fetch('/api/upload', { method: 'POST', body: fd });
    const d  = await r.json();
    if (!r.ok) throw new Error(d.error || 'Upload failed');

    S.sessionId = d.session_id;
    S.imageW    = d.width;
    S.imageH    = d.height;
    S.urls.source  = d.image_url;
    S.urls.current = d.image_url;

    // Show info
    document.getElementById('imageInfo').textContent =
      `${file.name}  •  ${d.width} × ${d.height}px  •  ${(file.size/1024).toFixed(0)} KB`;
    document.getElementById('imageInfo').classList.remove('hidden');
    document.getElementById('btnStep1Next').disabled = false;

    // Draw on canvas
    await loadImageToCanvas(d.image_url);
    showCanvasStack();
    document.getElementById('zoomControls').classList.remove('hidden');

    toast('Image uploaded ✓', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

// ── Canvas rendering ─────────────────────────────────────────
async function loadImageToCanvas(url) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      resizeCanvases(img.width, img.height);
      ctxBase.drawImage(img, 0, 0);
      clearMask();
      res();
    };
    img.onerror = rej;
    img.src = url + '?t=' + Date.now();
  });
}

function resizeCanvases(w, h) {
  const wrapper  = document.getElementById('canvasWrapper');
  const wrapW    = wrapper.clientWidth  - 40;
  const wrapH    = wrapper.clientHeight - 40;
  S.scale        = Math.min(wrapW / w, wrapH / h, 1.0);

  const dw = Math.round(w * S.scale);
  const dh = Math.round(h * S.scale);

  for (const cv of [cvBase, cvMask, cvText]) {
    cv.width  = dw;
    cv.height = dh;
    cv.style.width  = dw + 'px';
    cv.style.height = dh + 'px';
  }

  const stack = document.getElementById('canvasStack');
  stack.style.width  = dw + 'px';
  stack.style.height = dh + 'px';

  document.getElementById('zoomLabel').textContent = Math.round(S.scale * 100) + '%';
}

function showCanvasStack() {
  document.getElementById('dropZone').classList.add('hidden');
  document.getElementById('canvasStack').classList.remove('hidden');
}

// ── Zoom ─────────────────────────────────────────────────────
function zoom(delta) {
  if (!S.sessionId) return;
  S.scale = Math.max(0.1, Math.min(4.0, S.scale + delta));
  applyZoom();
}

function zoomFit() {
  S.scale = 1.0;
  resizeCanvases(S.imageW, S.imageH);
  reloadBase();
}

function applyZoom() {
  const dw = Math.round(S.imageW * S.scale);
  const dh = Math.round(S.imageH * S.scale);
  for (const cv of [cvBase, cvMask, cvText]) {
    cv.width  = dw; cv.height = dh;
    cv.style.width = dw+'px'; cv.style.height = dh+'px';
  }
  document.getElementById('canvasStack').style.width  = dw+'px';
  document.getElementById('canvasStack').style.height = dh+'px';
  document.getElementById('zoomLabel').textContent = Math.round(S.scale*100)+'%';
  reloadBase();
}

async function reloadBase() {
  if (!S.urls.current) return;
  const img = new Image();
  img.onload = () => ctxBase.drawImage(img, 0, 0, cvBase.width, cvBase.height);
  img.src = S.urls.current + '?t=' + Date.now();
}

// ── Mask tool setup ──────────────────────────────────────────
function activateMaskTool() {
  cvMask.style.pointerEvents = 'auto';
  cvMask.style.cursor = 'crosshair';
}

function setupMaskCanvas() {
  // Mouse
  cvMask.addEventListener('mousedown',  maskStart);
  cvMask.addEventListener('mousemove',  maskMove);
  cvMask.addEventListener('mouseup',    maskEnd);
  cvMask.addEventListener('mouseleave', maskEnd);

  // Touch
  cvMask.addEventListener('touchstart',  e => { e.preventDefault(); maskStart(e.touches[0]); }, {passive:false});
  cvMask.addEventListener('touchmove',   e => { e.preventDefault(); maskMove(e.touches[0]);  }, {passive:false});
  cvMask.addEventListener('touchend',    maskEnd);
}

function getCanvasPos(e) {
  const rect = cvMask.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left),
    y: (e.clientY - rect.top),
  };
}

function maskStart(e) {
  if (S.currentStep !== 2) return;
  S.isDrawing = true;
  const pos = getCanvasPos(e);
  S.lastX = pos.x; S.lastY = pos.y;
  // Push current state for undo
  S.maskHistory.push(ctxMask.getImageData(0, 0, cvMask.width, cvMask.height));
  S.maskRedo = [];
  drawMaskDot(pos.x, pos.y);
}

function maskMove(e) {
  if (!S.isDrawing || S.currentStep !== 2) return;
  const pos = getCanvasPos(e);
  drawMaskLine(S.lastX, S.lastY, pos.x, pos.y);
  S.lastX = pos.x; S.lastY = pos.y;
}

function maskEnd() { S.isDrawing = false; }

function drawMaskDot(x, y) {
  const r = +document.getElementById('brushSize').value / 2;
  ctxMask.save();
  if (S.tool === 'eraser') {
    ctxMask.globalCompositeOperation = 'destination-out';
    ctxMask.fillStyle = 'rgba(255,255,255,1)';
  } else {
    ctxMask.globalCompositeOperation = 'source-over';
    ctxMask.fillStyle = `rgba(235, 87, 87, ${S.maskOpacity})`;
  }
  ctxMask.beginPath();
  ctxMask.arc(x, y, r, 0, Math.PI * 2);
  ctxMask.fill();
  ctxMask.restore();
}

function drawMaskLine(x1, y1, x2, y2) {
  const r = +document.getElementById('brushSize').value / 2;
  ctxMask.save();
  if (S.tool === 'eraser') {
    ctxMask.globalCompositeOperation = 'destination-out';
    ctxMask.strokeStyle = 'rgba(255,255,255,1)';
  } else {
    ctxMask.globalCompositeOperation = 'source-over';
    ctxMask.strokeStyle = `rgba(235, 87, 87, ${S.maskOpacity})`;
  }
  ctxMask.lineWidth  = r * 2;
  ctxMask.lineCap    = 'round';
  ctxMask.lineJoin   = 'round';
  ctxMask.beginPath();
  ctxMask.moveTo(x1, y1);
  ctxMask.lineTo(x2, y2);
  ctxMask.stroke();
  ctxMask.restore();
  drawMaskDot(x2, y2);
}

function setTool(t) {
  S.tool = t;
  document.getElementById('btnBrush').classList.toggle('active',  t === 'brush');
  document.getElementById('btnEraser').classList.toggle('active', t === 'eraser');
}

function updateMaskOpacity(val) {
  S.maskOpacity = val / 100;
  document.getElementById('maskOpVal').textContent = val;
}

function toggleMaskVisibility(visible) {
  S.maskVisible = visible;
  cvMask.style.opacity = visible ? '1' : '0';
}

function clearMask() {
  S.maskHistory.push(ctxMask.getImageData(0, 0, cvMask.width, cvMask.height));
  ctxMask.clearRect(0, 0, cvMask.width, cvMask.height);
}

function undoMask() {
  if (!S.maskHistory.length) return;
  S.maskRedo.push(ctxMask.getImageData(0, 0, cvMask.width, cvMask.height));
  ctxMask.putImageData(S.maskHistory.pop(), 0, 0);
}

function redoMask() {
  if (!S.maskRedo.length) return;
  S.maskHistory.push(ctxMask.getImageData(0, 0, cvMask.width, cvMask.height));
  ctxMask.putImageData(S.maskRedo.pop(), 0, 0);
}

// Export mask as B&W PNG (data URL)
function getMaskDataURL() {
  // Create offscreen canvas at ORIGINAL image resolution
  const off = document.createElement('canvas');
  off.width  = S.imageW;
  off.height = S.imageH;
  const ctx = off.getContext('2d');

  // Draw the displayed mask scaled back up to original size
  ctx.drawImage(cvMask, 0, 0, S.imageW, S.imageH);

  // Convert the alpha channel to B&W (white = painted, black = clear)
  const id = ctx.getImageData(0, 0, S.imageW, S.imageH);
  const px = id.data;
  for (let i = 0; i < px.length; i += 4) {
    const a = px[i + 3];
    const v = a > 20 ? 255 : 0;
    px[i] = px[i+1] = px[i+2] = v;
    px[i+3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return off.toDataURL('image/png');
}

// ── Inpainting ───────────────────────────────────────────────
function selectEngine(btn) {
  document.querySelectorAll('#engineSeg .seg-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  S.engine = btn.dataset.val;
}

async function runInpaint() {
  if (!S.sessionId) { toast('No image loaded.', 'error'); return; }

  const maskData = getMaskDataURL();
  // Check mask is not empty
  if (isMaskEmpty()) { toast('Paint a mask over the text first.', 'error'); return; }

  const model  = document.getElementById('modelSelect').value;
  const prompt = document.getElementById('inpaintPrompt').value ||
                 'seamless decorative background, high quality texture';
  const neg    = document.getElementById('inpaintNeg').value ||
                 'text, watermark, letters, blurry';

  const endpoint = S.engine === 'comfyui' ? '/api/comfyui/inpaint' : '/api/inpaint';

  showProgress('Inpainting… this may take 10–60 seconds');
  document.getElementById('btnInpaint').disabled = true;

  try {
    const r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:    S.sessionId,
        mask_data_url: maskData,
        model,
        prompt,
        negative_prompt: neg,
      }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Inpainting failed');

    S.urls.inpainted = d.image_url;
    S.urls.current   = d.image_url;
    S.inpaintDone    = true;

    // Save sampled font/style info (used to auto-fill Step 4)
    if (d.sample_info) {
      S.sample    = d.sample_info;
      S.sampleUrl = d.sample_url;
      S._sampleApplied = false;   // force re-apply on (next) entry to Step 4
    }

    await loadImageToCanvas(d.image_url);
    toast('Inpainting complete ✓', 'success');

    document.getElementById('btnStep3Next').style.display = 'block';

    // ── Ask user what to replace the text with ──
    // Modal pops up: shows OCR'd original, asks for replacement text.
    // On confirm → auto-place the new text into the masked area (auto-fit).
    if (S.sample && S.sample.bbox) {
      openReplaceModal();
    } else {
      // No sample (no text detected) — just jump to Step 4
      setTimeout(() => goStep(4), 400);
    }
  } catch (err) {
    hideProgress();
    toast(err.message, 'error');
  } finally {
    document.getElementById('btnInpaint').disabled = false;
  }
}

// ── Reset back to the AI-placed version (button in Step 4) ──
async function rePlaceAIText() {
  if (!S.sample || !S.sample.ocr_text) {
    toast('No AI sample available — run inpainting first.', 'error');
    return;
  }
  showLoading('Re-placing text with AI style…');
  try {
    // Re-fill the form with sampled values, then re-bake the text
    S._sampleApplied = false;
    applySampleStyle();
    await autoPlaceSampledText();
    toast('Reset to AI auto-placement ✓', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

// ── Render the current Step-4 text styling onto an offscreen
//    canvas at FULL image resolution, return data URL.
//    This is what the user sees in the ghost preview, exactly.
function renderTextLayerDataURL(textOverride, opts) {
  opts = opts || {};
  const text     = (textOverride !== undefined && textOverride !== null)
                   ? textOverride
                   : document.getElementById('textInput').value;
  if (!text.trim()) return null;

  const fontSize = +(opts.fontSize  ?? document.getElementById('fontSize').value);
  const family   = (opts.family     ?? document.getElementById('fontFamily').value)
                   || 'sans-serif';
  const color    = opts.color       ?? document.getElementById('textColor').value;
  const bold     = (opts.bold       ?? S.textBold)   ? 'bold '   : '';
  const italic   = (opts.italic     ?? S.textItalic) ? 'italic ' : '';
  const sw       = +(opts.strokeWidth ?? document.getElementById('strokeWidth').value);
  const sc       = opts.strokeColor ?? document.getElementById('strokeColor').value;
  const opacity  = +(opts.opacity   ?? document.getElementById('textOpacity').value) / 100;
  const align    = opts.align       ?? S.textAlign;
  const tx       = (opts.x          ?? S.textX);
  const ty       = (opts.y          ?? S.textY);

  // Offscreen canvas at ORIGINAL image resolution
  const off = document.createElement('canvas');
  off.width  = S.imageW;
  off.height = S.imageH;
  const ctx  = off.getContext('2d');

  ctx.save();
  ctx.globalAlpha   = opacity;
  ctx.font          = `${italic}${bold}${fontSize}px ${family}`;
  ctx.fillStyle     = color;
  ctx.strokeStyle   = sc;
  ctx.lineJoin      = 'round';
  ctx.lineWidth     = sw * 2;
  ctx.textAlign     = align;
  ctx.textBaseline  = 'top';

  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const y = ty + i * fontSize * 1.2;
    if (sw > 0) ctx.strokeText(lines[i], tx, y);
    ctx.fillText(lines[i], tx, y);
  }
  ctx.restore();

  return off.toDataURL('image/png');
}

// ── Send the rendered text layer to backend for compositing ──
async function composeTextOnBackend(textOverride, opts) {
  const dataURL = renderTextLayerDataURL(textOverride, opts);
  if (!dataURL) throw new Error('No text to compose');

  const r = await fetch('/api/compose-text', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      session_id:           S.sessionId,
      text_layer_data_url:  dataURL,
    }),
  });
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || 'Compose failed');

  S.urls.withText = d.image_url;
  S.urls.current  = d.image_url;
  await loadImageToCanvas(d.image_url);
  return d;
}

// ── Auto-place text on the inpainted image ─────────────────
// Uses the BROWSER's canvas rendering (same as ghost preview) so
// what you see is what you get — no PIL/Canvas font mismatch.
// `customText` — optional override; falls back to OCR text.
async function autoPlaceSampledText(customText) {
  const s = S.sample;
  if (!s || !s.bbox) return false;

  const text = (customText !== undefined && customText !== null)
    ? customText
    : (s.ocr_text || '');
  if (!text.trim()) return false;

  // ── Place at the MASK bounding box (where the user actually painted) ──
  // This is exactly where the previous text was removed.
  const maskBox = (s.mask_bbox && s.mask_bbox.length === 4) ? s.mask_bbox : s.bbox;
  const [mx0, my0, mx1, my1] = maskBox;
  const maskW = Math.max(1, mx1 - mx0);
  const maskH = Math.max(1, my1 - my0);

  // Auto-fit the font size so the rendered text fills the masked area
  // (using browser canvas metrics so it matches the ghost preview).
  const family = selectFontInDropdown(s.best_family) || 'sans-serif';
  S.textBold   = !!s.bold;
  S.textItalic = !!s.italic;
  const fittedSize = findCanvasFitSize(text, maskW, maskH);

  // Center the rendered text vertically within the masked area
  const measuredH = measureTextHeight(text, fittedSize, family);
  const placeX = mx0;
  const placeY = my0 + Math.max(0, Math.floor((maskH - measuredH) / 2));

  // Sync state for the ghost preview
  S.textX = placeX;
  S.textY = placeY;

  // Pre-fill the Step-4 form fields so user can manipulate from here
  document.getElementById('textInput').value         = text;
  document.getElementById('fontSize').value          = fittedSize;
  document.getElementById('fontSizeVal').textContent = fittedSize;
  document.getElementById('textColor').value         = s.color || '#000000';
  document.getElementById('textX').value             = placeX;
  document.getElementById('textY').value             = placeY;

  // ── Render via BROWSER canvas + composite on backend ──
  await composeTextOnBackend(text, {
    x:           placeX,
    y:           placeY,
    fontSize:    fittedSize,
    family:      family,
    color:       s.color || '#000000',
    bold:        !!s.bold,
    italic:      !!s.italic,
    align:       'left',
    opacity:     1.0,
    strokeWidth: 0,
  });

  S._lastPlacedText = text;
  return true;
}

// Measure the rendered height (px) of text in a given font.
// Uses canvas.measureText with actualBoundingBox* if available.
function measureTextHeight(text, fontSize, family) {
  const off = document.createElement('canvas');
  const ctx = off.getContext('2d');
  const bold   = S.textBold   ? 'bold '   : '';
  const italic = S.textItalic ? 'italic ' : '';
  ctx.font = `${italic}${bold}${fontSize}px ${family || 'sans-serif'}`;
  const lines = text.split('\n');
  let maxH = 0;
  for (const line of lines) {
    const m = ctx.measureText(line || 'M');
    const ascent  = m.actualBoundingBoxAscent  || fontSize * 0.8;
    const descent = m.actualBoundingBoxDescent || fontSize * 0.2;
    maxH = Math.max(maxH, ascent + descent);
  }
  return Math.ceil(maxH * lines.length * (lines.length > 1 ? 1.2 : 1.0));
}

// Pick a font in the dropdown by family name (case-insensitive substring).
// Returns the canonical name (or the input if not found).
function selectFontInDropdown(family) {
  if (!family) return '';
  const sel = document.getElementById('fontFamily');
  if (!sel) return family;
  const f = family.toLowerCase();
  for (const opt of sel.options) {
    if (opt.value && opt.value.toLowerCase().includes(f)) {
      sel.value = opt.value;
      return opt.value;
    }
  }
  return family;
}

// ── Replacement-text modal ─────────────────────────────────
function openReplaceModal() {
  const s = S.sample;
  if (!s) return;

  const orig = (s.ocr_text || '').trim();

  // Defensive: if modal HTML isn't loaded (stale cache, etc.),
  // fall back to a native window prompt so the workflow still completes.
  const modal           = document.getElementById('replaceModal');
  const originalTextEl  = document.getElementById('modalOriginalText');
  const newTextEl       = document.getElementById('modalNewText');
  const summaryEl       = document.getElementById('modalSampleSummary');

  if (!modal || !originalTextEl || !newTextEl) {
    console.warn('Modal not in DOM — using native prompt fallback. Hard-refresh (Ctrl+Shift+R) to load the new UI.');
    const promptMsg =
      `Original text (detected): ${orig || '(none)'}\n\n` +
      `What should the new text say? It will be placed in the masked area.`;
    const newText = window.prompt(promptMsg, orig);
    if (newText && newText.trim()) {
      showLoading('Placing text…');
      autoPlaceSampledText(newText.trim())
        .then(() => {
          document.getElementById('textInput').value = newText.trim();
          toast('Text placed ✓', 'success');
          goStep(4);
        })
        .catch(err => toast(err.message, 'error'))
        .finally(() => hideLoading());
    } else {
      goStep(4);
    }
    return;
  }

  // Fill in the original OCR'd text + sampled style chips
  originalTextEl.textContent = orig || '(no text detected)';
  newTextEl.value = orig;

  // Sample summary
  const w = s.bbox ? (s.bbox[2] - s.bbox[0]) : 0;
  const h = s.bbox ? (s.bbox[3] - s.bbox[1]) : 0;
  if (summaryEl) {
    summaryEl.innerHTML = `
      <strong>Style detected:</strong>
      <span class="chip"><span class="chip-color" style="background:${s.color}"></span>${s.color}</span>
      <span class="chip">${s.best_family || '?'}</span>
      ${s.bold   ? '<span class="chip">Bold</span>'   : ''}
      ${s.italic ? '<span class="chip">Italic</span>' : ''}
      <span class="chip">${w}×${h}px area</span>
    `;
  }

  modal.classList.remove('hidden');
  setTimeout(() => newTextEl.focus(), 100);

  // Submit on Enter (Shift+Enter = newline)
  newTextEl.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      confirmReplaceText();
    } else if (e.key === 'Escape') {
      closeReplaceModal(false);
    }
  };
}

function closeReplaceModal(jumpToStep4 = true) {
  document.getElementById('replaceModal').classList.add('hidden');
  if (jumpToStep4) setTimeout(() => goStep(4), 250);
}

async function confirmReplaceText() {
  const newText = document.getElementById('modalNewText').value.trim();
  if (!newText) {
    toast('Enter the replacement text first.', 'error');
    return;
  }

  // Disable button during processing
  const btn = document.getElementById('modalPlaceBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Placing…';

  try {
    await autoPlaceSampledText(newText);
    toast(`Text "${newText.slice(0, 40)}${newText.length > 40 ? '…' : ''}" placed ✓`, 'success');
    // Pre-fill Step 4 textarea with the placed text
    document.getElementById('textInput').value = newText;
    closeReplaceModal(true);
  } catch (err) {
    toast(err.message, 'error');
    btn.disabled = false;
    btn.textContent = '✨ Place Text in Masked Area';
  }
}

function isMaskEmpty() {
  const d = ctxMask.getImageData(0, 0, cvMask.width, cvMask.height).data;
  for (let i = 3; i < d.length; i += 4) {
    if (d[i] > 10) return false;
  }
  return true;
}

// ── Text tool ─────────────────────────────────────────────────
function activateTextTool() {
  S.placingText = true;
  cvText.classList.remove('hidden');
  cvText.style.pointerEvents = 'auto';
  cvText.style.cursor = 'crosshair';
  cvText.addEventListener('click', onCanvasClickForText);

  // Display sampled original text reference + auto-fill form on first entry
  renderSampleBox();
  if (S.sample && !S._sampleApplied) {
    applySampleStyle();
    S._sampleApplied = true;
  }

  updateTextPreview();
}

function renderSampleBox() {
  const box = document.getElementById('sampleBox');
  if (!S.sample || !S.sampleUrl) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  document.getElementById('sampleImg').src = S.sampleUrl + '?t=' + Date.now();

  const s = S.sample;
  const matches = (s.matches || []).slice(0, 3)
    .map(m => `<span class="chip">${m.family}</span>`).join('');

  const groqLine = s.groq
    ? `<div style="margin-top:6px"><strong>🤖 Groq says:</strong>
         <span class="chip">${s.groq.family || '?'}</span>
         <span class="chip">${s.groq.weight || ''}</span>
         <span class="chip">${s.groq.style || ''}</span>
         <span class="chip">conf: ${s.groq.confidence || '?'}</span>
       </div>`
    : '';

  const ocrLine = s.ocr_text
    ? `<div style="margin-top:6px"><strong>OCR text:</strong> "${s.ocr_text}"</div>`
    : '';

  const posLine = (s.bbox && s.bbox.length === 4)
    ? `<div style="margin-top:4px;font-size:.72rem;opacity:.75">
         Position: (${s.bbox[0]}, ${s.bbox[1]}) · size ${s.bbox[2]-s.bbox[0]}×${s.bbox[3]-s.bbox[1]}px
       </div>`
    : '';

  document.getElementById('sampleMeta').innerHTML = `
    <div>
      <span class="chip"><span class="chip-color" style="background:${s.color}"></span>${s.color}</span>
      <span class="chip">${s.font_size}px</span>
      ${s.bold   ? '<span class="chip">Bold</span>'   : ''}
      ${s.italic ? '<span class="chip">Italic</span>' : ''}
      ${s.is_serif ? '<span class="chip">Serif</span>' : '<span class="chip">Sans-serif</span>'}
    </div>
    ${groqLine}
    ${ocrLine}
    <div style="margin-top:6px"><strong>Best matches:</strong> ${matches || '(none)'}</div>
    ${posLine}
  `;
}

function applySampleStyle() {
  if (!S.sample) { toast('No sample available — paint a mask and run inpainting first.', 'error'); return; }
  const s = S.sample;

  document.getElementById('textColor').value = s.color;
  document.getElementById('fontSize').value  = s.font_size;
  document.getElementById('fontSizeVal').textContent = s.font_size;

  S.textBold   = !!s.bold;
  S.textItalic = !!s.italic;
  document.getElementById('btnBold').classList.toggle('active',   S.textBold);
  document.getElementById('btnItalic').classList.toggle('active', S.textItalic);

  // ── Auto-position text at the exact original location ──
  if (s.bbox && Array.isArray(s.bbox) && s.bbox.length === 4) {
    const [x0, y0, x1, y1] = s.bbox;
    S.textX = x0;
    S.textY = y0;
    document.getElementById('textX').value = x0;
    document.getElementById('textY').value = y0;
  }

  // ── Pre-fill OCR text from Groq if textarea is empty ──
  const textInput = document.getElementById('textInput');
  if (s.ocr_text && !textInput.value.trim()) {
    textInput.value = s.ocr_text;
  }

  // Set font family — load fonts first if dropdown is empty
  const sel = document.getElementById('fontFamily');
  const setFamily = () => {
    const target = s.best_family;
    // Try exact match; otherwise pick the first option that contains the family name
    let matched = false;
    for (const opt of sel.options) {
      if (opt.value === target) { sel.value = target; matched = true; break; }
    }
    if (!matched) {
      for (const opt of sel.options) {
        if (opt.value.toLowerCase().includes(target.toLowerCase())) {
          sel.value = opt.value; matched = true; break;
        }
      }
    }
    if (!matched && sel.options.length > 1) {
      // Add it as a custom option so it sticks
      const o = document.createElement('option');
      o.value = o.textContent = target;
      sel.insertBefore(o, sel.options[1]);
      sel.value = target;
    }
    updateTextPreview();
  };

  if (sel.options.length <= 1) {
    // Lazy-load fonts
    fetch('/api/fonts').then(r => r.json()).then(d => {
      sel.innerHTML = '<option value="">System default</option>';
      d.fonts.forEach(f => {
        const o = document.createElement('option');
        o.value = o.textContent = f;
        sel.appendChild(o);
      });
      setFamily();
    });
  } else {
    setFamily();
  }

  toast(`Applied sampled style: ${s.best_family}`, 'success');
}

function deactivateTextTool() {
  S.placingText = false;
  cvText.classList.add('hidden');
  cvText.style.pointerEvents = 'none';
  cvText.removeEventListener('click', onCanvasClickForText);
}

function onCanvasClickForText(e) {
  const rect = cvText.getBoundingClientRect();
  // Convert display coords → original image coords
  const x = Math.round((e.clientX - rect.left) / S.scale);
  const y = Math.round((e.clientY - rect.top)  / S.scale);
  S.textX = x; S.textY = y;
  document.getElementById('textX').value = x;
  document.getElementById('textY').value = y;
  updateTextPreview();
}

function updateTextPreview() {
  if (S.currentStep !== 4) return;
  const text     = document.getElementById('textInput').value;
  const fontSize = +document.getElementById('fontSize').value * S.scale;
  const color    = document.getElementById('textColor').value;
  const family   = document.getElementById('fontFamily').value || 'sans-serif';
  const bold     = S.textBold   ? 'bold '   : '';
  const italic   = S.textItalic ? 'italic ' : '';
  const sw       = +document.getElementById('strokeWidth').value * S.scale;
  const sc       = document.getElementById('strokeColor').value;
  const opacity  = +document.getElementById('textOpacity').value / 100;
  const align    = S.textAlign;

  const dispX = S.textX * S.scale;
  const dispY = S.textY * S.scale;

  ctxText.clearRect(0, 0, cvText.width, cvText.height);
  if (!text) return;

  ctxText.save();
  ctxText.globalAlpha = opacity;
  ctxText.font        = `${italic}${bold}${fontSize}px ${family}`;
  ctxText.fillStyle   = color;
  ctxText.textAlign   = align;
  ctxText.textBaseline = 'top';

  if (sw > 0) {
    ctxText.strokeStyle = sc;
    ctxText.lineWidth   = sw * 2;
    ctxText.lineJoin    = 'round';
    for (const line of text.split('\n')) {
      ctxText.strokeText(line, dispX, dispY + text.split('\n').indexOf(line) * fontSize * 1.2);
    }
  }
  for (const [i, line] of text.split('\n').entries()) {
    ctxText.fillText(line, dispX, dispY + i * fontSize * 1.2);
  }
  ctxText.restore();

  // Show crosshair
  const cur = document.getElementById('textCursor');
  cur.classList.remove('hidden');
  cur.style.left = dispX + 'px';
  cur.style.top  = dispY + 'px';
}

async function applyText() {
  if (!S.sessionId) return;
  const text = document.getElementById('textInput').value.trim();
  if (!text) { toast('Enter some text first.', 'error'); return; }

  const autoFit = document.getElementById('chkAutoFit')?.checked;

  // ── If auto-fit is on AND sample exists, compute a new font_size
  //    that makes the rendered text fit S.sample.mask_bbox (browser-measured).
  if (autoFit && S.sample && (S.sample.mask_bbox || S.sample.bbox)) {
    const bb = S.sample.mask_bbox || S.sample.bbox;
    const targetW = bb[2] - bb[0];
    const targetH = bb[3] - bb[1];
    const fittedSize = findCanvasFitSize(text, targetW, targetH);
    document.getElementById('fontSize').value = fittedSize;
    document.getElementById('fontSizeVal').textContent = fittedSize;
    S.textX = bb[0];
    S.textY = bb[1];
    document.getElementById('textX').value = bb[0];
    document.getElementById('textY').value = bb[1];
  }

  showLoading(autoFit ? 'Fitting text to masked area…' : 'Applying text…');
  try {
    // ── Use the BROWSER canvas as the source of truth ──
    // composeTextOnBackend renders text via HTML5 Canvas, sends to backend,
    // backend composites onto inpainted.png → with_text.png matches ghost.
    await composeTextOnBackend(text);

    // Clear preview overlay (since the same text is now baked in)
    ctxText.clearRect(0, 0, cvText.width, cvText.height);
    document.getElementById('textCursor').classList.add('hidden');

    toast('Text applied ✓', 'success');
  } catch (err) {
    toast(err.message || 'Text overlay failed', 'error');
  } finally {
    hideLoading();
  }
}

// ── Binary-search a font size that makes browser-rendered text
//    fit within (targetW × targetH). Uses canvas.measureText. ──
function findCanvasFitSize(text, targetW, targetH) {
  const family = document.getElementById('fontFamily').value || 'sans-serif';
  const bold   = S.textBold   ? 'bold '   : '';
  const italic = S.textItalic ? 'italic ' : '';

  const off = document.createElement('canvas');
  const ctx = off.getContext('2d');

  let low = 6, high = Math.max(targetH * 4, 400);
  let best = Math.max(8, targetH);
  const margin = 0.95;

  for (let i = 0; i < 40 && low <= high; i++) {
    const mid = Math.floor((low + high) / 2);
    ctx.font = `${italic}${bold}${mid}px ${family}`;
    // measureText doesn't give height reliably; approximate with metrics
    const lines = text.split('\n');
    let maxW = 0;
    for (const line of lines) maxW = Math.max(maxW, ctx.measureText(line).width);
    const m = ctx.measureText('Mg');
    // Approximate text block height: lineCount × (ascent + descent) × 1.05
    const ascent  = m.actualBoundingBoxAscent  || mid * 0.8;
    const descent = m.actualBoundingBoxDescent || mid * 0.2;
    const lineH   = (ascent + descent) * 1.05;
    const totalH  = lineH * lines.length;

    if (maxW <= targetW * margin && totalH <= targetH * margin) {
      best = mid;
      low  = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return best;
}

async function undoText() {
  if (!S.sessionId) return;
  showLoading('Undoing…');
  try {
    const r = await fetch('/api/undo-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: S.sessionId }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error);
    S.urls.current = d.image_url;
    await loadImageToCanvas(d.image_url);
    toast('Text undone', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

function toggleStyle(which) {
  if (which === 'bold') {
    S.textBold = !S.textBold;
    document.getElementById('btnBold').classList.toggle('active', S.textBold);
  } else {
    S.textItalic = !S.textItalic;
    document.getElementById('btnItalic').classList.toggle('active', S.textItalic);
  }
  updateTextPreview();
}

function setAlign(a) {
  S.textAlign = a;
  ['L','C','R'].forEach(l =>
    document.getElementById(`btnAlign${l}`).classList.toggle('active', a === {L:'left',C:'center',R:'right'}[l])
  );
  updateTextPreview();
}

async function loadFonts() {
  try {
    const r = await fetch('/api/fonts');
    const d = await r.json();
    const sel = document.getElementById('fontFamily');
    sel.innerHTML = '<option value="">System default</option>';
    d.fonts.forEach(f => {
      const o = document.createElement('option');
      o.value = o.textContent = f;
      sel.appendChild(o);
    });
    toast(`${d.fonts.length} fonts loaded`, 'success');
  } catch (err) {
    toast('Could not load fonts: ' + err.message, 'error');
  }
}

// ── Save / Download ──────────────────────────────────────────
function setupSavePanel() {
  const fname = S.sessionId ? `edited_${S.sessionId.slice(0,8)}.png` : 'output.png';
  document.getElementById('saveFilename').value = fname;
}

async function saveImage() {
  if (!S.sessionId) return;
  showLoading('Saving…');
  try {
    const r = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: S.sessionId,
        filename:   document.getElementById('saveFilename').value,
      }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Save failed');

    S.downloadUrl = d.download_url;

    const box = document.getElementById('saveResult');
    box.innerHTML = `✓ Saved to <code>output/${d.filename}</code>`;
    box.className = 'info-box success';
    box.classList.remove('hidden');

    document.getElementById('btnDownload').style.display = 'flex';
    toast('Image saved ✓', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

function downloadImage() {
  if (!S.downloadUrl) return;
  const a = document.createElement('a');
  a.href     = S.downloadUrl;
  a.download = document.getElementById('saveFilename').value || 'output.png';
  a.click();
}

function showComparison(which) {
  document.querySelectorAll('#compareSeg .seg-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.val === which)
  );
  const urlMap = {
    final:     S.urls.withText  || S.urls.inpainted || S.urls.source,
    original:  S.urls.source,
    inpainted: S.urls.inpainted || S.urls.source,
  };
  const url = urlMap[which];
  if (url) {
    S.urls.current = url;
    reloadBase();
  }
}

function startOver() {
  if (!confirm('Start over? This clears the current session.')) return;
  S.sessionId   = null;
  S.currentStep = 1;
  S.inpaintDone = false;
  S.urls        = { source: null, inpainted: null, withText: null, current: null };
  S.sample      = null;
  S.sampleUrl   = null;
  S._sampleApplied = false;

  document.getElementById('canvasStack').classList.add('hidden');
  document.getElementById('dropZone').classList.remove('hidden');
  document.getElementById('zoomControls').classList.add('hidden');
  document.getElementById('imageInfo').classList.add('hidden');
  document.getElementById('btnStep1Next').disabled = true;

  ctxBase.clearRect(0, 0, cvBase.width, cvBase.height);
  ctxMask.clearRect(0, 0, cvMask.width, cvMask.height);
  ctxText.clearRect(0, 0, cvText.width, cvText.height);
  S.maskHistory = []; S.maskRedo = [];

  goStep(1);
}

// ── Progress bar ─────────────────────────────────────────────
let _progressTimer = null;
function showProgress(msg) {
  const wrap  = document.getElementById('inpaintProgress');
  const fill  = document.getElementById('progressFill');
  const label = document.getElementById('progressLabel');
  wrap.classList.remove('hidden');
  label.textContent = msg;
  fill.style.width  = '0%';
  let w = 0;
  _progressTimer = setInterval(() => {
    w = Math.min(w + (w < 60 ? 2 : 0.3), 92);
    fill.style.width = w + '%';
  }, 500);
}

function hideProgress() {
  clearInterval(_progressTimer);
  const fill = document.getElementById('progressFill');
  fill.style.width = '100%';
  setTimeout(() => {
    document.getElementById('inpaintProgress').classList.add('hidden');
    fill.style.width = '0%';
  }, 600);
}

// ── Loading overlay ───────────────────────────────────────────
function showLoading(msg = 'Processing…') {
  document.getElementById('loadingMsg').textContent = msg;
  document.getElementById('loadingOverlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loadingOverlay').classList.add('hidden');
}

// ═════════════════════════════════════════════════════════════
// ADVANCED TEXT REPLACEMENT — Auto-Detect + Batch Mode
// ═════════════════════════════════════════════════════════════

// ── Auto-Detect shared state ──────────────────────────────────
const AD = {
  boxes:       [],   // [[x0,y0,x1,y1], ...]  (image coords)
  ocr:         [],   // ["text", ...] per box
  maskUrl:     null,
  boxOverlays: [],   // {el, x0,y0,x1,y1} drawn on canvas wrapper
};

// ── Batch tab step routing (injected into existing goStep at runtime) ─
// We extend the existing goStep via a flag check at the call site below.
// The actual hook is inserted into goStep() via _patchGoStepForBatch()
// which runs once on DOMContentLoaded and re-routes 'batch' before the
// normal step machinery runs.
function _patchGoStepForBatch() {
  const _orig = goStep;
  goStep = function goStepPatched(n) {   // named for stack traces
    if (n === 'batch') {
      document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
      document.querySelectorAll('.step').forEach(s => s.classList.remove('active', 'done'));
      document.querySelector('.step.batch-tab')?.classList.add('active');
      document.getElementById('panelBatch').classList.remove('hidden');
      S.currentStep = 'batch';
      initBatchPanel();
      return;
    }
    document.querySelector('.step.batch-tab')?.classList.remove('active');
    _orig(n);
  };
}
document.addEventListener('DOMContentLoaded', _patchGoStepForBatch, { once: true });

// ── Batch panel init ─────────────────────────────────────────
function initBatchPanel() {
  const folderInput = document.getElementById('batchFolder');
  if (!folderInput.value) folderInput.value = '~/Desktop';
}

function toggleBatchColorAuto(auto) {
  document.getElementById('batchColor').disabled = auto;
}
toggleBatchColorAuto(true);   // run on load

// ── Scan folder ──────────────────────────────────────────────
async function batchScanFolder() {
  const folder = document.getElementById('batchFolder').value.trim() || '~/Desktop';
  try {
    const r = await fetch(`/api/batch/scan?folder=${encodeURIComponent(folder)}`);
    const d = await r.json();
    if (!r.ok) { toast(d.error || 'Scan failed', 'error'); return; }

    const el = document.getElementById('batchScanResult');
    if (!d.images.length) {
      el.innerHTML = '<p style="color:var(--text-dim)">No processable images found.</p>';
    } else {
      el.innerHTML =
        `<div class="scan-header">Found <strong>${d.count}</strong> image(s) in <code>${d.folder}</code></div>` +
        '<ul class="scan-list">' +
        d.images.map(img =>
          `<li><span class="scan-name">${img.name}</span><span class="scan-size">${img.size_kb} KB</span></li>`
        ).join('') +
        '</ul>';
    }
    el.classList.remove('hidden');
  } catch (err) {
    toast('Scan error: ' + err.message, 'error');
  }
}

// ── Run batch ─────────────────────────────────────────────────
async function runBatch() {
  const replText = document.getElementById('batchReplaceText').value.trim();
  if (!replText) { toast('Enter replacement text first.', 'error'); return; }

  const folder    = document.getElementById('batchFolder').value.trim() || '~/Desktop';
  const model     = document.getElementById('batchModel').value;
  const prompt    = document.getElementById('batchPrompt').value.trim();
  const fontSizeV = document.getElementById('batchFontSize').value;
  const colorAuto = document.getElementById('batchColorAuto').checked;
  const colorVal  = document.getElementById('batchColor').value;

  const body = {
    replacement_text:   replText,
    folder:             folder,
    model:              model,
    prompt:             prompt,
    font_size_override: fontSizeV ? parseInt(fontSizeV) : null,
    color_override:     colorAuto ? null : colorVal,
  };

  // Show progress
  document.getElementById('batchProgress').classList.remove('hidden');
  document.getElementById('batchProgressFill').style.width = '0%';
  document.getElementById('batchProgressLabel').textContent = 'Sending to server…';
  document.getElementById('btnRunBatch').disabled = true;

  let pct = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + (pct < 60 ? 1.2 : 0.2), 90);
    document.getElementById('batchProgressFill').style.width = pct + '%';
  }, 800);

  try {
    document.getElementById('batchProgressLabel').textContent =
      'Processing… this can take several minutes';

    const r = await fetch('/api/batch/process', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify(body),
    });
    const d = await r.json();

    clearInterval(ticker);
    document.getElementById('batchProgressFill').style.width = '100%';
    document.getElementById('batchProgressLabel').textContent = 'Done ✓';

    if (!r.ok) throw new Error(d.error || 'Batch failed');
    renderBatchResults(d);
    toast(`Batch done: ${d.succeeded}/${d.processed} succeeded`, d.failed > 0 ? 'error' : 'success');
  } catch (err) {
    clearInterval(ticker);
    document.getElementById('batchProgressLabel').textContent = 'Error';
    toast(err.message, 'error');
  } finally {
    document.getElementById('btnRunBatch').disabled = false;
    setTimeout(() => {
      document.getElementById('batchProgress').classList.add('hidden');
    }, 2000);
  }
}

function renderBatchResults(summary) {
  const el = document.getElementById('batchResults');
  el.classList.remove('hidden');

  const statusIcon = summary.failed === 0 ? '✅' : (summary.succeeded === 0 ? '❌' : '⚠️');

  let html = `
    <div class="batch-summary">
      <span>${statusIcon}</span>
      <span><strong>${summary.succeeded}</strong> succeeded</span>
      <span><strong>${summary.no_text || 0}</strong> no text</span>
      <span><strong>${summary.failed}</strong> failed</span>
      <span class="dim">${summary.timestamp}</span>
    </div>
    <ul class="batch-file-list">
  `;

  for (const f of (summary.files || [])) {
    const icon  = f.status === 'success' ? '✓' : (f.status === 'no_text' ? '—' : '✗');
    const cls   = f.status === 'success' ? 'ok' : (f.status === 'no_text' ? 'dim' : 'err');
    const name  = f.file  ? f.file.split('/').pop() : '?';
    const out   = f.output_path ? f.output_path.split('/').pop() : '';
    const extra = f.status === 'success'
      ? `→ ${out}  (${f.replaced || 0}/${f.regions} regions)`
      : (f.error ? f.error.slice(0, 80) : '');

    html += `<li class="bf-item ${cls}">
      <span class="bf-icon">${icon}</span>
      <span class="bf-name">${name}</span>
      <span class="bf-detail">${extra}</span>
    </li>`;
  }

  html += '</ul>';
  el.innerHTML = html;
}

// ── View log ─────────────────────────────────────────────────
async function viewBatchLog() {
  const el = document.getElementById('batchLogView');
  if (!el.classList.contains('hidden')) { el.classList.add('hidden'); return; }

  try {
    const r = await fetch('/api/batch/log');
    const d = await r.json();
    if (!r.ok) { toast(d.error, 'error'); return; }

    if (!d.runs.length) {
      el.innerHTML = '<p style="color:var(--text-dim)">No batch runs recorded yet.</p>';
    } else {
      const last5 = d.runs.slice(-5).reverse();
      el.innerHTML = '<div class="log-path">Log: <code>' + d.log_path + '</code></div>' +
        last5.map(run => `
          <div class="log-run">
            <span class="log-ts">${run.timestamp}</span>
            <span class="log-model">${run.model}</span>
            <span class="log-stat">✓${run.succeeded} ✗${run.failed}</span>
          </div>
        `).join('');
    }
    el.classList.remove('hidden');
  } catch (err) {
    toast('Log error: ' + err.message, 'error');
  }
}

// ── Auto-Detect text in current session ──────────────────────
async function runAutoDetect() {
  if (!S.sessionId) { toast('Upload an image first.', 'error'); return; }

  const btn = document.getElementById('btnAutoDetect');
  btn.disabled    = true;
  btn.textContent = '⏳ Detecting…';

  try {
    const r = await fetch('/api/auto-detect', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ session_id: S.sessionId, use_groq: true }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Detection failed');

    AD.boxes   = d.boxes   || [];
    AD.ocr     = d.ocr     || [];
    AD.maskUrl = d.mask_url;

    if (!AD.boxes.length) {
      document.getElementById('autoDetectResult').innerHTML =
        '<span style="color:var(--warn)">⚠ No text detected — try painting the mask manually.</span>';
      document.getElementById('autoDetectResult').classList.remove('hidden');
      return;
    }

    // Visualise bounding boxes on the canvas
    drawDetectedBoxes(AD.boxes);

    // Show summary
    const infoEl = document.getElementById('autoDetectResult');
    infoEl.innerHTML =
      `🔍 Detected <strong>${d.total_regions}</strong> region(s) via <em>${d.method}</em>. ` +
      `<button class="btn btn-sm" onclick="openAutoReplaceModal()" style="margin-left:8px">Replace…</button>`;
    infoEl.classList.remove('hidden');

    // Load the generated mask into cvMask so user can proceed normally
    if (d.mask_url) await applyAutoMaskToCanvas(d.mask_url);

    toast(`Detected ${d.total_regions} text region(s) ✓`, 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    btn.disabled    = false;
    btn.textContent = '🔍 Auto-Detect Text';
  }
}

// Draw green bounding boxes on the canvas wrapper
function drawDetectedBoxes(boxes) {
  // Remove old overlays
  AD.boxOverlays.forEach(o => o.el.remove());
  AD.boxOverlays = [];

  const wrapper = document.getElementById('canvasWrapper');

  boxes.forEach((b, i) => {
    const [x0, y0, x1, y1] = b;
    const el = document.createElement('div');
    el.className = 'bbox-overlay';
    el.style.left   = (x0 * S.scale) + 'px';
    el.style.top    = (y0 * S.scale) + 'px';
    el.style.width  = ((x1 - x0) * S.scale) + 'px';
    el.style.height = ((y1 - y0) * S.scale) + 'px';

    const ocr = AD.ocr[i] || '';
    if (ocr) {
      const label = document.createElement('div');
      label.className   = 'bbox-label';
      label.textContent = ocr.slice(0, 30) + (ocr.length > 30 ? '…' : '');
      el.appendChild(label);
    }

    const stack = document.getElementById('canvasStack');
    stack.style.position = 'relative';
    stack.appendChild(el);
    AD.boxOverlays.push({ el, x0, y0, x1, y1 });
  });
}

// Load auto-generated mask URL into the mask canvas
async function applyAutoMaskToCanvas(maskUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      ctxMask.clearRect(0, 0, cvMask.width, cvMask.height);
      // Draw mask: white pixels → semi-transparent red overlay
      const offscreen = document.createElement('canvas');
      offscreen.width  = img.width;
      offscreen.height = img.height;
      const octx = offscreen.getContext('2d');
      octx.drawImage(img, 0, 0);
      const id = octx.getImageData(0, 0, img.width, img.height);
      const px = id.data;
      for (let i = 0; i < px.length; i += 4) {
        const v = px[i];  // grayscale value (white = 255 = mask)
        if (v > 128) {
          px[i]   = 235;   // R
          px[i+1] = 87;    // G
          px[i+2] = 87;    // B
          px[i+3] = Math.round(S.maskOpacity * 255);
        } else {
          px[i+3] = 0;   // transparent
        }
      }
      octx.putImageData(id, 0, 0);
      ctxMask.drawImage(offscreen, 0, 0, cvMask.width, cvMask.height);
      resolve();
    };
    img.onerror = reject;
    img.src = maskUrl + '?t=' + Date.now();
  });
}

// ── Auto-Replace Modal ───────────────────────────────────────
function openAutoReplaceModal() {
  const modal = document.getElementById('autoReplaceModal');
  document.getElementById('arModalCount').textContent = AD.boxes.length;

  // Build per-region rows
  const list = document.getElementById('arRegionList');
  list.innerHTML = AD.boxes.map((b, i) => {
    const [x0, y0, x1, y1] = b;
    const ocr = AD.ocr[i] || '';
    return `
      <div class="ar-region-row">
        <div class="ar-region-info">
          <span class="ar-idx">${i + 1}</span>
          <span class="ar-bbox">${x0},${y0} → ${x1},${y1}</span>
          ${ocr ? `<span class="ar-ocr">"${ocr}"</span>` : ''}
        </div>
        <input class="ar-text-input" id="arText_${i}"
               placeholder="${ocr || 'Replacement text…'}"
               oninput="arClearGlobalIfCustom()">
      </div>
    `;
  }).join('');

  modal.classList.remove('hidden');
}

function closeAutoReplaceModal() {
  document.getElementById('autoReplaceModal').classList.add('hidden');
}

function toggleArColorAuto(auto) {
  document.getElementById('arColor').disabled = auto;
}

function arSyncGlobal(val) {
  // Fill all empty per-region inputs with the global value
  AD.boxes.forEach((_, i) => {
    const inp = document.getElementById(`arText_${i}`);
    if (inp && !inp.dataset.custom) inp.value = val;
  });
}

function arClearGlobalIfCustom() {
  // Mark this input as customised
  event.target.dataset.custom = '1';
}

async function applyAutoReplace() {
  const globalText = document.getElementById('arGlobalText').value.trim();
  const model      = document.getElementById('arModel').value;
  const fontSizeV  = document.getElementById('arFontSize').value;
  const colorAuto  = document.getElementById('arColorAuto').checked;
  const colorVal   = document.getElementById('arColor').value;

  // Build replacements list
  const replacements = AD.boxes.map((b, i) => {
    const inp  = document.getElementById(`arText_${i}`);
    const text = (inp?.value.trim()) || globalText;
    return { box: b, text };
  }).filter(r => r.text);

  if (!replacements.length) {
    toast('Enter replacement text for at least one region.', 'error');
    return;
  }

  closeAutoReplaceModal();
  showLoading('Inpainting + rendering replacement text…');

  try {
    const r = await fetch('/api/auto-replace', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id:          S.sessionId,
        replacements,
        model,
        font_size_override:  fontSizeV ? parseInt(fontSizeV) : null,
        color_override:      colorAuto ? null : colorVal,
      }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Auto-replace failed');

    S.urls.withText = d.image_url;
    S.urls.current  = d.image_url;
    S.inpaintDone   = true;

    // Clear bbox overlays (text is now baked in)
    AD.boxOverlays.forEach(o => o.el.remove());
    AD.boxOverlays = [];

    await loadImageToCanvas(d.image_url);
    toast(`Replaced ${d.results?.filter(r=>r.status==='ok').length || '?'} region(s) ✓`, 'success');
    goStep(5);   // jump to Save
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

// ── Toast ─────────────────────────────────────────────────────
let _toastTimer;
function toast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'toast show' + (type ? ' ' + type : '');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}
