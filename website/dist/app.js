'use strict';
const $ = (id) => document.getElementById(id);
const state = { file: null, buffer: null, objectURL: null, status: null, result: null, busy: false, sequence: 0 };
const names = { mobile: 'MobileNetV2', svm: 'MFCC + SVM', boosting: 'MFCC + boosting' };
let toastTimer;
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500); }
function selectedModel() { return document.querySelector('input[name="model"]:checked').value; }
function ready() { return !!state.status?.models?.[selectedModel()]?.some((r) => r.seed === Number($('seed').value)); }
function updateButton() {
  $('analyze-button').disabled = !state.file || !ready() || state.busy;
  $('analyze-button').firstChild.textContent = state.busy ? 'Examining recording… ' : 'Analyze recording ';
  $('analyze-note').textContent = !state.status ? 'Start the local Python app to enable real inference.' : !ready() ? 'This model and seed do not have a completed checkpoint yet.' : !state.file ? 'Choose a recording to begin.' : 'Ready. Your recording will be processed locally.';
}
function switchView() {
  const value = location.hash.slice(1);
  const view = ['analyze', 'experiments', 'method'].includes(value) ? value : 'analyze';
  document.querySelectorAll('.view').forEach((el) => { el.hidden = el.id !== `view-${view}`; });
  document.querySelectorAll('[data-view]').forEach((el) => { el.classList.toggle('active', el.dataset.view === view); if (el.dataset.view === view) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current'); });
  $('page-title').textContent = { analyze: 'Audio analysis', experiments: 'Experiments', method: 'How it works' }[view];
  if (view === 'analyze') requestAnimationFrame(drawWaveform);
}
function drawWaveform() {
  const canvas = $('waveform'), rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const w = rect.width, h = rect.height, center = (h - 12) / 2;
  ctx.strokeStyle = '#e6edde'; ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 12; y < h; y += 20) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  ctx.strokeStyle = '#afc5a2'; ctx.beginPath(); ctx.moveTo(0, center); ctx.lineTo(w, center); ctx.stroke();
  if (!state.buffer) return;
  const data = state.buffer.getChannelData(0), bars = Math.max(1, Math.floor(w / 3));
  let peak = 0; for (let i = 0; i < data.length; i += Math.max(1, Math.floor(data.length / 50000))) peak = Math.max(peak, Math.abs(data[i]));
  ctx.strokeStyle = '#6f9764'; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
  for (let i = 0; i < bars; i++) {
    const start = Math.floor(i * data.length / bars), end = Math.floor((i + 1) * data.length / bars);
    let amplitude = 0; for (let j = start; j < end; j++) amplitude = Math.max(amplitude, Math.abs(data[j]));
    const height = Math.max(1, amplitude / (peak || 1) * (h - 28) / 2);
    ctx.beginPath(); ctx.moveTo(i * 3 + 1, center - height); ctx.lineTo(i * 3 + 1, center + height); ctx.stroke();
  }
}
function resetResult() { state.result = null; $('result-content').replaceChildren(); $('result-content').hidden = true; $('result-empty').hidden = false; $('export-result').disabled = true; }
async function chooseFile(file) {
  if (!file || state.busy) return;
  if (!file.size || file.size > 25 * 1024 * 1024) { toast('Choose a nonempty audio file up to 25 MB.'); return; }
  const sequence = ++state.sequence;
  state.file = file; state.buffer = null; resetResult();
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  state.objectURL = URL.createObjectURL(file);
  $('audio-player').src = state.objectURL; $('audio-player').hidden = false;
  $('upload-title').textContent = file.name; $('upload-caption').textContent = 'Drop another file here to replace this recording.';
  $('fact-size').textContent = `${(file.size / 1024 ** 2).toFixed(2)} MB`;
  ['fact-duration', 'fact-rate', 'fact-channels', 'duration-axis'].forEach((id) => { $(id).textContent = '—'; });
  $('preview-label').textContent = 'Preparing waveform…'; $('waveform-empty').hidden = false; $('waveform-empty').textContent = 'Reading the recording…';
  drawWaveform(); updateButton();
  let context;
  try {
    context = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (sequence !== state.sequence) return;
    state.buffer = decoded; $('fact-duration').textContent = `${decoded.duration.toFixed(2)} sec`;
    // Web Audio may resample to the browser context's rate; never call it the file's original rate.
    $('fact-rate').textContent = `${(decoded.sampleRate / 1000).toFixed(1)} kHz*`;
    $('fact-rate').title = 'Browser preview sample rate; it may differ from the original recording. Detector input is resampled to 16 kHz.';
    $('fact-channels').textContent = decoded.numberOfChannels === 1 ? 'Mono' : `${decoded.numberOfChannels} channels`;
    $('duration-axis').textContent = `${Math.floor(decoded.duration / 60)}:${String(Math.floor(decoded.duration % 60)).padStart(2, '0')}`;
    $('waveform-empty').hidden = true; $('preview-label').textContent = 'Waveform · browser preview'; drawWaveform();
  } catch {
    if (sequence !== state.sequence) return;
    $('waveform-empty').textContent = 'Browser preview unavailable for this format.';
    $('preview-label').textContent = 'The local engine can still attempt to read this file.';
  } finally { if (context) await context.close().catch(() => {}); }
}
function formatMetric(metric) { return metric && Number.isFinite(metric.mean) && Number.isFinite(metric.sample_std) ? `${(metric.mean * 100).toFixed(2)} ± ${(metric.sample_std * 100).toFixed(2)}%` : 'Not measured'; }
function renderMetrics(data) {
  const table = $('metrics-table'); table.replaceChildren();
  for (const model of ['mobile', 'svm', 'boosting']) {
    const value = data?.aggregate?.[model], row = document.createElement('tr');
    for (const text of [names[model], formatMetric(value?.dev?.eer), formatMetric(value?.eval?.eer), formatMetric(value?.eval?.accuracy), String(data?.models?.[model]?.length || 0) + ' / 3']) { const td = document.createElement('td'); td.textContent = text; row.append(td); }
    table.append(row);
  }
}
async function refreshStatus() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store', signal: AbortSignal.timeout(5000) });
    if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('Local app unavailable');
    const data = await response.json(); if (data.service !== 'voxguard-local') throw new Error('Wrong service');
    state.status = data;
    const count = Object.values(data.models).reduce((sum, rows) => sum + rows.length, 0);
    $('engine-banner').classList.toggle('ready', count > 0);
    $('engine-message').textContent = count ? `Local engine connected · ${count} completed model run${count === 1 ? '' : 's'} available` : 'Local engine connected · Training checkpoints are not ready yet.';
    document.querySelectorAll('[data-model-state]').forEach((el) => { const n = data.models[el.dataset.modelState]?.length || 0; el.textContent = n ? `${n} ready` : 'Not trained'; el.classList.toggle('available', n > 0); });
    $('pipeline-state').textContent = data.pipeline?.state?.replaceAll('_', ' ') || 'Not started';
    $('pipeline-description').textContent = data.pipeline?.current_step ? `Current step: ${data.pipeline.current_step.replaceAll('_', ' ')}. Completed stages: ${data.pipeline.completed?.length || 0}.` : 'Start or resume the training pipeline in a separate terminal. Completed runs are detected automatically.';
    renderMetrics(data);
  } catch {
    state.status = null; $('engine-banner').classList.remove('ready');
    $('engine-message').textContent = 'Interface preview · Open the local app to connect your trained detectors.';
    document.querySelectorAll('[data-model-state]').forEach((el) => { el.textContent = 'Offline'; el.classList.remove('available'); });
    $('pipeline-state').textContent = 'Local engine offline'; renderMetrics(null);
  }
  updateButton();
}
function addResultFact(grid, label, value) { const cell = document.createElement('div'), title = document.createElement('span'), strong = document.createElement('strong'); title.textContent = label; strong.textContent = value; cell.append(title, strong); grid.append(cell); }
async function analyze() {
  if (!state.file || !ready() || state.busy) return;
  state.busy = true; updateButton();
  const file = state.file, model = selectedModel(), seed = Number($('seed').value);
  try {
    const response = await fetch(`/api/analyze?model=${encodeURIComponent(model)}&seed=${seed}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file, signal: AbortSignal.timeout(120000) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Analysis could not complete.');
    state.result = { ...data, filename: file.name, model, seed, analyzed_at: new Date().toISOString() };
    const content = $('result-content'); content.replaceChildren();
    if (data.status === 'rejected') { const warning = document.createElement('p'); warning.className = 'result-warning'; warning.textContent = `Recording rejected: ${data.reason}`; content.append(warning); }
    else {
      const grid = document.createElement('div'); grid.className = 'result-grid';
      addResultFact(grid, 'MODEL LABEL', data.prediction === 'bonafide' ? 'Bona fide' : 'Spoof');
      addResultFact(grid, 'RAW BONA FIDE SCORE', data.bonafide_score.toFixed(4));
      addResultFact(grid, 'DEV THRESHOLD', data.threshold_from_dev.toFixed(4)); content.append(grid);
      const note = document.createElement('p'); note.className = 'result-details'; note.textContent = `${names[model]} · Seed ${seed} · ${data.model_seconds.toFixed(3)}s model time. This score is not a probability or proof of identity.`; content.append(note);
      if (data.noise_warning) { const warning = document.createElement('p'); warning.className = 'result-warning'; warning.textContent = 'Noise-like signal detected. The model label may be unreliable; try a clear speech recording.'; content.append(warning); }
    }
    content.hidden = false; $('result-empty').hidden = true; $('export-result').disabled = false;
  } catch (error) { toast(error.name === 'TimeoutError' ? 'Analysis timed out. Check the local app terminal.' : error.message); }
  finally { state.busy = false; updateButton(); }
}
$('audio-file').addEventListener('change', (event) => chooseFile(event.target.files[0]));
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));
document.querySelectorAll('input[name="model"]').forEach((el) => el.addEventListener('change', () => { document.querySelectorAll('.model-option').forEach((item) => item.classList.toggle('selected', item.contains(document.querySelector('input[name="model"]:checked')))); updateButton(); }));
$('seed').addEventListener('change', updateButton); $('refresh-status').addEventListener('click', refreshStatus); $('analyze-button').addEventListener('click', analyze);
for (const id of ['open-guide', 'help-button']) $(id).addEventListener('click', () => $('guide').showModal());
$('close-guide').addEventListener('click', () => $('guide').close());
$('guide').addEventListener('click', (event) => { if (event.target === $('guide')) { const r = $('guide').getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $('guide').close(); } });
$('copy-command').addEventListener('click', async () => { try { await navigator.clipboard.writeText('.\\.venv\\Scripts\\python.exe web_server.py'); toast('Launch command copied.'); } catch { toast('Select and copy the launch command above.'); } });
$('export-result').addEventListener('click', () => { if (!state.result) return; const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' }), url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = 'voxguard-analysis.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
window.addEventListener('hashchange', switchView); window.addEventListener('resize', drawWaveform);
switchView(); refreshStatus(); setInterval(refreshStatus, 30000);
