/**
 * SonicPure DSP Workstation - Client-Side Application
 * Handles Web Audio API visualizations, interactive canvas waveforms,
 * frequency response graphs, REST API interactions, and audio playback.
 */

// ============================================================================
// AUDIO & STATE MANAGEMENT
// ============================================================================

const audioNoisy = new Audio('/api/audio/noisy');
const audioProc = new Audio('/api/audio/processed');
const audioClean = new Audio('/api/audio/clean');
let activeAudio = audioNoisy;
let isPlayingNoisy = false;
let isPlayingProc = false;
let isPlayingClean = false;

let audioCtx = null;
let analyser = null;
let sourceNoisy = null;
let sourceProc = null;
let sourceClean = null;
let animationFrameId = null;

// DOM Elements
const canvasOsc = document.getElementById('canvas-oscilloscope');
const ctxOsc = canvasOsc.getContext('2d');
const canvasNoisy = document.getElementById('canvas-noisy-wave');
const ctxNoisy = canvasNoisy.getContext('2d');
const canvasProc = document.getElementById('canvas-proc-wave');
const ctxProc = canvasProc.getContext('2d');
const canvasFreqz = document.getElementById('canvas-freqz');
const ctxFreqz = canvasFreqz.getContext('2d');

const btnPlayNoisy = document.getElementById('btn-play-noisy');
const btnPlayProc = document.getElementById('btn-play-proc');
const scrubNoisy = document.getElementById('scrub-noisy');
const scrubProc = document.getElementById('scrub-proc');
const timeCurrNoisy = document.getElementById('time-curr-noisy');
const timeDurNoisy = document.getElementById('time-dur-noisy');
const timeCurrProc = document.getElementById('time-curr-proc');
const timeDurProc = document.getElementById('time-dur-proc');


// ============================================================================
// INITIALIZATION & RESIZE HANDLING
// ============================================================================

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  [canvasOsc, canvasNoisy, canvasProc, canvasFreqz].forEach(canvas => {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
  });
  renderStaticWaveforms();
  fetchFilterFrequencyResponse();
}

window.addEventListener('resize', resizeCanvases);


// ============================================================================
// WEB AUDIO API REAL-TIME OSCILLOSCOPE
// ============================================================================

function initWebAudio() {
  if (audioCtx) return;
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;

  sourceNoisy = audioCtx.createMediaElementSource(audioNoisy);
  sourceProc = audioCtx.createMediaElementSource(audioProc);
  sourceClean = audioCtx.createMediaElementSource(audioClean);

  sourceNoisy.connect(analyser);
  sourceProc.connect(analyser);
  sourceClean.connect(analyser);
  analyser.connect(audioCtx.destination);

  drawOscilloscope();
}

function drawOscilloscope() {
  animationFrameId = requestAnimationFrame(drawOscilloscope);

  const w = canvasOsc.width;
  const h = canvasOsc.height;
  ctxOsc.clearRect(0, 0, w, h);

  if (!analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const timeData = new Uint8Array(bufferLength);
  const freqData = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(timeData);
  analyser.getByteFrequencyData(freqData);

  // Background grid
  ctxOsc.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctxOsc.lineWidth = 1;
  ctxOsc.beginPath();
  for (let x = 0; x < w; x += 60) { ctxOsc.moveTo(x, 0); ctxOsc.lineTo(x, h); }
  for (let y = 0; y < h; y += 30) { ctxOsc.moveTo(0, y); ctxOsc.lineTo(w, y); }
  ctxOsc.stroke();

  // Draw Frequency Spectrum Bars (Background Neon Glow)
  const barWidth = (w / bufferLength) * 2.5;
  let barX = 0;
  for (let i = 0; i < bufferLength; i += 2) {
    const barHeight = (freqData[i] / 255) * h * 0.75;
    ctxOsc.fillStyle = 'rgba(99, 102, 241, 0.15)';
    ctxOsc.fillRect(barX, h - barHeight, barWidth, barHeight);
    barX += barWidth + 1;
  }

  // Draw Smooth Time-Domain Waveform Oscilloscope
  ctxOsc.lineWidth = 2.5;
  if (activeAudio === audioClean) {
    ctxOsc.strokeStyle = '#06b6d4';
    ctxOsc.shadowColor = 'rgba(6, 182, 212, 0.6)';
  } else if (activeAudio === audioProc) {
    ctxOsc.strokeStyle = '#10b981';
    ctxOsc.shadowColor = 'rgba(16, 185, 129, 0.5)';
  } else {
    ctxOsc.strokeStyle = '#6366f1';
    ctxOsc.shadowColor = 'rgba(99, 102, 241, 0.5)';
  }
  ctxOsc.shadowBlur = 10;

  ctxOsc.beginPath();
  const sliceWidth = w / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = timeData[i] / 128.0;
    const y = (v * h) / 2;
    if (i === 0) ctxOsc.moveTo(x, y);
    else ctxOsc.lineTo(x, y);
    x += sliceWidth;
  }
  ctxOsc.stroke();
  ctxOsc.shadowBlur = 0; // reset
}


// ============================================================================
// CANVAS WAVEFORM RENDERING (BEFORE / AFTER)
// ============================================================================

let cachedWaveformData = null;

async function renderStaticWaveforms() {
  try {
    const res = await fetch('/api/waveform-data');
    cachedWaveformData = await res.json();
    drawSingleWaveform(ctxNoisy, canvasNoisy, cachedWaveformData.noisy, '#f43f5e', 'rgba(244, 63, 94, 0.08)');
    if (cachedWaveformData.processed) {
      drawSingleWaveform(ctxProc, canvasProc, cachedWaveformData.processed, '#10b981', 'rgba(16, 185, 129, 0.08)');
    } else {
      drawSingleWaveform(ctxProc, canvasProc, cachedWaveformData.noisy, '#64748b', 'rgba(100, 116, 139, 0.05)');
    }
  } catch (e) {
    console.error('Error fetching waveforms:', e);
  }
}

function drawSingleWaveform(ctx, canvas, samples, strokeColor, fillColor) {
  if (!ctx || !canvas || !samples) return;
  const w = canvas.width;
  const h = canvas.height;
  const midY = h / 2;

  ctx.clearRect(0, 0, w, h);

  // Center zero-line
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(w, midY);
  ctx.stroke();

  // Waveform fill & outline
  ctx.fillStyle = fillColor;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.8;

  ctx.beginPath();
  const step = w / samples.length;

  // Upper envelope
  ctx.moveTo(0, midY);
  for (let i = 0; i < samples.length; i++) {
    const val = samples[i];
    const y = midY - val * (h * 0.45);
    ctx.lineTo(i * step, y);
  }
  // Lower envelope mirrored
  for (let i = samples.length - 1; i >= 0; i--) {
    const val = samples[i];
    const y = midY + val * (h * 0.45);
    ctx.lineTo(i * step, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}


// ============================================================================
// CANVAS FILTER FREQUENCY RESPONSE CURVE
// ============================================================================

async function fetchFilterFrequencyResponse() {
  const fType = document.getElementById('sel-filter-type').value;
  let cut = 1000;
  let cutHigh = 3400;
  let btype = 'lowpass';
  let order = 4;
  let q = 35;

  if (fType === 'fir') {
    cut = parseFloat(document.getElementById('slider-fir-cut').value);
    btype = document.getElementById('fir-btype').value;
  } else if (fType === 'iir') {
    cut = parseFloat(document.getElementById('slider-iir-cut').value);
    order = parseInt(document.getElementById('slider-iir-order').value);
    btype = document.getElementById('iir-btype').value;
  } else if (fType === 'notch') {
    cut = parseFloat(document.getElementById('slider-notch-f0').value);
    q = parseFloat(document.getElementById('slider-notch-q').value);
  }

  try {
    const url = `/api/filter-response?filter_type=${fType}&cutoff=${cut}&cutoff_high=${cutHigh}&btype=${btype}&order=${order}&q=${q}`;
    const res = await fetch(url);
    const data = await res.json();
    drawFrequencyResponse(data.frequencies, data.magnitude_db, cut);
  } catch (e) {
    console.error('Error fetching frequency response:', e);
  }
}

function drawFrequencyResponse(freqs, magDb, cutoffLine) {
  const w = canvasFreqz.width;
  const h = canvasFreqz.height;
  ctxFreqz.clearRect(0, 0, w, h);

  // Background Grid & Axis Lines
  ctxFreqz.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctxFreqz.lineWidth = 1;
  ctxFreqz.beginPath();
  for (let x = 0; x < w; x += 80) { ctxFreqz.moveTo(x, 0); ctxFreqz.lineTo(x, h); }
  for (let y = 0; y < h; y += 40) { ctxFreqz.moveTo(0, y); ctxFreqz.lineTo(w, y); }
  ctxFreqz.stroke();

  // 0 dB reference line (approx 20% from top)
  const y0 = h * 0.18;
  ctxFreqz.strokeStyle = 'rgba(255, 255, 255, 0.2)';
  ctxFreqz.setLineDash([4, 4]);
  ctxFreqz.beginPath();
  ctxFreqz.moveTo(0, y0);
  ctxFreqz.lineTo(w, y0);
  ctxFreqz.stroke();

  // -3 dB line
  const yMinus3 = y0 + (h * 0.06);
  ctxFreqz.strokeStyle = 'rgba(244, 63, 94, 0.5)';
  ctxFreqz.beginPath();
  ctxFreqz.moveTo(0, yMinus3);
  ctxFreqz.lineTo(w, yMinus3);
  ctxFreqz.stroke();
  ctxFreqz.setLineDash([]); // reset

  // Magnitude Response Curve
  ctxFreqz.beginPath();
  ctxFreqz.lineWidth = 2.5;
  ctxFreqz.strokeStyle = '#4f46e5';

  const minDb = -60;
  const maxDb = 5;
  const nPoints = freqs.length;

  for (let i = 0; i < nPoints; i++) {
    const x = (i / (nPoints - 1)) * w;
    const db = Math.max(minDb, Math.min(maxDb, magDb[i]));
    const normY = (maxDb - db) / (maxDb - minDb);
    const y = normY * (h * 0.85) + y0;

    if (i === 0) ctxFreqz.moveTo(x, y);
    else ctxFreqz.lineTo(x, y);
  }
  ctxFreqz.stroke();

  // Labels
  ctxFreqz.fillStyle = '#94a3b8';
  ctxFreqz.font = '11px JetBrains Mono';
  ctxFreqz.fillText('0 dB', 8, y0 - 4);
  ctxFreqz.fillStyle = '#f43f5e';
  ctxFreqz.fillText('-3 dB Cutoff', 8, yMinus3 + 12);
}


// ============================================================================
// AUDIO PLAYBACK CONTROLS
// ============================================================================

btnPlayNoisy.addEventListener('click', () => {
  initWebAudio();
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

  if (isPlayingNoisy) {
    audioNoisy.pause();
    isPlayingNoisy = false;
    btnPlayNoisy.textContent = '▶';
  } else {
    if (isPlayingProc) { audioProc.pause(); isPlayingProc = false; btnPlayProc.textContent = '▶'; }
    if (isPlayingClean) { audioClean.pause(); isPlayingClean = false; updateCleanBtnUI(false); }
    activeAudio = audioNoisy;
    audioNoisy.play();
    isPlayingNoisy = true;
    btnPlayNoisy.textContent = '⏸';
  }
});

btnPlayProc.addEventListener('click', () => {
  initWebAudio();
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

  if (isPlayingProc) {
    audioProc.pause();
    isPlayingProc = false;
    btnPlayProc.textContent = '▶';
  } else {
    if (isPlayingNoisy) { audioNoisy.pause(); isPlayingNoisy = false; btnPlayNoisy.textContent = '▶'; }
    if (isPlayingClean) { audioClean.pause(); isPlayingClean = false; updateCleanBtnUI(false); }
    activeAudio = audioProc;
    audioProc.play();
    isPlayingProc = true;
    btnPlayProc.textContent = '⏸';
  }
});

audioNoisy.addEventListener('timeupdate', () => {
  if (audioNoisy.duration) {
    scrubNoisy.value = (audioNoisy.currentTime / audioNoisy.duration) * 100;
    timeCurrNoisy.textContent = formatTime(audioNoisy.currentTime);
    timeDurNoisy.textContent = formatTime(audioNoisy.duration);
    document.getElementById('osc-timecode').textContent = `${audioNoisy.currentTime.toFixed(2)}s / ${audioNoisy.duration.toFixed(2)}s`;
  }
});

audioProc.addEventListener('timeupdate', () => {
  if (audioProc.duration) {
    scrubProc.value = (audioProc.currentTime / audioProc.duration) * 100;
    timeCurrProc.textContent = formatTime(audioProc.currentTime);
    timeDurProc.textContent = formatTime(audioProc.duration);
    document.getElementById('osc-timecode').textContent = `${audioProc.currentTime.toFixed(2)}s / ${audioProc.duration.toFixed(2)}s`;
  }
});

audioNoisy.addEventListener('ended', () => { isPlayingNoisy = false; btnPlayNoisy.textContent = '▶'; });
audioProc.addEventListener('ended', () => {
  isPlayingProc = false;
  btnPlayProc.textContent = '▶';
  document.querySelectorAll('.play-unnoised-btn').forEach(b => b.textContent = '▶ Listen Denoised');
  document.querySelectorAll('.unnoised-card').forEach(c => c.classList.remove('active-playback'));
});

scrubNoisy.addEventListener('input', (e) => {
  if (audioNoisy.duration) audioNoisy.currentTime = (e.target.value / 100) * audioNoisy.duration;
});

scrubProc.addEventListener('input', (e) => {
  if (audioProc.duration) audioProc.currentTime = (e.target.value / 100) * audioProc.duration;
});

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}


// ============================================================================
// CLEAN REFERENCE SAMPLE PICKER & QUICK PRESETS
// ============================================================================

const selectClean = document.getElementById('select-clean-sample');
const btnPlayClean = document.getElementById('btn-play-clean-ref');
const cleanPlayIcon = document.getElementById('clean-play-icon');
const cleanPlayText = document.getElementById('clean-play-text');

function updateCleanBtnUI(playing) {
  if (cleanPlayIcon) cleanPlayIcon.textContent = playing ? '⏸' : '🔊';
  if (cleanPlayText) cleanPlayText.textContent = playing ? 'Pause Clean' : 'Audition Clean';
}

if (selectClean) {
  selectClean.addEventListener('change', async () => {
    const sampleId = selectClean.value;
    const activePresetBtn = document.querySelector('.preset-btn[data-preset].active');
    const preset = activePresetBtn ? activePresetBtn.getAttribute('data-preset') : 'hum';

    const formData = new FormData();
    formData.append('sample_id', sampleId);
    formData.append('preset', preset);

    try {
      const res = await fetch('/api/select-sample', { method: 'POST', body: formData });
      const status = await res.json();
      updateStatusUI(status);
      refreshAudioSources();
      renderStaticWaveforms();
    } catch (e) {
      console.error('Error selecting clean sample:', e);
    }
  });
}

if (btnPlayClean) {
  btnPlayClean.addEventListener('click', () => {
    initWebAudio();
    if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

    if (isPlayingClean) {
      audioClean.pause();
      isPlayingClean = false;
      updateCleanBtnUI(false);
    } else {
      if (isPlayingNoisy) { audioNoisy.pause(); isPlayingNoisy = false; btnPlayNoisy.textContent = '▶'; }
      if (isPlayingProc) { audioProc.pause(); isPlayingProc = false; btnPlayProc.textContent = '▶'; }
      activeAudio = audioClean;
      audioClean.play();
      isPlayingClean = true;
      updateCleanBtnUI(true);
    }
  });

  audioClean.addEventListener('ended', () => {
    isPlayingClean = false;
    updateCleanBtnUI(false);
  });

  audioClean.addEventListener('timeupdate', () => {
    if (activeAudio === audioClean && audioClean.duration) {
      document.getElementById('osc-timecode').textContent = `${audioClean.currentTime.toFixed(2)}s / ${audioClean.duration.toFixed(2)}s`;
    }
  });
}

document.querySelectorAll('.preset-btn[data-preset]').forEach(btn => {
  btn.addEventListener('click', async () => {
    document.querySelectorAll('.preset-btn[data-preset]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const preset = btn.getAttribute('data-preset');
    const formData = new FormData();
    formData.append('preset', preset);

    try {
      const res = await fetch('/api/quick-preset', { method: 'POST', body: formData });
      const status = await res.json();
      updateStatusUI(status);
      refreshAudioSources();
      renderStaticWaveforms();
    } catch (e) {
      console.error('Error applying preset:', e);
    }
  });
});

function refreshAudioSources() {
  const timestamp = new Date().getTime();

  // Stop any active playback immediately
  audioNoisy.pause();
  audioProc.pause();
  audioClean.pause();
  isPlayingNoisy = false;
  isPlayingProc = false;
  isPlayingClean = false;
  if (btnPlayNoisy) btnPlayNoisy.textContent = '▶';
  if (btnPlayProc) btnPlayProc.textContent = '▶';
  updateCleanBtnUI(false);

  // Reset scrubber positions
  if (scrubNoisy) scrubNoisy.value = 0;
  if (scrubProc) scrubProc.value = 0;
  if (timeCurrNoisy) timeCurrNoisy.textContent = '0:00';
  if (timeCurrProc) timeCurrProc.textContent = '0:00';

  // Reset benchmark / unnoised card play buttons
  document.querySelectorAll('.play-unnoised-btn').forEach(b => b.textContent = '▶ Listen Denoised');
  document.querySelectorAll('.unnoised-card').forEach(c => c.classList.remove('active-playback'));

  audioNoisy.src = `/api/audio/noisy?t=${timestamp}`;
  audioProc.src = `/api/audio/processed?t=${timestamp}`;
  audioClean.src = `/api/audio/clean?t=${timestamp}`;
  audioNoisy.load();
  audioProc.load();
  audioClean.load();
}

function updateStatusUI(status) {
  document.getElementById('active-track-name').textContent = status.audio_source_name;
  document.getElementById('sr-status-text').textContent = `${status.sample_rate.toLocaleString()} Hz`;
  if (status.audio_metadata) {
    document.getElementById('noisy-rms').textContent = `${status.audio_metadata.rms_dbfs} dBFS`;
  }
  if (status.active_sample_id && selectClean && selectClean.value !== status.active_sample_id) {
    selectClean.value = status.active_sample_id;
  }

  // Handle clean reference availability
  if (btnPlayClean) {
    if (status.has_clean_ref) {
      btnPlayClean.disabled = false;
      btnPlayClean.style.opacity = '1.0';
      btnPlayClean.style.cursor = 'pointer';
      btnPlayClean.title = 'Listen to Active Clean Reference Audio';
      updateCleanBtnUI(isPlayingClean);
    } else {
      btnPlayClean.disabled = true;
      btnPlayClean.style.opacity = '0.5';
      btnPlayClean.style.cursor = 'not-allowed';
      btnPlayClean.title = 'No clean ground-truth reference available for uploaded file';
      if (cleanPlayText) cleanPlayText.textContent = 'Clean Ref: N/A';
    }
  }

  if (status.has_processed && status.last_method) {
    const procTag = document.getElementById('proc-tag');
    if (procTag) procTag.textContent = `🟢 Denoised (${status.last_method})`;
  }
}


// ============================================================================
// TABS SWITCHING
// ============================================================================

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    const tabId = `tab-${btn.getAttribute('data-tab')}`;
    const targetPanel = document.getElementById(tabId);
    if (targetPanel) targetPanel.classList.add('active');

    // Trigger canvas redraw on tab show
    setTimeout(resizeCanvases, 50);
  });
});


// ============================================================================
// WORKBENCH FILTER CONTROLS & PARAMS
// ============================================================================

const selFilterType = document.getElementById('sel-filter-type');
selFilterType.addEventListener('change', () => {
  const val = selFilterType.value;
  document.getElementById('panel-params-fir').style.display = (val === 'fir') ? 'block' : 'none';
  document.getElementById('panel-params-iir').style.display = (val === 'iir') ? 'block' : 'none';
  document.getElementById('panel-params-notch').style.display = (val === 'notch') ? 'block' : 'none';
  document.getElementById('panel-params-sg').style.display = (val === 'spectral_gating') ? 'block' : 'none';
  document.getElementById('panel-params-wiener').style.display = (val === 'wiener') ? 'block' : 'none';
  fetchFilterFrequencyResponse();
});

// Slider value listeners
function bindSlider(sliderId, badgeId, suffix = '') {
  const s = document.getElementById(sliderId);
  const b = document.getElementById(badgeId);
  if (!s || !b) return;
  s.addEventListener('input', () => {
    b.textContent = `${s.value}${suffix}`;
    fetchFilterFrequencyResponse();
  });
}

bindSlider('slider-fir-cut', 'val-fir-cut', ' Hz');
bindSlider('slider-fir-taps', 'val-fir-taps', '');
bindSlider('slider-iir-cut', 'val-iir-cut', ' Hz');
bindSlider('slider-iir-order', 'val-iir-order', '');
bindSlider('slider-notch-f0', 'val-notch-f0', ' Hz');
bindSlider('slider-notch-q', 'val-notch-q', '');
bindSlider('slider-sg-floor', 'val-sg-floor', ' dB');
bindSlider('slider-sg-over', 'val-sg-over', 'x');
bindSlider('slider-w-alpha', 'val-w-alpha', '');
bindSlider('slider-w-beta', 'val-w-beta', '');
bindSlider('slider-target-snr', 'val-target-snr', ' dB');


// ============================================================================
// EXECUTE DIGITAL FILTER ACTION
// ============================================================================

document.getElementById('btn-execute-filter').addEventListener('click', async () => {
  const fType = selFilterType.value;
  const formData = new FormData();
  formData.append('filter_type', fType);

  if (fType === 'fir') {
    formData.append('cutoff_hz', document.getElementById('slider-fir-cut').value);
    formData.append('fir_taps', document.getElementById('slider-fir-taps').value);
    formData.append('btype', document.getElementById('fir-btype').value);
  } else if (fType === 'iir') {
    formData.append('cutoff_hz', document.getElementById('slider-iir-cut').value);
    formData.append('iir_order', document.getElementById('slider-iir-order').value);
    formData.append('btype', document.getElementById('iir-btype').value);
  } else if (fType === 'notch') {
    formData.append('notch_freq', document.getElementById('slider-notch-f0').value);
    formData.append('notch_q', document.getElementById('slider-notch-q').value);
    formData.append('notch_harmonics', document.getElementById('chk-notch-harmonics').checked);
  } else if (fType === 'spectral_gating') {
    formData.append('suppression_db', document.getElementById('slider-sg-floor').value);
    formData.append('over_subtraction', document.getElementById('slider-sg-over').value);
  } else if (fType === 'wiener') {
    formData.append('wiener_alpha', document.getElementById('slider-w-alpha').value);
    formData.append('wiener_beta', document.getElementById('slider-w-beta').value);
  }

  try {
    const res = await fetch('/api/apply-filter', { method: 'POST', body: formData });
    const result = await res.json();

    // Update UI Metrics
    if (result.metrics) {
      const m = result.metrics;
      document.getElementById('metric-snr-imp').textContent = m.snr_improvement_db !== undefined ? `+${m.snr_improvement_db} dB` : `${m.total_attenuation_db} dB`;
      document.getElementById('metric-snr-out').textContent = m.output_snr_db !== undefined ? `${m.output_snr_db} dB` : `${m.output_rms_dbfs} dBFS`;
      document.getElementById('metric-rmse').textContent = m.rmse !== undefined ? m.rmse.toFixed(4) : m.total_attenuation_db;
      document.getElementById('metric-latency').textContent = `${result.processing_time_ms} ms`;
      document.getElementById('proc-snr-badge').textContent = m.snr_improvement_db !== undefined ? `+${m.snr_improvement_db} dB SNR Imp.` : 'Processed';
    }

    document.getElementById('proc-tag').textContent = `🟢 Denoised (${result.method})`;
    refreshAudioSources();
    renderStaticWaveforms();
  } catch (e) {
    console.error('Error applying filter:', e);
  }
});


// ============================================================================
// ML ADAPTIVE ASSISTANT ACTION
// ============================================================================

document.getElementById('btn-run-adaptive-full').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/adaptive-ml', { method: 'POST' });
    const result = await res.json();

    if (result.status === 'success') {
      const pred = result.prediction;
      const strat = result.strategy;

      document.getElementById('ml-predicted-label').textContent = pred.class_label;
      document.getElementById('ml-confidence-badge').textContent = `${(pred.confidence * 100).toFixed(1)}% Confidence`;
      document.getElementById('ml-rationale').textContent = strat.reason;

      // Update probability bars
      const probs = pred.class_probabilities;
      updateProbBar('hum', probs.electrical_hum);
      updateProbBar('rumble', probs.low_frequency_rumble);
      updateProbBar('hiss', probs.high_frequency_hiss);
      updateProbBar('broadband', probs.broadband_noise);
      updateProbBar('clean', probs.clean_speech);

      // Highlight winning class bar
      document.querySelectorAll('.prob-bar').forEach(b => b.classList.remove('winner'));
      if (pred.predicted_class === 'electrical_hum') document.getElementById('bar-hum').classList.add('winner');
      else if (pred.predicted_class === 'low_frequency_rumble') document.getElementById('bar-rumble').classList.add('winner');
      else if (pred.predicted_class === 'high_frequency_hiss') document.getElementById('bar-hiss').classList.add('winner');
      else if (pred.predicted_class === 'broadband_noise') document.getElementById('bar-broadband').classList.add('winner');
      else if (pred.predicted_class === 'clean_speech') document.getElementById('bar-clean').classList.add('winner');

      document.getElementById('proc-tag').textContent = `🟢 Denoised (${strat.strategy})`;
      refreshAudioSources();
      renderStaticWaveforms();
    }
  } catch (e) {
    console.error('Error running adaptive ML:', e);
  }
});

function updateProbBar(key, val) {
  const pct = (val * 100).toFixed(1);
  const bar = document.getElementById(`bar-${key}`);
  const txt = document.getElementById(`prob-val-${key}`);
  if (bar && txt) {
    bar.style.width = `${pct}%`;
    txt.textContent = `${pct}%`;
  }
}


// ============================================================================
// MULTI-METHOD BENCHMARK ACTION
// ============================================================================

document.getElementById('btn-run-benchmark-full').addEventListener('click', async () => {
  const btn = document.getElementById('btn-run-benchmark-full');
  btn.innerHTML = '<span>⏳ Evaluating All Methods...</span>';

  try {
    const res = await fetch('/api/benchmark', { method: 'POST' });
    const data = await res.json();

    if (data.status === 'success') {
      const tbody = document.getElementById('benchmark-tbody');
      tbody.innerHTML = '';

      // Find highest SNR improvement
      let maxSnr = -999;
      data.results.forEach(r => {
        if (r.snr_improvement_db && r.snr_improvement_db > maxSnr) maxSnr = r.snr_improvement_db;
      });

      data.results.forEach(r => {
        const tr = document.createElement('tr');
        const isWinner = r.snr_improvement_db && Math.abs(r.snr_improvement_db - maxSnr) < 0.1 && r.method_id !== 'baseline';
        if (isWinner) tr.classList.add('winner-row');

        const snrImpText = r.snr_improvement_db !== undefined ? `+${r.snr_improvement_db.toFixed(2)} dB` : `${r.total_attenuation_db.toFixed(2)} dB`;
        const outSnrText = r.output_snr_db !== undefined ? `${r.output_snr_db.toFixed(2)} dB` : `${r.output_rms_dbfs.toFixed(2)} dBFS`;
        const rmseText = r.rmse !== undefined ? r.rmse.toFixed(4) : '-';

        tr.innerHTML = `
          <td><button class="table-audio-btn" data-audio-id="${r.method_id}">▶</button></td>
          <td>${r.method_name} ${isWinner ? '🏆' : ''}</td>
          <td style="${r.snr_improvement_db && r.snr_improvement_db >= 0 ? 'color: var(--accent-emerald); font-weight:700;' : ''}">${snrImpText}</td>
          <td>${outSnrText}</td>
          <td>${rmseText}</td>
          <td>${(r.processing_time_sec * 1000).toFixed(1)} ms</td>
          <td>${r.real_time_factor.toFixed(3)}x</td>
        `;
        tbody.appendChild(tr);
      });

      // Bind in-table audio players
      tbody.querySelectorAll('.table-audio-btn').forEach(b => {
        b.addEventListener('click', () => {
          const mId = b.getAttribute('data-audio-id');
          audioProc.src = `/api/audio/${mId}`;
          audioProc.play();
          isPlayingProc = true;
          btnPlayProc.textContent = '⏸';
          document.getElementById('proc-tag').textContent = `🟢 Benchmark Track: ${mId}`;
        });
      });
    }
  } catch (e) {
    console.error('Error running benchmark:', e);
  } finally {
    btn.innerHTML = '<span>🚀 Run All 6 Methods</span>';
  }
});


// ============================================================================
// UNNOISED AUDIO LISTENING RACK & BATCH EXPORT
// ============================================================================

document.querySelectorAll('.play-unnoised-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    initWebAudio();
    if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

    const mId = btn.getAttribute('data-audio-id');
    const card = btn.closest('.unnoised-card');
    const isCurrentPlaying = (isPlayingProc && activeAudio === audioProc && audioProc.src.endsWith(`/api/audio/${mId}`));

    if (isCurrentPlaying) {
      audioProc.pause();
      isPlayingProc = false;
      btn.textContent = '▶ Listen Denoised';
      if (card) card.classList.remove('active-playback');
      btnPlayProc.textContent = '▶';
    } else {
      if (isPlayingNoisy) { audioNoisy.pause(); isPlayingNoisy = false; btnPlayNoisy.textContent = '▶'; }
      if (isPlayingClean) { audioClean.pause(); isPlayingClean = false; updateCleanBtnUI(false); }

      // Reset other unnoised buttons
      document.querySelectorAll('.play-unnoised-btn').forEach(b => b.textContent = '▶ Listen Denoised');
      document.querySelectorAll('.unnoised-card').forEach(c => c.classList.remove('active-playback'));

      audioProc.src = `/api/audio/${mId}`;
      activeAudio = audioProc;
      audioProc.play().then(() => {
        isPlayingProc = true;
        btn.textContent = '⏸ Pause';
        if (card) card.classList.add('active-playback');
        btnPlayProc.textContent = '⏸';
        const tag = document.getElementById('proc-tag');
        if (tag) tag.textContent = `🟢 Denoised: ${mId.toUpperCase()}`;
      }).catch(err => console.error('Audio play error:', err));
    }
  });
});

const btnExportAll = document.getElementById('btn-export-all-wavs');
if (btnExportAll) {
  btnExportAll.addEventListener('click', async () => {
    const originalText = btnExportAll.innerHTML;
    btnExportAll.innerHTML = '<span>⏳ Exporting All 6 Topologies...</span>';
    btnExportAll.disabled = true;

    try {
      const res = await fetch('/api/export-all-unnoised', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        const fileList = data.files.map(f => `  • ${f.filename} (${f.name})`).join('\n');
        alert(`Exported all unnoised audio files successfully to:\n${data.output_directory}\n\nFiles Generated:\n${fileList}`);
      } else {
        alert('Export completed with status: ' + data.status);
      }
    } catch (err) {
      console.error('Error exporting unnoised audio:', err);
      alert('Error exporting unnoised audio: ' + err.message);
    } finally {
      btnExportAll.innerHTML = originalText;
      btnExportAll.disabled = false;
    }
  });
}


// ============================================================================
// NOISE SIMULATION LAB ACTIONS
// ============================================================================

document.getElementById('btn-mix-custom').addEventListener('click', async () => {
  const noiseType = document.getElementById('sel-noise-type').value;
  const snr = document.getElementById('slider-target-snr').value;

  const formData = new FormData();
  formData.append('noise_type', noiseType);
  formData.append('snr_db', snr);

  try {
    const res = await fetch('/api/mix-noise', { method: 'POST', body: formData });
    const data = await res.json();
    refreshAudioSources();
    renderStaticWaveforms();
    alert(`Synthesized custom mixture at ${data.measured_snr_db.toFixed(1)} dB SNR!`);
  } catch (e) {
    console.error('Error mixing noise:', e);
  }
});

document.getElementById('btn-upload-files').addEventListener('click', async () => {
  const noisyFile = document.getElementById('input-file-noisy').files[0];
  const refFile = document.getElementById('input-file-ref').files[0];

  if (!noisyFile) {
    alert('Please choose a noisy WAV file to upload.');
    return;
  }

  const uploadBtn = document.getElementById('btn-upload-files');
  const originalHtml = uploadBtn.innerHTML;
  uploadBtn.innerHTML = '<span>⏳ Processing Audio...</span>';
  uploadBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', noisyFile);
  if (refFile) formData.append('ref_file', refFile);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const status = await res.json();
    updateStatusUI(status);
    refreshAudioSources();
    renderStaticWaveforms();
    alert(`Audio file "${noisyFile.name}" loaded and automatically denoised into Studio!`);
  } catch (e) {
    console.error('Error uploading file:', e);
    alert('Error uploading file: ' + e.message);
  } finally {
    uploadBtn.innerHTML = originalHtml;
    uploadBtn.disabled = false;
  }
});


// ============================================================================
// STARTUP BOOTSTRAP
// ============================================================================

window.addEventListener('DOMContentLoaded', async () => {
  resizeCanvases();
  try {
    const res = await fetch('/api/status');
    const status = await res.json();
    updateStatusUI(status);
    refreshAudioSources();
    renderStaticWaveforms();
    fetchFilterFrequencyResponse();
  } catch (e) {
    console.error('Error booting studio:', e);
  }
});
