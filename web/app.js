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
let activeUploadedBlob = null;
let activeReferenceBlob = null;
let benchmarkDataUris = {};

function dataURItoBlob(dataURI) {
  if (!dataURI) return null;
  try {
    const parts = dataURI.split(',');
    const byteString = atob(parts[1]);
    const mimeString = parts[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
  } catch (e) {
    console.error('Error converting data URI to blob:', e);
    return null;
  }
}

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

function handleAudioMetadata(audioEl, durEl) {
  if (audioEl.duration && isFinite(audioEl.duration)) {
    durEl.textContent = formatTime(audioEl.duration);
    const oscEl = document.getElementById('osc-timecode');
    if (oscEl && (!activeAudio || activeAudio === audioEl)) {
      oscEl.textContent = `${audioEl.currentTime.toFixed(2)}s / ${audioEl.duration.toFixed(2)}s`;
    }
  }
}

audioNoisy.addEventListener('loadedmetadata', () => handleAudioMetadata(audioNoisy, timeDurNoisy));
audioNoisy.addEventListener('durationchange', () => handleAudioMetadata(audioNoisy, timeDurNoisy));
audioProc.addEventListener('loadedmetadata', () => handleAudioMetadata(audioProc, timeDurProc));
audioProc.addEventListener('durationchange', () => handleAudioMetadata(audioProc, timeDurProc));

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
      if (status.noisy_audio_data_uri) {
        activeUploadedBlob = dataURItoBlob(status.noisy_audio_data_uri);
        activeReferenceBlob = null;
        audioNoisy.src = status.noisy_audio_data_uri;
        audioNoisy.load();
      }
      if (status.processed_audio_data_uri) {
        audioProc.src = status.processed_audio_data_uri;
        audioProc.load();
      }
      if (status.noisy_waveform && status.noisy_waveform.length) {
        drawSingleWaveform(ctxNoisy, canvasNoisy, status.noisy_waveform, '#f43f5e', 'rgba(244, 63, 94, 0.08)');
      }
      if (status.processed_waveform && status.processed_waveform.length) {
        drawSingleWaveform(ctxProc, canvasProc, status.processed_waveform, '#10b981', 'rgba(16, 185, 129, 0.08)');
      }
      refreshAudioSources();
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
      if (status.noisy_audio_data_uri) {
        activeUploadedBlob = dataURItoBlob(status.noisy_audio_data_uri);
        activeReferenceBlob = null;
        audioNoisy.src = status.noisy_audio_data_uri;
        audioNoisy.load();
      }
      if (status.processed_audio_data_uri) {
        audioProc.src = status.processed_audio_data_uri;
        audioProc.load();
      }
      if (status.noisy_waveform && status.noisy_waveform.length) {
        drawSingleWaveform(ctxNoisy, canvasNoisy, status.noisy_waveform, '#f43f5e', 'rgba(244, 63, 94, 0.08)');
      }
      if (status.processed_waveform && status.processed_waveform.length) {
        drawSingleWaveform(ctxProc, canvasProc, status.processed_waveform, '#10b981', 'rgba(16, 185, 129, 0.08)');
      }
      refreshAudioSources();
    } catch (e) {
      console.error('Error applying preset:', e);
    }
  });
});

// Shortcut button to open Noise Simulation Lab tab
const btnCustomMixModal = document.getElementById('btn-custom-mix-modal');
if (btnCustomMixModal) {
  btnCustomMixModal.addEventListener('click', () => {
    const simTabBtn = document.querySelector('.tab-btn[data-tab="simulation"]');
    if (simTabBtn) {
      simTabBtn.click();
      simTabBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
}

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

function showToast(message, type = 'success') {
  let toast = document.getElementById('studio-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'studio-toast';
    toast.style.position = 'fixed';
    toast.style.bottom = '24px';
    toast.style.right = '24px';
    toast.style.zIndex = '9999';
    toast.style.padding = '12px 20px';
    toast.style.borderRadius = '8px';
    toast.style.fontSize = '0.90rem';
    toast.style.fontWeight = '600';
    toast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.5)';
    toast.style.transition = 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)';
    toast.style.display = 'flex';
    toast.style.alignItems = 'center';
    toast.style.gap = '8px';
    toast.style.backdropFilter = 'blur(10px)';
    document.body.appendChild(toast);
  }

  if (type === 'error') {
    toast.style.background = 'rgba(239, 68, 68, 0.95)';
    toast.style.color = '#ffffff';
    toast.style.border = '1px solid #f87171';
  } else if (type === 'info') {
    toast.style.background = 'rgba(59, 130, 246, 0.95)';
    toast.style.color = '#ffffff';
    toast.style.border = '1px solid #60a5fa';
  } else {
    toast.style.background = 'rgba(16, 185, 129, 0.95)';
    toast.style.color = '#ffffff';
    toast.style.border = '1px solid #34d399';
  }

  toast.textContent = message;
  toast.style.opacity = '1';
  toast.style.transform = 'translateY(0)';

  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
  }, 4000);
}

function updateStatusUI(status) {
  if (!status) return;

  if (status.audio_source_name) {
    const trackEl = document.getElementById('active-track-name');
    if (trackEl) trackEl.textContent = status.audio_source_name;
  }
  if (status.sample_rate) {
    const srEl = document.getElementById('sr-status-text');
    if (srEl) srEl.textContent = `${Number(status.sample_rate).toLocaleString()} Hz`;
  }
  if (status.audio_metadata && status.audio_metadata.rms_dbfs !== undefined) {
    const rmsEl = document.getElementById('noisy-rms');
    if (rmsEl) rmsEl.textContent = `${status.audio_metadata.rms_dbfs} dBFS`;
  }
  if (status.audio_metadata && status.audio_metadata.duration_seconds !== undefined) {
    const durSec = Number(status.audio_metadata.duration_seconds);
    const durFormatted = formatTime(durSec);
    if (timeDurNoisy) timeDurNoisy.textContent = durFormatted;
    if (timeDurProc) timeDurProc.textContent = durFormatted;
    const oscEl = document.getElementById('osc-timecode');
    if (oscEl) oscEl.textContent = `0.00s / ${durSec.toFixed(2)}s`;
  }
  if (status.noisy_audio_data_uri) {
    const dlNoisy = document.getElementById('btn-download-noisy');
    if (dlNoisy) dlNoisy.href = status.noisy_audio_data_uri;
  }
  if (status.processed_audio_data_uri) {
    const dlProc = document.getElementById('btn-download-proc');
    if (dlProc) dlProc.href = status.processed_audio_data_uri;
  }
  if (status.noisy_audio_data_uri && !activeUploadedBlob) {
    activeUploadedBlob = dataURItoBlob(status.noisy_audio_data_uri);
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
  if (activeUploadedBlob) {
    formData.append('file', activeUploadedBlob, 'current_audio.wav');
  }
  if (activeReferenceBlob) {
    formData.append('ref_file', activeReferenceBlob, 'clean_ref.wav');
  }
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

    if (result.processed_audio_data_uri) {
      audioProc.pause();
      isPlayingProc = false;
      btnPlayProc.textContent = '▶';
      audioProc.src = result.processed_audio_data_uri;
      audioProc.load();
    } else {
      refreshAudioSources();
    }

    if (result.processed_waveform && result.processed_waveform.length) {
      drawSingleWaveform(ctxProc, canvasProc, result.processed_waveform, '#10b981', 'rgba(16, 185, 129, 0.08)');
    } else {
      renderStaticWaveforms();
    }

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
    showToast(`Applied ${result.method}!`, 'success');
  } catch (e) {
    console.error('Error applying filter:', e);
  }
});


// ============================================================================
// ML ADAPTIVE ASSISTANT ACTION
// ============================================================================

document.getElementById('btn-run-adaptive-full').addEventListener('click', async () => {
  const btn = document.getElementById('btn-run-adaptive-full');
  const origHtml = btn.innerHTML;
  btn.innerHTML = '<span>⏳ Diagnosing & Denoising...</span>';
  btn.disabled = true;

  try {
    const formData = new FormData();
    if (activeUploadedBlob) {
      formData.append('file', activeUploadedBlob, 'current_audio.wav');
    }
    if (activeReferenceBlob) {
      formData.append('ref_file', activeReferenceBlob, 'clean_ref.wav');
    }
    const res = await fetch('/api/adaptive-ml', { method: 'POST', body: formData });
    const result = await res.json();

    if (result.status === 'success') {
      const pred = result.prediction;
      const strat = result.strategy;

      if (result.processed_audio_data_uri) {
        audioProc.pause();
        isPlayingProc = false;
        btnPlayProc.textContent = '▶';
        audioProc.src = result.processed_audio_data_uri;
        audioProc.load();
      } else {
        refreshAudioSources();
      }

      if (result.processed_waveform && result.processed_waveform.length) {
        drawSingleWaveform(ctxProc, canvasProc, result.processed_waveform, '#10b981', 'rgba(16, 185, 129, 0.08)');
      } else {
        renderStaticWaveforms();
      }

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
      showToast(`Cleaned with ${strat.strategy}!`, 'success');
    }
  } catch (e) {
    console.error('Error running adaptive ML:', e);
    showToast('Failed to run Adaptive ML', 'error');
  } finally {
    btn.innerHTML = origHtml;
    btn.disabled = false;
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
// ON-DEMAND METHOD AUDIO RESOLVER & UNIFIED PLAYBACK CONTROLLER
// ============================================================================

let currentlyPlayingMethodId = null;

async function getOrComputeMethodAudioUri(mId) {
  if (benchmarkDataUris[mId]) {
    return benchmarkDataUris[mId];
  }

  // 1. Baseline: simply use active noisy audio URI
  if (mId === 'baseline' || mId === 'input' || mId === 'noisy') {
    if (audioNoisy.src && audioNoisy.src.startsWith('data:')) {
      benchmarkDataUris[mId] = audioNoisy.src;
      return audioNoisy.src;
    }
  }

  // 2. Adaptive ML
  if (mId === 'adaptive') {
    const formData = new FormData();
    if (activeUploadedBlob) formData.append('file', activeUploadedBlob, 'current_audio.wav');
    if (activeReferenceBlob) formData.append('ref_file', activeReferenceBlob, 'clean_ref.wav');
    const res = await fetch('/api/adaptive-ml', { method: 'POST', body: formData });
    const result = await res.json();
    if (result.processed_audio_data_uri) {
      benchmarkDataUris[mId] = result.processed_audio_data_uri;
      return result.processed_audio_data_uri;
    }
  }

  // 3. Individual DSP Filters executed on active audio payload
  const formData = new FormData();
  if (activeUploadedBlob) formData.append('file', activeUploadedBlob, 'current_audio.wav');
  if (activeReferenceBlob) formData.append('ref_file', activeReferenceBlob, 'clean_ref.wav');

  if (mId === 'fir') {
    formData.append('filter_type', 'fir');
    formData.append('btype', 'bandpass');
    formData.append('cutoff_hz', '300');
    formData.append('cutoff_high_hz', '3400');
    formData.append('fir_taps', '101');
  } else if (mId === 'butterworth') {
    formData.append('filter_type', 'iir');
    formData.append('btype', 'highpass');
    formData.append('cutoff_hz', '150');
    formData.append('iir_order', '4');
  } else if (mId === 'notch') {
    formData.append('filter_type', 'notch');
    formData.append('notch_freq', '50');
    formData.append('notch_q', '30');
    formData.append('notch_harmonics', 'true');
  } else if (mId === 'spec_gate') {
    formData.append('filter_type', 'spectral_gating');
    formData.append('suppression_db', '-18');
    formData.append('over_subtraction', '1.5');
  } else if (mId === 'wiener') {
    formData.append('filter_type', 'wiener');
    formData.append('wiener_alpha', '0.95');
    formData.append('wiener_beta', '1.25');
    formData.append('wiener_floor', '-18');
  } else {
    formData.append('filter_type', mId);
  }

  const res = await fetch('/api/apply-filter', { method: 'POST', body: formData });
  const result = await res.json();
  if (result.processed_audio_data_uri) {
    benchmarkDataUris[mId] = result.processed_audio_data_uri;
    return result.processed_audio_data_uri;
  }
  return null;
}

async function playMethodAudio(mId, btn, card) {
  initWebAudio();
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

  // If this method track is already playing: pause it
  if (isPlayingProc && currentlyPlayingMethodId === mId) {
    audioProc.pause();
    isPlayingProc = false;
    currentlyPlayingMethodId = null;
    if (btn) btn.textContent = btn.classList.contains('play-unnoised-btn') ? '▶ Listen Denoised' : '▶';
    if (card) card.classList.remove('active-playback');
    btnPlayProc.textContent = '▶';
    return;
  }

  // Stop noisy or clean playback
  if (isPlayingNoisy) { audioNoisy.pause(); isPlayingNoisy = false; btnPlayNoisy.textContent = '▶'; }
  if (isPlayingClean) { audioClean.pause(); isPlayingClean = false; updateCleanBtnUI(false); }

  // Reset all other button labels
  document.querySelectorAll('.table-audio-btn').forEach(b => b.textContent = '▶');
  document.querySelectorAll('.play-unnoised-btn').forEach(b => b.textContent = '▶ Listen Denoised');
  document.querySelectorAll('.unnoised-card').forEach(c => c.classList.remove('active-playback'));

  // Show loading indicator if generating
  const isTableBtn = btn && btn.classList.contains('table-audio-btn');
  const origBtnText = btn ? btn.textContent : '▶';
  if (btn && !benchmarkDataUris[mId]) {
    btn.textContent = isTableBtn ? '⏳' : '⏳ Processing...';
  }

  try {
    const uri = await getOrComputeMethodAudioUri(mId);
    if (!uri) throw new Error('Could not produce audio for ' + mId);

    activeAudio = audioProc;
    audioProc.src = uri;
    audioProc.load();
    await audioProc.play();

    isPlayingProc = true;
    currentlyPlayingMethodId = mId;
    if (btn) btn.textContent = isTableBtn ? '⏸' : '⏸ Pause';
    if (card) card.classList.add('active-playback');
    btnPlayProc.textContent = '⏸';
    const tag = document.getElementById('proc-tag');
    if (tag) tag.textContent = `🟢 Denoised: ${mId.toUpperCase()}`;

    // Update download links for this method
    const dlProc = document.getElementById('btn-download-proc');
    if (dlProc) dlProc.href = uri;

    const rackLink = document.querySelector(`a[href*="/api/audio/${mId}"]`);
    if (rackLink) rackLink.href = uri;
  } catch (err) {
    console.error('Error playing method audio:', err);
    if (btn) btn.textContent = origBtnText;
    showToast('Failed to play ' + mId + ': ' + err.message, 'error');
  }
}

// Bind initial static benchmark table play buttons on page load
document.querySelectorAll('.table-audio-btn').forEach(b => {
  b.addEventListener('click', () => {
    const mId = b.getAttribute('data-audio-id');
    playMethodAudio(mId, b, null);
  });
});

// Bind Unnoised Audio Listening Rack buttons
document.querySelectorAll('.play-unnoised-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const mId = btn.getAttribute('data-audio-id');
    const card = btn.closest('.unnoised-card');
    playMethodAudio(mId, btn, card);
  });
});

// ============================================================================
// MULTI-METHOD BENCHMARK ACTION
// ============================================================================

document.getElementById('btn-run-benchmark-full').addEventListener('click', async () => {
  const btn = document.getElementById('btn-run-benchmark-full');
  btn.innerHTML = '<span>⏳ Evaluating All Methods...</span>';

  try {
    const formData = new FormData();
    if (activeUploadedBlob) {
      formData.append('file', activeUploadedBlob, 'current_audio.wav');
    }
    if (activeReferenceBlob) {
      formData.append('ref_file', activeReferenceBlob, 'clean_ref.wav');
    }
    const res = await fetch('/api/benchmark', { method: 'POST', body: formData });
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

      // Cache data URIs for all methods
      data.results.forEach(r => {
        if (r.audio_data_uri) {
          benchmarkDataUris[r.method_id] = r.audio_data_uri;
          const rackLink = document.querySelector(`a[href*="/api/audio/${r.method_id}"]`);
          if (rackLink) rackLink.href = r.audio_data_uri;
        }
      });

      // Bind dynamically created in-table audio players
      tbody.querySelectorAll('.table-audio-btn').forEach(b => {
        b.addEventListener('click', () => {
          const mId = b.getAttribute('data-audio-id');
          playMethodAudio(mId, b, null);
        });
      });
    }
  } catch (e) {
    console.error('Error running benchmark:', e);
    showToast('Failed to evaluate methods: ' + e.message, 'error');
  } finally {
    btn.innerHTML = '<span>🚀 Run All 6 Methods</span>';
  }
});

const btnExportAll = document.getElementById('btn-export-all-wavs');
if (btnExportAll) {
  btnExportAll.addEventListener('click', async () => {
    const originalText = btnExportAll.innerHTML;
    btnExportAll.innerHTML = '<span>⏳ Exporting All 6 Topologies...</span>';
    btnExportAll.disabled = true;

    try {
      const formData = new FormData();
      if (activeUploadedBlob) formData.append('file', activeUploadedBlob, 'current_audio.wav');
      if (activeReferenceBlob) formData.append('ref_file', activeReferenceBlob, 'clean_ref.wav');

      const res = await fetch('/api/export-all-unnoised', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.status === 'success') {
        showToast(`Exported all 6 unnoised WAVs to outputs/denoised_showcase/!`, 'success');
      } else {
        showToast('Export status: ' + data.status, 'info');
      }
    } catch (err) {
      console.error('Error exporting unnoised audio:', err);
      showToast('Error exporting audio: ' + err.message, 'error');
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
    showToast(`Synthesized custom mixture at ${data.measured_snr_db.toFixed(1)} dB SNR!`, 'success');
  } catch (e) {
    console.error('Error mixing noise:', e);
    showToast(`Failed to mix noise: ${e.message}`, 'error');
  }
});

document.getElementById('btn-upload-files').addEventListener('click', async () => {
  const noisyFileInput = document.getElementById('input-file-noisy');
  const refFileInput = document.getElementById('input-file-ref');
  const noisyFile = (noisyFileInput && noisyFileInput.files) ? noisyFileInput.files[0] : null;
  const refFile = (refFileInput && refFileInput.files) ? refFileInput.files[0] : null;

  if (!noisyFile) {
    showToast('Please select a noisy WAV file to upload.', 'error');
    return;
  }

  const uploadBtn = document.getElementById('btn-upload-files');
  const originalHtml = uploadBtn.innerHTML;
  uploadBtn.innerHTML = '<span>⏳ Processing Audio...</span>';
  uploadBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', noisyFile);
  if (refFile && refFile.size > 0) {
    formData.append('ref_file', refFile);
  }

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      let errMsg = `Upload failed (HTTP ${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errMsg = errJson.detail;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const result = await res.json();

    // Store active uploaded blobs for subsequent operations
    activeUploadedBlob = noisyFile;
    activeReferenceBlob = (refFile && refFile.size > 0) ? refFile : null;

    // Reset benchmark cache and currently playing method
    benchmarkDataUris = {};
    if (result.noisy_audio_data_uri) benchmarkDataUris['baseline'] = result.noisy_audio_data_uri;
    if (result.processed_audio_data_uri) benchmarkDataUris['adaptive'] = result.processed_audio_data_uri;
    currentlyPlayingMethodId = null;
    document.querySelectorAll('.table-audio-btn').forEach(b => b.textContent = '▶');
    document.querySelectorAll('.play-unnoised-btn').forEach(b => b.textContent = '▶ Listen Denoised');

    // 1. Instantly bind uploaded noisy audio and denoised audio
    audioNoisy.pause();
    audioProc.pause();
    audioClean.pause();
    isPlayingNoisy = false;
    isPlayingProc = false;
    isPlayingClean = false;
    btnPlayNoisy.textContent = '▶';
    btnPlayProc.textContent = '▶';

    if (result.noisy_audio_data_uri) {
      audioNoisy.src = result.noisy_audio_data_uri;
    } else {
      audioNoisy.src = URL.createObjectURL(noisyFile);
    }
    if (result.processed_audio_data_uri) {
      audioProc.src = result.processed_audio_data_uri;
    }
    audioNoisy.load();
    audioProc.load();

    // 2. Instantly draw waveforms
    if (result.noisy_waveform && result.noisy_waveform.length) {
      drawSingleWaveform(ctxNoisy, canvasNoisy, result.noisy_waveform, '#f43f5e', 'rgba(244, 63, 94, 0.08)');
    }
    if (result.processed_waveform && result.processed_waveform.length) {
      drawSingleWaveform(ctxProc, canvasProc, result.processed_waveform, '#10b981', 'rgba(16, 185, 129, 0.08)');
    }

    // 3. Update status UI
    updateStatusUI(result);

    // 4. Update ML Diagnosis card
    if (result.prediction) {
      const pred = result.prediction;
      document.getElementById('ml-predicted-label').textContent = pred.class_label;
      document.getElementById('ml-confidence-badge').textContent = `${(pred.confidence * 100).toFixed(1)}% Confidence`;
      if (result.strategy && result.strategy.reason) {
        document.getElementById('ml-rationale').textContent = result.strategy.reason;
      }
      const probs = pred.class_probabilities;
      if (probs) {
        updateProbBar('hum', probs.electrical_hum || 0);
        updateProbBar('rumble', probs.low_frequency_rumble || 0);
        updateProbBar('hiss', probs.high_frequency_hiss || 0);
        updateProbBar('broadband', probs.broadband_noise || 0);
        updateProbBar('clean', probs.clean_speech || 0);
      }
      document.querySelectorAll('.prob-bar').forEach(b => b.classList.remove('winner'));
      if (pred.predicted_class === 'electrical_hum') document.getElementById('bar-hum').classList.add('winner');
      else if (pred.predicted_class === 'low_frequency_rumble') document.getElementById('bar-rumble').classList.add('winner');
      else if (pred.predicted_class === 'high_frequency_hiss') document.getElementById('bar-hiss').classList.add('winner');
      else if (pred.predicted_class === 'broadband_noise') document.getElementById('bar-broadband').classList.add('winner');
      else if (pred.predicted_class === 'clean_speech') document.getElementById('bar-clean').classList.add('winner');
    }

    // 5. Update processed deck badge & tag
    if (result.metrics) {
      const m = result.metrics;
      document.getElementById('metric-snr-imp').textContent = m.snr_improvement_db !== undefined ? `+${m.snr_improvement_db} dB` : `${m.total_attenuation_db} dB`;
      document.getElementById('metric-snr-out').textContent = m.output_snr_db !== undefined ? `${m.output_snr_db} dB` : `${m.output_rms_dbfs} dBFS`;
      document.getElementById('metric-rmse').textContent = m.rmse !== undefined ? m.rmse.toFixed(4) : '-';
      document.getElementById('proc-snr-badge').textContent = m.snr_improvement_db !== undefined ? `+${m.snr_improvement_db} dB SNR Imp.` : 'Denoised';
    }
    const tag = result.last_method || (result.strategy && result.strategy.strategy) || 'Adaptive ML';
    document.getElementById('proc-tag').textContent = `🟢 Denoised (${tag})`;

    showToast(`Loaded "${noisyFile.name}" — auto-denoised! Click Play above.`, 'success');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (e) {
    console.error('Error uploading file:', e);
    showToast(`Upload failed: ${e.message}`, 'error');
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

  // Initialize GradientWaves (React Bits WebGL2 Component)
  const wavesContainer = document.getElementById('gradient-waves-container');
  if (wavesContainer && window.initGradientWaves) {
    try {
      window.initGradientWaves(wavesContainer, {
        horizonColor: '#5227FF',
        waveColor: '#FF9FFC',
        crestColor: '#FFFFFF',
        speed: 0.4,
        amplitude: 2.5,
        relief: 1,
        lightAngle: 120,
        mouseInteraction: true,
        swellOnClick: true,
        quality: window.innerWidth < 768 ? 'low' : 'medium',
        dpr: Math.min(window.devicePixelRatio || 1, 1.5)
      });
    } catch (err) {
      console.warn('GradientWaves initialization warning:', err);
    }
  }

  // Initialize Interference Ribbons (React Bits Pro Canvas Component)
  const ribbonsContainer = document.getElementById('ribbons-oscilloscope-container');
  if (ribbonsContainer && window.initInterferenceRibbons) {
    try {
      window.initInterferenceRibbons(ribbonsContainer, {
        numWires: window.innerWidth < 768 ? 10 : 16,
        wireColor1: '#06b6d4',
        wireColor2: '#8b5cf6',
        wireColor3: '#ec4899',
        speed: 0.0018,
        amplitude: 35,
        tension: 0.003,
        opacity: 0.55
      });
    } catch (err) {
      console.warn('Interference Ribbons initialization warning:', err);
    }
  }

  // Initialize Axial Beam (React Bits Pro Canvas Component)
  const beamContainer = document.getElementById('axial-beam-container');
  if (beamContainer && window.initAxialBeam) {
    try {
      window.initAxialBeam(beamContainer, {
        beamCount: window.innerWidth < 768 ? 4 : 7,
        baseColor: '#4f46e5',
        accentColor: '#06b6d4',
        glowIntensity: 0.55,
        speed: 0.0012
      });
    } catch (err) {
      console.warn('Axial Beam initialization warning:', err);
    }
  }

  // Initialize Satin Ridges (React Bits Pro Canvas Component)
  const ridgesContainer = document.getElementById('satin-ridges-container');
  if (ridgesContainer && window.initSatinRidges) {
    try {
      window.initSatinRidges(ridgesContainer, {
        ridgeCount: window.innerWidth < 768 ? 3 : 5,
        speed: 0.0009,
        opacity: 0.4
      });
    } catch (err) {
      console.warn('Satin Ridges initialization warning:', err);
    }
  }

  // Mobile Bottom Bar Quick-Action Listeners
  const mobNoisy = document.getElementById('mob-btn-noisy');
  const mobProc = document.getElementById('mob-btn-proc');
  const mobClean = document.getElementById('mob-btn-clean');
  const mobAdaptive = document.getElementById('mob-btn-adaptive');

  if (mobNoisy) mobNoisy.addEventListener('click', () => btnPlayNoisy.click());
  if (mobProc) mobProc.addEventListener('click', () => btnPlayProc.click());
  if (mobClean) {
    const btnCleanRef = document.getElementById('btn-play-clean-ref');
    mobClean.addEventListener('click', () => btnCleanRef && btnCleanRef.click());
  }
  if (mobAdaptive) {
    const btnAdaptive = document.getElementById('btn-run-adaptive-full');
    mobAdaptive.addEventListener('click', () => btnAdaptive && btnAdaptive.click());
  }

  // Global Keyboard Shortcuts (WCAG Accessibility)
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

    if (e.code === 'Space') {
      e.preventDefault();
      if (isPlayingNoisy) btnPlayNoisy.click();
      else if (isPlayingProc) btnPlayProc.click();
      else if (isPlayingClean) {
        const btnClean = document.getElementById('btn-play-clean-ref');
        if (btnClean) btnClean.click();
      } else {
        btnPlayProc.click();
      }
    } else if (e.key === '1') {
      e.preventDefault();
      btnPlayNoisy.click();
    } else if (e.key === '2') {
      e.preventDefault();
      btnPlayProc.click();
    } else if (e.key === '3') {
      e.preventDefault();
      const btnClean = document.getElementById('btn-play-clean-ref');
      if (btnClean) btnClean.click();
    } else if (e.key.toLowerCase() === 'm') {
      e.preventDefault();
      const btnAdaptive = document.getElementById('btn-run-adaptive-full');
      if (btnAdaptive) btnAdaptive.click();
    } else if (e.key.toLowerCase() === 'b') {
      e.preventDefault();
      const btnBench = document.getElementById('btn-run-benchmark-full');
      if (btnBench) btnBench.click();
    }
  });

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
