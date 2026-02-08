/**
 * main.js - RuVector VWM Viewer entry point
 *
 * Initializes WebGPU, sets up the camera and renderer, generates demo data
 * (or connects to WASM when available), and runs the render loop.
 */

import { OrbitCamera } from './camera.js';
import { GaussianRenderer, projectGaussians } from './renderer.js';
import { generateDemoGaussians, samplePosition } from './demo-data.js';
import { UIController } from './ui.js';

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function main() {
  // ---- WebGPU availability check ----
  if (!navigator.gpu) {
    document.getElementById('no-webgpu').style.display = 'flex';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    document.getElementById('no-webgpu').style.display = 'flex';
    return;
  }

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  // ---- Canvas & context setup ----
  const canvas = document.getElementById('viewport');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // Handle DPR and resize
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
  }
  resize();
  window.addEventListener('resize', resize);

  // ---- UI ----
  const ui = new UIController();

  // ---- Camera ----
  const camera = new OrbitCamera({
    position: [0, 3, 10],
    target: [0, 0, 0],
    fov: Math.PI / 4,
    aspect: canvas.width / canvas.height,
    near: 0.1,
    far: 200,
  });
  camera.attach(canvas);

  // ---- Renderer ----
  const renderer = new GaussianRenderer(device, context, format);

  // ---- Data source ----
  let wasmMode = false;
  let wasmModule = null;

  // Attempt to load WASM module (optional)
  try {
    wasmModule = await import('../pkg/ruvector_vwm_wasm.js');
    await wasmModule.default(); // init WASM
    wasmMode = true;
    document.getElementById('mode-label').textContent = 'wasm';
    ui.setStatus('WASM module loaded');
  } catch (_) {
    // WASM not available - proceed in demo mode
    wasmMode = false;
    document.getElementById('mode-label').textContent = 'demo';
    ui.setStatus('Demo mode (synthetic data)');
  }

  // ---- Demo data ----
  const DEMO_COUNT = 2000;
  const DEMO_TIME_STEPS = 120;
  const demo = generateDemoGaussians(DEMO_COUNT, DEMO_TIME_STEPS);
  ui.setGaussianCount(demo.gaussians.length);
  ui.setCoherenceState('coherent');

  // Active mask for entity filtering
  let activeMask = new Array(demo.gaussians.length).fill(true);

  // Handle search filtering
  ui.onSearchChange((query) => {
    if (!query) {
      activeMask.fill(true);
    } else {
      for (let i = 0; i < demo.labels.length; i++) {
        activeMask[i] = demo.labels[i].includes(query);
      }
    }
    // Update visible count
    const visible = activeMask.filter(Boolean).length;
    ui.setGaussianCount(visible);
  });

  // ---- Animation state ----
  let animTime = 0;      // normalized [0, 1)
  const animSpeed = 0.15; // full cycles per second

  // ---- Coherence simulation ----
  // In demo mode we simulate coherence state changes
  let coherenceTimer = 0;
  const coherenceStates = ['coherent', 'coherent', 'coherent', 'degraded', 'coherent'];
  let coherenceIdx = 0;

  // ---- Render loop ----
  let lastTime = performance.now();

  function frame(now) {
    requestAnimationFrame(frame);

    const dt = (now - lastTime) / 1000;
    lastTime = now;

    // Update canvas size
    resize();
    camera.setAspect(canvas.width / canvas.height);

    // Advance animation time
    if (ui.playing) {
      animTime = (animTime + dt * animSpeed) % 1.0;
      ui.setTime(animTime);
    } else {
      animTime = ui.normalizedTime;
    }

    // Coherence state cycling (every ~5 seconds in demo mode)
    if (!wasmMode) {
      coherenceTimer += dt;
      if (coherenceTimer > 5.0) {
        coherenceTimer = 0;
        coherenceIdx = (coherenceIdx + 1) % coherenceStates.length;
        ui.setCoherenceState(coherenceStates[coherenceIdx]);
      }
    }

    // ---- Build per-frame Gaussian data ----
    const positions = [];
    const colors = [];
    const opacities = [];
    const scales = [];

    if (wasmMode && wasmModule) {
      // TODO: Use WasmDrawList and WasmCoherenceGate when WASM API is ready
      // For now, fall through to demo path
    }

    // Demo data path
    for (let i = 0; i < demo.gaussians.length; i++) {
      const g = demo.gaussians[i];
      positions.push(samplePosition(g, animTime));
      colors.push(g.color);
      opacities.push(g.opacity);
      scales.push(g.scale);
    }

    // ---- Project & render ----
    const viewProj = camera.getViewProjectionMatrix();

    const { data, count } = projectGaussians({
      positions,
      colors,
      opacities,
      scales,
      activeMask,
      viewProj,
      width: canvas.width,
      height: canvas.height,
      fovY: camera.fov,
    });

    renderer.render(data, count, canvas.width, canvas.height);

    // ---- UI updates ----
    ui.recordFrame(now);
    if (!ui.searchQuery) {
      ui.setGaussianCount(count);
    }
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error('VWM Viewer initialization failed:', err);
  const status = document.getElementById('status-text');
  if (status) status.textContent = `Error: ${err.message}`;
});
