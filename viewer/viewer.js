/**
 * MurimSim Replay Viewer — viewer.js
 *
 * Responsibilities:
 *   - Parse a run_*.jsonl file loaded via the file picker
 *   - Render the 2D grid (resources + agents) onto a canvas each tick
 *   - Playback controls: play/pause, step, speed slider, scrub bar
 *   - Click an agent dot → populate the inspector sidebar
 *   - Show a running event log for the current tick
 *   - Mark generation boundaries on the scrub bar
 *
 * Display rule (canonical invariant):
 *   All internal values are [0,1]. Displayed as 0–100 by multiplying ×100.
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RESOURCE_COLORS = {
  food:      "rgba(52, 168, 83, 0.75)",
  qi:        "rgba(66, 133, 244, 0.75)",
  materials: "rgba(234, 179, 8, 0.85)",
  poison:    "rgba(147, 51, 234, 0.85)",
  flame:     "rgba(239, 68, 68, 0.85)",
  mountain:  "rgba(120, 80, 40, 0.85)",
};

const ACTION_COLORS = {
  TRAIN:       "#f59e0b",
  GATHER:      "#22c55e",
  EAT:         "#86efac",
  REST:        "#60a5fa",
  DEPOSIT:     "#34d399",
  WITHDRAW:    "#6ee7b7",
  COLLABORATE: "#818cf8",
  WALK_AWAY:   "#94a3b8",
  ATTACK:      "#ef4444",
  DEFEND:      "#f97316",
  STEAL:       "#c084fc",
};

const DEAD_AGENT_COLOR    = "#64748b";   // slate-500 — clearly visible on dark bg
const DEAD_AGENT_BORDER   = "#94a3b8";   // slate-400 ring
const AGENT_COLOR         = "#a78bfa";
const AGENT_COMBAT_COLOR  = "#ef4444";
const AGENT_RADIUS_FRAC   = 0.28;
const DEFAULT_FPS         = 3;
const MIN_FPS             = 0.25;
const MAX_FPS             = 60;
const TRAIL_TICKS         = 8;
const SPARKLINE_TICKS     = 50;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  ticks:              [],
  currentIndex:       0,
  playing:            false,
  fps:                DEFAULT_FPS,
  intervalId:         null,
  gridSize:           30,
  selectedAgentId:    null,
  selectedTile:       null,
  generationMarkers:  [],
  // New fields
  aliveCounts:        [],   // [tickIndex] => number of alive agents
  agentHistories:     {},   // [agentId] => {health: [], hunger: []}
  deathLocations:     [],   // [tickIndex] => [{x, y, id}]
  showQiOverlay:      false,
  showTrail:          false,
};

// ---------------------------------------------------------------------------
// DOM references (populated after DOMContentLoaded)
// ---------------------------------------------------------------------------

let canvas, ctx;
let scrubBar, tickLabel, genLabel, fpsDisplay;
let playBtn, stepBtn, stepBackBtn, rewindBtn;
let fpsSlider;
let agentPanel, eventLog;
let populationCanvas, popCtx;
let tooltipEl;
let toggleQiEl, toggleTrailEl;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  canvas          = document.getElementById("sim-canvas");
  ctx             = canvas.getContext("2d");
  scrubBar        = document.getElementById("scrub");
  tickLabel       = document.getElementById("tick-label");
  genLabel        = document.getElementById("gen-label");
  fpsDisplay      = document.getElementById("fps-display");
  playBtn         = document.getElementById("btn-play");
  stepBtn         = document.getElementById("btn-step");
  stepBackBtn     = document.getElementById("btn-step-back");
  rewindBtn       = document.getElementById("btn-rewind");
  fpsSlider       = document.getElementById("fps-slider");
  agentPanel      = document.getElementById("agent-panel");
  eventLog        = document.getElementById("event-log");
  populationCanvas = document.getElementById("population-bar");
  toggleQiEl      = document.getElementById("toggle-qi");
  toggleTrailEl   = document.getElementById("toggle-trail");

  // Create tooltip element dynamically if not in HTML
  tooltipEl = document.getElementById("agent-tooltip");
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.id = "agent-tooltip";
    tooltipEl.style.cssText = [
      "position:fixed", "display:none", "padding:6px 10px",
      "background:rgba(15,15,30,0.92)", "color:#e2e8f0",
      "border:1px solid rgba(167,139,250,0.4)", "border-radius:6px",
      "font:12px/1.5 monospace", "pointer-events:none", "z-index:100",
      "white-space:pre",
    ].join(";");
    document.body.appendChild(tooltipEl);
  }

  if (populationCanvas) {
    popCtx = populationCanvas.getContext("2d");
  }

  document.getElementById("file-input").addEventListener("change", onFileLoaded);
  playBtn.addEventListener("click", togglePlay);
  stepBtn.addEventListener("click", stepForward);
  if (stepBackBtn) stepBackBtn.addEventListener("click", stepBack);
  rewindBtn.addEventListener("click", rewind);
  scrubBar.addEventListener("input", onScrub);
  fpsSlider.addEventListener("input", onFpsChange);
  canvas.addEventListener("click", onCanvasClick);
  canvas.addEventListener("mousemove", onCanvasHover);
  canvas.addEventListener("mouseleave", () => { tooltipEl.style.display = "none"; });

  document.querySelectorAll(".speed-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const fps = parseFloat(btn.dataset.fps);
      fpsSlider.value = fps;
      onFpsChange();
      document.querySelectorAll(".speed-preset").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
  // Mark default active preset
  const defaultPreset = document.querySelector(`.speed-preset[data-fps="${DEFAULT_FPS}"]`);
  if (defaultPreset) defaultPreset.classList.add("active");

  if (toggleQiEl) {
    toggleQiEl.addEventListener("change", () => {
      state.showQiOverlay = toggleQiEl.checked;
      render();
    });
  }
  if (toggleTrailEl) {
    toggleTrailEl.addEventListener("change", () => {
      state.showTrail = toggleTrailEl.checked;
      render();
    });
  }

  document.addEventListener("keydown", onKeyDown);

  fpsSlider.min   = MIN_FPS;
  fpsSlider.max   = MAX_FPS;
  fpsSlider.step  = 0.25;
  fpsSlider.value = DEFAULT_FPS;
  fpsDisplay.textContent = DEFAULT_FPS;

  drawPlaceholder();
});

// ---------------------------------------------------------------------------
// File loading & parsing
// ---------------------------------------------------------------------------

function onFileLoaded(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => parseJSONL(ev.target.result);
  reader.readAsText(file);
}

function parseJSONL(text) {
  const lines = text.trim().split("\n");
  state.ticks = lines
    .map((line, i) => {
      try { return JSON.parse(line); }
      catch { console.warn(`Skipping invalid JSON on line ${i}`); return null; }
    })
    .filter(Boolean);

  if (state.ticks.length === 0) {
    alert("No valid tick data found in file.");
    return;
  }

  // Infer grid size from resource positions
  let maxCoord = 29;
  for (const tick of state.ticks) {
    for (const tiles of Object.values(tick.resources || {})) {
      for (const [x, y] of tiles) {
        if (x > maxCoord) maxCoord = x;
        if (y > maxCoord) maxCoord = y;
      }
    }
  }
  state.gridSize = maxCoord + 1;

  // Find generation boundaries
  state.generationMarkers = [];
  let lastGen = -1;
  for (let i = 0; i < state.ticks.length; i++) {
    const g = state.ticks[i].generation ?? 0;
    if (g !== lastGen) { state.generationMarkers.push(i); lastGen = g; }
  }

  // Precompute alive counts
  state.aliveCounts = state.ticks.map(tick =>
    (tick.agents || []).filter(a => a.alive).length
  );

  // Precompute agent histories (last SPARKLINE_TICKS ticks per agent)
  state.agentHistories = {};
  for (let i = 0; i < state.ticks.length; i++) {
    for (const agent of (state.ticks[i].agents || [])) {
      if (!state.agentHistories[agent.id]) {
        state.agentHistories[agent.id] = { health: [], hunger: [] };
      }
      state.agentHistories[agent.id].health.push(agent.health ?? 0);
      state.agentHistories[agent.id].hunger.push(agent.hunger ?? 0);
    }
  }

  // Precompute death locations per tick
  state.deathLocations = state.ticks.map(tick =>
    (tick.events || [])
      .filter(ev => ev.type === "death")
      .map(ev => {
        const agent = (tick.agents || []).find(a => a.id === ev.agent);
        if (!agent) return null;
        return { x: agent.pos[0], y: agent.pos[1], id: ev.agent };
      })
      .filter(Boolean)
  );

  // Reset playback
  state.currentIndex  = 0;
  state.playing       = false;
  clearInterval(state.intervalId);
  state.intervalId    = null;
  playBtn.textContent = "▶ Play";

  scrubBar.max   = state.ticks.length - 1;
  scrubBar.value = 0;

  render();
  drawPopulationBar();
  document.getElementById("controls").style.display = "flex";
  document.getElementById("drop-hint").style.display = "none";
}

// ---------------------------------------------------------------------------
// Playback controls
// ---------------------------------------------------------------------------

function togglePlay() {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? "⏸ Pause" : "▶ Play";
  if (state.playing) {
    scheduleNext();
  } else {
    clearInterval(state.intervalId);
    state.intervalId = null;
  }
}

function scheduleNext() {
  clearInterval(state.intervalId);
  state.intervalId = setInterval(() => {
    if (state.currentIndex < state.ticks.length - 1) {
      state.currentIndex++;
      scrubBar.value = state.currentIndex;
      render();
      drawPopulationBar();
    } else {
      togglePlay();
    }
  }, 1000 / state.fps);
}

function stepForward() {
  if (state.currentIndex < state.ticks.length - 1) {
    state.currentIndex++;
    scrubBar.value = state.currentIndex;
    render();
    drawPopulationBar();
  }
}

function stepBack() {
  if (state.currentIndex > 0) {
    state.currentIndex--;
    scrubBar.value = state.currentIndex;
    render();
    drawPopulationBar();
  }
}

function rewind() {
  state.currentIndex = 0;
  scrubBar.value     = 0;
  render();
  drawPopulationBar();
}

function onScrub() {
  state.currentIndex = parseInt(scrubBar.value, 10);
  render();
  drawPopulationBar();
  if (state.playing) scheduleNext();
}

function onFpsChange() {
  state.fps = parseFloat(fpsSlider.value);
  fpsDisplay.textContent = state.fps < 1 ? state.fps.toFixed(2) : state.fps;
  // Highlight matching preset button if any
  document.querySelectorAll(".speed-preset").forEach(btn => {
    btn.classList.toggle("active", parseFloat(btn.dataset.fps) === state.fps);
  });
  if (state.playing) scheduleNext();
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

function onKeyDown(e) {
  if (e.target && e.target.tagName === "INPUT") return;
  switch (e.code) {
    case "Space":      e.preventDefault(); togglePlay();    break;
    case "ArrowRight": e.preventDefault(); stepForward();   break;
    case "ArrowLeft":  e.preventDefault(); stepBack();      break;
    case "KeyR":       e.preventDefault(); rewind();        break;
  }
}

// ---------------------------------------------------------------------------
// Agent hit-test helper
// ---------------------------------------------------------------------------

function _agentAtMouse(mx, my) {
  if (state.ticks.length === 0) return null;
  const tick    = state.ticks[state.currentIndex];
  const cellW   = canvas.width  / state.gridSize;
  const cellH   = canvas.height / state.gridSize;
  const baseR   = Math.min(cellW, cellH) * AGENT_RADIUS_FRAC;

  for (const agent of (tick.agents || [])) {
    const strength = agent.traits ? (agent.traits.strength ?? 0.5) : 0.5;
    const r        = baseR * (0.8 + 0.6 * strength);
    const [ax, ay] = agent.pos;
    const cx       = ax * cellW + cellW / 2;
    const cy       = ay * cellH + cellH / 2;
    if (Math.hypot(mx - cx, my - cy) <= r + 4) return agent;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  if (state.ticks.length === 0) return;
  const tick = state.ticks[state.currentIndex];

  const W     = canvas.width;
  const H     = canvas.height;
  const cellW = W / state.gridSize;
  const cellH = H / state.gridSize;
  const baseR = Math.min(cellW, cellH) * AGENT_RADIUS_FRAC;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);

  // Grid lines (subtle)
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth   = 0.5;
  for (let i = 0; i <= state.gridSize; i++) {
    ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(W, i * cellH); ctx.stroke();
  }

  // Resources
  for (const [rid, tiles] of Object.entries(tick.resources || {})) {
    if (rid === "qi" && state.showQiOverlay) continue; // drawn separately below
    const color = RESOURCE_COLORS[rid] || "rgba(200,200,200,0.5)";
    for (const [x, y, intensity] of tiles) {
      const alpha = Math.min(1, Math.max(0.2, intensity ?? 1));
      ctx.fillStyle = color.replace(/[\d.]+\)$/, `${alpha})`);
      ctx.fillRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2);
    }
  }

  // Qi field overlay
  if (state.showQiOverlay) {
    const qiTiles = (tick.resources || {}).qi || [];
    for (const [x, y, intensity] of qiTiles) {
      const cx   = x * cellW + cellW / 2;
      const cy   = y * cellH + cellH / 2;
      const r    = 2.5 * Math.min(cellW, cellH);
      const alpha = (intensity ?? 1) * 0.18;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grad.addColorStop(0, `rgba(66,133,244,${alpha})`);
      grad.addColorStop(1, "rgba(66,133,244,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
    }
  }

  // Tile selection highlight
  if (state.selectedTile) {
    const { x, y } = state.selectedTile;
    ctx.strokeStyle = "#ffe066";
    ctx.lineWidth   = 2;
    ctx.strokeRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2);
  }

  // Stash markers
  if (Array.isArray(tick.stashes)) {
    for (const stash of tick.stashes) {
      const sx     = stash.x * cellW + cellW / 2;
      const sy     = stash.y * cellH + cellH / 2;
      const half   = Math.min(cellW, cellH) * 0.28;
      const color  = stash.is_own ? "#2dd4bf" : "#fb923c";
      ctx.beginPath();
      ctx.moveTo(sx,        sy - half);
      ctx.lineTo(sx + half, sy);
      ctx.lineTo(sx,        sy + half);
      ctx.lineTo(sx - half, sy);
      ctx.closePath();
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      ctx.stroke();
      ctx.fillStyle = stash.is_own ? "rgba(45,212,191,0.15)" : "rgba(251,146,60,0.15)";
      ctx.fill();
    }
  }

  // Movement trail
  if (state.showTrail) {
    const trailDotR = Math.min(cellW, cellH) * 0.08;
    for (let age = TRAIL_TICKS; age >= 1; age--) {
      const ti = state.currentIndex - age;
      if (ti < 0) continue;
      const alpha = 0.55 * (1 - age / TRAIL_TICKS);
      for (const agent of (state.ticks[ti].agents || [])) {
        if (!agent.alive) continue;
        const [ax, ay] = agent.pos;
        const tx = ax * cellW + cellW / 2;
        const ty = ay * cellH + cellH / 2;
        ctx.beginPath();
        ctx.arc(tx, ty, trailDotR, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(167,139,250,${alpha})`;
        ctx.fill();
      }
    }
  }

  // Death skull fade (last 10 ticks)
  for (let age = 10; age >= 1; age--) {
    const ti = state.currentIndex - age + 1;
    if (ti < 0 || ti >= state.deathLocations.length) continue;
    const alpha = 1 - age / 10;
    for (const { x, y } of state.deathLocations[ti]) {
      const dx   = x * cellW + cellW / 2;
      const dy   = y * cellH + cellH / 2;
      const half = Math.min(cellW, cellH) * 0.22;
      ctx.strokeStyle = `rgba(255,80,80,${alpha})`;
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.moveTo(dx - half, dy - half);
      ctx.lineTo(dx + half, dy + half);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(dx + half, dy - half);
      ctx.lineTo(dx - half, dy + half);
      ctx.stroke();
    }
  }

  // Agents
  const agents = tick.agents || [];
  for (const agent of agents) {
    const strength   = agent.traits ? (agent.traits.strength ?? 0.5) : 0.5;
    const agentRadius = baseR * (0.8 + 0.6 * strength);
    const [ax, ay]   = agent.pos;
    const cx         = ax * cellW + cellW / 2;
    const cy         = ay * cellH + cellH / 2;
    const health     = agent.health ?? 1;
    const hunger     = agent.hunger ?? 0;
    const actionKey  = (agent.action || "").toUpperCase();

    // --- Dead agent: gray corpse circle, skip all living indicators ---
    if (!agent.alive) {
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.arc(cx, cy, agentRadius, 0, 2 * Math.PI);
      ctx.fillStyle = DEAD_AGENT_COLOR;
      ctx.fill();
      // Thin border ring
      ctx.beginPath();
      ctx.arc(cx, cy, agentRadius + 1.5, 0, 2 * Math.PI);
      ctx.strokeStyle = DEAD_AGENT_BORDER;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
      continue;
    }

    // Health arc ring (faint background arc)
    const arcStart = -Math.PI / 2;
    const arcEnd   = arcStart + 2 * Math.PI * health;
    const ringR    = agentRadius + 2.5;

    ctx.beginPath();
    ctx.arc(cx, cy, ringR, arcEnd, arcStart + 2 * Math.PI);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth   = 2;
    ctx.stroke();

    // Colored health arc
    const hue = health * 120; // 0=red, 120=green
    ctx.beginPath();
    ctx.arc(cx, cy, ringR, arcStart, arcEnd);
    ctx.strokeStyle = `hsl(${hue},90%,55%)`;
    ctx.lineWidth   = 2;
    ctx.stroke();

    // Agent fill — action-based tint
    ctx.beginPath();
    ctx.arc(cx, cy, agentRadius, 0, 2 * Math.PI);
    ctx.fillStyle = ACTION_COLORS[actionKey] || AGENT_COLOR;
    ctx.fill();

    // Combat outer glow
    if (actionKey === "ATTACK" || actionKey === "DEFEND") {
      ctx.beginPath();
      ctx.arc(cx, cy, agentRadius + 5, 0, 2 * Math.PI);
      ctx.strokeStyle = AGENT_COMBAT_COLOR;
      ctx.lineWidth   = 2;
      ctx.stroke();
    }

    // Selection highlight
    if (agent.id === state.selectedAgentId) {
      ctx.beginPath();
      ctx.arc(cx, cy, agentRadius + 5, 0, 2 * Math.PI);
      ctx.strokeStyle = "#ffe066";
      ctx.lineWidth   = 2;
      ctx.stroke();
    }

    // Hunger indicator dot
    const dotY    = cy - agentRadius - 5;
    const hungerC = hunger > 0.7 ? "#ef4444"
                  : hunger > 0.4 ? "#f97316"
                  :                "#94a3b8";
    ctx.beginPath();
    ctx.arc(cx, dotY, 3, 0, 2 * Math.PI);
    ctx.fillStyle = hungerC;
    ctx.fill();
  }

  updateHUD(tick);
  updateInspectorPanel(tick);
  updateEventLog(tick);
}

// ---------------------------------------------------------------------------
// Population bar chart
// ---------------------------------------------------------------------------

function drawPopulationBar() {
  if (!popCtx || state.aliveCounts.length === 0) return;
  const W   = populationCanvas.width;
  const H   = populationCanvas.height;
  const n   = state.aliveCounts.length;
  const max = Math.max(1, ...state.aliveCounts);

  popCtx.clearRect(0, 0, W, H);
  popCtx.fillStyle = "#0f0f1e";
  popCtx.fillRect(0, 0, W, H);

  const barW = W / n;
  for (let i = 0; i < n; i++) {
    const bH = (state.aliveCounts[i] / max) * (H - 2);
    popCtx.fillStyle = (i === state.currentIndex) ? "#ffe066" : "rgba(167,139,250,0.55)";
    popCtx.fillRect(i * barW, H - bH, Math.max(1, barW - 0.5), bH);
  }
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

function updateHUD(tick) {
  tickLabel.textContent = `Tick: ${tick.tick}`;
  genLabel.textContent  = `Generation: ${tick.generation ?? 0}`;

  const total   = state.ticks.length;
  const markers = state.generationMarkers
    .map(i => `${Math.round((i / total) * 100)}%`)
    .join(", ");
  document.getElementById("gen-markers").textContent =
    markers ? `Gen boundaries at: ${markers}` : "";
}

// ---------------------------------------------------------------------------
// Inspector panel — dispatches to agent or tile view
// ---------------------------------------------------------------------------

function onCanvasClick(e) {
  if (state.ticks.length === 0) return;
  const rect = canvas.getBoundingClientRect();
  const mx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
  const my   = (e.clientY - rect.top)  * (canvas.height / rect.height);
  const hitAgent = _agentAtMouse(mx, my);

  if (hitAgent) {
    state.selectedAgentId = hitAgent.id;
    state.selectedTile    = null;
  } else {
    state.selectedAgentId = null;
    const cellW = canvas.width  / state.gridSize;
    const cellH = canvas.height / state.gridSize;
    state.selectedTile = { x: Math.floor(mx / cellW), y: Math.floor(my / cellH) };
  }
  render();
}

function onCanvasHover(e) {
  if (state.ticks.length === 0) return;
  const rect = canvas.getBoundingClientRect();
  const mx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
  const my   = (e.clientY - rect.top)  * (canvas.height / rect.height);
  const agent = _agentAtMouse(mx, my);

  if (agent && agent.alive) {
    const hp  = Math.round((agent.health  ?? 0) * 100);
    const hun = Math.round((agent.hunger  ?? 0) * 100);
    tooltipEl.textContent = `${agent.id}\nHP: ${hp}%  Hunger: ${hun}%\n${agent.action || "—"}`;
    tooltipEl.style.left    = `${e.clientX + 14}px`;
    tooltipEl.style.top     = `${e.clientY - 10}px`;
    tooltipEl.style.display = "block";
  } else {
    tooltipEl.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// Inspector panel — agent + tile views
// ---------------------------------------------------------------------------

const RESOURCE_LABELS = {
  food:      "🌾 Food",
  qi:        "💠 Qi",
  materials: "🪨 Materials",
  flame:     "🔥 Flame",
  poison:    "☠️ Poison",
  mountain:  "⛰️ Mountain",
};

function updateInspectorPanel(tick) {
  if (state.selectedAgentId) {
    updateAgentPanel(tick);
  } else if (state.selectedTile) {
    updateTilePanel(tick);
  } else {
    agentPanel.innerHTML = "<p class='muted'>Click a tile or agent to inspect</p>";
  }
}

function updateAgentPanel(tick) {
  const agent = (tick.agents || []).find(a => a.id === state.selectedAgentId);
  if (!agent) {
    agentPanel.innerHTML = "<p class='muted'>Agent not present this tick</p>";
    return;
  }

  const res  = agent.resistances || {};
  const fmt  = v => (v == null || isNaN(v)) ? "—" : v >= 1.0 ? "Immune" : `${Math.round(v * 100)}`;
  const barW = v => `${Math.min(100, Math.round((v ?? 0) * 100))}`;

  const resRows = Object.entries(res)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => {
      const color = k === "poison"   ? "#9b59b6"
                  : k === "flame"    ? "#ef4444"
                  : k === "qi_drain" ? "#3b82f6"
                  : "#94a3b8";
      const label = k === "qi_drain" ? "Qi Drain Res."
                  : `${k[0].toUpperCase() + k.slice(1)} Res.`;
      return `
      <tr><th>${label}</th><td>
        <div class="bar-wrap"><div class="bar" style="width:${barW(v)}%;background:${color}"></div></div>
        ${fmt(v)}
      </td></tr>`;
    }).join("");

  // Action target row — humanize the detail
  const detail = agent.action_detail || "";
  let targetRow = "";
  if (detail && detail !== "no_target") {
    const label = detail.startsWith("stash") ? "Stash" : "Target";
    targetRow = `<tr><th>${label}</th><td style="color:#fbbf24">${detail}</td></tr>`;
  } else if (detail === "no_target") {
    targetRow = `<tr><th>Target</th><td style="color:#ef4444">none (wasted)</td></tr>`;
  }

  // Collaborator map — agents currently in the same group
  const collabs = agent.collaborators || [];
  let collabSection = "";
  if (collabs.length > 0) {
    const tags = collabs.map(id => `<span style="
        display:inline-block;padding:1px 6px;margin:2px;border-radius:10px;
        background:#1e3a5f;border:1px solid #3b82f6;font-size:10px;color:#93c5fd
      ">${id}</span>`).join("");
    collabSection = `
      <div style="margin-top:8px">
        <div style="font-size:10px;color:#64748b;margin-bottom:3px">▸ GROUP MEMBERS</div>
        <div>${tags}</div>
      </div>`;
  } else {
    collabSection = `<div style="margin-top:6px;font-size:10px;color:#475569">No group / solo</div>`;
  }

  agentPanel.innerHTML = `
    <h3>${agent.id}</h3>
    <table class="stat-table">
      <tr><th>Position</th><td>(${agent.pos[0]}, ${agent.pos[1]})</td></tr>
      <tr><th>Alive</th><td>${agent.alive ? "✅" : "💀"}</td></tr>
      ${!agent.alive && agent.death_cause ? `<tr><th>Cause</th><td>${agent.death_cause}</td></tr>` : ""}
      <tr><th>Health</th><td>
        <div class="bar-wrap"><div class="bar health-bar" style="width:${barW(agent.health)}%"></div></div>
        ${fmt(agent.health)}
      </td></tr>
      <tr><th>Hunger</th><td>
        <div class="bar-wrap"><div class="bar hunger-bar" style="width:${barW(agent.hunger ?? 0)}%"></div></div>
        ${Math.round((agent.hunger ?? 0) * 100)}
      </td></tr>
      ${resRows}
      <tr><th>Action</th><td>${agent.action}</td></tr>
      ${targetRow}
    </table>
    ${collabSection}
    ${_buildSparkline(agent.id, state.currentIndex)}
  `;
}

/**
 * Build an inline SVG sparkline showing last SPARKLINE_TICKS of health + hunger.
 */
function _buildSparkline(agentId, upToTick) {
  const hist = state.agentHistories[agentId];
  if (!hist) return "";

  const end      = Math.min(upToTick + 1, hist.health.length);
  const start    = Math.max(0, end - SPARKLINE_TICKS);
  const healths  = hist.health.slice(start, end);
  const hungers  = hist.hunger.slice(start, end);
  if (healths.length < 2) return "";

  const W = 190, H = 34, PAD = 2;
  const IW = W - PAD * 2, IH = H - PAD * 2;

  function toPath(values) {
    if (values.length === 0) return "";
    const step = IW / (values.length - 1);
    return values.map((v, i) => {
      const px = PAD + i * step;
      const py = PAD + IH * (1 - v);
      return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(" ");
  }

  const healthPath = toPath(healths);
  const hungerPath = toPath(hungers);

  return `
  <div style="margin-top:8px">
    <svg width="${W}" height="${H}" style="background:#0f172a;border-radius:4px;display:block">
      <path d="${healthPath}" stroke="#22c55e" stroke-width="1.5" fill="none"/>
      <path d="${hungerPath}" stroke="#f59e0b" stroke-width="1.5" fill="none"/>
    </svg>
    <div style="font:10px monospace;color:#64748b;margin-top:2px">
      <span style="color:#22c55e">■</span> health &nbsp;
      <span style="color:#f59e0b">■</span> hunger &nbsp; (last ${healths.length} ticks)
    </div>
  </div>`;
}

function updateTilePanel(tick) {
  const { x, y } = state.selectedTile;

  const present = [];
  for (const [rid, tiles] of Object.entries(tick.resources || {})) {
    for (const [tx, ty, intensity] of tiles) {
      if (tx === x && ty === y) {
        present.push({ type: rid, intensity: intensity ?? 1.0 });
        break;
      }
    }
  }

  const here = (tick.agents || []).filter(a => a.pos[0] === x && a.pos[1] === y);

  const resourceRows = present.length > 0
    ? present.map(({ type, intensity }) => {
        const solidColor = (RESOURCE_COLORS[type] || "rgba(180,180,180,0.85)")
          .replace(/[\d.]+\)$/, "1)");
        return `
        <tr><th>${RESOURCE_LABELS[type] || type}</th><td>
          <div class="bar-wrap"><div class="bar" style="width:${Math.round(intensity * 100)}%;background:${solidColor}"></div></div>
          ${Math.round(intensity * 100)}
        </td></tr>`;
      }).join("")
    : `<tr><td colspan="2" class="muted" style="padding:4px 0">Empty tile</td></tr>`;

  const agentRows = here.map(a =>
    `<tr><th>Agent</th><td style="color:#fff">${a.id} ${a.alive ? "✅" : "💀"}</td></tr>`
  ).join("");

  agentPanel.innerHTML = `
    <h3>Tile (${x}, ${y})</h3>
    <table class="stat-table">
      ${resourceRows}
      ${agentRows}
    </table>
  `;
}

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------

function updateEventLog(tick) {
  const events = tick.events || [];
  if (events.length === 0) {
    eventLog.innerHTML = "<p class='muted'>No events this tick</p>";
    return;
  }
  eventLog.innerHTML = events.map(ev => {
    if (ev.type === "combat") {
      return `<div class="event combat">⚔️ <b>${ev.attacker}</b> → <b>${ev.defender}</b> (dmg: ${(ev.damage ?? 0).toFixed(2)})</div>`;
    }
    if (ev.type === "death") {
      return `<div class="event death">💀 <b>${ev.agent}</b> — ${ev.cause}</div>`;
    }
    return `<div class="event">${JSON.stringify(ev)}</div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// Placeholder (shown before a file is loaded)
// ---------------------------------------------------------------------------

function drawPlaceholder() {
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "rgba(255,255,255,0.15)";
  ctx.font      = "bold 16px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Load a run_*.jsonl file to begin", W / 2, H / 2);
}
