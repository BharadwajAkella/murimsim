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
  food:      "rgba(52, 168, 83, 0.75)",   // green
  qi:        "rgba(66, 133, 244, 0.75)",  // blue
  materials: "rgba(234, 179, 8, 0.85)",   // yellow
  poison:    "rgba(147, 51, 234, 0.85)",  // purple
  flame:     "rgba(239, 68, 68, 0.85)",   // red
  mountain:  "rgba(120, 80, 40, 0.85)",   // brown
};

// Per-sect agent colors — Phase 6a
const SECT_COLORS = {
  iron_fang:   "#e74c3c",   // red
  jade_lotus:  "#2ecc71",   // green
  shadow_root: "#9b59b6",   // purple
};
const DEAD_AGENT_COLOR = "#555";
const AGENT_COLOR = "#a78bfa";         // fallback color when sect is "none"
const AGENT_COMBAT_COLOR = "#ef4444";  // bright red during attack or defend
const AGENT_RADIUS_FRAC = 0.28;        // fraction of cell size
const DEFAULT_FPS = 3;
const MIN_FPS = 1;
const MAX_FPS = 60;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  ticks: [],           // parsed tick records (array of objects)
  currentIndex: 0,     // index into ticks[]
  playing: false,
  fps: DEFAULT_FPS,
  intervalId: null,
  gridSize: 30,        // inferred from data
  selectedAgentId: null,
  selectedTile: null,  // {x, y} of last clicked tile (when no agent was hit)
  generationMarkers: [], // tick indices where generation changes
};

// ---------------------------------------------------------------------------
// DOM references (populated after DOMContentLoaded)
// ---------------------------------------------------------------------------

let canvas, ctx;
let scrubBar, tickLabel, genLabel, fpsDisplay;
let playBtn, stepBtn, rewindBtn;
let fpsSlider;
let agentPanel, eventLog;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  canvas    = document.getElementById("sim-canvas");
  ctx       = canvas.getContext("2d");
  scrubBar  = document.getElementById("scrub");
  tickLabel = document.getElementById("tick-label");
  genLabel  = document.getElementById("gen-label");
  fpsDisplay = document.getElementById("fps-display");
  playBtn   = document.getElementById("btn-play");
  stepBtn   = document.getElementById("btn-step");
  rewindBtn = document.getElementById("btn-rewind");
  fpsSlider = document.getElementById("fps-slider");
  agentPanel = document.getElementById("agent-panel");
  eventLog  = document.getElementById("event-log");

  document.getElementById("file-input").addEventListener("change", onFileLoaded);
  playBtn.addEventListener("click", togglePlay);
  stepBtn.addEventListener("click", stepForward);
  rewindBtn.addEventListener("click", rewind);
  scrubBar.addEventListener("input", onScrub);
  fpsSlider.addEventListener("input", onFpsChange);
  canvas.addEventListener("click", onCanvasClick);

  fpsSlider.min = MIN_FPS;
  fpsSlider.max = MAX_FPS;
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

  // Reset playback
  state.currentIndex = 0;
  state.playing = false;
  clearInterval(state.intervalId);
  state.intervalId = null;
  playBtn.textContent = "▶ Play";

  scrubBar.max = state.ticks.length - 1;
  scrubBar.value = 0;

  render();
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
    } else {
      togglePlay(); // stop at end
    }
  }, 1000 / state.fps);
}

function stepForward() {
  if (state.currentIndex < state.ticks.length - 1) {
    state.currentIndex++;
    scrubBar.value = state.currentIndex;
    render();
  }
}

function rewind() {
  state.currentIndex = 0;
  scrubBar.value = 0;
  render();
}

function onScrub() {
  state.currentIndex = parseInt(scrubBar.value, 10);
  render();
  if (state.playing) scheduleNext(); // reset interval to avoid drift
}

function onFpsChange() {
  state.fps = parseInt(fpsSlider.value, 10);
  fpsDisplay.textContent = state.fps;
  if (state.playing) scheduleNext();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  if (state.ticks.length === 0) return;
  const tick = state.ticks[state.currentIndex];

  const W = canvas.width;
  const H = canvas.height;
  const cellW = W / state.gridSize;
  const cellH = H / state.gridSize;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);

  // Sect home-region overlays (subtle tinted bands — Phase 6a)
  const SECT_HOME_REGIONS = [
    { sect: "iron_fang",   y_lo: 0,  y_hi: 9 },
    { sect: "jade_lotus",  y_lo: 10, y_hi: 19 },
    { sect: "shadow_root", y_lo: 20, y_hi: 29 },
  ];
  for (const region of SECT_HOME_REGIONS) {
    const color = SECT_COLORS[region.sect];
    ctx.fillStyle = color.replace("#", "rgba(") + "19)"; // ~10% opacity
    // Convert hex to rgba properly
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    ctx.fillStyle = `rgba(${r},${g},${b},0.07)`;
    ctx.fillRect(0, region.y_lo * cellH, W, (region.y_hi - region.y_lo + 1) * cellH);
    // Thin border on the bottom edge of each band (except last)
    if (region.y_hi < state.gridSize - 1) {
      ctx.strokeStyle = `rgba(${r},${g},${b},0.25)`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, (region.y_hi + 1) * cellH);
      ctx.lineTo(W, (region.y_hi + 1) * cellH);
      ctx.stroke();
    }
  }

  // Grid lines (subtle)
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= state.gridSize; i++) {
    ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(W, i * cellH); ctx.stroke();
  }

  // Resources
  for (const [rid, tiles] of Object.entries(tick.resources || {})) {
    const color = RESOURCE_COLORS[rid] || "rgba(200,200,200,0.5)";
    for (const [x, y, intensity] of tiles) {
      const alpha = Math.min(1, Math.max(0.2, intensity ?? 1));
      ctx.fillStyle = color.replace(/[\d.]+\)$/, `${alpha})`);
      ctx.fillRect(
        x * cellW + 1, y * cellH + 1,
        cellW - 2, cellH - 2
      );
    }
  }

  // Tile selection highlight
  if (state.selectedTile) {
    const { x, y } = state.selectedTile;
    ctx.strokeStyle = "#ffe066";
    ctx.lineWidth = 2;
    ctx.strokeRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2);
  }

  // Agents
  const agents = tick.agents || [];
  const agentRadius = Math.min(cellW, cellH) * AGENT_RADIUS_FRAC;
  for (const agent of agents) {
    const [ax, ay] = agent.pos;
    const cx = ax * cellW + cellW / 2;
    const cy = ay * cellH + cellH / 2;

    // Outer ring: health indicator
    ctx.beginPath();
    ctx.arc(cx, cy, agentRadius + 2, 0, 2 * Math.PI);
    ctx.strokeStyle = agent.alive ? "rgba(255,255,255,0.35)" : "#333";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Agent dot
    ctx.beginPath();
    ctx.arc(cx, cy, agentRadius, 0, 2 * Math.PI);
    const inCombat = agent.alive && (agent.action === "attack" || agent.action === "defend");
    if (!agent.alive) {
      ctx.fillStyle = DEAD_AGENT_COLOR;
    } else if (inCombat) {
      // Flash between two reds: bright on even render frames, darker on odd
      ctx.fillStyle = (state.currentIndex % 2 === 0) ? AGENT_COMBAT_COLOR : "#b91c1c";
    } else {
      ctx.fillStyle = SECT_COLORS[agent.sect] || AGENT_COLOR;
    }
    ctx.fill();

    // Combat ring: extra glow when fighting
    if (inCombat) {
      ctx.beginPath();
      ctx.arc(cx, cy, agentRadius + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = AGENT_COMBAT_COLOR;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Selection highlight
    if (agent.id === state.selectedAgentId) {
      ctx.beginPath();
      ctx.arc(cx, cy, agentRadius + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = "#ffe066";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  updateHUD(tick);
  updateInspectorPanel(tick);
  updateEventLog(tick);
}

// ---------------------------------------------------------------------------
// HUD (tick / generation labels + scrub markers)
// ---------------------------------------------------------------------------

function updateHUD(tick) {
  tickLabel.textContent = `Tick: ${tick.tick}`;
  genLabel.textContent  = `Generation: ${tick.generation ?? 0}`;

  // Draw generation markers on scrub bar via a custom background gradient
  // (CSS only approach — overlay divs would require DOM mutation each frame,
  //  so we just annotate the label instead)
  const idx = state.currentIndex;
  const total = state.ticks.length;
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
  const tick = state.ticks[state.currentIndex];
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const my = (e.clientY - rect.top)  * (canvas.height / rect.height);
  const cellW = canvas.width  / state.gridSize;
  const cellH = canvas.height / state.gridSize;
  const agentRadius = Math.min(cellW, cellH) * AGENT_RADIUS_FRAC;

  // Agent hit-test takes priority
  let hitAgent = null;
  for (const agent of (tick.agents || [])) {
    const [ax, ay] = agent.pos;
    const cx = ax * cellW + cellW / 2;
    const cy = ay * cellH + cellH / 2;
    if (Math.hypot(mx - cx, my - cy) <= agentRadius + 4) { hitAgent = agent; break; }
  }

  if (hitAgent) {
    state.selectedAgentId = hitAgent.id;
    state.selectedTile = null;
  } else {
    state.selectedAgentId = null;
    state.selectedTile = {
      x: Math.floor(mx / cellW),
      y: Math.floor(my / cellH),
    };
  }
  render();
}

// ---------------------------------------------------------------------------
// Inspector panel — dispatches to agent or tile view
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

  const res = agent.resistances || {};

  // Render a [0,1] value as 0–100, or "Immune" at 1.0
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

  agentPanel.innerHTML = `
    <h3>${agent.id}</h3>
    <table class="stat-table">
      <tr><th>Position</th><td>(${agent.pos[0]}, ${agent.pos[1]})</td></tr>
      <tr><th>Alive</th><td>${agent.alive ? "✅" : "💀"}</td></tr>
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
      <tr><th>Detail</th><td>${agent.action_detail || "—"}</td></tr>
    </table>
  `;
}

function updateTilePanel(tick) {
  const { x, y } = state.selectedTile;

  // Collect all resources present at this cell
  const present = [];
  for (const [rid, tiles] of Object.entries(tick.resources || {})) {
    for (const [tx, ty, intensity] of tiles) {
      if (tx === x && ty === y) {
        present.push({ type: rid, intensity: intensity ?? 1.0 });
        break;
      }
    }
  }

  // Agents standing on this cell
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
  ctx.font = "bold 16px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Load a run_*.jsonl file to begin", W / 2, H / 2);
}
