"""
core/styles.py — CSS stylesheet and JavaScript for the TAE web UI.

Pure string constants — no FastHTML, no business logic.
Imported by main.py and passed directly into the fast_app() headers.
"""

from fasthtml.common import Style, Script

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = Style("""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Rajdhani:wght@500;700&display=swap');
 
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
:root {
  --bg0:        #080910;
  --bg1:        #0f1020;
  --bg2:        #14162a;
  --bg3:        #1b1d30;
  --bg4:        #22253a;
  --border:     #2b2e48;
  --border-hi:  #3d4270;
  --accent:     #4ade80;
  --accent-dim: rgba(74,222,128,.12);
  --blue:       #7aa2f7;
  --blue-dim:   rgba(122,162,247,.12);
  --cyan:       #22d3ee;
  --text:       #c0caf5;
  --muted:      #525880;
  --danger:     #f7768e;
  --amber:      #e0af68;
  --font-mono:  'JetBrains Mono', monospace;
  --font-head:  'Rajdhani', sans-serif;
  --panel-r:    12px;
}
 
html, body {
  height: 100%;
  background: var(--bg0);
  color: var(--text);
  font-family: var(--font-mono);
  overflow: hidden;
}
 
/* ── Navbar ──────────────────────────────────────────────────────────────── */
.tae-nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 900;
  height: 52px;
  background: rgba(15, 16, 32, 0.97);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 18px;
  gap: 14px;
  backdrop-filter: blur(12px);
}
 
.brand {
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .12em;
  text-transform: uppercase;
}
.brand-sub {
  font-size: 9px;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: .15em;
  text-transform: uppercase;
  align-self: flex-end;
  margin-bottom: 3px;
}
.sep { width: 1px; height: 28px; background: var(--border); margin: 0 4px; }
 
.upload-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--bg4);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 13px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text);
  transition: border-color .18s, background .18s;
}
.upload-btn:hover { border-color: var(--blue); background: var(--blue-dim); }
.upload-btn input[type="file"] { display: none; }
 
.status-badge {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
}
.sdot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--danger);
  flex-shrink: 0;
}
.sdot.active {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
}
 
/* nav toolbar pills on the right */
.nav-pill {
  display: flex;
  align-items: center;
  gap: 5px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 6px;
}
.nav-pill-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 14px;
  cursor: pointer;
  font-size: 10px;
  font-family: var(--font-mono);
  font-weight: 600;
  letter-spacing: .05em;
  color: var(--muted);
  border: none;
  background: transparent;
  transition: background .15s, color .15s;
  white-space: nowrap;
}
.nav-pill-btn:hover { background: var(--bg4); color: var(--text); }
.nav-pill-btn.active { background: var(--bg0); color: var(--blue); }
 
/* ── Workspace ────────────────────────────────────────────────────────────── */
.tae-workspace {
  position: fixed;
  top: 52px; left: 0; right: 0; bottom: 0;
  overflow: hidden;
}
 
#bg-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
 
/* ── Panel system ─────────────────────────────────────────────────────────── */
.tae-panel {
  position: absolute;
  min-width: 260px;
  min-height: 44px;
  background: rgba(14, 15, 26, 0.94);
  border: 1px solid var(--border);
  border-radius: var(--panel-r);
  box-shadow:
    0 0 0 0.5px rgba(255,255,255,.04) inset,
    0 12px 48px rgba(0,0,0,.75);
  backdrop-filter: blur(18px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  z-index: 100;
  transition: box-shadow .18s, border-color .18s;
}
 
.tae-panel.is-focused {
  z-index: 300;
  border-color: var(--border-hi);
  box-shadow:
    0 0 0 0.5px rgba(255,255,255,.06) inset,
    0 16px 60px rgba(0,0,0,.9);
}
 
.tae-panel.is-collapsed .panel-body  { display: none !important; }
.tae-panel.is-collapsed .panel-foot  { display: none !important; }
.tae-panel.is-collapsed .panel-resize { display: none !important; }
 
/* ── Panel header ─────────────────────────────────────────────────────────── */
.panel-header {
  height: 44px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 0 10px 0 12px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  cursor: grab;
  user-select: none;
}
.panel-header:active { cursor: grabbing; }
 
.panel-icon {
  width: 26px; height: 26px;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px;
  flex-shrink: 0;
  opacity: .9;
}
.pi-map   { background: rgba(122,162,247,.18); color: var(--blue); }
.pi-chat  { background: rgba(74,222,128,.16);  color: var(--accent); }
.pi-video { background: rgba(34,211,238,.16);  color: var(--cyan); }
.pi-frame { background: rgba(224,175,104,.16); color: var(--amber); }
 
.panel-title {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text);
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1;
}
 
.panel-controls {
  display: flex;
  align-items: center;
  gap: 2px;
  margin-left: auto;
  flex-shrink: 0;
}
 
.panel-btn {
  width: 26px; height: 26px;
  border-radius: 6px;
  background: transparent;
  border: none;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted);
  font-size: 10px;
  transition: background .12s, color .12s;
}
.panel-btn:hover  { background: var(--bg4); color: var(--text); }
 
/* ── Panel body ───────────────────────────────────────────────────────────── */
.panel-body {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  min-height: 0;
}
 
/* ── Resize handle ────────────────────────────────────────────────────────── */
.panel-resize {
  position: absolute;
  right: 0; bottom: 0;
  width: 18px; height: 18px;
  cursor: nwse-resize;
  z-index: 10;
}
.panel-resize::before {
  content: '';
  position: absolute;
  right: 3px; bottom: 3px;
  width: 9px; height: 9px;
  border-right: 2px solid var(--border-hi);
  border-bottom: 2px solid var(--border-hi);
  border-radius: 0 0 3px 0;
  opacity: .6;
}
 
/* ── Map panel ────────────────────────────────────────────────────────────── */
#tae-map-panel { top: 20px; left: 20px; width: 640px; height: 480px; }
 
.map-panel-body iframe {
  width: 100%; height: 100%;
  border: none; display: block;
}
 
/* ── Chat panel ───────────────────────────────────────────────────────────── */
#tae-chat-panel { bottom: 20px; left: 20px; width: 370px; height: 420px; top: auto; }
 
.chat-msgs {
  flex: 1;
  overflow-y: auto;
  padding: 12px 14px;
  min-height: 0;
}
.chat-msgs::-webkit-scrollbar { width: 3px; }
.chat-msgs::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
 
.panel-foot { flex-shrink: 0; }
.chat-divider { height: 1px; background: var(--border); }
.chat-input-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
}
.chat-input-row input {
  flex: 1;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 7px 11px;
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text);
  outline: none;
  transition: border-color .15s;
}
.chat-input-row input::placeholder { color: var(--muted); }
.chat-input-row input:focus { border-color: var(--blue); }
 
.send-btn {
  background: var(--blue);
  color: var(--bg0);
  border: none;
  border-radius: 7px;
  padding: 7px 14px;
  cursor: pointer;
  font-weight: 700;
  font-size: 13px;
  font-family: var(--font-mono);
  transition: background .15s;
}
.send-btn:hover { background: #89b4fa; }
 
/* ── Video panel ──────────────────────────────────────────────────────────── */
#tae-video-panel { top: 20px; right: 20px; width: 480px; height: 300px; }
 
.video-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 14px;
  color: var(--muted);
}
.video-placeholder i { font-size: 40px; opacity: .4; }
.video-placeholder p { font-size: 11px; letter-spacing: .04em; opacity: .6; }
 
/* ── Frame / detection panel ──────────────────────────────────────────────── */
#tae-frame-panel { bottom: 20px; right: 20px; width: 400px; height: 480px; top: auto; }
 
.frame-panel-body {
  overflow-y: auto;
  min-height: 0;
}
.frame-panel-body::-webkit-scrollbar { width: 4px; }
.frame-panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
 
/* ── Message styles ───────────────────────────────────────────────────────── */
.msg { margin-bottom: 10px; }
.msg-time { font-size: 9px; color: var(--muted); margin-bottom: 3px; }
.msg-bubble {
  display: inline-block;
  padding: 7px 11px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.6;
  max-width: 94%;
}
.msg.user { text-align: right; }
.msg.user .msg-time { text-align: right; }
.msg.user .msg-bubble { background: var(--bg4); color: var(--text); text-align: left; }
.msg.sys  .msg-bubble { background: var(--bg3); color: #a9b1d6; }
 
/* ── Detection cards ──────────────────────────────────────────────────────── */
.det-card {
  margin: 12px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.det-img { width: 100%; display: block; }
.det-meta {
  padding: 8px 12px;
  font-size: 10px;
  color: var(--muted);
  background: var(--bg3);
  line-height: 1.7;
}
.det-meta .label {
  color: var(--accent);
  font-weight: 600;
  font-size: 11px;
  margin-bottom: 3px;
  text-transform: uppercase;
  letter-spacing: .05em;
}
 
/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  gap: 12px;
  color: var(--muted);
  text-align: center;
  height: 100%;
}
.empty-state i  { font-size: 32px; opacity: .3; }
.empty-txt { font-size: 11px; line-height: 1.7; opacity: .6; }
 
/* ── Upload indicator ─────────────────────────────────────────────────────── */
.htmx-indicator { display: none; }
.htmx-request ~ .htmx-indicator,
.htmx-request.htmx-indicator { display: inline-flex; align-items: center; gap: 6px; }
 
@keyframes spin  { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
 
.spinner {
  width: 12px; height: 12px;
  border: 2px solid var(--accent);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
.pulse-txt { font-size: 11px; color: var(--accent); animation: pulse 1.4s ease infinite; }
/* ── Mission selector ────────────────────────────────────────────────────── */
.mission-selector {
  position: relative;
  display: flex;
  align-items: center;
  gap: 4px;
}

.mission-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text);
  transition: border-color .15s, background .15s;
  max-width: 200px;
}
.mission-btn:hover { border-color: var(--blue); background: var(--blue-dim); }

.m-active-name {
  flex: 1;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  max-width: 140px;
}

.mission-cog {
  width: 28px; height: 28px;
  border-radius: 6px;
  background: transparent;
  border: 1px solid var(--border);
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted);
  font-size: 11px;
  transition: background .12s, color .12s, border-color .15s;
}
.mission-cog:hover { background: var(--bg4); color: var(--text); border-color: var(--border-hi); }

/* Mission dropdown */
.mission-drop {
  display: none;
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  min-width: 240px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0,0,0,.7);
  z-index: 800;
  padding: 4px 0;
  overflow: hidden;
}
.mission-drop.open { display: block; }

.m-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  cursor: pointer;
  transition: background .12s;
  font-size: 11px;
}
.m-item:hover { background: var(--bg3); }
.m-item-active { background: var(--bg4); }

.m-item-name {
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.m-item-date {
  color: var(--muted);
  font-size: 9px;
  margin-top: 1px;
}

/* ── Settings drawer ─────────────────────────────────────────────────────── */
.drawer-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.45);
  z-index: 800;
  backdrop-filter: blur(2px);
}
.drawer-overlay.open { display: block; }

.drawer-panel {
  display: none;
  position: fixed;
  top: 0; right: 0; bottom: 0;
  width: 340px;
  background: var(--bg1);
  border-left: 1px solid var(--border);
  z-index: 801;
  flex-direction: column;
  overflow-y: auto;
  box-shadow: -8px 0 40px rgba(0,0,0,.6);
}
.drawer-panel.open { display: flex; }

.drawer-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.drawer-title {
  font-family: var(--font-head);
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  flex: 1;
}
.drawer-close {
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--muted);
  font-size: 16px;
  padding: 2px 6px;
  border-radius: 4px;
  transition: color .12s, background .12s;
}
.drawer-close:hover { color: var(--text); background: var(--bg4); }

.drawer-section {
  padding: 14px 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.drawer-sep { height: 1px; background: var(--border); flex-shrink: 0; }
.drawer-label {
  font-size: 9px;
  color: var(--muted);
  letter-spacing: .1em;
  text-transform: uppercase;
  font-family: var(--font-mono);
}
.drawer-input, .drawer-textarea {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 11px;
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text);
  outline: none;
  transition: border-color .15s;
  width: 100%;
}
.drawer-input:focus, .drawer-textarea:focus { border-color: var(--blue); }
.drawer-textarea { resize: vertical; min-height: 80px; }
.drawer-hint { font-size: 10px; color: var(--muted); line-height: 1.6; }

.drawer-save-btn {
  background: var(--blue);
  color: var(--bg0);
  border: none;
  border-radius: 6px;
  padding: 8px 18px;
  cursor: pointer;
  font-weight: 700;
  font-size: 12px;
  font-family: var(--font-mono);
  transition: background .15s;
  align-self: flex-start;
  margin-top: 4px;
}
.drawer-save-btn:hover { background: #89b4fa; }

.intent-row {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  padding: 6px 0;
  user-select: none;
}
/* Hard-reset checkbox — overrides MonsterUI/Theme.slate global input styles */
.intent-row input[type="checkbox"] {
  width: 15px !important;
  height: 15px !important;
  min-width: 15px !important;
  max-width: 15px !important;
  flex-shrink: 0 !important;
  accent-color: var(--blue);
  cursor: pointer;
  appearance: auto !important;
  -webkit-appearance: checkbox !important;
  background: transparent !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  padding: 0 !important;
  margin: 0 !important;
  box-shadow: none !important;
}
.intent-row span { font-size: 12px; color: var(--text); line-height: 1; }

.arch-btn, .del-btn {
  background: var(--bg4);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px 14px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--muted);
  transition: background .12s, color .12s;
}
.arch-btn:hover { color: var(--text); background: var(--bg3); }
.del-btn:hover  { color: var(--danger); border-color: var(--danger); background: rgba(247,118,142,.08); }

.del-confirm { display: none; flex-direction: column; gap: 6px; margin-top: 6px; }
.del-confirm.open { display: flex; }
.del-hint { font-size: 10px; color: var(--muted); }
.del-confirm-input {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 10px;
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text);
  outline: none;
}
.del-go {
  background: rgba(247,118,142,.15);
  border: 1px solid var(--danger);
  border-radius: 5px;
  padding: 7px 14px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--danger);
  transition: background .12s;
}
.del-go:hover { background: rgba(247,118,142,.25); }

/* ── Detection panel ─────────────────────────────────────────────────────── */
#tae-det-panel { top: 340px; right: 20px; width: 480px; height: 280px; }

.pi-det { background: rgba(247,118,142,.16); color: var(--danger); }

.det-panel-body {
  overflow-y: auto;
  min-height: 0;
  flex: 1;
}
.det-panel-body::-webkit-scrollbar { width: 4px; }
.det-panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.det-rows-wrap { padding: 6px 0; }

.det-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 14px;
  cursor: pointer;
  transition: background .12s;
  border-bottom: 1px solid rgba(43,46,72,.5);
}
.det-row:hover       { background: var(--bg3); }
.det-row-active      { background: var(--bg4); border-left: 2px solid var(--blue); padding-left: 12px; }
.det-row-label       { font-size: 11px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.det-row-meta        { font-size: 9px;  color: var(--muted); margin-top: 2px; }

/* Timeline tooltip */
#timeline-tooltip {
  position: fixed;
  background: rgba(14,15,26,.97);
  border: 1px solid var(--border-hi);
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text);
  pointer-events: none;
  display: none;
  z-index: 9999;
  white-space: nowrap;
  box-shadow: 0 4px 20px rgba(0,0,0,.7);
  line-height: 1.7;
}

/* ── Video panel content (output of /video_panel route) ─────────────────── */
.video-panel-inner {
  flex: 1; overflow-y: auto; display: flex; flex-direction: column; min-height: 0;
}
.video-panel-inner::-webkit-scrollbar { width: 4px; }
.video-panel-inner::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.video-panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px; border-bottom: 1px solid var(--border);
  background: var(--bg3); flex-shrink: 0;
  font-size: 11px; color: var(--muted);
}
.video-panel-close {
  cursor: pointer; color: var(--muted); font-size: 16px;
  line-height: 1; transition: color .15s;
}
.video-panel-close:hover { color: var(--text); }

.video-wrap { position: relative; flex: 1; min-height: 0; background: #000; }
.video-wrap video { max-height: none !important; width: 100%; height: 100%; display: block; object-fit: contain; }
.bbox-overlay {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%; pointer-events: none;
}
.video-empty { padding: 24px; font-size: 12px; color: var(--muted); text-align: center; line-height: 1.7; }
.timeline-wrap { padding: 6px 14px 8px; flex-shrink: 0; }
.timeline-canvas { cursor: pointer; display: block; }
.stream-status { flex-shrink: 0; }
.video-detlist { padding: 0 14px 10px; flex-shrink: 0; font-size: 10px; color: var(--muted); }

/* Detection images — below video in video panel */
.tae-imgpanel-inner { border-top: 1px solid var(--border); overflow-y: auto; max-height: 340px; }
.tae-imgpanel-inner::-webkit-scrollbar { width: 4px; }
.tae-imgpanel-inner::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* panel-close used by /images/ route output */
.panel-close { cursor: pointer; color: var(--muted); font-size: 16px; transition: color .15s; }
.panel-close:hover { color: var(--text); }

""")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# JS
# ─────────────────────────────────────────────────────────────────────────────
_JS = Script("""
// ══ Background dot-grid canvas ════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', function() {

(function initBg() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
 
  // Perfectly regular grid — no jitter, no random placement
  const SPACING = 38;   // px between dots
  const BASE_R  = 1.4;  // dim dot radius
  const HI_R    = 2.4;  // bright node radius
  const CONN    = 120;  // max connection distance between bright nodes
 
  let W, H, dots = [], t = 0;
 
  function buildDots() {
    dots = [];
    const cols = Math.ceil(W / SPACING) + 2;
    const rows = Math.ceil(H / SPACING) + 2;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Deterministic bright pattern — every ~11th dot (spread evenly)
        const bright = ((r * 7 + c * 3) % 11 === 0);
        dots.push({
          x:     c * SPACING,
          y:     r * SPACING,
          bright,
          phase: (r * 0.7 + c * 1.3) % (Math.PI * 2),  // spread phases
        });
      }
    }
  }
 
  function resize() {
    W = canvas.width  = canvas.parentElement.offsetWidth;
    H = canvas.height = canvas.parentElement.offsetHeight;
    buildDots();
  }
 
  function draw() {
    ctx.clearRect(0, 0, W, H);
    t += 0.012;
 
    // Connections between bright nodes
    ctx.lineWidth = 0.6;
    for (let i = 0; i < dots.length; i++) {
      const a = dots[i];
      if (!a.bright) continue;
      for (let j = i + 1; j < dots.length; j++) {
        const b = dots[j];
        if (!b.bright) continue;
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONN) {
          const alpha = (1 - dist / CONN) * 0.13;
          ctx.strokeStyle = `rgba(122,162,247,${alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
 
    // Dots
    for (const d of dots) {
      if (d.bright) {
        const pulse = 0.55 + 0.45 * Math.sin(t + d.phase);
        ctx.fillStyle = `rgba(122,162,247,${(0.55 * pulse).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(d.x, d.y, HI_R * (0.8 + 0.25 * pulse), 0, Math.PI * 2);
        ctx.fill();
      } else {
        const breathe = 0.85 + 0.15 * Math.sin(t * 0.4 + d.phase);
        ctx.fillStyle = `rgba(72,82,138,${(0.32 * breathe).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(d.x, d.y, BASE_R, 0, Math.PI * 2);
        ctx.fill();
      }
    }
 
    requestAnimationFrame(draw);
  }
 
  const ro = new ResizeObserver(resize);
  ro.observe(canvas.parentElement);
  resize();
  draw();
})();
 
 
// ══ Panel system ══════════════════════════════════════════════════════════
(function initPanels() {
  let topZ = 200;
 
  function focusPanel(panel) {
    document.querySelectorAll('.tae-panel').forEach(p => p.classList.remove('is-focused'));
    panel.classList.add('is-focused');
    panel.style.zIndex = ++topZ;
  }
 
  function initPanel(panel) {
    const header      = panel.querySelector('.panel-header');
    const collapseBtn = panel.querySelector('[data-collapse]');
    const resizeEl    = panel.querySelector('.panel-resize');
 
    /* Focus */
    panel.addEventListener('mousedown', () => focusPanel(panel), true);
 
    /* Drag ---------------------------------------------------------------- */
    let dragging = false, dragSX = 0, dragSY = 0, origX = 0, origY = 0;
 
    header.addEventListener('mousedown', (e) => {
      if (e.target.closest('.panel-controls')) return;
      dragging = true;
      dragSX   = e.clientX;
      dragSY   = e.clientY;
      origX    = panel.offsetLeft;
      origY    = panel.offsetTop;
      header.style.cursor = 'grabbing';
      e.preventDefault();
    });
 
    /* Collapse ------------------------------------------------------------ */
    if (collapseBtn) {
      collapseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const isCollapsed = panel.classList.toggle('is-collapsed');
 
        if (isCollapsed) {
          // Pin height to header only; stash the previous height for restore
          panel.dataset.prevH = panel.offsetHeight + 'px';
          panel.style.height  = '44px';
        } else {
          // Restore — fall back to empty string if nothing was stored
          panel.style.height = panel.dataset.prevH || '';
        }
 
        const icon = collapseBtn.querySelector('i');
        if (icon) icon.className = isCollapsed ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
      });
    }
 
    /* Resize -------------------------------------------------------------- */
    let resizing = false, rSX = 0, rSY = 0, rW = 0, rH = 0;
    if (resizeEl) {
      resizeEl.addEventListener('mousedown', (e) => {
        resizing = true;
        rSX = e.clientX; rSY = e.clientY;
        rW  = panel.offsetWidth; rH = panel.offsetHeight;
        e.preventDefault(); e.stopPropagation();
      });
    }
 
    /* Global move/up ―― registered once per panel ----------------------- */
    document.addEventListener('mousemove', (e) => {
      if (dragging) {
        let nx = origX + e.clientX - dragSX;
        let ny = origY + e.clientY - dragSY;
        nx = Math.max(0, Math.min(window.innerWidth  - panel.offsetWidth,  nx));
        ny = Math.max(0, Math.min(window.innerHeight - 44,                 ny));
        panel.style.left = nx + 'px';
        panel.style.top  = ny + 'px';
        panel.style.bottom = 'auto';  /* clear bottom/right anchors */
        panel.style.right  = 'auto';
      }
      if (resizing) {
        panel.style.width  = Math.max(260, rW + e.clientX - rSX) + 'px';
        panel.style.height = Math.max(100, rH + e.clientY - rSY) + 'px';
      }
    });
    document.addEventListener('mouseup', () => {
      if (dragging) { dragging = false; header.style.cursor = 'grab'; }
      resizing = false;
    });
  }
 
  document.querySelectorAll('.tae-panel').forEach(initPanel);
 
  /* Expose for panels injected later (e.g. frame panel after HTMX swap) */
  window.taeInitPanel = initPanel;
})();

}); // DOMContentLoaded


// ══ Map iframe → detection images ═════════════════════════════════════════
// The map popup button does: window.parent.postMessage({type:'show_images', id:det_id})
// We intercept it here and load the detection images into #tae-imgpanel (inside frame panel)
window.addEventListener('message', function(e) {
  if (!e.data || e.data.type !== 'show_images') return;
  htmx.ajax('GET', '/images/' + e.data.id, {target: '#tae-imgpanel', swap: 'innerHTML'});
  // Only reveal panel if currently hidden — never resize an already-visible panel
  var fp = document.getElementById('tae-frame-panel');
  if (fp && (fp.style.display === 'none' || fp.classList.contains('collapsed'))) {
    showPanel('tae-frame-panel');
  }
});
             
// Ctrl+P = toggle coverage polygon
document.addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'p') {
    e.preventDefault();
    htmx.ajax('GET', '/toggle_coverage', {target: '#tae-msgs', swap: 'beforeend'});
    setTimeout(function() { scrollChat(); refreshMap(); }, 400);
  }
});
 
 
// ══ Helpers ═══════════════════════════════════════════════════════════════
function scrollChat() {
  const el = document.getElementById('tae-msgs');
  if (el) el.scrollTop = el.scrollHeight;
}
 
function refreshMap() {
  const f = document.getElementById('tae-map-frame');
  if (f) { const s = f.src; f.src = ''; f.src = s; }
}
 
function showPanel(id) {
  const p = document.getElementById(id);
  if (!p) return;
  p.style.display = 'flex';
  p.classList.remove('is-collapsed');
  p.style.height = p.dataset.prevH || '';
  // bring to front
  document.querySelectorAll('.tae-panel').forEach(function(x){x.classList.remove('is-focused');});
  p.classList.add('is-focused');
}
 
function closeImages() {
  const p = document.getElementById('tae-frame-panel');
  if (p) p.style.display = 'none';
}
 
/* Called by map marker click */
function openImages(detId) {
  showPanel('tae-frame-panel');
}
 
/* Timestamp used by map-refresh query param to bust cache */

// ══ Feed / Video panel helpers ═══════════════════════════════════════════════
function openFeedPanel() {
  showPanel('tae-video-panel');
  htmx.ajax('GET', '/feed_panel', {target: '#video-panel', swap: 'innerHTML'});
}

function closeFeedPanel() {
  htmx.ajax('GET', '/video_panel', {target: '#video-panel', swap: 'innerHTML'});
}

function closeVideoPanel() {
  var p = document.getElementById('tae-video-panel');
  if (p) p.style.display = 'none';
}

function startStream(event) {
  var url = (document.getElementById('stream-url') || {}).value || '';
  var lat = parseFloat((document.getElementById('stream-lat') || {}).value || '0');
  var lon = parseFloat((document.getElementById('stream-lon') || {}).value || '0');
  fetch('/stream/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: url, lat: lat, lon: lon})
  }).then(function(r){return r.json();}).then(function(d){
    if (d.ok) htmx.ajax('GET', '/stream/panel', {target: '#video-panel', swap: 'innerHTML'});
    else alert('Stream error: ' + d.error);
  });
}

function stopStream() {
  fetch('/stream/stop', {method:'POST'})
    .then(function(){htmx.ajax('GET','/stream/panel',{target:'#video-panel',swap:'innerHTML'});});
}

// ── Video playback ────────────────────────────────────────────────────────────
var _videoDets = [];

function onVideoMeta() {
  fetch('/video_detections').then(function(r){return r.json();}).then(function(data){
    _videoDets = data;
    _drawTimeline();
    _initTimelineTooltip();
    refreshDetPanel();
    // If no detections yet (server still restoring state), retry once after 3s
    if (!data.length) {
      setTimeout(function() {
        fetch('/video_detections').then(function(r){return r.json();}).then(function(d){
          if (d.length) { _videoDets = d; _drawTimeline(); refreshDetPanel(); }
        });
      }, 3000);
    }
  });
}

function onVideoTime() {
  var v = document.getElementById('tae-video');
  var el = document.getElementById('vtime');
  if (v && el) {
    var s = Math.floor(v.currentTime);
    el.textContent = Math.floor(s/60) + ':' + String(s%60).padStart(2,'0');
  }
  _drawBbox();
}

function seekVideo(event) {
  var canvas = document.getElementById('timeline-canvas');
  var v = document.getElementById('tae-video');
  if (!canvas || !v || !v.duration) return;
  var rect = canvas.getBoundingClientRect();
  v.currentTime = ((event.clientX - rect.left) / rect.width) * v.duration;
}

function _drawTimeline() {
  var canvas = document.getElementById('timeline-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var v = document.getElementById('tae-video');
  if (!v || !v.duration) return;
  var W = canvas.offsetWidth, H = 36;
  canvas.width = W; canvas.height = H;
  ctx.fillStyle = '#1b1d30';
  ctx.fillRect(0,0,W,H);
  _videoDets.forEach(function(d) {
    if (d.timestamp_ms == null) return;
    var x = (d.timestamp_ms / (v.duration * 1000)) * W;
    ctx.fillStyle = d.color || '#4ade80';
    ctx.fillRect(Math.max(0, x-2), 4, 4, H-8);
  });
}

function _drawBbox() {
  var canvas = document.getElementById('bbox-canvas');
  var v = document.getElementById('tae-video');
  if (!canvas || !v) return;
  var ctx = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  var nowMs = v.currentTime * 1000, tol = 2000;
  var frameW = v.videoWidth  || 1920;
  var frameH = v.videoHeight || 1080;

  // Compute the actual video-content rect inside the canvas
  // (object-fit:contain adds letterbox / pillarbox bars)
  var cW = canvas.width, cH = canvas.height;
  var vAspect = frameW / frameH;
  var cAspect = cW / cH;
  var vx, vy, vw, vh;
  if (vAspect > cAspect) {
    // wider than canvas — bars top & bottom
    vw = cW; vh = cW / vAspect;
    vx = 0;  vy = (cH - vh) / 2;
  } else {
    // taller than canvas — bars left & right
    vh = cH; vw = cH * vAspect;
    vy = 0;  vx = (cW - vw) / 2;
  }
  var sx = vw / frameW;
  var sy = vh / frameH;

  _videoDets.forEach(function(d) {
    if (d.timestamp_ms == null) return;
    if (Math.abs(d.timestamp_ms - nowMs) > tol) return;
    var b = d.bbox;
    if (!b || b.length !== 4) return;

    // bbox is relative to the tile; tile is positioned in the full frame
    var tx = d.tile_x || 0, ty = d.tile_y || 0;
    var x1 = vx + (tx + b[0]) * sx;
    var y1 = vy + (ty + b[1]) * sy;
    var x2 = vx + (tx + b[2]) * sx;
    var y2 = vy + (ty + b[3]) * sy;

    ctx.strokeStyle = d.color || '#4ade80';
    ctx.lineWidth = Math.max(1.5, 2 * sx);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });
}


// ══ Detection selection — syncs Frame, Video, Map ═════════════════════════
function selectDetection(detId, lat, lon, tsMs) {
  // Frame panel: load detection image
  htmx.ajax('GET', '/images/' + detId, {target: '#tae-imgpanel', swap: 'innerHTML'});
  showPanel('tae-frame-panel');

  // Video: seek to timestamp; only reveal panel if hidden (don't reset its size)
  var v = document.getElementById('tae-video');
  if (v && tsMs != null) {
    v.currentTime = tsMs / 1000;
    var _vp = document.getElementById('tae-video-panel');
    if (_vp && (_vp.style.display === 'none' || _vp.classList.contains('is-collapsed'))) {
      showPanel('tae-video-panel');
    }
  }

  // Map: recentre (rebuilds map server-side, then refreshes iframe)
  fetch('/set_map_focus?det_id=' + encodeURIComponent(detId))
    .then(function(r) { return r.json(); })
    .then(function(d) { if (d.ok) refreshMap(); });

  // Highlight selected row
  document.querySelectorAll('.det-row').forEach(function(r) {
    r.classList.remove('det-row-active');
  });
  var row = document.getElementById('det-row-' + detId);
  if (row) {
    row.classList.add('det-row-active');
    row.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }
}

// ══ Detection panel refresh (called after a query completes) ══════════════
function refreshDetPanel() {
  var el = document.getElementById('det-panel-list');
  if (el) htmx.ajax('GET', '/detections_panel_content', {target: '#det-panel-list', swap: 'innerHTML'});
}

// ══ Timeline tooltip ═══════════════════════════════════════════════════════
var _ttEl = null;

function _initTimelineTooltip() {
  var canvas = document.getElementById('timeline-canvas');
  if (!canvas || canvas._ttInit) return;
  canvas._ttInit = true;

  if (!_ttEl) {
    _ttEl = document.createElement('div');
    _ttEl.id = 'timeline-tooltip';
    document.body.appendChild(_ttEl);
  }

  canvas.addEventListener('mousemove', function(e) {
    var v = document.getElementById('tae-video');
    if (!v || !v.duration || !_videoDets.length) {
      _ttEl.style.display = 'none'; return;
    }
    var rect   = canvas.getBoundingClientRect();
    var pct    = (e.clientX - rect.left) / rect.width;
    var hoverMs = pct * v.duration * 1000;
    var tol    = v.duration * 1000 * 0.025;  // 2.5% of total duration

    var near = _videoDets.filter(function(d) {
      return d.timestamp_ms != null && Math.abs(d.timestamp_ms - hoverMs) < tol;
    });

    if (!near.length) { _ttEl.style.display = 'none'; return; }

    _ttEl.innerHTML = near.map(function(d) {
      var ts = '';
      if (d.timestamp_ms != null) {
        var s = Math.floor(d.timestamp_ms / 1000);
        ts = ' <span style="color:var(--muted)">'+Math.floor(s/60)+':'+(''+(s%60)).padStart(2,'0')+'</span>';
      }
      var sym = d.confirmed ? '●' : '○';
      return '<span style="color:' + (d.color||'#4ade80') + '">' + sym + '</span> ' + d.label + ts;
    }).join('<br>');

    _ttEl.style.display = 'block';
    _ttEl.style.left    = (e.clientX + 14) + 'px';
    _ttEl.style.top     = (e.clientY - _ttEl.offsetHeight - 8) + 'px';
  });

  canvas.addEventListener('mouseleave', function() {
    if (_ttEl) _ttEl.style.display = 'none';
  });

  // Also make timeline clickable to both seek AND open nearest detection
  canvas.addEventListener('click', function(e) {
    var v = document.getElementById('tae-video');
    if (!v || !v.duration) return;
    var rect   = canvas.getBoundingClientRect();
    var pct    = (e.clientX - rect.left) / rect.width;
    var hoverMs = pct * v.duration * 1000;
    var tol    = v.duration * 1000 * 0.025;

    var near = _videoDets.filter(function(d) {
      return d.timestamp_ms != null && Math.abs(d.timestamp_ms - hoverMs) < tol;
    });
    if (near.length) {
      var d = near[0];
      selectDetection(d.id, d.lat, d.lon, d.timestamp_ms);
    }
  }, true);  // capture so it fires before the existing onclick seekVideo
}


// ══ Mission selector ══════════════════════════════════════════════════════
function toggleMissionDrop(e) {
  e.stopPropagation();
  var drop = document.getElementById('mission-drop');
  if (!drop) return;
  drop.classList.toggle('open');
}

// Close mission drop when clicking elsewhere
document.addEventListener('click', function(e) {
  var drop = document.getElementById('mission-drop');
  if (drop && !drop.closest('.mission-selector').contains(e.target)) {
    drop.classList.remove('open');
  }
});

// ══ Settings drawer ════════════════════════════════════════════════════════
function openDrawer(missionId) {
  document.getElementById('mission-drop').classList.remove('open');
  var overlay = document.getElementById('drawer-overlay');
  var panel   = document.getElementById('drawer-panel');
  var content = document.getElementById('drawer-content');
  if (!overlay || !panel || !content) return;
  overlay.classList.add('open');
  panel.classList.add('open');
  htmx.ajax('GET', '/missions/' + missionId + '/drawer',
    {target: '#drawer-content', swap: 'innerHTML'});
}

function closeDrawer() {
  var overlay = document.getElementById('drawer-overlay');
  var panel   = document.getElementById('drawer-panel');
  if (overlay) overlay.classList.remove('open');
  if (panel)   panel.classList.remove('open');
}

function toggleDeleteConfirm() {
  var box = document.getElementById('del-confirm-box');
  if (box) box.classList.toggle('open');
}

""" + "window._mapTs = () => Date.now();" + """
""")