"""
core/styles.py — CSS stylesheet and JavaScript for the TAE web UI.

Pure string constants — no FastHTML, no business logic.
Imported by main.py and passed directly into the fast_app() headers.
"""

from fasthtml.common import Style, Script

_CSS = Style("""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Rajdhani:wght@500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg0: #0d0e1a;
  --bg1: #12131f;
  --bg2: #181929;
  --bg3: #1e1f32;
  --bg4: #252637;
  --border: #2a2c45;
  --accent: #4ade80;
  --accent-dim: rgba(74,222,128,.15);
  --blue: #7aa2f7;
  --blue-dim: rgba(122,162,247,.12);
  --text: #c0caf5;
  --muted: #6272a4;
  --danger: #f7768e;
  --danger-dim: rgba(247,118,142,.12);
  --font-mono: 'JetBrains Mono', monospace;
  --font-head: 'Rajdhani', sans-serif;
}

html, body { height:100%; background: var(--bg0); color: var(--text); font-family: var(--font-mono); overflow: hidden; }

/* ── Navbar ─────────────────────────────────────────────────────────────── */
.tae-nav {
  position: fixed; top:0; left:0; right:0; z-index:200;
  height: 52px;
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 14px;
  gap: 10px;
}

/* ── Mission chip ────────────────────────────────────────────────────────── */
.m-chip-wrap { position: relative; }

.m-chip {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 5px 10px;
  font-size: 10px;
  cursor: pointer;
  min-width: 150px;
  max-width: 220px;
  user-select: none;
  transition: border-color .15s, background .15s;
}
.m-chip:hover, .m-chip.open { border-color: var(--blue); background: var(--blue-dim); }
.m-chip-name { font-weight: 600; color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.m-chip-chev { font-size: 8px; color: var(--muted); transition: transform .15s; flex-shrink: 0; }
.m-chip.open .m-chip-chev { transform: rotate(180deg); }

/* ── Mission dropdown ────────────────────────────────────────────────────── */
.m-dropdown {
  display: none;
  position: absolute;
  top: calc(100% + 5px);
  left: 0;
  width: 272px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 8px 28px rgba(0,0,0,.55);
  overflow: hidden;
  z-index: 300;
}
.m-dropdown.open { display: block; }

.m-item {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 9px 12px;
  font-size: 10px;
  cursor: pointer;
  transition: background .1s;
}
.m-item:hover { background: var(--bg3); }
.m-item.cur { background: var(--accent-dim); }
.m-item-name { flex: 1; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.m-item-date { color: var(--muted); font-size: 9px; white-space: nowrap; }

.m-sep { height: 1px; background: var(--border); margin: 2px 0; }

.m-new {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  font-size: 10px;
  color: var(--accent);
  cursor: pointer;
  font-weight: 600;
}
.m-new:hover { background: var(--accent-dim); }

.m-archived {
  padding: 7px 12px;
  font-size: 9px;
  color: var(--muted);
  cursor: pointer;
}
.m-archived:hover { color: var(--text); }

/* ── Settings (config) button ────────────────────────────────────────────── */
.cfg-btn {
  background: none;
  border: none;
  color: var(--muted);
  cursor: pointer;
  font-size: 15px;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 5px;
  transition: color .15s, background .15s;
  flex-shrink: 0;
}
.cfg-btn:hover, .cfg-btn.active { color: var(--text); background: var(--bg4); }

/* ── Brand ───────────────────────────────────────────────────────────────── */
.brand {
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .12em;
  text-transform: uppercase;
  flex-shrink: 0;
}

.sep { width:1px; height:28px; background: var(--border); margin: 0 2px; flex-shrink:0; }

/* ── Upload button ───────────────────────────────────────────────────────── */
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
.upload-btn input[type="file"] { display:none; }

/* ── Status badge ────────────────────────────────────────────────────────── */
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

/* ── Main layout ─────────────────────────────────────────────────────────── */
.tae-main {
  display: flex;
  height: calc(100vh - 52px);
  margin-top: 52px;
}

/* ── Map ─────────────────────────────────────────────────────────────────── */
.map-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
}
.map-wrap iframe {
  width: 100%; height: 100%;
  border: none; display: block;
}

/* ── Settings drawer ─────────────────────────────────────────────────────── */
.settings-drawer {
  width: 300px;
  min-width: 300px;
  background: var(--bg1);
  border-left: 1px solid var(--border);
  display: none;
  flex-direction: column;
  overflow-y: auto;
  transition: width .22s ease;
}
.settings-drawer.open { display: flex; }
.settings-drawer::-webkit-scrollbar { width: 4px; }
.settings-drawer::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.drawer-header {
  position: sticky; top: 0; z-index: 1;
  background: var(--bg1);
  padding: 13px 15px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}
.drawer-title { font-size: 11px; font-weight: 600; color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.drawer-close { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 15px; line-height: 1; padding: 2px 4px; }
.drawer-close:hover { color: var(--text); }

.drawer-section { padding: 13px 15px; }
.drawer-label { font-size: 9px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 7px; }
.drawer-sep { height: 1px; background: var(--border); }

.drawer-input {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 7px 9px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 10px;
  outline: none;
}
.drawer-input:focus { border-color: var(--blue); }

.drawer-textarea {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 7px 9px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 10px;
  outline: none;
  resize: vertical;
  min-height: 72px;
  line-height: 1.6;
}
.drawer-textarea:focus { border-color: var(--blue); }
.drawer-hint { font-size: 9px; color: var(--muted); margin-top: 4px; line-height: 1.5; }

.intent-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  cursor: pointer;
  font-size: 10px;
  color: var(--text);
  border: none !important;
  box-shadow: none !important;
  background: none !important;
  padding: 0 !important;
}
.intent-row input[type="checkbox"] {
  accent-color: var(--accent);
  cursor: pointer;
  width: 15px !important;
  height: 15px !important;
  min-width: 15px;
  flex-shrink: 0;
  margin: 0;
  padding: 0;
  border: none;
  background: none;
  box-shadow: none;
  appearance: auto;
  -webkit-appearance: checkbox;
}

.drawer-save-btn {
  margin-top: 10px;
  background: var(--accent-dim);
  border: 1px solid var(--accent);
  border-radius: 5px;
  padding: 6px 12px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--accent);
  cursor: pointer;
  font-weight: 600;
}
.drawer-save-btn:hover { background: var(--accent); color: var(--bg0); }

.arch-btn {
  width: 100%;
  background: none;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 11px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  cursor: pointer;
  text-align: left;
  margin-bottom: 7px;
}
.arch-btn:hover { border-color: var(--text); color: var(--text); }

.del-btn {
  width: 100%;
  background: none;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 11px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--danger);
  cursor: pointer;
  text-align: left;
}
.del-btn:hover { border-color: var(--danger); background: var(--danger-dim); }

.del-confirm { margin-top: 8px; display: none; }
.del-confirm.show { display: block; }
.del-hint { font-size: 9px; color: var(--muted); margin-bottom: 5px; }
.del-confirm-input {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--danger);
  border-radius: 5px;
  padding: 6px 9px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 9px;
  outline: none;
  margin-bottom: 6px;
}
.del-go {
  width: 100%;
  background: var(--danger-dim);
  border: 1px solid var(--danger);
  border-radius: 5px;
  padding: 6px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--danger);
  cursor: pointer;
  font-weight: 600;
}
.del-go:hover { background: var(--danger); color: var(--bg0); }

/* ── Image panel (right side) ────────────────────────────────────────────── */
.img-panel {
  width: 380px;
  min-width: 380px;
  background: var(--bg2);
  border-left: 1px solid var(--border);
  display: none;
  flex-direction: column;
  overflow-y: auto;
}
.img-panel.open { display: flex; }
.img-panel::-webkit-scrollbar { width: 4px; }
.img-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.panel-header {
  position: sticky; top: 0; z-index: 1;
  background: var(--bg2);
  padding: 13px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  color: var(--blue);
  letter-spacing: .06em;
  text-transform: uppercase;
}
.panel-close {
  cursor: pointer; font-size: 20px; line-height: 1;
  color: var(--muted); transition: color .15s;
}
.panel-close:hover { color: var(--text); }

.det-card {
  margin: 12px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.det-meta {
  padding: 8px 12px;
  font-size: 10px;
  color: var(--muted);
  background: var(--bg3);
  line-height: 1.7;
}
.det-meta .label {
  color: var(--accent); font-weight: 600; font-size: 11px;
  margin-bottom: 3px; text-transform: uppercase; letter-spacing: .05em;
}

/* ── Chat ────────────────────────────────────────────────────────────────── */
.chat-wrap {
  position: fixed;
  bottom: 20px; left: 20px;
  z-index: 100;
  width: 370px;
}
.chat-box {
  background: rgba(18,19,31,.97);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,.7);
  backdrop-filter: blur(14px);
}
.chat-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  cursor: pointer; user-select: none;
}
.chat-head-label {
  font-family: var(--font-head); font-size: 12px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--blue);
}
.chat-head-chevron { color: var(--muted); font-size: 11px; transition: transform .2s; }
.chat-msgs {
  height: 260px; overflow-y: auto; padding: 12px 14px;
}
.chat-msgs::-webkit-scrollbar { width: 4px; }
.chat-msgs::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.chat-divider { height: 1px; background: var(--border); }
.chat-input-row { padding: 10px 12px; }
.chat-input-row form { display: flex; gap: 8px; }
.chat-input-row input {
  flex: 1; background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; color: var(--text);
  font-family: var(--font-mono); font-size: 11px; outline: none;
}
.chat-input-row input:focus { border-color: var(--blue); }
.send-btn {
  background: var(--accent-dim); border: 1px solid var(--accent);
  border-radius: 6px; padding: 8px 14px; cursor: pointer;
  color: var(--accent); font-size: 14px; font-weight: 700;
}
.send-btn:hover { background: var(--accent); color: var(--bg0); }

.msg {
  display: flex; flex-direction: column; margin-bottom: 10px;
}
.msg.user { align-items: flex-end; }
.msg.sys  { align-items: flex-start; }
.msg-time { font-size: 9px; color: var(--muted); margin-bottom: 3px; }
.msg-bubble {
  max-width: 92%; padding: 8px 11px; border-radius: 8px;
  font-size: 11px; line-height: 1.6;
  background: var(--bg3); color: var(--text);
}
.msg.user .msg-bubble { background: var(--blue-dim); border: 1px solid var(--blue); }

/* ── Video panel (left side) ─────────────────────────────────────────────── */
.video-panel {
  width: 420px;
  min-width: 420px;
  background: var(--bg1);
  border-right: 1px solid var(--border);
  display: none;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
}
.video-panel.open { display: flex; }

.video-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 700;
  color: var(--blue);
  letter-spacing: .06em;
  text-transform: uppercase;
  flex-shrink: 0;
}
.video-panel-close {
  cursor: pointer; font-size: 18px; color: var(--muted);
}
.video-panel-close:hover { color: var(--text); }

.video-wrap {
  position: relative;
  background: #000;
  flex-shrink: 0;
}
.video-wrap video {
  width: 100%;
  display: block;
  max-height: 280px;
  object-fit: contain;
}
.bbox-overlay {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}

.timeline-wrap {
  padding: 10px 14px 6px;
  flex-shrink: 0;
}
.timeline-label {
  font-size: 9px;
  color: var(--muted);
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-bottom: 5px;
  display: flex;
  justify-content: space-between;
}
.timeline-canvas {
  width: 100%;
  height: 36px;
  display: block;
  cursor: pointer;
  border-radius: 4px;
  background: var(--bg3);
  border: 1px solid var(--border);
}

.video-detlist {
  flex: 1;
  overflow-y: auto;
  padding: 0 14px 12px;
}
.video-detlist::-webkit-scrollbar { width: 4px; }
.video-detlist::-webkit-scrollbar-thumb { background: var(--border); border-radius:2px; }

.video-det-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 10px;
  cursor: pointer;
}
.video-det-row:hover { background: var(--bg3); margin: 0 -4px; padding: 6px 4px; border-radius:4px; }
.video-det-time { color: var(--muted); font-size: 9px; white-space: nowrap; min-width: 44px; }
.video-det-label { flex: 1; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.video-empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  flex: 1; gap: 10px; padding: 30px;
  text-align: center; color: var(--muted); font-size: 11px; line-height: 1.7;
}

.video-btn {
  background: none;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 11px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--muted);
  cursor: pointer;
  transition: border-color .15s, color .15s;
}
.video-btn:hover, .video-btn.active {
  border-color: var(--blue);
  color: var(--text);
  background: var(--blue-dim);
}

/* ── Misc ────────────────────────────────────────────────────────────────── */
.htmx-indicator { display: none; }
.htmx-request .htmx-indicator { display: flex; align-items: center; gap: 7px; }
@keyframes spin  { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.5; } }
.spinner {
  width: 14px; height: 14px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
.pulse-txt { font-size: 11px; color: var(--accent); animation: pulse 1.5s ease-in-out infinite; }

.empty-state {
  flex:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:10px; padding:30px; text-align:center;
}
.empty-txt { font-size:12px; color:var(--muted); line-height:1.7; }
""")


# ─────────────────────────────────────────────────────────────────────────────
# JavaScript
# ─────────────────────────────────────────────────────────────────────────────

_JS = Script("""
function scrollChat() {
    const el = document.getElementById('tae-msgs');
    if (el) el.scrollTop = el.scrollHeight;
}

function refreshMap() {
    const f = document.getElementById('tae-map-frame');
    if (f) f.src = '/map?' + Date.now();
}

// ── Mission dropdown toggle ───────────────────────────────────────────────
function toggleMissionDropdown() {
    const chip = document.querySelector('.m-chip');
    const dd   = document.getElementById('m-dropdown');
    const open = dd.classList.toggle('open');
    chip.classList.toggle('open', open);
}

document.addEventListener('click', function(e) {
    if (!e.target.closest('.m-chip-wrap')) {
        const dd   = document.getElementById('m-dropdown');
        const chip = document.querySelector('.m-chip');
        if (dd)   dd.classList.remove('open');
        if (chip) chip.classList.remove('open');
    }
});

// ── Settings drawer ───────────────────────────────────────────────────────
function openDrawer() {
    const dd   = document.getElementById('m-dropdown');
    const chip = document.querySelector('.m-chip');
    if (dd)   dd.classList.remove('open');
    if (chip) chip.classList.remove('open');
    document.getElementById('tae-imgpanel').classList.remove('open');
    document.getElementById('settings-drawer').classList.add('open');
    document.querySelector('.cfg-btn').classList.add('active');
}

function closeDrawer() {
    document.getElementById('settings-drawer').classList.remove('open');
    document.querySelector('.cfg-btn').classList.remove('active');
}

function toggleDeleteConfirm() {
    document.getElementById('del-confirm-box').classList.toggle('show');
}

// ── Map / image panel messages ────────────────────────────────────────────
window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'show_images') return;
    closeDrawer();
    htmx.ajax('GET', '/images/' + e.data.id, {
        target: '#tae-imgpanel', swap: 'innerHTML'
    });
    document.getElementById('tae-imgpanel').classList.add('open');
});

function closeImages() {
    document.getElementById('tae-imgpanel').classList.remove('open');
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        htmx.ajax('GET', '/toggle_coverage', {
            target: '#tae-msgs', swap: 'beforeend'
        });
        setTimeout(() => { scrollChat(); refreshMap(); }, 400);
    }
    if (e.key === 'Escape') { closeDrawer(); }
});

// ── Video panel ───────────────────────────────────────────────────────────
let _vDets   = [];   // [{id,timestamp_ms,color,confirmed,label,bbox,tile_x,tile_y,tile_w,tile_h}]
let _vDurMs  = 0;

function openVideoPanel() {
    // Close other panels safely, guarding against missing elements
    const sd = document.getElementById('settings-drawer');
    if (sd) sd.classList.remove('open');
    const cfg = document.querySelector('.cfg-btn');
    if (cfg) cfg.classList.remove('active');
    document.getElementById('tae-imgpanel').classList.remove('open');
    document.getElementById('video-panel').classList.add('open');
    document.querySelector('.video-btn').classList.add('active');
    htmx.ajax('GET', '/video_panel', {
        target: '#video-panel', swap: 'innerHTML'
    });
}

function closeVideoPanel() {
    document.getElementById('video-panel').classList.remove('open');
    const btn = document.querySelector('.video-btn');
    if (btn) btn.classList.remove('active');
}

function onVideoMeta() {
    const v = document.getElementById('tae-video');
    if (!v) return;
    _vDurMs = v.duration * 1000;
    fetch('/video_detections')
        .then(r => r.json())
        .then(data => { _vDets = data; _drawTimeline(0); _updateDetList(); });
}

function onVideoTime() {
    const v = document.getElementById('tae-video');
    if (!v) return;
    const ms = v.currentTime * 1000;
    _drawTimeline(ms);
    _drawBboxes(ms);
    // Update time display
    const el = document.getElementById('vtime');
    if (el) el.textContent = _fmtTime(v.currentTime) + ' / ' + _fmtTime(v.duration || 0);
}

function _fmtTime(s) {
    if (!isFinite(s)) return '0:00';
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ':' + String(sec).padStart(2, '0');
}

function _drawTimeline(curMs) {
    const c = document.getElementById('timeline-canvas');
    if (!c || !_vDurMs) return;
    const ctx = c.getContext('2d');
    c.width = c.offsetWidth; c.height = c.offsetHeight;
    const w = c.width, h = c.height, pad = 8;

    // Track
    ctx.fillStyle = '#2a2c45';
    ctx.fillRect(pad, h/2 - 2, w - pad*2, 4);

    // Detection markers
    for (const d of _vDets) {
        if (!d.timestamp_ms && d.timestamp_ms !== 0) continue;
        const x = pad + (d.timestamp_ms / _vDurMs) * (w - pad*2);
        ctx.fillStyle = d.color || '#4ade80';
        ctx.beginPath();
        ctx.arc(x, h/2, d.confirmed ? 6 : 4, 0, Math.PI*2);
        ctx.fill();
        if (!d.confirmed) {
            ctx.strokeStyle = d.color || '#4ade80';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    // Playhead
    if (_vDurMs > 0) {
        const x = pad + (curMs / _vDurMs) * (w - pad*2);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x - 1, 2, 2, h - 4);
    }
}

function seekVideo(e) {
    const c = document.getElementById('timeline-canvas');
    const v = document.getElementById('tae-video');
    if (!c || !v || !_vDurMs) return;
    const r = c.getBoundingClientRect();
    const pad = 8;
    const frac = Math.max(0, Math.min(1, (e.clientX - r.left - pad) / (c.offsetWidth - pad*2)));
    v.currentTime = frac * _vDurMs / 1000;
}

function seekToDet(ms) {
    const v = document.getElementById('tae-video');
    if (!v || !ms) return;
    v.currentTime = ms / 1000;
    v.pause();
}

function _drawBboxes(curMs) {
    const c = document.getElementById('bbox-canvas');
    const v = document.getElementById('tae-video');
    if (!c || !v) return;
    c.width  = v.clientWidth;
    c.height = v.clientHeight;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (!v.videoWidth) return;
    const sx = c.width  / v.videoWidth;
    const sy = c.height / v.videoHeight;
    const WIN = 800;   // ms window around detection timestamp
    for (const d of _vDets) {
        if (!d.bbox || d.timestamp_ms === undefined) continue;
        if (Math.abs((d.timestamp_ms || 0) - curMs) > WIN) continue;
        const [b0,b1,b2,b3] = d.bbox;
        const x1 = ((d.tile_x||0) + b0) * sx;
        const y1 = ((d.tile_y||0) + b1) * sy;
        const x2 = ((d.tile_x||0) + b2) * sx;
        const y2 = ((d.tile_y||0) + b3) * sy;
        const alpha = 1 - Math.abs((d.timestamp_ms||0) - curMs) / WIN;
        ctx.globalAlpha = 0.4 + 0.6 * alpha;
        ctx.strokeStyle = d.color || '#4ade80';
        ctx.lineWidth   = 2;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
        if (d.label) {
            ctx.fillStyle = d.color || '#4ade80';
            ctx.font = '11px JetBrains Mono, monospace';
            ctx.fillText(d.label.slice(0,24), x1+2, y1 > 14 ? y1-4 : y1+14);
        }
    }
    ctx.globalAlpha = 1;
}

function _updateDetList() {
    const el = document.getElementById('video-det-list');
    if (!el) return;
    if (!_vDets.length) { el.innerHTML = '<div style="color:var(--muted);font-size:10px;padding:8px 0">No detections yet</div>'; return; }
    const sorted = [..._vDets].filter(d => d.timestamp_ms !== undefined)
                              .sort((a,b) => a.timestamp_ms - b.timestamp_ms);
    el.innerHTML = sorted.map(d => {
        const t = _fmtTime((d.timestamp_ms||0)/1000);
        const dot = d.confirmed ? '●' : '○';
        return '<div class="video-det-row" onclick="seekToDet(' + d.timestamp_ms + ')">'
             + '<span class="video-det-time">' + t + '</span>'
             + '<span style="color:' + (d.color||'#4ade80') + ';margin-right:5px">' + dot + '</span>'
             + '<span class="video-det-label">' + (d.label||'').slice(0,40) + '</span>'
             + '</div>';
    }).join('');
}

""")