"""
AI Podcast Studio — Fixed & Improved
- Fixed edge-tts rate/pitch string formatting
- Dynamic Ollama model selection in UI
- Direct TTS mode (no Ollama) as fallback
- Script preview panel
- Improved voice/speed/pitch controls
"""

import asyncio
import io
import json
import socket
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, Response, send_file
import requests as http_requests
import edge_tts

app = Flask(__name__)

JOBS = {}
BASE_TEMP_DIR = Path(tempfile.gettempdir()) / "podcast_temp"
BASE_TEMP_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

VOICES = [
    {"id": "en-US-GuyNeural",        "label": "Guy — US Male"},
    {"id": "en-US-AriaNeural",       "label": "Aria — US Female"},
    {"id": "en-US-JennyNeural",      "label": "Jenny — US Female"},
    {"id": "en-US-ChristopherNeural","label": "Christopher — US Male"},
    {"id": "en-US-EricNeural",       "label": "Eric — US Male"},
    {"id": "en-US-MichelleNeural",   "label": "Michelle — US Female"},
    {"id": "en-GB-RyanNeural",       "label": "Ryan — UK Male"},
    {"id": "en-GB-SoniaNeural",      "label": "Sonia — UK Female"},
    {"id": "en-GB-LibbyNeural",      "label": "Libby — UK Female"},
    {"id": "en-AU-NatashaNeural",    "label": "Natasha — AU Female"},
    {"id": "en-AU-WilliamNeural",    "label": "William — AU Male"},
    {"id": "en-IN-NeerjaNeural",     "label": "Neerja — IN Female"},
    {"id": "en-IN-PrabhatNeural",    "label": "Prabhat — IN Male"},
]


# ── Ollama Helpers ─────────────────────────────────────────────────────
def get_available_models():
    try:
        r = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def pick_model(available):
    prefs = ["gemma3:4b", "gemma3", "llama3", "mistral", "phi3", "qwen"]
    for p in prefs:
        for m in available:
            if m.lower().startswith(p.lower()):
                return m
    return available[0] if available else None


def ollama_generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.85, "num_predict": 2000},
    }
    text = []
    try:
        with http_requests.post(
            f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=(30, 600)
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    text.append(chunk.get("response", ""))
                    if chunk.get("done"):
                        break
    except Exception as e:
        print(f"Ollama error: {e}")
    return "".join(text).strip()


SYSTEM_PROMPT = """You are an excellent podcast narrator.
Analyse the source material carefully and turn it into a natural, engaging spoken podcast episode.
Use warm, conversational tone. Make it interesting and easy to listen to.
Do NOT include any stage directions, speaker labels, or markdown formatting.
Output ONLY the final narration text, ready to be spoken aloud."""


def build_prompt(text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\nSOURCE MATERIAL:\n\"\"\"\n{text[:4000]}\n\"\"\"\n\nPODCAST NARRATION:"
    )


# ── Edge TTS ───────────────────────────────────────────────────────────
# FIX: edge-tts requires rate/pitch as strings like "+10%" and "+5Hz"
# If value is 0, we must still pass a valid string ("+0%" not None).
async def generate_audio(text: str, voice: str, rate: int = 0, pitch: int = 0) -> bytes:
    rate_str = f"{rate:+d}%"      # always "+0%", "+10%", "-20%" etc.
    pitch_str = f"{pitch:+d}Hz"   # always "+0Hz", "+5Hz", "-10Hz" etc.

    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    data = buf.read()
    if len(data) < 1000:
        raise ValueError("TTS returned no audio — check the voice ID and text.")
    return data


# ── Job Runner ────────────────────────────────────────────────────────
def set_job(job_id, **kwargs):
    JOBS.setdefault(job_id, {}).update(kwargs)


def run_job(
    job_id: str,
    source_text: str,
    voice: str,
    rate: int,
    pitch: int,
    model: str,
    direct_tts: bool,
):
    try:
        if direct_tts or not model:
            # Skip Ollama — read the text directly
            set_job(job_id, status="running", progress=30, message="Generating audio directly from your text...")
            script = source_text
        else:
            set_job(job_id, status="running", progress=15, message="Connecting to Ollama...")
            available = get_available_models()
            if not available:
                raise ValueError(
                    "Ollama is not running or has no models. "
                    "Start Ollama or use 'Direct TTS' mode."
                )
            if model not in available:
                model = pick_model(available) or model

            set_job(job_id, progress=25, message=f"Writing podcast script with {model}...")
            script = ollama_generate(model, build_prompt(source_text))

            if len(script) < 80:
                raise ValueError(
                    "Ollama returned a very short response. "
                    "Try a different model or use Direct TTS mode."
                )

        set_job(job_id, progress=60, message="Synthesising voice audio...", script=script)

        audio_bytes = asyncio.run(generate_audio(script, voice, rate, pitch))

        audio_path = BASE_TEMP_DIR / f"{job_id}.mp3"
        audio_path.write_bytes(audio_bytes)

        set_job(
            job_id,
            status="done",
            progress=100,
            message="Podcast ready!",
            audio_path=str(audio_path),
            script=script,
        )

    except Exception as e:
        err_msg = str(e)
        print(f"[Job {job_id}] FAILED: {err_msg}")
        set_job(job_id, status="error", progress=0, message=f"Error: {err_msg}")


# ── Flask Routes ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/api/models")
def api_models():
    models = get_available_models()
    return jsonify({"models": models, "best": pick_model(models)})


@app.route("/api/voices")
def api_voices():
    return jsonify(VOICES)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 30:
        return jsonify({"error": "Text is too short (minimum 30 characters)."}), 400

    job_id = str(uuid.uuid4())
    set_job(job_id, status="queued", progress=0, message="Starting...")

    threading.Thread(
        target=run_job,
        daemon=True,
        args=(
            job_id,
            text,
            data.get("voice", "en-US-GuyNeural"),
            int(data.get("rate", 0)),
            int(data.get("pitch", 0)),
            data.get("model", ""),
            bool(data.get("direct_tts", False)),
        ),
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = JOBS.get(job_id, {})
    return jsonify(
        {
            "status": job.get("status"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "script": job.get("script"),
        }
    )


@app.route("/api/audio/<job_id>")
def api_audio(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("audio_path"):
        return jsonify({"error": "Audio not ready"}), 404
    path = Path(job["audio_path"])
    if not path.exists():
        return jsonify({"error": "Audio file missing"}), 404
    return send_file(str(path), mimetype="audio/mpeg", download_name="podcast.mp3")


# ── GUI ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Podcast Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Anybody:wght@300;400;600;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:       #080c10;
  --panel:    #0f1419;
  --card:     #141b24;
  --border:   #1e2d3d;
  --accent:   #00e5a0;
  --accent2:  #00aaff;
  --warn:     #ffb800;
  --danger:   #ff4d6d;
  --muted:    #4a6070;
  --text:     #cce0f0;
  --text2:    #7a9ab5;
  --r:        10px;
  --font:     'Anybody', sans-serif;
  --mono:     'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  min-height: 100dvh;
  padding: 28px 16px 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  background-image:
    radial-gradient(ellipse 60% 40% at 20% 0%, rgba(0,229,160,.06) 0%, transparent 60%),
    radial-gradient(ellipse 50% 30% at 80% 100%, rgba(0,170,255,.05) 0%, transparent 60%);
}

/* ── Header ── */
header {
  width: 100%; max-width: 760px;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 28px; padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.logo {
  display: flex; align-items: baseline; gap: 10px;
}
.logo h1 {
  font-size: clamp(1.6rem, 4vw, 2.4rem);
  font-weight: 800; letter-spacing: -.04em;
}
.logo h1 em { color: var(--accent); font-style: normal; }
.badge {
  font-family: var(--mono); font-size: .62rem;
  background: rgba(0,229,160,.12); color: var(--accent);
  border: 1px solid rgba(0,229,160,.25);
  padding: 3px 8px; border-radius: 99px; letter-spacing: .1em;
}

/* ── Cards ── */
.card {
  width: 100%; max-width: 760px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--r); padding: 20px; margin-bottom: 14px;
}
.card-title {
  font-family: var(--mono); font-size: .65rem;
  text-transform: uppercase; letter-spacing: .14em;
  color: var(--muted); margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.card-title::before {
  content: ''; display: inline-block;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent);
}

/* ── Textarea ── */
textarea {
  width: 100%; min-height: 190px; resize: vertical;
  background: var(--panel); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 14px; font-family: var(--mono);
  font-size: .88rem; line-height: 1.7; outline: none;
  transition: border-color .2s;
}
textarea:focus { border-color: var(--accent); }
.char-count {
  font-family: var(--mono); font-size: .65rem;
  color: var(--muted); text-align: right; margin-top: 6px;
}

/* ── Grid ── */
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media(max-width:520px) { .grid2 { grid-template-columns: 1fr; } }

/* ── Select ── */
.sel-wrap { position: relative; }
.sel-wrap::after {
  content: '▾'; position: absolute; right: 12px; top: 50%;
  transform: translateY(-50%); color: var(--accent2); pointer-events: none;
  font-size: .75rem;
}
select {
  width: 100%; background: var(--panel); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 11px 32px 11px 12px; font-family: var(--font);
  font-size: .88rem; outline: none; cursor: pointer;
  appearance: none; -webkit-appearance: none;
  transition: border-color .2s;
}
select:focus { border-color: var(--accent2); }

/* ── Sliders ── */
.slider-group { display: flex; flex-direction: column; gap: 10px; }
.slider-row {
  display: flex; align-items: center; gap: 10px;
}
.slider-label { font-family: var(--mono); font-size: .72rem; color: var(--text2); min-width: 46px; }
input[type=range] {
  flex: 1; height: 4px; border-radius: 99px;
  background: var(--border); outline: none; cursor: pointer;
  -webkit-appearance: none;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--bg);
  cursor: pointer;
}
input[type=range]::-webkit-slider-runnable-track { height: 4px; border-radius: 99px; }
.slider-val {
  font-family: var(--mono); font-size: .72rem;
  color: var(--accent2); min-width: 58px; text-align: right;
}

/* ── Toggle ── */
.toggle-row {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 0;
}
.toggle-row label { font-size: .88rem; color: var(--text2); cursor: pointer; flex: 1; }
.toggle-row label span { color: var(--text); font-weight: 600; }
.switch { position: relative; display: inline-block; width: 40px; height: 22px; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider-sw {
  position: absolute; inset: 0; background: var(--border);
  border-radius: 99px; cursor: pointer; transition: .25s;
}
.slider-sw::before {
  content: ''; position: absolute;
  width: 16px; height: 16px; left: 3px; top: 3px;
  background: var(--muted); border-radius: 50%; transition: .25s;
}
.switch input:checked + .slider-sw { background: rgba(0,229,160,.3); }
.switch input:checked + .slider-sw::before {
  transform: translateX(18px); background: var(--accent);
}

/* ── Tip box ── */
.tip {
  font-family: var(--mono); font-size: .7rem; color: var(--warn);
  background: rgba(255,184,0,.07); border: 1px solid rgba(255,184,0,.2);
  border-radius: 6px; padding: 8px 12px; margin-top: 10px;
  display: none;
}
.tip.show { display: block; }

/* ── Generate button ── */
.gen-btn {
  width: 100%; max-width: 760px;
  padding: 17px; margin-top: 8px;
  background: linear-gradient(135deg, var(--accent), #00c87a);
  color: #030a06; font-family: var(--font); font-size: 1.1rem;
  font-weight: 800; border: none; border-radius: var(--r);
  cursor: pointer; letter-spacing: .02em;
  transition: opacity .2s, transform .1s;
  box-shadow: 0 0 24px rgba(0,229,160,.2);
}
.gen-btn:hover { opacity: .9; }
.gen-btn:active { transform: scale(.99); }
.gen-btn:disabled { opacity: .35; cursor: not-allowed; transform: none; }

/* ── Progress ── */
.progress-wrap {
  width: 100%; max-width: 760px; margin: 12px 0 0;
}
.progress-bar-bg {
  height: 4px; background: var(--border); border-radius: 99px; overflow: hidden;
}
.progress-bar-fill {
  height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2));
  border-radius: 99px; transition: width .4s ease;
  width: 0%;
}
.status-text {
  font-family: var(--mono); font-size: .72rem; color: var(--muted);
  margin-top: 7px; min-height: 16px;
}
.status-text.err { color: var(--danger); }
.status-text.ok  { color: var(--accent); }

/* ── Script preview ── */
.script-box {
  margin-top: 14px; padding: 14px;
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; font-family: var(--mono);
  font-size: .78rem; line-height: 1.75; color: var(--text2);
  max-height: 200px; overflow-y: auto; display: none;
  white-space: pre-wrap;
}
.script-box.show { display: block; }

/* ── Player ── */
.player-card { display: none; }
.player-card.show { display: block; }
audio {
  width: 100%; border-radius: 8px; margin-top: 4px;
  filter: invert(1) hue-rotate(130deg) brightness(.85);
}
.dl-link {
  display: inline-flex; align-items: center; gap: 6px;
  margin-top: 12px; font-family: var(--mono); font-size: .75rem;
  color: var(--accent2); text-decoration: none;
}
.dl-link:hover { color: var(--accent); }
</style>
</head>
<body>

<header>
  <div class="logo">
    <h1>AI <em>Podcast</em> Studio</h1>
    <span class="badge">edge-tts</span>
  </div>
</header>

<!-- Text input -->
<div class="card">
  <div class="card-title">Source Text</div>
  <textarea id="text" placeholder="Paste an article, blog post, research paper, notes... anything you want turned into a podcast."></textarea>
  <div class="char-count" id="char-count">0 characters</div>
</div>

<!-- Mode toggle -->
<div class="card">
  <div class="card-title">Mode</div>
  <div class="toggle-row">
    <label for="direct-toggle">
      <span>Direct TTS</span> — read text as-is, no Ollama needed
    </label>
    <label class="switch">
      <input type="checkbox" id="direct-toggle" onchange="toggleMode()"/>
      <span class="slider-sw"></span>
    </label>
  </div>
  <div class="toggle-row" id="ollama-row">
    <label for="model-sel">Ollama model</label>
    <div class="sel-wrap" style="width:220px">
      <select id="model-sel"><option value="">Loading models…</option></select>
    </div>
  </div>
  <div class="tip" id="ollama-tip">
    ⚠ Ollama not detected. Enable "Direct TTS" or start Ollama with a model.
  </div>
</div>

<!-- Voice -->
<div class="card">
  <div class="card-title">Voice</div>
  <div class="sel-wrap">
    <select id="voice"></select>
  </div>
</div>

<!-- Speed / Pitch / Volume -->
<div class="card">
  <div class="card-title">Voice Settings</div>
  <div class="slider-group">

    <div class="slider-row">
      <span class="slider-label">Speed</span>
      <input type="range" id="rate" min="-50" max="100" value="0" step="5" oninput="updateSlider('rate','rate-val','%')">
      <span class="slider-val" id="rate-val">+0%</span>
    </div>

    <div class="slider-row">
      <span class="slider-label">Pitch</span>
      <input type="range" id="pitch" min="-50" max="50" value="0" step="5" oninput="updateSlider('pitch','pitch-val',' Hz')">
      <span class="slider-val" id="pitch-val">+0 Hz</span>
    </div>

    <div class="slider-row">
      <span class="slider-label">Volume</span>
      <input type="range" id="volume" min="0" max="100" value="90" step="5" oninput="updateVolume(this)">
      <span class="slider-val" id="volume-val">90%</span>
    </div>

  </div>
</div>

<!-- Progress -->
<div class="progress-wrap" id="progress-wrap" style="display:none">
  <div class="progress-bar-bg"><div class="progress-bar-fill" id="prog-fill"></div></div>
  <div class="status-text" id="status-text"></div>
</div>

<button class="gen-btn" onclick="generate()" id="gen-btn">⚡ Generate Podcast</button>

<!-- Player -->
<div class="card player-card" id="player-card" style="margin-top:18px">
  <div class="card-title">Your Podcast</div>
  <audio id="audio-el" controls></audio>
  <a class="dl-link" id="dl-link" href="#" download="podcast.mp3">⬇ Download MP3</a>

  <div id="script-label" style="font-family:var(--mono);font-size:.65rem;color:var(--muted);margin-top:16px;letter-spacing:.1em;text-transform:uppercase;display:none">
    Generated Script
  </div>
  <div class="script-box" id="script-box"></div>
</div>

<script>
/* ── Init ───────────────────────────────────────────── */
const $ = id => document.getElementById(id);

// Load voices
fetch('/api/voices').then(r=>r.json()).then(voices=>{
  const sel = $('voice');
  voices.forEach(v=>{
    const o = document.createElement('option');
    o.value = v.id; o.textContent = v.label; sel.appendChild(o);
  });
});

// Load Ollama models
fetch('/api/models').then(r=>r.json()).then(data=>{
  const sel = $('model-sel');
  sel.innerHTML = '';
  if (!data.models.length) {
    sel.innerHTML = '<option value="">No models found</option>';
    $('ollama-tip').classList.add('show');
    return;
  }
  data.models.forEach(m=>{
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    if (m === data.best) o.selected = true;
    sel.appendChild(o);
  });
});

// Char counter
$('text').addEventListener('input', function(){
  $('char-count').textContent = this.value.length.toLocaleString() + ' characters';
});

/* ── Slider helpers ─────────────────────────────────── */
function updateSlider(id, valId, unit) {
  const v = parseInt($(id).value);
  $(valId).textContent = (v >= 0 ? '+' : '') + v + unit;
}
function updateVolume(slider) {
  const v = parseInt(slider.value);
  $('volume-val').textContent = v + '%';
  const audio = $('audio-el');
  if (audio) audio.volume = v / 100;
}

/* ── Mode toggle ────────────────────────────────────── */
function toggleMode() {
  const direct = $('direct-toggle').checked;
  $('ollama-row').style.display = direct ? 'none' : 'flex';
}

/* ── Generate ───────────────────────────────────────── */
let pollTimer = null;

async function generate() {
  const text  = $('text').value.trim();
  const voice = $('voice').value;
  const rate  = parseInt($('rate').value);
  const pitch = parseInt($('pitch').value);
  const model = $('model-sel').value;
  const directTts = $('direct-toggle').checked;

  const btn = $('gen-btn');
  const pw  = $('progress-wrap');

  if (!text) { showStatus('Please paste some text first.', 'err'); return; }

  btn.disabled = true;
  $('player-card').classList.remove('show');
  pw.style.display = 'block';
  setProgress(5, 'Submitting job...');

  try {
    const res  = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text, voice, rate, pitch, model, direct_tts: directTts })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollStatus(data.job_id, btn), 1200);

  } catch(err) {
    showStatus('Error: ' + err.message, 'err');
    btn.disabled = false;
  }
}

async function pollStatus(jobId, btn) {
  try {
    const s = await fetch('/api/status/' + jobId).then(r=>r.json());
    setProgress(s.progress, s.message);

    if (s.status === 'done') {
      clearInterval(pollTimer);
      const url = '/api/audio/' + jobId;
      $('audio-el').src = url;
      $('audio-el').volume = parseInt($('volume').value) / 100;
      $('dl-link').href = url;
      $('player-card').classList.add('show');
      btn.disabled = false;
      showStatus('✓ Podcast ready!', 'ok');

      if (s.script) {
        $('script-box').textContent = s.script;
        $('script-box').classList.add('show');
        $('script-label').style.display = 'block';
      }

    } else if (s.status === 'error') {
      clearInterval(pollTimer);
      showStatus(s.message, 'err');
      btn.disabled = false;
    }
  } catch(e) {
    /* network blip — keep polling */
  }
}

function setProgress(pct, msg) {
  $('prog-fill').style.width = pct + '%';
  $('status-text').textContent = msg;
  $('status-text').className = 'status-text';
}

function showStatus(msg, cls) {
  $('status-text').textContent = msg;
  $('status-text').className = 'status-text ' + (cls || '');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"

    print("\n╔══════════════════════════════════════════╗")
    print("║         AI Podcast Studio v2             ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Local : http://localhost:5050           ║")
    print(f"║  LAN   : http://{local_ip}:5050      ║")
    print("╚══════════════════════════════════════════╝\n")
    print("Tip: If Ollama isn't running, enable 'Direct TTS' mode in the UI.\n")

    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
