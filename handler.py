import base64, hashlib, io, os, re, sys, time, urllib.request

T0 = time.time()
sys.path.insert(0, os.environ.get("LONGCAT_CODE", "/app/longcat"))
import numpy as np, soundfile as sf, librosa, torch, torch.nn.functional as F
import runpod
import audiodit  # registers the model with transformers
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, approx_duration_from_text, normalize_volume

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/longcat")
WHISPER_DIR = os.environ.get("WHISPER_DIR", "/models/whisper")
VDIR = os.environ.get("VOICES_DIR", "/runpod-volume/voices")  # legacy network volume, only used for export
CACHE = "/tmp/voices"
MAX_REF = 28.0
# create_clones: a recording (up to CLONE_TOTAL s) is split into consecutive references of up to CLONE_LEN s,
# one voice clone each. Reference + output share LongCat's 60 s window (max_wav_duration), and similarity
# stops improving at ~20 s of reference, so 30 s leaves 30 s for the output. Under CLONE_MIN s a clone
# sounds clearly less like the speaker, so a shorter last piece is dropped (measured, see README).
CLONE_LEN, CLONE_TOTAL, CLONE_MIN = 30.0, 180.0, 8.0
PAUSE_SEARCH = 3.0   # look this far back from a window's end for a pause to cut at

model = AudioDiTModel.from_pretrained(MODEL_DIR, dtype=torch.float32).to("cuda")
model.vae.to_half()
model.eval()
tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
sr, hop, maxd = model.config.sampling_rate, model.config.latent_hop, model.config.max_wav_duration
print(f"model ready in {time.time() - T0:.1f}s", flush=True)

_whisper = None
def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_DIR, device="cuda", compute_type="float16")
    return _whisper

def safe_name(n):
    return re.sub(r"[^\w\- ]", "", n or "").strip()

def list_voices():
    if not os.path.isdir(VDIR):
        return []
    return sorted(d for d in os.listdir(VDIR) if os.path.exists(f"{VDIR}/{d}/ref.wav"))

def wav_b64(x):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()

def create_voice(name, audio_b64):
    name = safe_name(name) or "voice"
    audio, _ = librosa.load(io.BytesIO(base64.b64decode(audio_b64)), sr=sr, mono=True)
    audio = audio[: int(60 * sr)]
    # faster-whisper expects 16 kHz when given an array; timestamps are in seconds either way
    segs, _ = whisper().transcribe(librosa.resample(audio, orig_sr=sr, target_sr=16000), beam_size=5, vad_filter=True)
    segs = [s for s in segs if s.text.strip()]
    if not segs:
        return {"error": "No speech found in the file"}
    pieces, texts, total = [], [], 0.0
    for s in segs:  # whole sentences until ~28s
        a, b = max(0, s.start - 0.1), min(len(audio) / sr, s.end + 0.15)
        if total + (b - a) > MAX_REF:
            if pieces:
                break
            b = a + MAX_REF
        pieces += [audio[int(a * sr): int(b * sr)], np.zeros(int(0.15 * sr), np.float32)]
        texts.append(s.text.strip())
        total += b - a
    ref = np.concatenate(pieces)
    ref = ref / (np.abs(ref).max() + 1e-6) * 0.9
    # the Vercel proxy stores the voice (Vercel Blob); the worker keeps no state
    return {"name": name, "text": " ".join(texts), "ref_wav": wav_b64(ref)}

def split_windows(words, total, length=CLONE_LEN, min_len=CLONE_MIN, search=PAUSE_SEARCH):
    """Split timed words [(start, end, word)] into consecutive windows of whole words, each at most `length` s.
    A window ends in a pause: the latest gap of at least 0.2 s between two words in its last `search` seconds,
    else the longest gap there. The first window starts just before the first word; a last window shorter than
    `min_len` is dropped (unless it is the only one). Returns [(start, end, [words])]."""
    # Whisper's last words can have timestamps past the end of the audio or out of order: clamp and sort
    clean, last = [], 0.0
    for a, b, w in words:
        a = min(max(a, last), total)
        if a >= total - 0.05:
            break
        clean.append((a, min(max(b, a), total), w))
        last = a
    words = clean
    out, i = [], 0
    while i < len(words):
        start = max(0.0, words[i][0] - 0.15, out[-1][1] if out else 0.0)
        limit = min(total, start + length)
        j = i  # end (exclusive) of the words that fit
        if total <= start + length:
            j = len(words)  # the rest of the recording fits: one last window
        while j < len(words) and words[j][1] + 0.05 <= limit:
            j += 1
        if j == i:  # one "word" longer than the window (rare ASR artefact): take it, cut off
            j = i + 1
        elif j < len(words):  # more speech follows: cut in a pause near the end
            gaps = [(k, words[k][0] - words[k - 1][1]) for k in range(i + 1, j + 1) if words[k - 1][1] >= limit - search]
            if gaps:
                pauses = [g for g in gaps if g[1] >= 0.2]
                j = max(pauses)[0] if pauses else max(gaps, key=lambda g: g[1])[0]
        # end halfway into the pause (at most 0.25 s after the last word), or shortly after the last word
        tail = min(0.25, (words[j][0] - words[j - 1][1]) / 2) if j < len(words) else 0.3
        end = min(limit, total, words[j - 1][1] + max(0.0, tail))
        out.append((round(start, 2), round(end, 2), words[i:j]))
        i = j
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_len:
        out.pop()  # too short to sound like the speaker
    return out

def create_clones(audio_url, length=CLONE_LEN, total=CLONE_TOTAL):
    """Transcribe a recording once and split it into voice-clone references. Returns only the cut points
    and transcripts; the Vercel proxy cuts the references from the same file (it already has it)."""
    if not audio_url:
        return {"error": "No recording"}
    audio = librosa.load(fetch(audio_url), sr=16000, mono=True)[0][: int(total * 16000)]
    segs, _ = whisper().transcribe(audio, beam_size=5, vad_filter=True, word_timestamps=True)
    words = [(w.start, w.end, w.word) for s in segs for w in (s.words or []) if w.word.strip()]
    if not words:
        return {"error": "No speech found in the recording"}
    wins = split_windows(words, len(audio) / 16000, length)
    return {"clones": [{"start": a, "end": b, "text": "".join(w[2] for w in ws).strip()} for a, b, ws in wins]}

def fetch(url):
    # voice files are immutable blobs (random-suffixed names), so cache them for the worker's lifetime
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{hashlib.sha1(url.encode()).hexdigest()}"
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        open(path + ".part", "wb").write(data); os.replace(path + ".part", path)
    return path

def load_voice(inp):
    if inp.get("ref_url") and inp.get("text_url"):
        return librosa.load(fetch(inp["ref_url"]), sr=sr, mono=True)[0], open(fetch(inp["text_url"]), encoding="utf-8").read()
    return None, None

@torch.no_grad()
def generate(a, pt, text, steps=32):
    if a is None:
        return {"error": "Voice not found"}
    if not (text or "").strip():
        return {"error": "Empty text"}
    seed = int(np.random.randint(0, 2**31))
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    pt_n, tx_n = normalize_text(pt), normalize_text(text)
    inp = tok([f"{pt_n} {tx_n}"], padding="longest", return_tensors="pt")
    # same -6 dBFS peak normalisation the model was trained with (upstream utils.load_audio fix, 2026-10-15)
    pw = normalize_volume(torch.from_numpy(a).unsqueeze(0), -6)
    x = F.pad(pw, (0, (-pw.shape[-1]) % hop)); x = F.pad(x, (0, hop * 3))
    pdur = model.vae.encode(x.unsqueeze(0).cuda())[..., :-3].shape[-1]
    ptime = pdur * hop / sr
    d = approx_duration_from_text(tx_n, max_duration=maxd - ptime) * np.clip(ptime / approx_duration_from_text(pt_n, maxd), 1.0, 1.5)
    dur = min(int(d * sr // hop) + pdur, int(maxd * sr // hop))
    out = model(input_ids=inp.input_ids, attention_mask=inp.attention_mask, prompt_audio=pw.unsqueeze(0),
                duration=dur, steps=steps, cfg_strength=4.0, guidance_method="apg")
    return {"audio": wav_b64(out.waveform.squeeze().float().cpu().numpy())}

def handler(job):
    inp = job["input"]
    action = inp.get("action")
    if action == "warmup":
        return {"ok": True}
    if action == "generate":
        a, pt = load_voice(inp)
        return generate(a, pt, inp.get("text"))
    if action == "create_voice":
        return create_voice(inp.get("name"), inp.get("audio"))
    if action == "create_clones":
        out = create_clones(inp.get("audio_url"), min(max(float(inp.get("length") or CLONE_LEN), CLONE_MIN), CLONE_LEN))
        return out if "error" in out else {"name": safe_name(inp.get("name")) or "voice", **out}
    if action == "list_volume_voices":  # one-off migration from the old network volume
        return {"voices": list_voices()}
    if action == "export_voice":
        n = safe_name(inp.get("name"))
        if not os.path.exists(f"{VDIR}/{n}/ref.wav"):
            return {"error": f"Voice '{n}' not found"}
        return {"name": n, "text": open(f"{VDIR}/{n}/text.txt", encoding="utf-8").read(),
                "ref_wav": base64.b64encode(open(f"{VDIR}/{n}/ref.wav", "rb").read()).decode()}
    return {"error": f"unknown action {action}"}

def warmup():
    # the first generation after boot is ~3x slower (CUDA kernel loading); pay that before taking jobs
    t = time.time()
    noise = (np.random.randn(3 * sr) * 0.01).astype(np.float32)
    generate(noise, "Hello there.", "Hi.", steps=2)  # tiny: just loads the kernels
    print(f"warmup generation {time.time() - t:.1f}s, total boot {time.time() - T0:.1f}s", flush=True)

if __name__ == "__main__":
    if "--test_input" not in sys.argv:
        warmup()
    runpod.serverless.start({"handler": handler})
