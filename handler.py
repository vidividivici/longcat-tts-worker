import base64, io, os, re, shutil, sys, time

T0 = time.time()
sys.path.insert(0, os.environ.get("LONGCAT_CODE", "/app/longcat"))
import numpy as np, soundfile as sf, librosa, torch, torch.nn.functional as F
import runpod
import audiodit  # registers the model with transformers
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, approx_duration_from_text

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/longcat")
WHISPER_DIR = os.environ.get("WHISPER_DIR", "/models/whisper")
VDIR = os.environ.get("VOICES_DIR", "/runpod-volume/voices")
MAX_REF = 28.0

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
    os.makedirs(f"{VDIR}/{name}", exist_ok=True)
    sf.write(f"{VDIR}/{name}/ref.wav", ref, sr)
    open(f"{VDIR}/{name}/text.txt", "w").write(" ".join(texts))
    return {"name": name, "text": " ".join(texts), "voices": list_voices()}

@torch.no_grad()
def generate(voice, text):
    voice = safe_name(voice)
    if not os.path.exists(f"{VDIR}/{voice}/ref.wav"):
        return {"error": f"Voice '{voice}' not found"}
    if not (text or "").strip():
        return {"error": "Empty text"}
    a, _ = librosa.load(f"{VDIR}/{voice}/ref.wav", sr=sr, mono=True)
    pt = open(f"{VDIR}/{voice}/text.txt").read()
    seed = int(np.random.randint(0, 2**31))
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    pt_n, tx_n = normalize_text(pt), normalize_text(text)
    inp = tok([f"{pt_n} {tx_n}"], padding="longest", return_tensors="pt")
    pw = torch.from_numpy(a).unsqueeze(0)
    x = F.pad(pw, (0, (-pw.shape[-1]) % hop)); x = F.pad(x, (0, hop * 3))
    pdur = model.vae.encode(x.unsqueeze(0).cuda())[..., :-3].shape[-1]
    ptime = pdur * hop / sr
    d = approx_duration_from_text(tx_n, max_duration=maxd - ptime) * np.clip(ptime / approx_duration_from_text(pt_n, maxd), 1.0, 1.5)
    dur = min(int(d * sr // hop) + pdur, int(maxd * sr // hop))
    out = model(input_ids=inp.input_ids, attention_mask=inp.attention_mask, prompt_audio=pw.unsqueeze(0),
                duration=dur, steps=32, cfg_strength=4.0, guidance_method="apg")
    return {"audio": wav_b64(out.waveform.squeeze().float().cpu().numpy())}

def handler(job):
    inp = job["input"]
    action = inp.get("action")
    if action == "warmup" or action == "list_voices":
        return {"voices": list_voices()}
    if action == "generate":
        return generate(inp.get("voice"), inp.get("text"))
    if action == "create_voice":
        return create_voice(inp.get("name"), inp.get("audio"))
    if action == "delete_voice":
        n = safe_name(inp.get("name"))
        if n and os.path.isdir(f"{VDIR}/{n}"):
            shutil.rmtree(f"{VDIR}/{n}")
        return {"voices": list_voices()}
    return {"error": f"unknown action {action}"}

def warmup():
    # the first generation after boot is ~3x slower (CUDA kernel loading); pay that before taking jobs
    v = list_voices()
    if v:
        t = time.time()
        generate(v[0], "Warming up.")
        print(f"warmup generation {time.time() - t:.1f}s, total boot {time.time() - T0:.1f}s", flush=True)

if __name__ == "__main__":
    if "--test_input" not in sys.argv:
        warmup()
    runpod.serverless.start({"handler": handler})
