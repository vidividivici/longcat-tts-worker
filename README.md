# longcat-tts-worker

RunPod serverless worker for [LongCat-AudioDiT 3.5B](https://github.com/meituan-longcat/LongCat-AudioDiT) zero-shot voice cloning.
Weights (fp16) and Whisper large-v3-turbo are baked into the image for fast cold starts. Voices live on a RunPod network volume (`/runpod-volume/voices/<name>/{ref.wav,text.txt}`).

Actions (`input.action`): `warmup` / `list_voices`, `generate {voice, text}` → `{audio: base64 wav}`, `create_voice {name, audio: base64 audio}`, `delete_voice {name}`.

Image: `ghcr.io/vidividivici/longcat-tts-worker:latest` (built by GitHub Actions on push).
