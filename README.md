# longcat-tts-worker

RunPod serverless worker for [LongCat-AudioDiT 3.5B](https://github.com/meituan-longcat/LongCat-AudioDiT) zero-shot voice cloning.
Weights (fp16) and Whisper large-v3-turbo are baked into the image for fast cold starts. The worker is stateless: voices
(reference wav + transcript) live in Vercel Blob and are passed in as URLs.

Actions (`input.action`): `warmup`, `generate {text, ref_url, text_url}` → `{audio: base64 wav}`,
`create_voice {name, audio: base64 audio}` → `{name, text, ref_wav}` (Whisper transcribes the reference; the web proxy stores it).

The reference is peak-normalised to -6 dBFS before encoding, matching LongCat's training data (upstream fix of 2026-10-15).

Image: `ghcr.io/vidividivici/longcat-tts-worker` — `:latest` is built from main (the live endpoint); every push also gets a `:<sha>` tag,
and pushes to other branches get only the `:<sha>` tag, so a change can be tested on a temporary endpoint before it reaches `:latest`.
