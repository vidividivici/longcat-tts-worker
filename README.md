# longcat-tts-worker

RunPod serverless worker for [LongCat-AudioDiT 3.5B](https://github.com/meituan-longcat/LongCat-AudioDiT) zero-shot voice cloning.
Weights (fp16) and Whisper large-v3-turbo are baked into the image for fast cold starts. The worker is stateless: voices
(reference wav + transcript) live in Vercel Blob and are passed in as URLs.

Actions (`input.action`): `warmup`, `generate {text, ref_url, text_url}` → `{audio: base64 wav}`,
`create_voice {name, audio: base64 audio}` → `{name, text, ref_wav}` (Whisper transcribes the reference; the web proxy stores it),
`create_clones {name, audio_url}` → `{name, clones: [{start, end, text}]}`.

`create_clones` transcribes a recording (the first 180 s) once with word timestamps and splits it into consecutive
references of at most 30 s, one voice clone each; a last piece under 8 s is dropped. Each cut is placed in a pause
found in the audio itself (the latest quiet stretch of 0.12 s or more in the window's last 3 s, else its quietest 30 ms),
because Whisper's word times are often a few hundred ms off; words go to the window holding their middle. A word running
past 180 s is left out rather than cut (the browser sends a few seconds more for this). It returns only
the cut points and transcripts: the web proxy cuts the references from the same file. Why 30 s: the reference and the
output share LongCat's 60 s window (`max_wav_duration` = 60 in the model config; `generate()` gives the output
`60 s - reference`), so a 30 s reference leaves 30 s of speech per clip. Measured on Falk's recordings (Whisper WER,
ECAPA similarity to a separate take): similarity 0.63-0.70 at 3 s, 0.73-0.78 at 5 s, 0.81-0.87 at 10 s, a plateau of
0.85-0.88 from 15-20 s, no gain at 25-40 s (only slower: 21 s per clip at 30 s), and a squeezed output from 50 s.

The reference is peak-normalised to -6 dBFS before encoding, matching LongCat's training data (upstream fix of 2026-10-15).

Image: `ghcr.io/vidividivici/longcat-tts-worker` — `:latest` is built from main (the live endpoint); every push also gets a `:<sha>` tag,
and pushes to other branches get only the `:<sha>` tag, so a change can be tested on a temporary endpoint before it reaches `:latest`.
