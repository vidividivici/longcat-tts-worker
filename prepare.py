"""Build-time: download LongCat-AudioDiT-3.5B and store the weights as sharded fp16 safetensors
(half the size -> faster cold start, low build RAM). Also pre-downloads the umt5 tokenizer and the
Whisper model so the worker never touches the network at runtime."""
import json, os, shutil, sys
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

SRC_REPO = "meituan-longcat/LongCat-AudioDiT-3.5B"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/models/longcat"
TMP = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hf_src"
SHARD_BYTES = 2 * 1024**3

src = snapshot_download(SRC_REPO, cache_dir=TMP, allow_patterns=["*.json", "*.safetensors"])
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(src, "config.json"), os.path.join(OUT, "config.json"))

shards, cur, cur_bytes, kept = [], {}, 0, 0
with safe_open(os.path.join(src, "model.safetensors"), "pt") as sf:
    for k in sf.keys():
        t = sf.get_tensor(k)
        if t.dtype == torch.float32 and t.abs().max().item() < 60000:
            t = t.half()
        else:
            kept += 1
        cur[k] = t.contiguous()
        cur_bytes += t.numel() * t.element_size()
        if cur_bytes >= SHARD_BYTES:
            shards.append({k2: v.numel() * v.element_size() for k2, v in cur.items()})
            save_file(cur, os.path.join(OUT, f"part{len(shards)}.tmp"), metadata={"format": "pt"})
            cur, cur_bytes = {}, 0
    if cur:
        shards.append({k2: v.numel() * v.element_size() for k2, v in cur.items()})
        save_file(cur, os.path.join(OUT, f"part{len(shards)}.tmp"), metadata={"format": "pt"})

n = len(shards)
weight_map, total = {}, 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    os.rename(os.path.join(OUT, f"part{i}.tmp"), os.path.join(OUT, fname))
    for k, nbytes in shard.items():
        weight_map[k] = fname
        total += nbytes
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print(f"wrote {n} shards, {total / 1e9:.2f} GB, {kept} tensors kept in original dtype")
shutil.rmtree(TMP, ignore_errors=True)

if "--skip-extras" not in sys.argv:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(json.load(open(os.path.join(OUT, "config.json")))["text_encoder_model"])
    from faster_whisper import download_model
    download_model("large-v3-turbo", output_dir="/models/whisper")
