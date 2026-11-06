FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/models/hf HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers==5.18.0 librosa soundfile einops safetensors faster-whisper runpod huggingface_hub hf_transfer
# make pip-installed CUDA libs (cuDNN/cuBLAS) visible to ctranslate2 / faster-whisper
RUN python -c "import os, nvidia; r = list(nvidia.__path__)[0]; print('\n'.join(os.path.join(r, d, 'lib') for d in os.listdir(r) if os.path.isdir(os.path.join(r, d, 'lib'))))" > /etc/ld.so.conf.d/nvidia-pip.conf || true; ldconfig

COPY prepare.py /app/prepare.py
RUN python /app/prepare.py /models/longcat /tmp/hf_src

# model code after the weights layer, so bumping the pin does not rebuild/re-pull the weights
RUN git clone https://github.com/meituan-longcat/LongCat-AudioDiT /app/longcat && cd /app/longcat && git checkout 183bddedf1fd106431f0a36cf45f7be2d9b6d751 && rm -rf .git assets *.pdf

COPY handler.py /app/handler.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "-u", "/app/handler.py"]
