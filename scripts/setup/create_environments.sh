#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

create_vllm() {
  conda create -n vllm-env python=3.12 -y
  conda activate vllm-env
  python -m pip install -U pip setuptools wheel
  python -m pip install "vllm==0.25.1"
  conda deactivate
}

create_stirrup() {
  conda create -n stirrup-py312 python=3.12 -y
  conda activate stirrup-py312
  python -m pip install -U pip setuptools wheel
  python -m pip install \
    "stirrup==0.1.12" "openai==2.46.0" "transformers==5.14.1" \
    datasets pandas pyarrow openpyxl xlsxwriter numpy \
    python-docx reportlab pypdf pymupdf "pdfplumber<0.11.10" python-pptx \
    requests beautifulsoup4 lxml matplotlib moviepy pydub librosa soundfile \
    opencv-python-headless pyyaml nbformat nbconvert psd-tools markdown tabulate \
    filetype rich tqdm jsonlines pydantic jinja2 \
    "click>=8.4.0,<9" "pillow>=11.3,<12"
  conda install -c conda-forge ffmpeg -y
  python -m pip check
  conda deactivate
}

create_routing() {
  conda create -n routing-hf-py312 python=3.12 -y
  conda activate routing-hf-py312
  python -m pip install -U pip setuptools wheel
  python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
  python -m pip install \
    "transformers==5.14.1" accelerate safetensors sentencepiece protobuf \
    huggingface_hub einops numpy pandas pyarrow tqdm jsonlines rich
  python -m pip install -e .
  conda deactivate
}

for env in vllm-env stirrup-py312 routing-hf-py312; do
  if conda env list | awk '{print $1}' | grep -qx "$env"; then
    echo "[skip] conda environment already exists: $env"
  else
    case "$env" in
      vllm-env) create_vllm ;;
      stirrup-py312) create_stirrup ;;
      routing-hf-py312) create_routing ;;
    esac
  fi
done

echo "[done] environments created"
