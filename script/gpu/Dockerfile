# Training image for the instrumented finetune.py and pretokenize.py.

# Build & push (from script/gpu/):
#   docker build -t jmukaj/gpu-fine-tuner-hf:<tag> .
#   docker push jmukaj/gpu-fine-tuner-hf:<tag>
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install --no-deps \
      filelock==3.16.1 fsspec==2024.10.0 Jinja2==3.1.4 MarkupSafe==3.0.2 \
      networkx==3.4.2 sympy==1.13.3 mpmath==1.3.0 typing_extensions==4.12.2 \
      setuptools==75.6.0

RUN pip install --no-deps nvidia-cudnn-cu12==9.1.0.70
RUN pip install --no-deps nvidia-cublas-cu12==12.1.3.1 nvidia-cusparse-cu12==12.1.0.106 nvidia-nvjitlink-cu12==12.9.86
RUN pip install --no-deps nvidia-cusolver-cu12==11.4.5.107 nvidia-cufft-cu12==11.0.2.54 \
      nvidia-curand-cu12==10.3.2.106 nvidia-cuda-cupti-cu12==12.1.105 \
      nvidia-cuda-runtime-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-nvtx-cu12==12.1.105
RUN pip install --no-deps nvidia-nccl-cu12==2.20.5 triton==3.0.0
RUN pip install --no-deps torch==2.4.1

WORKDIR /workspace/scripts
COPY requirements.txt .
RUN grep -vE '^\s*torch==' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && python -c "import torch, transformers, datasets, peft, accelerate; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__)" \
    && torchrun --help >/dev/null

# The operator launches /workspace/scripts/finetune.py (spec.model.trainingScript).
COPY finetune.py pretokenize.py ./

CMD ["/bin/bash"]
