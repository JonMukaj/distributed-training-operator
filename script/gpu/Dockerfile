# Training image for the instrumented finetune.py (thesis Ch 4/6) and pretokenize.py.
#
# Runtime base already contains torch 2.4.1 + CUDA 12.1 + cuDNN 9. No devel
# toolchain and no second torch install: the previous devel-based image was
# ~12 GB compressed and a fresh node could not download + unpack its largest
# layers inside the kubelet's per-request timeout ("context canceled" pulls).
# Every job runs on a freshly created node with an empty image cache, and the
# pull time lands in the PyTorchJob's wall-clock training time, so size matters.
#
# Build & push (from script/gpu/):
#   docker build -t jmukaj/gpu-fine-tuner-hf:<tag> .
#   docker push jmukaj/gpu-fine-tuner-hf:<tag>
# then set spec.pytorchSpec.image in operator/config/samples/*hyperstack_gpu.yaml.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/scripts

# torch is pinned by the base image; install everything else from requirements.txt.
COPY requirements.txt .
RUN grep -vE '^\s*torch==' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && python -c "import torch, transformers, datasets, peft, accelerate; print('torch', torch.__version__, 'transformers', transformers.__version__)"

# The operator launches /workspace/scripts/finetune.py (spec.model.trainingScript).
COPY finetune.py pretokenize.py ./

CMD ["/bin/bash"]
