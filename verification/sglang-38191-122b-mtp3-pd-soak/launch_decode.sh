#!/bin/bash
source <HOME>/venv-main/bin/activate
export LD_LIBRARY_PATH=<HOME>/cudart12lib:/usr/local/cuda-13.0/compat
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export PYTHONPATH=<HOME>/sglang-newmain/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true
exec python -m sglang.launch_server \
  --model-path <pfs>/models/Qwen3.5-122B-A10B \
  --tp 8 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  --disaggregation-bootstrap-port 8998 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 --port 30002 \
  --mem-fraction-static 0.85
