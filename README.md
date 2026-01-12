### STEP: Step-level Trace Evaluation and Pruning
This is the official code implementation of paper: *Hidden States as Early Signals: Step-level Trace Evaluation and Pruning for Efficient Test-Time Scaling*

STEP is an efficient parallel thinking framework built upon vLLM, which evaluates reasoning steps using hidden states and dynamically prunes unpromising traces during generation. It supports various types of reasoning tasks, including math and science.

## Features
- **Step-level trace scoring**: STEP trains a lightweight step-level scorer based on hidden states to estimate reasoning trace quality online.
- **Memory-aware pruning**: STEP prunes low-quality traces only when KV-cache saturates GPU memory, directly reducing waiting time and end-to-end latency.
- **Effective efficiency–accuracy**: STEP reduces end-to-end inference latency by **45%–70%** compared to self-consistency while consistently improving reasoning accuracy by **2%-8%** across challenging benchmarks.


## Install 
This repo is built based on vLLM 0.11.1. 

```bash
# Install the official vLLM wheel once to pull in the correct PyTorch and CUDA dependencies.
uv pip install vllm --torch-backend=auto
# Install STEP in editable mode.
pip install -e . --no-build-isolation
```

For a more detailed instruction, please refer to [vLLM v0.11.1 document](https://docs.vllm.ai/en/v0.11.1/). 
## Quick Start 

``` python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
GPU_util = 0.9
step_scorer_path = "PATH_OF_STEP_SCORER"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
sampling_params = SamplingParams(
    n=64,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=600000,
)
llm = LLM(
    model=model,
    tensor_parallel_size=1,
    gpu_memory_utilization=GPU_util,
    STEP_enable=True,
    STEP_step_scorer_path=step_scorer_path,
    disable_log_stats=False,
    )

```

We currently provide step scorer for Qwen3-4B, DeepSeek-R1-0528-Qwen3-8B, Phi-4-reasoning-plus (15B)


## Evaluation
We provide a script at [benchmark_eval.py](STEP/tests/benchmark_eval.py) to evaluate HMMT-25, HMMT-24, GPQA-Diamond, AIME. Here is an example to evaluate DeepSeek-R1-0528-Qwen3-8B on HMMT-25.
```bash
python STEP/tests/benchmark_eval.py \
  --benchmark STEP/datasets/hmmt_2025.jsonl \
  --output-dir STEP/eval_result \
  --model-path DeepSeek-R1-0528-Qwen3-8B \
  --STEP-step-scorer-path STEP/step_scorer/Qwen3-4B_step_scorer.pt
```

## Citation 


