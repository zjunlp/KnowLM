# vLLM Serving

[vLLM](https://github.com/vllm-project/vllm) implements a simple and efficient framework for accelerating LLM reasoning and services. Now, we integrated vLLM to improve the reasoning speed and service response speed of the KnowLM family of models.

## Setup

For the environment configuration of the vLLM, vist its official installation and configuration document. ([Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html))

In addition, the original LLaMA model and LoRA parameters need to be merged, you can execute the following command:
```shell
python tools/export_hf_checkpoint.py \
    --base_model data/zhixi-13b \
    --lora_model data/zhixi-13b-lora \
    --output_dir data/zhixi-13b-merged
```

## Launch service

Run the following command to start the vLLM api service. Control the maximum number of tokens allowed in the batch by setting `max_num_batched_tokens`. In addition, 'tensor-parallel-size' is the number of GPUs used by tensor parallel. If set to '--tensor-parallel-size 1', tensor pallel is not enabled and the model will reason on a single GPU.

```shell
max_num_batched_tokens=8000

CUDA_VISIBLE_DEVICES=1,2 python inference/launch_vllm.py \
    --port 8090 \
    --model data/zhixi-13B \
    --use-np-weights \
    --max-num-batched-tokens $max_num_batched_tokens \
    --dtype half \
    --tensor-parallel-size 2
```