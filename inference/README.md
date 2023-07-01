:speaking_head: \[ **中文** | [English](./README_EN.md) \]
# vLLM Serving

[vLLM](https://github.com/vllm-project/vllm) 实现了简单高效加速LLM推理和服务的框架。为此，我们集成vLLM来提高KnowLM系列模型的推理速度和服务响应速度。

## 配置

vLLM的环境配置可见其官方安装配置文档 ([Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html))。

另外，需要将原LLaMA模型参数和LoRA参数进行合并，可以执行下面的命令：
```shell
python tools/export_hf_checkpoint.py \
    --base_model data/zhixi-13b \
    --lora_model data/zhixi-13b-lora \
    --output_dir data/zhixi-13b-merged
```

## 启动服务

通过以下命令启动vLLM api服务。通过设置 `max_num_batched_tokens` 控制允许batch内最大的token数量；另外，`tensor-parallel-size` 为 tensor parallel 所使用的GPUs数量，若设置为`--tensor-parallel-size 1` 则不启用 tensor pallel，模型将在单卡上进行推理。

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