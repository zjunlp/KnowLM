:speaking_head: \[ **中文** | [English](./README_EN.md) \]
# LoRA 指令微调
## 环境配置
使用下面的命令配置环境：
```shell
conda create -n lora python=3.9 -y
conda activate lora
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
对于 `Qwen`，支持 `flash attention 2`，建议安装 `flash-attention` 库：
```shell
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install .
```
**注意：我们目前仅考虑微调后的模型（如 `Qwen-7B`，`ChatGLM-6B-Base`）具备单轮对话的能力，多轮对话的情形目前还没考虑**。
## 运行
我们的代码基于 [alpaca-lora](https://github.com/tloen/alpaca-lora) 进行修改，仅修改了训练的超参数。我们在一个 Node 上（8张 32 GB的 V100 显卡）进行训练。所有的训练超参数都在已经在训练代码中体现。请根据自己的硬件情况修改训练参数，包括 `warmup_steps` `micro_batch_size` 等参数。
`Qwen` 训练示例：
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch finetune.py \
    --data_path './data/alpaca_data_cleaned.json' \
    --base_model './Qwen-7B' \
    --lora_target_modules '["c_attn", "c_proj", "w1", "w2"]' \
    --prompt_template_name "qwen" \
    --eval_steps 50 \
    --save_steps 50
```
`ChatGLM3` 训练示例：
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch finetune.py \
    --data_path './data/alpaca_data_cleaned.json' \
    --base_model './chatglm3-6b-base' \
    --lora_target_modules '["query_key_value"]' \
    --prompt_template_name "chatglm3" \
    --eval_steps 50 \
    --save_steps 50
```
如果希望在命令行中修改训练超参数，可以直接在命令后添加相应的参数即可。
关于如何获取和复原ZhiXi-13B的权重，请参考[此处](https://github.com/zjunlp/KnowLLM/tree/main#22-%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E8%8E%B7%E5%8F%96%E4%B8%8E%E6%81%A2%E5%A4%8D)。关于训练的数据格式，请参考[alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json)，下面是一个简单的例子：
```python
[
  {"instruction": "", "input":"", "output":""},
  {"instruction": "", "input":"", "output":""},
  ...
]
```