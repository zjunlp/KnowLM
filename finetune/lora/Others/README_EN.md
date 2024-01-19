:speaking_head: \[ [中文](./README.md) | **English** \]

# Instruction tuning (LoRA)
## Environment Configuration
Configure the environment with the following commands:
```shell
conda create -n lora python=3.9 -y
conda activate lora
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
For `Qwen`, `flash attention 2` is supported. We suggest you to install `flash-attention` library：
```shell
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install .
```
## Download Models
We have provided a simple script, `download_script.py`, to download models from [Hugging Face](https://huggingface.co/) to your local machine.

**Note that we are currently only considering fine-tuned models (such as `Qwen-7B`, `ChatGLM3-6B-Base`) to have the ability for single-turn dialogue. The scenario of multi-turn dialogue is not yet being considered**.
## Run
Our code is based on [alpaca-lora](https://github.com/tloen/alpaca-lora) with modifications made only to the training hyperparameters. We trained the model on a single node with 8 V100(32GB) GPUs. All the training hyperparameters are already reflected in the training code. Please modify the training parameters, including `warmup_steps`, `micro_batch_size`, etc., according to your hardware setup. 
An example for `Qwen`: 
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch finetune.py \
    --data_path './data/alpaca_data_cleaned.json' \
    --base_model './Qwen-7B' \
    --lora_target_modules '["c_attn", "c_proj", "w1", "w2"]' \
    --prompt_template_name "qwen" \
    --eval_steps 50 \
    --save_steps 50
```
An exmaple for `ChatGLM3`:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch finetune.py \
    --data_path './data/alpaca_data_cleaned.json' \
    --base_model './chatglm3-6b-base' \
    --lora_target_modules '["query_key_value"]' \
    --prompt_template_name "chatglm3" \
    --eval_steps 50 \
    --save_steps 50
```
If you want to modify the training hyperparameters in the command line, you can simply add the corresponding parameters after the command.
For instructions on how to obtain , please refer to this [link]([../README_EN.md](https://github.com/zjunlp/KnowLLM/blob/main/README_EN.md#22-pretraining-model-weight-acquisition-and-restoration)). Regarding the data format for training, please refer to [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json). Here is a simple example:
```python
[
  {"instruction": "", "input":"", "output":""},
  {"instruction": "", "input":"", "output":""},
  ...
]
```
