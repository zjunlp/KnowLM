:speaking_head: \[ [中文](./README.md) | **English** \]

# Instruction tuning (LoRA)
## Environment Configuration
Configure the environment with the following commands:
```shell
conda create -n lora python=3.9 -y
conda activate lora
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## Run
Our code is based on [alpaca-lora](https://github.com/tloen/alpaca-lora) with modifications made only to the training hyperparameters. We trained the model on a single node with 8 V100(32GB) GPUs. All the training hyperparameters are already reflected in the training code. Please modify the training parameters, including `warmup_steps`, `micro_batch_size`, etc., according to your hardware setup. To start the training, use the following command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch finetune.py --data_path /your/data/path --base_model /your/path/to/zhixi
```
If you want to modify the training hyperparameters in the command line, you can simply add the corresponding parameters after the command.
For instructions on how to obtain and restore the weights of `ZhiXi-13B`, please refer to this [link]([../README_EN.md](https://github.com/zjunlp/KnowLLM/blob/main/README_EN.md#22-pretraining-model-weight-acquisition-and-restoration)). Regarding the data format for training, please refer to [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json). Here is a simple example:
```python
[
  {"instruction": "", "input":"", "output":""},
  {"instruction": "", "input":"", "output":""},
  ...
]
```
