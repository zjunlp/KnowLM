:speaking_head: \[ **中文** | [English](./README_EN.md) \]
# LoRA指令微调
## 环境配置
使用下面的命令配置环境：
```shell
conda create -n lora python=3.9 -y
conda activate lora
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## 运行
我们的代码基于[alpaca-lora](https://github.com/tloen/alpaca-lora)进行修改，仅修改了训练的超参数。我们在一个Node上（8张32GB的V100显卡）进行训练。所有的训练超参数都在已经在训练代码中体现。请根据自己的硬件情况修改训练参数，包括`warmup_steps` `micro_batch_size`等参数。使用下面的命令开始训练：
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch finetune.py --data_path /your/data/path --base_model /your/path/to/cama
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
