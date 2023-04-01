<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/cama_logo.jpeg" alt="ZJU-CaMA" style="width: 30%; min-width: 30px; display: block; margin: auto;"></a>
</p>

# CaMA: A Chinese-English Bilingual LLaMA Model

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/zjunlp/cama/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/zjunlp/cama/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Welcome to the CaMA project repository! Our goal is to develop and distribute a Chinese-English bilingual LLaMA model. Our research is grateful to the community for providing feedback on CaMA and supporting us. Our model and live demo will be suspended until further notice.

**This project is under fast development (due to the limted GPU, it will take a long time for pre-training) and the full code & checkpoints will be released soon.**

Features of CaMA:

1) ***unsupervised further pre-training on Chinese-English bilingual data.***

2) ***multiple IPT versions including full fine-tuning and parameter-efficient fine-tuning (e.g., Lora).***

**Usage and License Notices**: Please note that CaMA is exclusively licensed for research purposes. The accompanying dataset is licensed under CC BY NC 4.0, which permits solely non-commercial usage. We strongly advise against employing models trained with this dataset for any purposes other than research.

## Overview

We performed unsupervised further pre-training on Chinese-English bilingual data (including the **latest Wikidata**) based on the LLaMA-13B weights. The data used included various Chinese Baidu Encyclopedia, Wudao corpora, and Wikipedia data. We trained for 4,000 steps on 32* V100 32G GPUs with a learning rate of 2e-5 and a batch size of 384, resulting in CaMA-13B. Our models have stronger Chinese language understanding and are updated continuously to incorporate recent knowledge.

On the CaMA-13B pre-trained weights, we further instruct-tuned  using a collection of existing self-instruction datasets in Chinese and English. And the dataset and preprocessing code were organized in the Chinese-English-instruction-dataset. The learning rate was set to 2e-5, and we trained for 2000 steps, resulting in Chat-CaMA. As for Chat-CaMA, we will provide multiple versions, including full fine-tuning and parameter-efficient fine-tuning (e.g., as Lora), for the community to use.

Currently, CaMA is still in its developmental phase, and there are several limitations that need to be addressed. One of the most crucial limitations is the fact that we have not yet fine-tuned the CaMA model to ensure safety and harmlessness. Thus, we highly recommend users to exercise caution while interacting with CaMA and report any questionable behavior to assist us in enhancing the safety and ethical considerations of the model.

As of now, we will release the data generation process, dataset, and training recipe, but we are yet to release the model weights, pending approval from the CaMA creators. However, we will release a live demo to help users better understand the scope and limitations of Alpaca, as well as to enable us to assess the model's performance on a broader audience.

## Train CaMA-13B on DeepSpeed Zero-2/3:

> We conduct pre-training using deepspeed. Please note that when using ZeRO2, we use version `deepspeed==0.7.3`. If you use the latest version `0.8.3`, an `overflow` issue may occur. Our code supports checkpointing for resuming training.

### ZeRO2

> On 8 V100 GPUs, llama-13B with a maximum sequence length of 1024, the batch size per card can reach 3.

To pretrain using `ZeRO2`, use the following command. Set the `--model_size` parameter to 13 for `Llama-13B`, and 7 for `Llama-7B`.

`--global_batch_distributed` is for training sampling. In the following example, its value is `[10,6]`, which means the `global batch size` is 16 (make sure that the product of `batch_per_gpu` and `num_gpu` is equal to the `global batch size`). Since Chinese and English data are distinguished in version `1.2`, the value of `[10,6]` represents 10 samples for Chinese data and 6 samples for English data.

--pretrained_path it denotes the path specified in `pretrained_path` for initialization.

`--save_steps` represents how many steps to save at a time.

`--accumulate` represents the maximum number of saved checkpoints. If it exceeds the limit, the oldest checkpoint will be overwritten. If there is no limit, set it to `-1`.

`--data_path_with_prefix` represents the path and file prefix of the preprocessed data.

`--save_path` represents the location to save the output."

```shell
deepspeed --num_nodes=1 --num_gpus=8 train.py --deepspeed --deepspeed_config zero2.json \
    --seq_len=1024 --model_size=13 --batch_size_per_gpu=2 --global_batch_distributed=[10,6] \
    --data_path_with_prefix="data/data" --save_steps=1000 \
    --accumulate=1 --save_path="./checkpoint/" \
    --pretrained_path="/llama/converted_llama13"
```

### ZeRO3

> On 8 V100 GPUs, LLaMA-13B with a maximum sequence length of 1024, the batch size per card can reach 32.

Change `zero2.json` to `zero3.json` in the `--deepspeed_config` parameter to use it.

### Authors

All authors below contributed equally and the order is determined by random draw.

- Xiang Chen
- Jintian Zhang
- Honghao Gui
- Zhen Bi
- Shengyu Mao
- Xiaohan Wang
- Jing Chen
- **Ningyu Zhang**
- Huajun Chen

### Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{cama,
  author = {Xiang Chen, Jintian Zhang, Honghao Gui, Zhen Bi, Shengyu Mao, Xiaohan Wang, Jing Chen Ningyu Zhang, Huajun Chen},
  title = {CaMA: A Chinese-English Bilingual LLaMA Model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/}},
}
```

### Acknowledgements

We thanks following multiple wonderful  open source projects for their help in providing pipeline code:

[Meta AI LLaMA](https://arxiv.org/abs/2302.13971v1)

[Huggingface Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

[Vicuna](https://vicuna.lmsys.org/)

[Llama-X](https://github.com/AetherCortex/Llama-X)
