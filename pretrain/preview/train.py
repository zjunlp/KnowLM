#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import os

from torch.utils.data import DataLoader
from typing import List
import math
from dataloader import MyDataset, dataloader_builder, sampler_builder
import time

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """filed的作用：如果有新值就会被替换掉，"""
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # resume_path: str = field(default=None, metadata={"help": "断点续训的路径"})
    resume_epoch: int = field(default=-1, metadata={"help": "-1表示不进行跳过数据，若为0，则表示第0个epoch还没训练完，接下来仍然从第0个epoch开始，即用epoch-0来生成sampler的iter"})
    resume_global_data: int = field(default=0, metadata={"help": "表示要跳过多少个数据，注意，此处可能需要手动计算。此外，请确保"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    global_batch_distributed: str = field(
        default=None,
        metadata={"help": "会调用eval将其转换成list，list的个数为不同数据集的个数，list的和必须和global batch size保持一致"}
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


@dataclass
class MyDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        """batch_first: 默认batch在第一个维度，padding_value:不够的填充为什么"""
        """input_ids:list，以里面最多的为基准"""
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        """label也是"""
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print("mike:test:labels",labels.shape)
        torch.cuda.empty_cache()
        return dict(
            input_ids=input_ids,
            labels=labels,
            # attention_mask=input_ids.ne(self.pad_token_id),
        )


def _make_supervised_data_module(data_prefix, seq_length=1024, pad_id=0) -> Dict:
    train_dataset = MyDataset(data_prefix=data_prefix, seq_length=seq_length, pad_id=pad_id)
    data_collator = MyDataCollatorForSupervisedDataset(pad_token_id=0)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class MyTrainer(Trainer):
    RESUME = False
    RESUME_EPOCH = -1
    RESUME_DATA = 0
    def get_train_dataloader(self) -> DataLoader:
        sampler = sampler_builder(self.args, self.train_dataset)
        if MyTrainer.RESUME:
            print(f"[{time.ctime()}] 启用了断点续训，正在进行数据跳过......")
            assert MyTrainer.RESUME_EPOCH >= 0
            sampler.set_epoch(MyTrainer.RESUME_EPOCH)
            sampler.jump(MyTrainer.RESUME_DATA)
        else:
            sampler.set_epoch(0)
        print("mike:collate:", self.data_collator)
        dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=self.args.per_device_train_batch_size,
            pin_memory=True,
            num_workers=0
            # batch_size=sum(eval(self.args.global_batch_distributed))
        )
        print("mike:dataloader:", len(dataloader))
        return dataloader

def train():
    """transformers.HfArgumentParser，传入的类必须有@dataclass进行装饰，这样这些参数会被自动添加到命令中"""
    """https://zhuanlan.zhihu.com/p/296535876?utm_id=0"""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    """命令行传过来的参数"""
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    resume_from_checkpoint = True if model_args.resume_epoch >=0 else False
    MyTrainer.RESUME = resume_from_checkpoint
    MyTrainer.RESUME_DATA = model_args.resume_global_data
    MyTrainer.RESUME_EPOCH = model_args.resume_epoch

    """实例化模型"""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    data_module = _make_supervised_data_module(data_prefix=data_args.data_path,
                                               seq_length=training_args.model_max_length)
    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = MyTrainer(model=model, args=training_args, **data_module)
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    """不走Trainer提供的断点续训"""
    """此处的断点续训仅加载模型权重，跳过已经训练的数据"""
    """不保存不加载优化器参数！！！"""
    """跳过数据这边，只要传入当前的epoch，然后调用sampler.set_epoch设置随机数；接着传入当前已经训练的数据条数，然后直接从那边开始索引即可"""
    """需要注意的是，请确保训练的参数包括global_batch_size、混合比例等保持一致，否则不支持"""
    train()


