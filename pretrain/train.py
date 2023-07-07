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

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import os

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """filed的作用：如果有新值就会被替换掉，"""
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    resume_path: str = field(default=None, metadata={"help": "断点续训的路径"})

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
        torch.cuda.empty_cache()
        return dict(
            input_ids=input_ids,
            labels=labels,
            # attention_mask=input_ids.ne(self.pad_token_id),
        )


class MyDataset(Dataset):
    def __init__(self, data_prefix, seq_length, pad_id):
        super(MyDataset, self).__init__()
        """这边要求data_prefix为完整的路径，但不包括后缀"""
        """比如：/llama/our/data"""
        """后面会根据需要自动的添加上/llama/our/data.idx"""
        """后面会根据需要自动的添加上/llama/our/data.bin"""
        """后面会根据需要自动的添加上/llama/our/data.dis"""
        self.idx_file_path = f"{data_prefix}.idx"
        self.bin_file_path = f"{data_prefix}.bin"
        self.dis_file_path = f"{data_prefix}.dis"
        self.seq_length = seq_length
        self.pad_id = pad_id

        self.index_start_pos = None  # 每个样本的起始位置
        self.index_length = None  # 每个样本的长度
        self._load_index()
        self._load_bin()
        self._load_dis()

        self._check()

    def _check(self):
        """验证数据是否正确"""
        assert self.index_length[-1] + self.index_start_pos[-1] == len(self.bin_buffer), \
            "数据错误校验错误！"

    def _load_index(self):
        """文件所占的字节大小"""
        file_size = os.stat(self.idx_file_path).st_size
        """样本总数"""
        assert file_size % 10 == 0  # 2B的length，8B的start pos
        self.total_sample = file_size // 10
        with open(self.idx_file_path, "rb") as f:
            self.index_start_pos = np.frombuffer(f.read(self.total_sample * 8), dtype=np.uint64).tolist()
            self.index_length = np.frombuffer(f.read(self.total_sample * 2), dtype=np.uint16).tolist()
            # print(self.index_length)

    def _load_bin(self):
        """以内存映射的方式进行加载大文件"""
        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint16, mode='r')

    def _load_dis(self):
        """仅当有多种类别的数据混合有效"""
        self.distributed = torch.load(self.dis_file_path)
        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample
        # print_log(f"数据的分布为：{self.distributed}",rank=0)
        print(f"数据的分布为：{self.distributed}")

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        """为了节省时间，采用动态长度"""
        start_idx = self.index_start_pos[idx]
        length = self.index_length[idx]
        if length > self.seq_length:
            """如果超出最大长度，则使用最大长度"""
            """否则使用原生长度"""
            length = self.seq_length
        data = torch.as_tensor(self.bin_buffer[start_idx:start_idx + length].tolist(), dtype=torch.long)
        labels = data.clone()
        """注意，此时都是没有padding的"""
        return dict(input_ids=data, labels=labels)


def _make_supervised_data_module(data_prefix, seq_length=1024, pad_id=0) -> Dict:
    train_dataset = MyDataset(data_prefix=data_prefix, seq_length=seq_length, pad_id=pad_id)
    data_collator = MyDataCollatorForSupervisedDataset(pad_token_id=0)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    """transformers.HfArgumentParser，传入的类必须有@dataclass进行装饰，这样这些参数会被自动添加到命令中"""
    """https://zhuanlan.zhihu.com/p/296535876?utm_id=0"""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    """命令行传过来的参数"""
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    trainer = Trainer(model=model, args=training_args, **data_module)
    model.config.use_cache = False

    resume_from_checkpoint = model_args.resume_path if model_args.resume_path != None else False
    print(resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=model_args.resume_path if model_args.resume_path != None else False)
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

