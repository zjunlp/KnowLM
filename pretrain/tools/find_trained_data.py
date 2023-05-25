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
from torch.utils.data.distributed import DistributedSampler

"""我的测试代码"""


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


class args:
    model_max_length = 1024
    pad_id = 0
    data_prefix = "/dev/shm/third_code_data/third"
    world_size = 24
    seed = 42
    steps = 3800 * 3  # 1100*3=3300 3表示梯度累积
    epoch = 0
    batch_size_per_gpu = 20
    save_path = "/your/path/remove.idx"


def train():
    train_dataset = MyDataset(data_prefix=args.data_prefix, seq_length=args.model_max_length, pad_id=args.pad_id)
    samplers = []
    world_size = args.world_size
    steps = args.steps
    batch_size_per_gpu = args.batch_size_per_gpu
    for rank in range(world_size):
        print(rank)
        samplers.append(
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
        )
        samplers[-1].set_epoch(args.epoch)
    _trained_id = []
    # for i in range(2):
    #     print(f"step:{i}")
    #     for j in range(args.world_size):
    #         _sampler = samplers[j]
    #         print(list(_sampler.__iter__())[i*args.batch_size_per_gpu:(i+1)*args.batch_size_per_gpu])

    for i in range(world_size):
        _sampler = samplers[i]
        _trained_id.extend(
            list(_sampler.__iter__())[0:steps * batch_size_per_gpu]
        )
    print("长度:", len(_trained_id))
    torch.save(_trained_id, args.save_path)


if __name__ == "__main__":
    train()

