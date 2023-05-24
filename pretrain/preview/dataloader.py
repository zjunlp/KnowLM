import torch
import math
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
import os
import time


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

class MyDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self,
                 dataset: Dataset,
                 size_distribute: List[int],
                 sampled_distribute: List[int],
                 batch_size_per_gpu: int,
                 num_replica: int,
                 rank: int,
                 seed=2023,
                 drop_mode=0):
        """

        :param dataset:             唯一的数据集
        :param size_distribute:     每个数据集的样本数
        :param sampled_distribute:  一个batch的样本数，global batch
        :param num_replica:         显卡总数
        :param rank:                当前显卡id
        :param drop_mode:           0-扔+扔 1-填+扔 2-扔+填 3-填+填      第一个表示单个数据集来讲的
        """
        assert len(size_distribute) == len(sampled_distribute)
        assert batch_size_per_gpu * num_replica == sum(sampled_distribute)  # 这边到时候可以加入一个自动分配sampled_distribute
        self.batch_szie_per_gpu = batch_size_per_gpu
        self.global_batch_size = sum(sampled_distribute)
        self.size_distribute = size_distribute
        self.sampled_distribute = sampled_distribute
        self.drop_mode = drop_mode
        self.num_replica = num_replica
        self.dataset = dataset
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.offset = [0]
        self.consumed_data = 0  # 消耗的数据，到时候需要跳过
        if len(size_distribute) > 1:
            for i in size_distribute[0:-1]:
                self.offset.append(self.offset[-1] + i)
        if self.drop_mode == 0:
            _n_batch = [math.floor(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            n_batch = _min_n_batch = min(_n_batch)
        elif self.drop_mode == 1:
            _n_batch = [math.ceil(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            n_batch = _min_n_batch = min(_n_batch)
        elif self.drop_mode == 2:
            _n_batch = [math.floor(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            n_batch = _max_n_batch = max(_n_batch)
        elif self.drop_mode == 3:
            _n_batch = [math.ceil(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            n_batch = max(_n_batch)
        else:
            assert False
        """每张卡的样本个数"""
        self._num_samples = self.num_samples = n_batch * sum(self.sampled_distribute) // self.num_replica
        assert np.ceil(n_batch * sum(self.sampled_distribute) // self.num_replica) == self.num_samples, "不能被整除"

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # print(f"seed={self.seed+self.epoch}")
        print(f"seed={self.seed + self.epoch}")
        """直接在这边先填充好，然后就直接打包就可以了"""
        _indices = [torch.randperm(size, generator=g).tolist() for size in self.size_distribute]
        """假设总的为[10,20,30]，采样的每个batch为[3,2,1]，则ratio为[3.3, 10, 30]，_n_batch就是单个数据集能够采样得到的batch个数"""
        if self.drop_mode == 3:
            """全部填充"""
            """_n_batch=[4,10,30]"""
            _n_batch = [math.ceil(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            """先确保每个数据集内能够被整除"""
            for i in range(len(_indices)):
                """这边会在原来的基础上进行添加的"""
                lack_num = self.sampled_distribute[i] - self.size_distribute[i] % self.sampled_distribute[i]
                if lack_num != self.sampled_distribute[i]:
                    """补全"""
                    _indices[i].extend(torch.randint(0, self.size_distribute[i], [lack_num], generator=g).tolist())
            """然后拓宽每个数据集使得其_n_batch一致，即让[4,10,30] -> [30,30,30]"""
            n_batch = _max_n_batch = max(_n_batch)
            for i in range(len(_indices)):
                lack_n_batch = _max_n_batch - _n_batch[i]  # 计算当前还差多少个batch
                if lack_n_batch != 0:
                    """开始补齐"""
                    lack_num = lack_n_batch * self.sampled_distribute[i]  # 需要补全
                    _indices[i].extend(torch.randint(0, self.size_distribute[i], [lack_num], generator=g).tolist())
        elif self.drop_mode == 2:
            """单个数据集扔，batch补全"""
            _n_batch = [math.floor(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            """删除多的"""
            for i in range(len(_indices)):
                del_num = self.size_distribute[i] % self.sampled_distribute[i]
                if del_num != 0:
                    _indices[i] = _indices[i][:-del_num]
            """batch补全"""
            n_batch = _max_n_batch = max(_n_batch)
            for i in range(len(_indices)):
                lack_n_batch = _max_n_batch - _n_batch[i]  # 计算当前还差多少个batch
                if lack_n_batch != 0:
                    """开始补齐"""
                    lack_num = lack_n_batch * self.sampled_distribute[i]  # 需要补全
                    _indices[i].extend(torch.randint(0, self.size_distribute[i], [lack_num], generator=g).tolist())
        elif self.drop_mode == 1:
            """单个数据集补全，batch扔"""
            _n_batch = [math.ceil(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            """数据集补全"""
            for i in range(len(_indices)):
                """这边会在原来的基础上进行添加的"""
                lack_num = self.sampled_distribute[i] - self.size_distribute[i] % self.sampled_distribute[i]
                if lack_num != self.sampled_distribute[i]:
                    """补全"""
                    _indices[i].extend(torch.randint(0, self.size_distribute[i], [lack_num], generator=g).tolist())
            """batch扔"""
            n_batch = _min_n_batch = min(_n_batch)
            for i in range(len(_indices)):
                del_n_batch = _n_batch[i] - _min_n_batch
                if del_n_batch != 0:
                    """开始删除"""
                    del_num = del_n_batch * self.sampled_distribute[i]  # 需要删除
                    _indices[i] = _indices[i][:-del_num]
        elif self.drop_mode == 0:
            """全部都扔掉"""
            _n_batch = [math.floor(self.size_distribute[i] / self.sampled_distribute[i]) for i in
                        range(len(self.size_distribute))]
            """删除多的"""
            for i in range(len(_indices)):
                del_num = self.size_distribute[i] % self.sampled_distribute[i]
                if del_num != 0:
                    _indices[i] = _indices[i][:-del_num]
            """batch扔"""
            n_batch = _min_n_batch = min(_n_batch)
            for i in range(len(_indices)):
                del_n_batch = _n_batch[i] - _min_n_batch
                if del_n_batch != 0:
                    """开始删除"""
                    del_num = del_n_batch * self.sampled_distribute[i]  # 需要删除
                    _indices[i] = _indices[i][:-del_num]
        else:
            assert False

        """获得全局的indices"""
        indices = []
        for i in range(n_batch):
            """数据集个数"""
            for j in range(len(self.size_distribute)):
                # 需要加上每个数据集的偏移值
                elements = torch.as_tensor(
                    _indices[j][i * self.sampled_distribute[j]:(i + 1) * self.sampled_distribute[j]])
                elements = (elements + self.offset[j]).tolist()
                if type(elements) == int:
                    elements = [elements]
                indices.extend(
                    elements
                )
        if self.consumed_data > 0:
            print(f"[Sampler] 跳过{self.consumed_data}这么多数据！")
            """_num_samples是每张卡的样本个数"""
            self._num_samples = self.num_samples - self.consumed_data // self.num_replica
        else:
            print(f"[Sampler] 不进行跳过，正常运行epoch！")
            self._num_samples = self.num_samples
        """不能这么写，因为每个设备是跳着取的"""
        # indices = indices[self.consumed_data:]

        """最后进行切分"""
        # return indices
        idx = self.rank / self.num_replica
        start = int(idx * self.global_batch_size)
        cur_indices = []
        i = 0
        while True:
            _start = start + i * self.global_batch_size
            if _start >= len(indices):
                break
            cur_indices.extend(indices[_start:_start + self.batch_szie_per_gpu])
            i += 1
        """跳过"""
        cur_indices = cur_indices[self.consumed_data // self.num_replica:]

        assert len(cur_indices) == self._num_samples, f"{len(cur_indices)}!={self._num_samples}"
        return iter(cur_indices)
        # return cur_indices
        # cur_indices = indices[self.rank::self.num_replica]
        # 每个设备
        # return iter(cur_indices)

    def __len__(self):
        print("mike:hhhhh", self._num_samples)
        return self._num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.jump(0)

    """当前epoch用完之后需要再设置，来防止减少，如果显示调用了set_epoch，则可以不用恢复应该"""

    def jump(self, consumed_data):
        self.consumed_data = consumed_data
        """每张卡的样本个数"""
        assert self.consumed_data % self.num_replica == 0
        print(f"每张卡减少{self.consumed_data // self.num_replica}个样本，理论上的值为batch_size_per_gpu*gradient_accumulation*steps")
        self._num_samples = self.num_samples - self.consumed_data // self.num_replica

def sampler_builder(args, dataset: MyDataset):
    """记录下卡的个数"""
    num_gpu = torch.distributed.get_world_size()
    cur_rank = torch.distributed.get_rank()
    print(f"当前的全局的global size={num_gpu}, 当前的全局的rank={cur_rank}")
    """一次batch的分布"""
    sampled_distribute: List = eval(args.global_batch_distributed)
    assert len(sampled_distribute) == len(dataset.distributed), \
        f"[{time.ctime()}] 输入的采样分布分为{len(sampled_distribute)}类，但是数据集中的采样分布为{len(dataset.distributed)}"
    assert num_gpu * args.per_device_train_batch_size == sum(sampled_distribute), \
        f"[{time.ctime()}] 请保证输入的采样分布和为global batch size的大小。当前全局显卡个数为{num_gpu}，每张显卡的batch size为{args.per_device_train_batch_size}，但是输入的采样的总数为{sum(sampled_distribute)}!"
    sampler = MyDistributedSampler(
        dataset=dataset,
        size_distribute=dataset.distributed,  # 整个数据集的分布
        sampled_distribute=sampled_distribute,
        batch_size_per_gpu=args.per_device_train_batch_size,
        num_replica=num_gpu,
        rank=cur_rank,
        seed=args.seed,
        drop_mode=3
    )
    return sampler


def dataloader_builder(args, dataset: MyDataset, RESUME, RESUME_EPOCH, RESUME_DATA):
    """
    args:
        - per_device_train_batch_size: 自带的
        - seed: 自带的
        - global_batch_distributed: "[1,2,3]"
    """
    sampler = sampler_builder(args, dataset)
    if RESUME:
        print(f"[{time.ctime()}] 启用了断点续训，正在进行数据跳过......")
        assert RESUME_EPOCH >= 0
        sampler.set_epoch(RESUME_EPOCH)
        sampler.jump(RESUME_DATA)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.per_device_train_batch_size,
        sampler=sampler
    )
    return dataset, sampler, dataloader