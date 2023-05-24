"""
数据处理：
1. 侵入transformer的Trainer，得到已经训练的样本id --> [id1, id2, ...]
2. 根据原来的hybrid，先全部读入到内存，然后根据id，进行重新保存。

讲一下如何侵入transformer的Trainer代码来获得已经训练的样本的id
return DistributedSampler(
    self.train_dataset,
    num_replicas=self.args.world_size,
    rank=self.args.process_index,
    seed=seed,
)

先回顾一下数据处理存储的格式：
.bin: 二进制文件，用于存储每个样本的token id，每个token id采用2个字节存储
.idx: 二进制文件，用于存储每个样本的长度和每个样本的起始位置，样本长度采用2个字节存储，位置采用8个字节存储。
        在二进制文件中，前面部分为起始位置，后面部分是长度
.dis: torch.save保存，存储了每种数据集的种类。也是按顺序存储

这份代码的主要功能是实现第2个功能，下面介绍这个功能的具体步骤：
1. 为了加快速度，可以先将原始的数据集移动到/dev/shm中来加快。
2. 采用mydataset加载。
3. 根据.dis文件，得到不同数据集的个数。然后根据得到的id个个数，更新.dis文件
4. .bin和.idx则根据for循环来即可
"""

from torch.utils.data import Dataset
import numpy as np
import os
import torch
import argparse
import time
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor

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
            f"数据错误校验错误！{self.index_length[-1] + self.index_start_pos[-1]}!={len(self.bin_buffer)}"

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
        return self.bin_buffer[start_idx:start_idx + length].tolist()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_data_path_prefix", default=None, type=str, help="会自动在结尾添加.idx/.bin/.dis")
    parser.add_argument("--write_data_path_prefix", default=None, type=str, help="也会自动在结尾添加.idx/.bin/.dis")
    parser.add_argument("--remove_idx_path", default=None, type=str, help="需要删除的索引文件，使用torch.save保存，list")
    parser.add_argument("--tokenizer_file", default=None, type=str, help="分词器文件，从meta的llama仓库下载")
    return parser.parse_args()

def _get_class(_offset, _cur_id):
    if len(_offset) == 1:
        return 0
    else:
        for i in range(len(_offset)):
            if _cur_id<_offset[i]:
                return i
        assert False

def remove_and_save_chinese(args):
    """只保留中文，而在之前的版本中，中文是在前面的"""
    """加载待删除的索引"""
    remove_idx_path = args.remove_idx_path
    remove_idx = torch.load(remove_idx_path)

    """读取数据"""
    print(f"[{time.ctime()}] 读取数据......")
    _dataset = MyDataset(args.read_data_path_prefix, seq_length=1024, pad_id=0)  # 后面两个参数没啥用
    _distributed = _dataset.distributed # 目前的数据集种类个数
    _offset = [_distributed[0]]
    for i in range(len(_distributed)-1):
        _offset.append(_offset[-1]+_distributed[i+1])
    print(f"[{time.ctime()}] 原始数据条数为{sum(_distributed)}, 待删除条数为{len(remove_idx)}, 分布为{_dataset.distributed}, 理论应剩余{sum(_distributed)-len(remove_idx)}")

    """写数据"""
    f_bin_out = open(f"{args.write_data_path_prefix}.bin","wb")
    start = []
    length = []
    pbar = tqdm(total=len(_dataset))
    start_pos = 0
    distributed = _distributed.copy()
    """待删除的记为1，不用删除的记为0"""
    _flag = torch.zeros([sum(distributed)], dtype=torch.int8)
    _flag[remove_idx] = 1
    for i in range(len(_dataset)):
        pbar.update(1)
        _length = _dataset.index_length[i]
        if _flag[i]==1:
            """为当前的元素减少"""
            _ = _get_class(_offset, i)
            distributed[_] -= 1
            continue
        # 如果是1，表示为英语，则不加入
        if _get_class(_offset, i) == 1:
            continue
        length.append(_length)
        start.append(start_pos)
        start_pos += _length
        f_bin_out.write(np.array(_dataset[i], dtype=np.uint16).tobytes(order='C'))
    pbar.close()
    f_bin_out.close()
    print(f"[{time.ctime()}] 写入idx和dis......")
    f_idx_out = open(f"{args.write_data_path_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint16).tobytes(order='C'))
    f_idx_out.close()

    torch.save([distributed[0]], f"{args.write_data_path_prefix}.dis")
    print(f"[{time.ctime()}] 写入完成!")

    del _dataset

def remove(args):
    """加载待删除的索引"""
    remove_idx_path = args.remove_idx_path
    remove_idx = torch.load(remove_idx_path)

    """读取数据"""
    print(f"[{time.ctime()}] 读取数据......")
    _dataset = MyDataset(args.read_data_path_prefix, seq_length=1024, pad_id=0)  # 后面两个参数没啥用
    _distributed = _dataset.distributed # 目前的数据集种类个数
    _offset = [_distributed[0]]
    for i in range(len(_distributed)-1):
        _offset.append(_offset[-1]+_distributed[i+1])
    print(f"[{time.ctime()}] 原始数据条数为{sum(_distributed)}, 待删除条数为{len(remove_idx)}, 分布为{_dataset.distributed}, 理论应剩余{sum(_distributed)-len(remove_idx)}")

    """写数据"""
    f_bin_out = open(f"{args.write_data_path_prefix}.bin","wb")
    start = []
    length = []
    pbar = tqdm(total=len(_dataset))
    start_pos = 0
    distributed = _distributed.copy()
    """待删除的记为1，不用删除的记为0"""
    _flag = torch.zeros([sum(distributed)], dtype=torch.int8)
    _flag[remove_idx] = 1
    for i in range(len(_dataset)):
        pbar.update(1)
        _length = _dataset.index_length[i]
        if _flag[i]==1:
            """为当前的元素减少"""
            _ = _get_class(_offset, i)
            distributed[_] -= 1
            continue
        length.append(_length)
        start.append(start_pos)
        start_pos += _length
        f_bin_out.write(np.array(_dataset[i], dtype=np.uint16).tobytes(order='C'))
    pbar.close()
    f_bin_out.close()
    print(f"[{time.ctime()}] 写入idx和dis......")
    f_idx_out = open(f"{args.write_data_path_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint16).tobytes(order='C'))
    f_idx_out.close()

    torch.save(distributed, f"{args.write_data_path_prefix}.dis")
    print(f"[{time.ctime()}] 写入完成!")

    del _dataset

def check(args):
    print(f"[{time.ctime()}] 开始校验{args.write_data_path_prefix}......")
    dataset = MyDataset(data_prefix=args.write_data_path_prefix, seq_length=1024, pad_id=0)
    dataset._check()
    print(f"[{time.ctime()}] 新生成的数据条数个数为{dataset.total_sample}, 分布为{dataset.distributed}")
    print(f"[{time.ctime()}] 校验完成！")
    print(f"[{time.ctime()}] decoding......")

    # sp_model = SentencePieceProcessor(model_file="./converted_llama13/tokenizer.model")
    sp_model = SentencePieceProcessor(model_file=args.tokenizer_file)
    cnt = 0
    for i in dataset:
        print(i)
        print(sp_model.decode(i))
        cnt += 1
        if cnt==10:
            break
    print(f"[{time.ctime()}] done!")


if __name__ == '__main__':
    args = get_args()
    remove(args)
    check(args)
