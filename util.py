from typing import List
import os
import torch
import logging
import random
import numpy as np
import time

def print_log(message:str, rank=-1):
    if rank==-1:
        print(f"[{time.ctime()}] {message}")
    elif rank>=0:
        if torch.distributed.get_rank()==rank:
            print(f"[{time.ctime()}] {message}")

def log_dist(message,
             logger,
             ranks: List[int] = [],
             level: int = logging.INFO,
             DEBUG=False,
             bucket:List=None
             ):
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if not DEBUG:
        if my_rank in ranks:
            if level == logging.INFO:
                logger.info(f'[Rank {my_rank}] {message}')
            if level == logging.ERROR:
                logger.error(f'[Rank {my_rank}] {message}')
            if level == logging.DEBUG:
                logger.debug(f'[Rank {my_rank}] {message}')

    else:
        """主要用于梯度检查，bucket用于存储传入的梯度"""
        if my_rank in ranks:
            grad = message
            bucket.append([torch.max(grad), torch.min(grad), torch.mean(grad)])

def set_seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
