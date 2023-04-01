import json
import logging
import time
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import loguru
import os
import argparse
from torch.utils.data import DataLoader
import deepspeed
from util import log_dist, set_seed, print_log
from model import MyModel, LLAMA_7B_CONFIG, LLAMA_DEBUG_CONFIG, LLAMA_13B_CONFIG
from data import MyDataset, MyDistributedSampler
import shutil

# ==============================Global START========================================
DEBUG = False
SAVE_STATE_NAME = "state.dict"
CUR_GRADIENT_BUCKET = []
GLOBAL_GRADIENT_BUCKET = []
"""用于记录已经存储了几个batch的梯度"""
DEBUG_GRADIENT_STEP = 0
logger = loguru.logger
MyDistributedSampler.logger = logger
NO_DECAY = ["bias", "layernorm"]
# ===============================Global END=========================================

"""梯度检查hook"""
def check_grad(grad):
    log_dist(grad.cpu(), logger=logger, ranks=[0], level=logging.INFO, DEBUG=True, bucket=CUR_GRADIENT_BUCKET)

"""保存梯度，到时候可以调用对应的文件将其转换成csv格式导出"""
def debug_gradient_save(filename="/home/grad.loss"):
    torch.save(GLOBAL_GRADIENT_BUCKET, filename)

def get_ds_config(args):
    """获取deepspeed的配置文件"""
    ds_config_path = args.deepspeed_config
    with open(ds_config_path, "r") as f:
        lines = f.readlines()
    config = ""
    for line in lines:
        config += line.strip()
    return json.loads(config)

def add_argument():
    parser = argparse.ArgumentParser(description="LLaMA")
    # ====================================一些常规的设置=========================================
    """预训练模型的地址。通常用在第一次训练的时候。另外一种情况用在当机器切换结点个数的时候。"""
    parser.add_argument('--pretrained_path', type=str, default=None)
    """当前设备的rank值，需要注意的是，这边是local_rank，在我写的Sampler中传入的需要是global_rank，因此到时候传入的时候可能需要变化"""
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    """模型所能支持的最大"""
    parser.add_argument('--seq_len', type=int, default=1024)
    """随机数种子，请确保在训练的时候不要更改，特别是在断点续训的时候"""
    parser.add_argument('--seed', type=int, default=2023)
    """模型规模，7表示7B，13表示13B，-1表示加载DEBUG参数"""
    parser.add_argument('--model_size', type=int, default=7)
    """是否启用DEBUG"""
    parser.add_argument('--do_debug', action='store_true', default=False)
    """梯度检查检查几个batch"""
    parser.add_argument('--gradient_check_step', type=int, default=5)
    """epoch个数"""
    parser.add_argument('--epoch', type=int, default=100)
    # ==============================训练的设置，包括batch size，比例===============================
    """在下面的时候会通过eval来转换成列表"""
    """元素的个数为不同数据集的个数，元素的和必须等于global batch size"""
    """这个参数表示不同数据集在同一个batch中采样的个数"""
    parser.add_argument('--global_batch_distributed', type=str, default=None, help="[1,2,3]")
    """单张显卡的batch size为多少"""
    parser.add_argument('--batch_size_per_gpu', type=int, default=2)
    """输入数据集的相关信息"""
    parser.add_argument('--data_path_with_prefix', type=str, default=None, help="建议使用绝对路径，且输入的时候带上数据集前缀，后缀的idx/bin/dis会自动补全")
    """隔多少步保存"""
    parser.add_argument('--save_steps', type=int, default=1000, help="隔多少步保存一次")
    # ======================================断点续训============================================
    """表示是否开启从chekpoint中加载模型"""
    parser.add_argument('--do_load_checkpoint', action='store_true', default=False)
    """当改变了显卡张数或者节点个数时，有用"""
    parser.add_argument('--do_change_node', action='store_true', default=False)
    """最多保存多少个checkpoint，如果是-1表示无限，主要是担心硬盘容量，因为保存一个checkpoint需要很多内存"""
    parser.add_argument('--accumulate', type=int, default=5)
    """保存的路径"""
    parser.add_argument('--save_path', type=str, default=None, help="需要结尾带有/")
    """加载checkpoint的位置，如果为None，则使用save_path"""
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    """从0开始，最大值为accumulate-1"""
    parser.add_argument('--tag', type=str, default=None, help="需要从")

    # =====================================Deepspeed===========================================
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def args_check(args):
    assert args.save_path[-1] == "/", \
        f"[{time.ctime()}] 请确保输入的save_path的结尾为'/'"
    if args.load_checkpoint_path == None:
        args.load_checkpoint_path = args.save_path
    else:
        assert args.load_checkpoint_path[-1] == "/", \
            f"[{time.ctime()}] 请确保输入的load_checkpoint_path的结尾为'/'"
    try:
        dirs = [int(f) for f in os.listdir(f"{args.save_path}") if os.path.isdir(f)]
    except:
        assert False, \
            "请确保保存的位置为空文件夹，或者文件夹的名称为连续的数字，默认从0开始"

    assert args.data_path_with_prefix[-1] != "/", \
        f"[{time.ctime()}] 请确保输入的数据集的格式是正确的，结尾不能包含`/`"

"""断点续训"""
"""在开始训练之前完成"""
def resume(args, model_engine):
    """难点在于当切换结点个数或者改变GPU个数的时候，如何让dataloader能够恢复"""
    """
    my_dict中参数的作用：
        step: 当前epoch的step
        epoch:  
        global_step:  从初始训练开始到现在所过的step
        global_epoch: 从初始训练开始所到现在所处的轮次
        consumed_data: 耗费了多少个数据
    对于不使用--do_change_node，则直接将sampler设置epoch，然后跳过step直到step
    对于使用--do_change_node，将sampler设为epoch之后，需要
    """
    if args.do_load_checkpoint:
        if args.do_change_node:
            log_dist("由于您选择了改变结点，即先前训练的环境发生了变化，因此仅加载模型参数，不加载优化器", logger=logger, ranks=[0], level=logging.INFO)
        state, client_sd = model_engine.load_checkpoint(args.load_checkpoint_path, args.tag, load_module_only=args.do_load_checkpoint)
        assert state != None, \
            f"[{time.ctime()}] 加载预训练模型失败，请检查保存的位置和tag参数！"
        """加载保存的其他信息"""
        my_dict = torch.load(f"{args.load_checkpoint_path}{args.tag}/{SAVE_STATE_NAME}")
        return model_engine, my_dict
    return None, None


"""模型保存"""
"""在for循环中运行"""
def save(args, model_engine, my_dict, cur_step):
    try:
        if cur_step % args.save_steps == 0 and cur_step!=0:
            """检查保存的地方是否还有位置"""
            dirs = [int(f) for f in os.listdir(f"{args.save_path}") if os.path.isdir(f)]
            if len(dirs) == 0:
                save_tag = 0
            else:
                save_tag = (max(dirs)+1) % args.accumulate
                if max(dirs) > save_tag:
                    """不确定是否删除"""
                    # shutil.rmtree(f"{args.save_path}{save_tag}")
                    log_dist(f"即将保存在`{save_tag}`文件夹中，但是会造成覆盖", ranks=[0], level=logging.INFO, logger=logger)
            model_engine.save_checkpoint(args.save_path, save_tag)
            torch.save(my_dict, f"{args.save_path}{save_tag}/{SAVE_STATE_NAME}")
            log_dist(f"保存成功，保存在`{save_tag}`文件夹中", ranks=[0], level=logging.INFO, logger=logger)
    except:
        log_dist(f"保存失败，{my_dict}",
                 logger=logger, ranks=[0], level=logging.WARNING)

"""模型加载"""
def model_builder(args):
    if args.model_size == 7:
        print_log("加载7B模型")
        config = LLAMA_7B_CONFIG
        config["pretrained_path"]=args.pretrained_path
    elif args.model_size == 13:
        print_log("加载13B模型")
        config = LLAMA_13B_CONFIG
        config["pretrained_path"]=args.pretrained_path
    elif args.model_size == -1:
        print_log("加载DEBUG模型")
        config = LLAMA_DEBUG_CONFIG
    else:
        assert False, "--model_size的值要在[7,13,-1]中选"
    return MyModel(config=config)

def sampler_builder(args, dataset:MyDataset, ds_config:Dict):
    """记录下卡的个数"""
    # num_gpu = deepspeed.comm.get_world_size()
    num_gpu = torch.distributed.get_world_size()
    """这边可能还有问题，因为多机多卡，不知道这个函数是否正确"""
    # cur_rank = deepspeed.comm.get_global_rank()
    cur_rank = torch.distributed.get_rank()
    log_dist(f"当前的全局的global size={num_gpu}, 当前的全局的rank={cur_rank}", logger=logger, ranks=[cur_rank], level=logging.INFO)
    """一次batch的分布"""
    sampled_distribute:List = eval(args.global_batch_distributed)
    assert len(sampled_distribute) == len(dataset.distributed), \
        f"[{time.ctime()}] 输入的采样分布分为{len(sampled_distribute)}类，但是数据集中的采样分布为{len(dataset.distributed)}"
    assert args.batch_size_per_gpu == int(ds_config["train_micro_batch_size_per_gpu"]), \
        f"[{time.ctime()}] 请保持命令行的参数与config文件的train_micro_batch_size_per_gpu保持一致，虽然当使用自己构建的迭代器时，train_micro_batch_size_per_gpu不起作用"
    assert num_gpu * args.batch_size_per_gpu == sum(sampled_distribute), \
        f"[{time.ctime()}] 请保证输入的采样分布和为global batch size的大小。当前全局显卡个数为{num_gpu}，每张显卡的batch size为{args.batch_size_per_gpu}，但是输入的采样的总数为{sum(sampled_distribute)}!"
    sampler = MyDistributedSampler(
        dataset=dataset,
        size_distribute=dataset.distributed,    # 整个数据集的分布
        sampled_distribute=sampled_distribute,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_replica=num_gpu,
        rank=cur_rank,
        seed=args.seed,
        drop_mode=3
    )
    return sampler

def dataset_builder(args):
    assert args.data_path_with_prefix != None, \
        f"[{time.ctime()}] 请输入数据集的位置！"
    dataset = MyDataset(
        data_prefix=args.data_path_with_prefix,
        seq_length=args.seq_len,
        pad_id=0
    )
    return dataset

def dataloader_builder(args, ds_config:Dict):
    dataset = dataset_builder(args)
    sampler = sampler_builder(args, dataset, ds_config)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size_per_gpu,
        sampler=sampler
    )
    return dataset, sampler, dataloader

def wrap_with_deepspeed(args, model:nn.Module, ds_config:Dict):
    parameters = model_parameter(ds_config, model)
    model_engine, _, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters
    )
    return model_engine

def model_parameter(ds_config:Dict, model:nn.Module):
    global DEBUG
    """先设置WEIGHT_DECAY"""
    optimized_grouped_parameters = [
        {'params': [], 'weight_decay':ds_config["optimizer"]["params"]["weight_decay"]},
        {'params': [], 'weight_decay':0.0}                                                  # no_decay
    ]
    _NO_DECAY_NAME = []
    for n, p in model.named_parameters():
        flag=True
        for name in NO_DECAY:
            if name in p:
                flag=False
                break
        if flag:
            optimized_grouped_parameters[1]["params"].append(p)
        else:
            _NO_DECAY_NAME.append(n)
            optimized_grouped_parameters[0]["params"].append(p)
    log_dist(f"下面的参数不参加Weight Decay:\n{_NO_DECAY_NAME}", logger=logger, ranks=[0], level=logging.INFO)
    """设置hook"""
    if DEBUG:
        print_log("正在构建梯度hook")
        for param in model.parameters():
            param.register_hook(check_grad)

    return optimized_grouped_parameters

"""多机多卡的加载、保存和断点续训还没有测试"""
if __name__ == '__main__':
    """准备"""
    # num_gpu = deepspeed.comm.get_world_size()
    num_gpu = torch.distributed.get_world_size()
    args = add_argument()
    print_log(args)
    args_check(args)
    """将config.json文件转换成json格式"""
    ds_config = get_ds_config(args)
    set_seed(seed=args.seed)
    # global DEBUG
    DEBUG = args.do_debug

    """构建模型、采样器、优化器"""
    model = model_builder(args)
    dataset, sampler, dataloader = dataloader_builder(args, ds_config)
    model_engine = wrap_with_deepspeed(args, model, ds_config)

    """检查是否需要断点续训，如果需要则启动"""
    _model_engine, _my_dict = resume(args, model_engine)
    if _model_engine != None:
        FLAG_CONSUME = True
        model_engine = _model_engine
        my_dict = _my_dict
        sampler.set_epoch(my_dict["global_epoch"])
        sampler.jump(my_dict["consumed_data"])
    else:
        FLAG_CONSUME = False
        my_dict = {
            "global_epoch": 0,
            "consumed_data": 0
        }

    """设为train模式"""
    model_engine.train()

    """获取当前进程的局部信息"""
    local_rank = args.local_rank
    device = (
        torch.device("cuda", local_rank) if (local_rank > -1) and torch.cuda.is_available() else torch.device("cpu"))

    """开始训练"""
    for epoch in range(my_dict["global_epoch"], args.epoch):
        my_dict["global_epoch"] = epoch
        sampler.set_epoch(epoch)
        if FLAG_CONSUME:
            FLAG_CONSUME = False
            sampler.jump(my_dict["consumed_data"])
        else:
            sampler.jump(0)

        for step, batch in enumerate(dataloader, start=0):
            """将当前的数据移入到显存"""
            batch = batch.to(device)
            """构建label"""
            labels = batch.clone()
            labels[torch.where(labels == 0)] = -100
            loss = model_engine(input_ids=batch, labels=labels)
            model_engine.backward(loss)
            log_dist(str(loss.item()), logger=logger, ranks=[local_rank], level=logging.INFO)

            if DEBUG:
                if int(os.environ.get("RANK", "0")) == 0:
                    GLOBAL_GRADIENT_BUCKET.append(CUR_GRADIENT_BUCKET.copy())
                    CUR_GRADIENT_BUCKET.clear()
                    DEBUG_GRADIENT_STEP += 1
                if DEBUG_GRADIENT_STEP >= args.gradient_check_step:
                    debug_gradient_save()
                    exit()

            model_engine.step()
            my_dict["consumed_data"] += num_gpu*args.batch_size_per_gpu
            save(args, model_engine, my_dict, step)

        my_dict["consumed_data"] = 0

