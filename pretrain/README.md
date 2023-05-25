:speaking_head: \[ **中文** | [English](./README_EN.md) \]
> 代码基于[Llama-X](https://github.com/AetherCortex/Llama-X)，Llama-X所使用的Huggingface提供的Trainer来进行训练（主要涉及的是全参数的指令微调），采用了Deepspeed的ZeRO-3策略，Llama-X额外提供数据预处理部分。因此对于全参数的指令微调，可以直接使用。
>
> 本代码更改了数据预处理部分使得其适配预训练任务。主要特点包括：
>
> 1. 离线处理预训练数据集，并设计了一种贪心策略对超出长度的文本进行分割，贪心的目标是“在每一个训练样本尽可能都是完整的句子的前提下，使得文本分段的段数最少，每个训练样本的长度尽可能长”；
> 2. 参考[Deepspeed-Megatron](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，采用`mmap`进行数据存储和读入，可以极大的降低内存占用；
> 3. 提供按比例采样，即每个数据源在每个batch内分别采样多少个样本；
> 4. 提供一整套数据预处理方案。

# 0. 环境

通过下面的命令来安装相应的包。需要注意的是，在数据预处理阶段使用了`nltk`包，可以参考[此处](https://blog.csdn.net/weixin_43409402/article/details/100012485)进行安装。

```shell
conda create -n train python=3.9 -y
conda activate train
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```



# 1. 数据预处理

## 1.1 数据准备

对于原始数据，一般有两种格式，第一种：每个txt文件的每一行都是一个完整的文档；第二种：每个txt文件的每一行都是一个json格式的键值对。

对于第一种，修改`preprocess.py`文件中的`write()`函数，在实例化`DistributedTokenizer`的时候将`collate_fn`设为`collate_fn_from_text`。

对于第二种，修改`preprocess.py`文件中的`write()`函数，在实例化`DistributedTokenizer`的时候将`collate_fn`设为`collate_fn_from_json`，并根据自己的需要修改`collate_fn_from_json`函数的值，比如我想对每个json文件的`instruction`和`output`键进行拼接，则`collate_fn_From_json`的定义为：

```python
def collate_fn_from_json(json_line: str):
    data = json.loads(json_line)
    # return data["content"]
    return data["instruction"] + "\n" + data["output"]
```

## 1.2 原理

对每个数据集进行处理时，基本思想是先对每个doc（一般指的是数据文件中的某一行）按照最大长度进行分段，然后对分段的结果进行分词，然后把分词的结果以某种特定的格式进行存储，下面对生成的格式进行介绍：

| 文件后缀 | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| `.bin`   | 二进制文件，用于存储每个样本的`token id`，采用2个字节存储（无符号整数），因为`llama`支持的`token`的最大个数为32000，`2^16=65536`。 |
| `.idx`   | 二进制文件，用于存储每个样本的长度和每个样本在`.bin`文件中的起始位置，其中长度采用2个字节无符号整数存储，起始位置采用8个字节无符号整数存储。 |
| `.dis`   | 由`torch.save`进行存储，存储的是样本个数。                   |
| `.tmp`   | 有`torch.save`进行存储，存储的是一个含有4个元素的列表，每个元素表示：开头为BOS的个数、结尾为EOS的个数、不包含EOS和BOS的个数、同时包含BOS和EOS的个数。四个元素相加为总样本个数。(不参与后续处理，仅做一个指标参考) |

对于长文本分段策略，首先对每个文档按照句子进行切割，然后将句子进行分词（这保证了每个样本都是一个完整的句子）。此时我们的任务是如何将各个句子进行融合。该问题等同于给定一个一维列表，长度为n（n个句子），每个列表中的元素为一个正整数，其值的范围为1~1024（最大长度）。只能将相邻的元素通过加法进行合并，合并后每个元素的值不能超过1024（最大长度）。目标是输出合并后的列表，要求合并后列表中的元素个数尽可能少（分段尽可能少），每个元素的值尽可能大（长度尽可能大，提供上下文信息）。

最后读取的时候，我们只需要将`.idx`加载到内存中，然后使用`numpy`提供的`memmap`对`.bin`文件进行读取即可，即对于需要读取的，去磁盘上查找然后读入内存。

## 1.3 使用方法

> ❗ 下面的所有命令中都忽略了`--tokenizer_path`，请将其设为正确的地址，该文件名为`tokenizer.model`，从Llama官方git仓库中下载。

### 1.3.1 单个数据集处理

```shell
python preprocess.py --mode="write" --file_path="./data/dataset1.txt" --save_prefix="dataset1" --save_path="./data/" --language "english" --do_split_sentences --do_keep_newlines --seq_length 1024 --num_workers 8
```

**参数说明：**

`--language`: 目前支持`chinese`和`english`，对于代码，建议使用`english`参数，该方法用于区分使用哪种方法对文档进行分句。

`--do_split_sentences`: 建议使用，否则可能会报错。

`--do_keep_newlines`: 建议开启，即对换行符进行保留。

`--seq_length`: 每个样本的长度上限。

`--num_workers`: 使用多少个进程进行处理。

`--mode`: 用于区分不同的任务，可选的包括`write` `read` `merge`。

`--file_path`: 原始文件的地址。

`--save_path`&`--save_prefix`: 最后保存的时候是将这两个直接拼接，然后在结尾添加上`.idx` `.bin` `.dis` `.tmp`作为输出文件地址，因此请确保`--save_path`以`/`结尾。

**其他：**

关于速度，中文比英文快，因为对文档进行分句的时候，英文采用的是`nltk`进行，中文则直接按照标点符号进行分割成句子。对于`25GB`的英文来说，在`num_workers`设为32时，耗时`20min`。

### 1.3.2 数据集合并

假设采用`1.3.1`的方法处理了多个数据源，现在要将其进行合并。假设处理了五个数据集，其中`dataset1`和`dataset2`为英文数据集，`dataset3` `dataset4` `dataset5`为中文数据集。

采用下面的命令可以对同一类型的数据集进行合并，比如将`dataset1`和`dataset2`合并为`english`:

```shell
python preprocess.py --mode="merge" --merge_path_prefix="['data/dataset1', 'data/dataset2']" --merge_path_type=[0,0] --new_path_prefix="./data/english"
```

**参数说明：**

`--mode`: `merge`表示当前为合并模式。

`--merge_path_prefix`: 传入的是字符串，在实际运行的时候会使用`eval`将其转换为List，里面的每一个值为`1.3.1`中的`--save_path`&`--save_prefix`，因为在读取的时候，是直接把`merge_path_prefix`中的每一个元素添加上后缀进行读取。

`--merge_path_type`: 传入的也是字符串，也是使用`eval`将其转换为`List`，里面的每一个相同的值表示属于统一类型的数据集。

`--new_path_prefix`: 保存的文件的路径和前缀。



对于是否合并为同一类型的数据集，主要体现在采样的时候是否要区分按比例采样。下面的命令将中英文数据进行合并：

```shell
python preprocess.py --mode="merge" --merge_path_prefix="['data/chinese', 'data/english']" --merge_path_type=[0,1] --new_path_prefix="./data/data"
```

此处的`merge_path_type`为`[0,1]`表示这两个数据集合并的时候是不同的数据集。如果认为其属于同一个数据集，可以将其设为`[0,0]`、`[1,1]`、`[2,2]`等。

### 1.3.3 数据集读取

使用下面的命令进行读取：

```shell
python preprocess.py --mode="read" --read_path_prefix="./data"
```

只是用来顺序解码，观察上面的操作是否出现错误等。可以通过直接修改`read`函数进行修改。



# 2. 训练脚本

> 下面的`--data_path`参数传入的是由`1.3`生成的文件的前缀，比如在`1.3`中生成的是`/a/b/c/d.bin` `/a/b/c/d.idx` `/a/b/c/d.dis`

## 2.1 混合采样

### 2.1.2 单机多卡

```shell
deepspeed train.py \
    --model_name_or_path /your/path/to/hf_llama/folder \
    --model_max_length 1024 \
    --data_path /your/path/to/data/with/prefix \
    --output_dir /your/output/folder \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1.5e-5 \
    --warmup_steps 300 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/config.json \
    --fp16 True \
    --log_on_each_node False \
    --lr_scheduler_type "cosine" \
    --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1
```



### 2.1.3 多机多卡

下面是在3机器、每张机器8张显卡上运行的命令。

```shell
deepspeed --num_gpus 8 --num_nodes 3 --hostfile=host.txt train.py \
    --gradient_accumulation_steps 3 \
    --model_name_or_path /your/path/to/hf_llama/folder \
    --model_max_length 1024 \
    --data_path /your/path/to/data/with/prefix \
    --output_dir /your/output/folder \
    --num_train_epochs 1 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1.5e-5 \
    --warmup_steps 300 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/config.json \
    --fp16 True \
    --log_on_each_node False \
    --lr_scheduler_type "cosine" \
    --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1
```

请注意，需要在主节点下面新建一个`host.txt`文件，文件的内容为：

```shell
127.0.0.1 slots=8
127.0.0.2 slots=8
127.0.0.3 slots=8
```

其中第一行为主节点的ip，第2、3行为其他两个机器的ip，`slots`表示每个机器的显卡个数。`model_name_or_path`的文件夹下需要包含两个文件：`pytorch_model.bin`和`config.json`。此外，确保所有机器上的路径保持一致，虚拟环境保持一致。



## 2.2 按比例采样

> 需要注意的是，按比例采样实际上是根据数据集的种类数来进行的。比如假设数据处理完成后，得到了三种类型的数据集，分别记为A、B、C，设A有100条，B为50条，C为10条，假设batch size是16，那么每次采样的时候都是A:B:C=10:5:1，同时这也是最佳的。如果batch size是8，则A:B:C的值需要手动指定，比如比例为5:2:1，此时数据集A在100/5=20次抽完，数据集B在50/2=25次抽完，数据集C在10/1=10次抽完，由于每个数据集抽取的次数不一样，因此为了确保一样，我们需要通过数据的扩增或者丢弃来保证此处相同，这个参数通过修改`./preview/dataloader.py`文件中类`MyDistributedSampler`的`drop_mode`来控制，我们提供了四种模式来确保抽取的次数一样。
>
> 按比例采样的原理在于自定义`Sampler`，具体来说，我们需要在训练之前，将数据集的每个样本的索引进行打乱，具体可以参见代码`./preview/dataloader.py`。

### 2.2.2 单机多卡

```shell
deepspeed ./preview/train.py \
    --model_name_or_path /your/path/to/hf_llama/folder \
    --model_max_length 1024 \
    --data_path /your/path/to/data/with/prefix \
    --output_dir /your/output/folder \
    --global_batch_distributed "[64,64]" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1.5e-5 \
    --warmup_steps 300 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/config.json \
    --fp16 True \
    --log_on_each_node False \
    --lr_scheduler_type "cosine" \
    --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1 
```

仅增加`--global_batch_distributed "[64,64]"`参数，该参数传入的List，元素个数为数据集类型个数，请确保该list的和为`global batch size`（如果使用了梯度累积，其和仍然为不启用梯度累积时的global batch size）。

### 2.2.3 多机多卡

```shell
deepspeed --num_gpus 8 --num_nodes 3 --hostfile=host.txt ./preview/train.py \
    --model_name_or_path /your/path/to/hf_llama/folder \
    --model_max_length 1024 \
    --data_path /your/path/to/data/with/prefix \
    --global_batch_distributed "[424,80]"\
    --output_dir ./output3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 21 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1.5e-5 \
    --warmup_steps 300 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/config.json \
    --fp16 True \
    --log_on_each_node False \
    --lr_scheduler_type "cosine" \
 --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1
```

仅增加`--global_batch_distributed "[64,64]"`参数，该参数传入的List，元素个数为数据集类型个数，请确保该list的和为`global batch size`（如果使用了梯度累积，其和仍然为不启用梯度累积时的global batch size）。

## 2.3 跳过训练的数据

> 在多机多卡训练的混合采样中，huggingface官方提供的断点续训是存在问题的（`4.28.0dev`），因此如果需要断点续训，需要手动的将数据跳过，下面提供了跳过已经训练的脚本。

**1. 找到训练的数据**

需要手动更改`tools/find_trained_data.py`中的`args`类：

```python
class args:
    # 训练的时候的最大长度
    model_max_length = 1024
    # padding字符的id
    pad_id = 0
    data_prefix = "/your/path/to/data/with/prefix"
    # 显卡总数
    world_size = 24
    # 这个是默认值，可以不改
    seed = 42
    # 总共取了多少次数据，计算方法为：step数*梯度累积数
    steps = 1100 * 3  # 1100*3=3300 1100表示现在已经训练了1100个step，3表示梯度累积
    # 当前是第几个epoch
    epoch = 0
    # 一张显卡的batch size数（不含梯度累积）
    batch_size_per_gpu = 20
    # 找到的训练过的索引，保存的位置
    save_path = "/your/path/remove.idx"
```

根据自己的情况，修改上面的值。然后执行下面的命令，可以得到

```shell
python tools/find_trained_data.py
```

**2. 剔除训练的数据**

执行下面的命令，即可得到未训练的数据。`--read_data_path_prefix`是原始数据，`--write_data_path_prefix`是新写入的数据，`--remove_idx_path`为第1步生成的路径。

```shell
python remove.py --read_data_path_prefix "/a/b/c/d" --write_data_path_prefix "/a/b/c/e" --remove_idx_path "/your/path/remove.idx"
```



# 3. 注意

1. 在多机多卡训练中，`--save_total_limit`仅对主节点有效，对于其他节点，请手动及时删除其它节点的保存的文件。（由于采用的是`huggingface`的`Trainer`进行训练，问题应该出自于`huggingface`，我们使用的版本是`4.28.0dev`）。

2. 在多机多卡训练中，断点续训是有问题的，因此不建议使用断点续训。如果非法中断了，则需要手动跳过数据，我们提供了脚本（参考`2.3`）来手动跳过这些已经训练的数据，然后再使用上面的多机多卡训练接着训练即可。

   对于按比例采样的多机多卡的断点续训，在代码层面我们实现了断点续训（仅仅是数据层面自动跳过，模型的优化器状态还是不会被保存），目前该功能仍然处于实验室阶段。具体方法是在训练的时候传入`--resume_epoch`和`--resume_global`，第一个参数用于传入一个非负数，用于指示这是第几个epoch的恢复，第二个参数用于传入需要跳过的数据条数，即已经训练了多少个数据。

