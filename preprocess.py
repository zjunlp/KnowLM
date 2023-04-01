
import json
import shutil
"""
合并不同的数据集：
    1. 如果是同类型的，则直接把两个文件拼接起来即可
    2. 如果是不同类型的，则需要额外生成一个文件，称为.dist，这个文件存储了每个类型的样本总数
    暂定.dist文件为torch.save保存的List文件
"""

import numpy as np
import os
from sentencepiece import SentencePieceProcessor
from typing import List
import argparse
import multiprocessing
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset
import re

"""来自于Megatron-Deepspeed/tools/preprocess_data.py"""
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False
# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class ChineseSplitter(object):
    def __init__(self, pattern):
        self.pattern = pattern
        if "(" in self.pattern and ")" in self.pattern:
            """保留分隔符"""
            self.keep = True
        else:
            self.keep = False

    def tokenize(self, text):
        '''
        import re
        def tokenize(pattern, text):
            new_list = []
            _list = re.split(pattern=pattern, string=text)
            print(_list)
            for item in _list:
                if len(new_list) == 0:
                    """如果是第一个元素，则直接进行一个保留"""
                    new_list.append(item)
                elif (len(item) == 1 and item in pattern) or (item == "\n" or item == r"\n") or len(item)==0:
                    """说明是分隔符"""
                    new_list[-1] += item
                else:
                    new_list.append(item)
            return new_list
        print(tokenize(r"([;!?；。！？\n])",r"这是第一句。\n这是另外一段"))
        print(tokenize(r"([;!?；。！？\n])","这是另外一句。\n这是另外一段"))
        '''
        if self.keep:
            new_list = []
            _list = re.split(pattern=self.pattern, string=text)
            for item in _list:
                if len(new_list) == 0:
                    """如果是第一个元素，则直接进行一个保留"""
                    new_list.append(item)
                elif (len(item) == 1 and item in self.pattern) or (item == "\n" or item == r"\n") or len(item)==0:
                    """说明是分隔符"""
                    new_list[-1] += item
                else:
                    new_list.append(item)
        else:
            new_list = re.split(pattern=self.pattern, string=text)
        return new_list

def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

"""llama官方的分词器"""
"""
处理逻辑：
    1. 对于一个样本（文档）来说，仅对开始的部分增加[BOS]，对结束的部分增加[EOS]，中间切断的部分不加
    2. 
        英文先用nltk进行分句，对于中文文档，直接用中文的正则表达式进行切分，用。？；！这几个进行划分
        然后对每个句子进行tokenizer，然后依据下面的规则进行合并（抽象出来就是下面这道算法题）：
                给定一个一维列表，长度为n，每个列表中的元素为一个正整数，其值的范围为1~1024。只能将相邻的元素通过加法进行合并，合并后每个元素的值不能超过1024。
                目标是输出合并后的列表，要求合并后列表中的元素个数尽可能少，每个元素的值尽可能大。
                def merge_list(lst, maxlen):
                    merged_lst = []
                    answer_lst = []
                    i = j = 0
                    while i < len(lst):
                        ans = [0, 0]
                        sum = lst[i]
                        j = i + 1
                        while j < len(lst) and sum + lst[j] <= maxlen:
                            sum += lst[j]
                            j += 1
                        k = j  # 记录一下终点(不包括这个点)
                        ans[1] = j
                        j = i  # 从起点往回走
                        while j - 1 >= 0 and sum + lst[j - 1] <= maxlen:
                            sum += lst[j - 1]
                            j -= 1
                        merged_lst.append(sum)
                        ans[0] = j
                        answer_lst.append(ans)
                        i = k  #
                    return merged_lst, answer_lst
                merge_list([100,200,300,400,500,600,700,800,900,1000,1100,1200],1024)
        这边有一个需要处理的点，就是每个句子tokenizer之后，会出现单个元素的值超过最大长度，解决：
            1.要么直接对整条样本去除。
            2.不能对tokenizer后的进行分块，而是需要对原文本进行拆分，这边可以随意点，先确定tokenzier后的长度，然后除以最大长度得到拆分成几块：
                对于中文，就直接对原始数据切片，然后送进去
                对于英文，直接对tokenizer后的进行切分
    
            
"""
class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

class DistributedTokenizer:
    def __init__(self, args, collate_fn=None):
        self.args = args
        self.max_seq_length = self.args.seq_length
        self.collate_fn = collate_fn
        # 用于将json格式转换成文本格式，就比如有的时候可能需要把"title"和"content"拼接起来，
        # 这个函数的输入是原始文本的一行(json)，输出也是一行(就是要处理的文档)

    def split(self, lst: List[int]):
        """这个函数就是采用贪心的策略进行，"""
        maxlen = self.max_seq_length
        merged_lst = []
        i = j = 0
        answer_lst = []
        while i < len(lst):
            ans = [0, 0]
            sums = lst[i]
            j = i + 1
            while j < len(lst) and sums + lst[j] <= maxlen:
                sums += lst[j]
                j += 1
            k = j  # 记录一下终点(不包括这个点)
            ans[1] = j
            j = i  # 从起点往回走
            while j - 1 >= 0 and sums + lst[j - 1] <= maxlen:
                sums += lst[j - 1]
                j -= 1
            merged_lst.append(sums)
            ans[0] = j
            answer_lst.append(ans)
            i = k  #
        return answer_lst   # 左闭右开

    """code from dEEPsPEED mEGAtRON"""
    def dsmt_initializer(self):
        """加载分词器"""
        """最后对文档进行处理的时候，采用的是DistributedTokenizer.spliiter.tokenize进行划分句子"""
        """然后使用DistributedTokenizer.tokenizer来对句子进行id化"""
        DistributedTokenizer.tokenizer = Tokenizer(self.args.tokenizer_path)
        if self.args.language.lower() == "english":
            if self.args.do_split_sentences:
                if not nltk_available:
                    print("NLTK is not available to split sentences.")
                    exit()
                """这个是英文的，将文档划分成句子"""
                splitter = nltk.load("tokenizers/punkt/english.pickle")
                if self.args.do_keep_newlines:
                    """下面的方法是在划分的时候保存换行符"""
                    DistributedTokenizer.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                        train_text = splitter._params,
                        lang_vars = CustomLanguageVars())
                else:
                    """不保存换行符"""
                    DistributedTokenizer.splitter = splitter
            else:
                DistributedTokenizer.splitter = IdentitySplitter()
        elif self.args.language.lower() == "chinese":
            if self.args.do_split_sentences:
                """中文也分割句子"""
                """采用正则表达式切割一下就行"""
                if self.args.do_keep_newlines:
                    pattern = r"([;!?；？。！\n])"  # 用分号、感叹号、问号、句号进行分割句子，左右圆括号表示保留分隔符
                    DistributedTokenizer.splitter = ChineseSplitter(pattern=pattern)
                else:
                    pattern = r"[;!?；？。！\n]"
                    DistributedTokenizer.splitter = ChineseSplitter(pattern=pattern)
            else:
                """不分割成句子"""
                DistributedTokenizer.splitter = IdentitySplitter()

        else:
            assert False, "目前支持的语言为english和chinese，请确保输入正确"

    def _re_split(self, src:str, tokenized:List, delete_29871=True, start_part=False, end_part=False):
        """

        :param src:         原始的句子
        :param tokenized:   分完词后的列表
        :param delete_29871: 具体可以看一下llama-main/llama/token_sentence.py文件
        :param start_part:  传入的src为开始部分，说明当前tokenized的开头有BOS
        :param end_part:    传入的src为结束部分，说明当前tokenized的结尾有EOS
        :return:
        """
        if len(tokenized)<=self.max_seq_length:
            return [tokenized]
        else:
            """超出最大长度"""
            n_block = int(np.ceil(len(tokenized)/self.max_seq_length).item())
            if self.args.language.lower() == "english":
                """英文就直接对tokenized均分"""
                new_tokenized = []
                for i in range(n_block):
                    new_tokenized.append(tokenized[i*self.max_seq_length:(i+1)*self.max_seq_length])
            elif self.args.language.lower() == "chinese":
                """中文需要对src进行分割，然后送入到tokenize进入"""
                """需要注意的是，中文好像对于每个句子的开头都会添加一个29871，
                测试可以参见llama-main/llama/token_sentence.py文件"""
                new_tokenized = []
                if len(tokenized) % self.max_seq_length >= self.max_seq_length*0.8:
                    """如果说余数超过了最大长度的80%，则多进行一次分块，避免此处分完块后还会出现多的部分"""
                    n_block += 1
                length_per_block = int(np.ceil(len(src) / n_block).item())
                for i in range(n_block):
                    new_src = src[i*length_per_block:(i+1)*length_per_block]
                    bos = True if i==0 and start_part==True else False          # 是否添加bos
                    eos = True if i==n_block-1 and end_part==True else False    # 是否添加eos
                    _tokenized = DistributedTokenizer.tokenizer.encode(new_src, bos=bos, eos=eos)
                    if delete_29871 and _tokenized[0] == 29871:
                        _tokenized = _tokenized[1:]
                    new_tokenized.append(_tokenized)
            else:
                assert False, "目前支持的语言为english和chinese，请确保输入正确"
            return new_tokenized

    def dsmt_encode(self, json_line):
        if self.collate_fn == None:
            text = json_line
        else:
            text = self.collate_fn(json_line)
        if text == "\n" or text.strip() == "" or text == r"\n":
            return []
        """将其切分成句子"""
        sentences = DistributedTokenizer.splitter.tokenize(text)
        """对句子进行分词，然后对于超过长度的再次分割，处理完成之后送入split进行融合即可"""
        if len(sentences) == 1:
            # 只有一句话
            _tokenized:List = DistributedTokenizer.tokenizer.encode(sentences[0], bos=True, eos=True)
            _tokenized = self._re_split(sentences[0], _tokenized, delete_29871=True, start_part=True, end_part=True)
        else:
            _tokenized = []
            for idx, sentence in enumerate(sentences):
                cur_tokenized = DistributedTokenizer.tokenizer.encode(sentence, bos= idx==0, eos= idx==len(sentences)-1)
                # print(cur_tokenized)
                _tokenized.extend(self._re_split(src=sentence, tokenized=cur_tokenized, delete_29871=True, start_part= idx==0, end_part= idx==len(sentences)-1))
        """记录下分句后每个句子token数目"""
        length_tokenized = [len(_) for _ in _tokenized]
        """得到合并的索引"""
        index = self.split(length_tokenized)
        ultra = []
        for pair in index:
            cur = []
            start, end = pair
            for i in range(start, end):
                cur.extend(_tokenized[i])
            ultra.append(cur)
        """
        ultra = [
            [block1 part tokens],
            [block2 part tokens],
            ...
        ]
        """
        return ultra


    def initializer(self):
        """加载分词器"""
        DistributedTokenizer.tokenizer = Tokenizer(self.args.tokenizer_path)

    def encode(self, text:str):
        return DistributedTokenizer.tokenizer.encode(text.strip(), self.bos, self.eos)

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

        self.index_start_pos = None       # 每个样本的起始位置
        self.index_length = None          # 每个样本的长度
        self._load_index()
        self._load_bin()
        self._load_dis()

    def _load_index(self):
        """文件所占的字节大小"""
        file_size = os.stat(self.idx_file_path).st_size
        """样本总数"""
        assert file_size % 10 == 0       # 2B的length，8B的start pos
        self.total_sample = file_size // 10
        with open(self.idx_file_path, "rb") as f:
            self.index_start_pos = np.frombuffer(f.read(self.total_sample*8), dtype=np.uint64).tolist()
            self.index_length = np.frombuffer(f.read(self.total_sample*2), dtype=np.uint16).tolist()
            # print(self.index_length)

    def _load_bin(self):
        """参考了Megatron-Deepspeed"""
        _warmup_mmap_file(self.bin_file_path)
        """以内存映射的方式进行加载大文件"""
        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint16, mode='r')

    def _load_dis(self):
        """仅当有多种类别的数据混合有效"""
        self.distributed = torch.load(self.dis_file_path)
        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        if self.pad_id == 0:
            data = torch.zeros([self.seq_length], dtype=torch.long)
        else:
            data = torch.ones([self.seq_length], dtype=torch.long) * self.pad_id
        start_idx = self.index_start_pos[idx]
        length = self.index_length[idx]
        if idx+1<self.total_sample:
            assert start_idx+length == self.index_start_pos[idx+1], \
                f"{start_idx+length}!={self.index_start_pos[idx+1]}, idx={idx}"
        if length>self.seq_length:
            length = self.seq_length
        # data[0:length] = torch.as_tensor(self.bin_buffer[start_idx:start_idx+length].tolist(), dtype=torch.long)
        # return data
        return self.bin_buffer[start_idx:start_idx+length].tolist()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="write", type=str, help="有merge,write,read三种模式")
    parser.add_argument("--seq_length", default=1024, type=int, help="最大长度")
    parser.add_argument("--language", default="english", type=str, help="english, chinese")
    parser.add_argument("--do_split_sentences", action="store_true", default=False, help="是否将文档划分成句子")
    parser.add_argument("--do_keep_newlines", action="store_true", default=False, help="划分的时候是否保留换行符")
    parser.add_argument("--file_path", default="testdata.txt", type=str, help="源文件，每一行都是一个样本")                          # 写入
    parser.add_argument("--num_workers", default=1, type=int, help="并行处理数量")                                                # 写入
    parser.add_argument("--tokenizer_path", default="tokenizer.model", type=str, help="sentencepiece文件")                      # 写入
    parser.add_argument("--save_prefix", default="hello", type=str, help="保存的时候叫什么,索引文件会添加上.idx,数据文件添加上.bin")        # 写入
    parser.add_argument("--save_path", default="./", type=str, help="保存的位置，需要结尾为/")                                     # 写入

    parser.add_argument("--read_path_prefix", default="./hello", type=str, help="读取的文件前缀，读取的时候会自动补全.idx/.bin/.dis") # 读取

    parser.add_argument("--merge_path_prefix", default=None, type=str, help="需要合并的文件前缀，['1','2','3']")
    parser.add_argument("--merge_path_type", default=None, type=str, help="如果不提供，则默认为同一类型的数据集，提供了，则以[1,1,0]这种格式给出")
    """下面这个暂时先取消掉自动合并的功能，必须要指定，也就是写入到新的文件"""
    parser.add_argument("--new_path_prefix", default=None, type=str, help="如果为None，则自动从上面的文件中选取最大的进行合并，如果不为None，则自动")

    """save_mode暂时没用"""
    parser.add_argument("--save_mode", default=1, type=int, help="0-不存储索引文件 1-存储索引文件")

    return parser.parse_args()

def collate_fn_from_json(json_line:str):
    data = json.loads(json_line)
    return data["content"]

def collate_fn_from_text(text:str):
    return text

def count_lines(path):
    """计算输入文件的行数"""
    with open(path, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1  # Remove this if a wc-like count is needed
    return count

def write(args):
    """这个函数用于处理我们爬取的文件"""
    # """单进程debug"""
    # with open(args.file_path, "r", encoding='utf-8') as f:
    #     lines = f.readlines()
    # encoder = DistributedTokenizer(args, eos=False, bos=True, collate_fn=collate_fn_from_text)
    # encoder.dsmt_initializer()
    # for line in lines:
    #     print(encoder.dsmt_encode(line))
    # assert False
    """统计文件行数"""
    count = count_lines(args.file_path)
    print(f"行数为:{count}")
    """打开文本文件"""
    fin = open(args.file_path, 'r', encoding='utf-8')
    """创建多进程"""
    encoder = DistributedTokenizer(args, collate_fn=collate_fn_from_text)
    pool = multiprocessing.Pool(args.num_workers, initializer=encoder.dsmt_initializer)
    """从输入流中进行读取"""
    """进度条：https://blog.csdn.net/weixin_39274659/article/details/107794635"""
    encoded_samples = list(
        (tqdm(pool.imap(encoder.dsmt_encode, fin, 25), total=count, desc="读取进度"))
    )
    # encoded_samples = pool.imap(encoder.dsmt_encode, fin, 25)
    """开始写入"""
    """起始位置:np.uint64: 8B"""
    """长度和token:np.uint16: 2B"""
    f_bin_out = open(f"{args.save_path}{args.save_prefix}.bin", "wb")
    encoded_samples = list(encoded_samples)
    pbar = tqdm(total=len(encoded_samples))
    start_pos = 0
    start = []
    length = []
    num_samples = 0
    for doc in encoded_samples:
        for target in doc:
            """如果没有的话，那就不加入"""
            if len(target) == 0:
                continue
            num_samples += 1
            f_bin_out.write(np.array(target, dtype=np.uint16).tobytes(order='C'))
            length.append(len(target))
            start.append(start_pos)
            start_pos += len(target)
        pbar.update(1)
    """下面这行是数据总数，其实感觉可以不用，直接用获取的索引文件大小除以10即可"""
    # f_idx_out.write(np.array([len(encoded_samples)], dtype=np.uint64).tobytes(order='C'))
    f_bin_out.close()
    f_idx_out = open(f"{args.save_path}{args.save_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint16).tobytes(order='C'))
    f_idx_out.close()

    dis = [num_samples]
    torch.save(dis, f"{args.save_path}{args.save_prefix}.dis")

def write_scratch(args):
    """打开文本文件"""
    fin = open(args.file_path, 'r', encoding='utf-8')
    """实例化分词器"""
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    """创建多进程"""
    encoder = DistributedTokenizer(args, eos=False, bos=True)
    pool = multiprocessing.Pool(args.num_workers, initializer=encoder.initializer)
    """从输入流中进行读取"""
    encoded_samples = pool.imap(encoder.encode, fin, 25)

    """开始写入"""
    """起始位置:np.uint64: 8B"""
    """长度和token:np.uint16: 2B"""
    f_bin_out = open(f"{args.save_path}{args.save_prefix}.bin", "wb")
    encoded_samples = list(encoded_samples)
    pbar = tqdm(total=len(encoded_samples))
    # start_pos = np.array([0], dtype=np.uint64)
    # length = np.array([0], dtype=np.uint16)
    start_pos = 0
    start = []
    length = []
    num_samples = 0
    for target in encoded_samples:
        """如果没有的话，那就不加入"""
        if len(target) == 0:
            continue
        num_samples += 1
        f_bin_out.write(np.array(target, dtype=np.uint16).tobytes(order='C'))
        pbar.update(1)
        length.append(len(target))
        start.append(start_pos)
        start_pos += len(target)
    """下面这行是数据总数，其实感觉可以不用，直接用获取的索引文件大小除以10即可"""
    # f_idx_out.write(np.array([len(encoded_samples)], dtype=np.uint64).tobytes(order='C'))
    f_bin_out.close()
    f_idx_out = open(f"{args.save_path}{args.save_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint16).tobytes(order='C'))
    f_idx_out.close()

    dis = [num_samples]
    torch.save(dis, f"{args.save_path}{args.save_prefix}.dis")

def read(args):
    ds = MyDataset(args.read_path_prefix, seq_length=args.seq_length, pad_id=0)
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    print(f"长度为{len(ds)}")
    for i in range(len(ds)):
        print(f"分句：{i}",tokenizer.decode(ds[i]))
    print(f"分布为:{ds.distributed}")

def merge(args):
    """不支持一个数据集中有不同类型的数据进行混合"""
    """只支持单类型的数据进行混合，从形式上来说，就是.dis文件只有一个数据"""
    if args.merge_path_prefix == None:
        assert False
    else:
        merge_path_prefix = eval(args.merge_path_prefix)

    """判断是否为同一类型"""
    if args.merge_path_type == None:
        print(f"[{time.ctime()}] 合并的数据集属于同一类型")
        merge_path_type = None
    else:
        merge_path_type = eval(args.merge_path_type)

    """合并后的文件叫什么"""
    # new_file_path_prefix = -1
    if args.new_path_prefix == None:
        # filesize = [os.stat(file+".bin").st_size for file in merge_path_prefix]
        # max_ = max(filesize)
        # for i in range(len(merge_path_prefix)):
        #     if max_ == filesize[i]:
        #         new_file_path_prefix=i
        #         break
        # print(
        #     f"[{time.ctime()}] 合并的数据集属于同一类型，"
        #     f"将文件合并到{merge_path_prefix[new_file_path_prefix]}中")
        assert False
    new_path_prefix = args.new_path_prefix

    """先对文件进行分类"""
    if merge_path_type != None:
        """merge_path_type=[0,0,1,1]"""
        """先构建一个dict用于索引"""
        classifier_prefix = {}
        for idx, types in enumerate(merge_path_type):
            if types not in classifier_prefix:
                classifier_prefix[types] = [merge_path_prefix[idx]]
            else:
                classifier_prefix[types].append(merge_path_prefix[idx])
        """二进制文件，直接添加到结尾即可"""
        new_file_bin = open(new_path_prefix+".bin","wb")
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                with open(file_prefix+".bin", "rb") as f:
                    shutil.copyfileobj(f, new_file_bin)
        new_file_bin.close()
        """接下来写入idx文件"""
        new_file_idx = open(new_path_prefix+".idx", "wb")
        index_start_pos = []
        index_length = []
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                file_size = os.stat(file_prefix+".idx").st_size
                """样本总数"""
                assert file_size % 10 == 0
                total_sample = file_size // 10
                with open(file_prefix+".idx", "rb") as f:
                    _index_start_pos = np.frombuffer(f.read(total_sample * 8), dtype=np.uint64)
                    _index_length = np.frombuffer(f.read(total_sample * 2), dtype=np.uint16).tolist()
                if len(index_start_pos) > 0:
                    """如果是不是第一个，则需要将当前得到的起始位置加上目前的最后一个样本的开始位置加上目前最后一个样本的长度"""
                    """比如说，第一个文件我已经拿过来了，然后它的最后一个样本的起始位置为100.长度为10，那么新的文件需要的偏移量为110"""
                    index_start_pos.extend((_index_start_pos+index_start_pos[-1]+index_length[-1]).tolist())
                else:
                    """就是最开始，那么直接加入即可"""
                    index_start_pos.extend(_index_start_pos)
                index_length.extend(_index_length)
        new_file_idx.write(np.array(index_start_pos, dtype=np.uint64).tobytes(order='C'))
        new_file_idx.write(np.array(index_length, dtype=np.uint16).tobytes(order='C'))
        new_file_idx.close()
        """写入dis文件"""
        _cur_size = 0
        new_dist = []
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                data = torch.load(file_prefix+".dis")
                assert len(data) == 1
                _cur_size += data[0]
            new_dist.append(_cur_size)
            _cur_size = 0
        torch.save(new_dist, new_path_prefix+".dis")
        # 做最后的check
        assert sum(new_dist) == len(index_start_pos)
    else:
        """二进制文件，直接添加到结尾即可"""
        new_file_bin = open(new_path_prefix+".bin","wb")
        for file in merge_path_prefix:
            with open(file+".bin","rb") as f:
                shutil.copyfileobj(f, new_file_bin)
        new_file_bin.close()
        """索引文件需要更新"""
        new_file_idx = open(new_path_prefix+".idx","wb")
        index_start_pos = []
        index_length = []
        for file in merge_path_prefix:
            file_size = os.stat(file+".idx").st_size
            """样本总数"""
            assert file_size % 10 == 0  # 2B的length，8B的start pos
            total_sample = file_size // 10
            with open(file+".idx", "rb") as f:
                _index_start_pos = np.frombuffer(f.read(total_sample * 8), dtype=np.uint64)
                _index_length = np.frombuffer(f.read(total_sample * 2), dtype=np.uint16).tolist()
            if len(index_start_pos) > 0:
                """如果是不是第一个，则需要将当前得到的起始位置加上目前的最后一个样本的开始位置加上目前最后一个样本的长度"""
                """比如说，第一个文件我已经拿过来了，然后它的最后一个样本的起始位置为100.长度为10，那么新的文件需要的偏移量为110"""
                index_start_pos.extend((_index_start_pos+index_start_pos[-1]+index_length[-1]).tolist())
            else:
                """就是最开始，那么直接加入即可"""
                index_start_pos.extend(_index_start_pos)
            index_length.extend(_index_length)
        new_file_idx.write(np.array(index_start_pos, dtype=np.uint64).tobytes(order='C'))
        new_file_idx.write(np.array(index_length, dtype=np.uint16).tobytes(order='C'))
        new_file_idx.close()
        assert len(index_start_pos) == len(index_length)
        """最后生成.dis文件"""
        torch.save([len(index_start_pos)], new_path_prefix+".dis")
        # 下面的代码用于check是否合理
        total = 0
        for file in merge_path_prefix:
            data = torch.load(file+".dis")
            """不支持一个数据集中有不同类型的数据进行混合"""
            assert len(data) == 1
            total += data[0]
        assert total == len(index_start_pos)

if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.mode.lower() == "read":
        read(args)
    elif args.mode.lower() == "write":
        write(args)
    elif args.mode.lower() == "merge":
        merge(args)
    else:
        assert False






