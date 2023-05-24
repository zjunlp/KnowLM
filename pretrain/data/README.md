# 数据

- 此处的`dataset1.txt` `dataset2.txt` ...... `dataset5.txt`仅用于提供示例，用于熟悉`pretrain/preprocess.py`的处理过程

- 支持的任意的数据格式，仅需要修改`pretrain/preprocess.py`文件中的`collate_fn_from_json`即可，下面列举了我们预训练所使用的数据格式：

  | 数据源        | 语言 | 格式                         |
  | ------------- | ---- | ---------------------------- |
  | 百度百科      | 中文 | {"content": "", "title": ""} |
  | 悟道          | 中文 | {"content": "", "title": ""} |
  | 中文维基百科  | 中文 | {"content": "", "title": ""} |
  | arxiv         | 英文 | {"content": ""}              |
  | books         | 英文 | {"text": ""}                 |
  | gutenberg     | 英文 | {"text": ""}                 |
  | stackexchange | 英文 | {"content": ""}              |
  | wikipedia     | 英文 | {"text": "", "title": ""}    |
  | 爬取          | 代码 | {"input":"", "output":""}    |

  我们仅取用`content` `text` `input` `output`字段的值。用户在获取元数据时，可以采用一行一个`json`格式进行，然后通过修改`collate_fn_from_json`函数，即可实现数据处理。