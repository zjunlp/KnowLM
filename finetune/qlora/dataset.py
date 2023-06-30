import copy
from dataclasses import dataclass
from typing import Sequence, Dict

from datasets import load_dataset, Dataset

import torch
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


def local_dataset(args):

    dataset_name = args.dataset
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_name)
    else:
        dataset = load_dataset(dataset_name)

    dataset = dataset["train"].train_test_split(
        test_size=args.eval_dataset_size, shuffle=True, seed=42
    )
    dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])

    eval_dataset = dataset['test']
    train_dataset = dataset['train']

    return train_dataset, eval_dataset


@dataclass
class DataCollatorForCausalLM(object):
    IGNORE_INDEX = -100

    tokenizer: PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            if not self.train_on_source:
                labels.append(
                    torch.tensor([self.IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                )
            else:
                labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
        
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

    