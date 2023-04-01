import torch.distributed
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn as nn
import time
from util import print_log

LLAMA_7B_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "pretrained_path": None
}
LLAMA_13B_CONFIG={
    "vocab_size": 32000,
    "hidden_size": 5120,
    "intermediate_size": 13824,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,
    "pretrained_path": None
}


LLAMA_DEBUG_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "pretrained_path": None
}



class MyModel(nn.Module):
    def __init__(self,
                 config,
                 gradient_checkpoint=True):
        super(MyModel, self).__init__()
        self.config = LlamaConfig(
            vocab_sizes=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            intermediate_size=config["intermediate_size"],
            num_attention_heads=config["num_attention_heads"],
            rms_norm_eps=1e-6
        )
        print_log(f"{self.config}", rank=0)
        if config["pretrained_path"] == None:
            self.model = LlamaForCausalLM(self.config)
            print_log("随机初始化")
        else:
            self.model = LlamaForCausalLM.from_pretrained(config["pretrained_path"])
            print_log(f"从{config['pretrained_path']}中加载文件")
        if gradient_checkpoint:
            self.model.gradient_checkpointing_enable()
        self.model.train()

    def forward(self,
               input_ids,
               labels):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            use_cache=False
        )[0]