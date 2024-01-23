# multi gpu inference utils
import torch
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from typing import List

def set_limit(allocate:List[int]=None):
    if allocate is None:
        return None
    map_list = {}

    total_memory = torch.cuda.get_device_properties(0).total_memory     # Byte
    total_mem = total_memory / 1024 / 1024 / 1024                       # GB

    for i, lim in enumerate(allocate):
        ratio = lim / total_mem
        map_list[i] = f"{lim}GB"
        torch.cuda.set_per_process_memory_fraction(ratio, i)
        print(f"set cuda:{i} usage: {lim:2d}GB({ratio*100:7.2f}%)", flush=True)
    torch.cuda.empty_cache()
    return map_list


def get_tokenizer_and_model(base_model:str, dtype:str, allocate:List[int]=None, use_cache:bool=True):
    dtype2torch = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    assert dtype in dtype2torch, f"`{dtype}` is not a valid type. The valid value of `dtype` must be `float16`, `float32` and `float16`."
    if dtype == "bfloat16":
        assert torch.cuda.is_bf16_supported(), "`bfloat16` is not supported on your device. Please set `dtype` to `float16` or `float32`"
    if torch.cuda.is_bf16_supported() and dtype != "bfloat16":
        print("Your device is support `bfloat16`. We suggest it.")
    config = AutoConfig.from_pretrained(base_model)
    config.use_cache = use_cache
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # device_map = infer_auto_device_map(
    #     model, 
    #     no_split_module_classes=model._no_split_modules, 
    #     dtype=dtype2torch[dtype],
    #     max_memory=set_limit(allocate)
    # )
    if allocate:
        max_memory = {i:f"{value}GiB" for i, value in enumerate(allocate)}
        max_memory.update({'cpu': "20GiB"})
    else:
        max_memory = None
    device_map = infer_auto_device_map(
        model, 
        max_memory=max_memory,
        no_split_module_classes=model._no_split_modules, 
        dtype=dtype2torch[dtype],
    )
    load_checkpoint_and_dispatch(
        model,
        base_model,
        device_map=device_map,
        # offload_folder=None,
        # offload_state_dict=False,
        dtype=dtype
    )
    return model, tokenizer

