import os
import logging
import bitsandbytes as bnb

import torch
import transformers
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

from training_args import TrainingArguments, DataArguments, ModelArguments
from training_utils import SavePeftModelCallback, print_trainable_parameters
from dataset import DataCollatorForCausalLM, local_dataset


torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args):

    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.lora_path is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        print(f'adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, train_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(train_args)
    )

    model = get_accelerate_model(args)
    model.config.use_cache = False

    print_trainable_parameters(args, model)
    print('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        use_auth_token=args.use_auth_token,
    )

    train_dataset, eval_dataset = local_dataset(args)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Callbacks
    trainer.add_callback(SavePeftModelCallback)
    
    # Training
    logger.info("*** Train ***")
    # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
    # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
    train_result = trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()