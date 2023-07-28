import os
import fire
import time
from datasets import Dataset, load_dataset
from transformers import LlamaTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from utils import Prompter


def main(
    pretrained_model_dir: str = 'data/zhixi-13B',
    quantized_model_dir: str = 'data/zhixi-13B-4bit',
    data_path: str = './data/training_data/',
    num_samples: int = 128,
    quant_batch_size: int = 4,
    prompt_template_dir: str = 'data/templates',
    prompt_template_name: str = 'alpaca',
    cutoff_len: int = 512,
    train_on_inputs: bool = False,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
):

    quantize_config = BaseQuantizeConfig(
        bits=bits,  # quantize model to 4-bit
        group_size=group_size,  # it is recommended to set the value to 128
        desc_act=desc_act,  # set to False can significantly speed up inference but the perplexity may slightly bad 
    )

    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_dir)
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    prompter = Prompter(prompt_template_dir, prompt_template_name)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt
    
    data_paths = []
    data_path = data_path if data_path[-1] == "/" else data_path+"/"
    for i in os.listdir(data_path):
        data_paths.append(os.path.join(data_path, i))
    print(f"data includes: {data_paths}")
    dataset = load_dataset("json", data_files=data_paths)['train'].shuffle().map(generate_and_tokenize_prompt)
    dataset = dataset.to_list()[:num_samples]

    start = time.time()
    model.quantize(
        examples=dataset,
        batch_size=quant_batch_size,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    model.save_quantized(quantized_model_dir)


if __name__ == '__main__':

    fire.Fire(main)