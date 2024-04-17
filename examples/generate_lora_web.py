import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from multi_gpu_inference import get_tokenizer_and_model
from typing import List

from callbacks import Iteratorize, Stream
from prompter import Prompter
from utils import Web

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# model = None
# tokenizer = None
# web_config = None

def get_kwargs(kwargs, *args):
    return [kwargs.get(arg, None) for arg in args]


def main(
    load_8bit: bool = True,
    load_4bit: bool = False,
    base_model: str = None,
    model_tag: str = 'zhixi',
    # lora_weights: str = "zjunlp/CaMA-13B-LoRA",
    # prompt_template: str = "finetune/lora/knowlm/templates/alpaca.json",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    multi_gpu: bool = False,
    allocate: List[int] = None
):
    # global model
    # global tokenizer
    # global web_config
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        if multi_gpu:
            model, tokenizer = get_tokenizer_and_model(base_model=base_model, dtype="float16", allocate=allocate)
        else:
            if load_8bit or load_4bit:
                device_map = {"":0}
            else:
                device_map = {"": device}
            if load_4bit:
                load_8bit = False
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=load_8bit,
                load_in_8bit=load_4bit,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        #     torch_dtype=torch.float16,
        # )
    elif device == 'cpu':
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     device_map={"": device},
        # )


    if 'oneke' in base_model.lower():
        model_tag = 'oneke'
    elif 'knowlm' in base_model.lower():
        model_tag = 'zhixi'
    assert model_tag in  Web.__SUPPORT_MODEL__
    web_config = Web.get_ui(model_tag)
    prompter = Prompter(model_name=model_tag)
    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # pad
    # model.config.bos_token_id = tokenizer.pad_token_id = 1
    # model.config.eos_token_id = tokenizer.pad_token_id = 2

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    def evaluate(
        # instruction,
        # input=None,
        # temperature=0.4,
        # top_p=0.75,
        # top_k=40,
        # num_beams=2,
        # max_new_tokens=512,
        # repetition_penalty=1.3,
        # stream_output=False,
        *args
        # **kwargs,
    ):
        kwargs = {}
        assert len(web_config['var_name']) == len(args), f"{en(web_config['var_name'])} == {len(args)}"
        for key, value in zip(web_config['var_name'], args):
            kwargs[key] = value
        print(kwargs)
        generation_var = [
            'temperature', 
            'top_p', 
            'top_k', 
            'num_beams', 
            'max_new_tokens', 
            'repetition_penalty',
            'stream_output'
        ]
        input_var = [
            'instruction', 
            'input', 
            'schema',
            'system_prompt',
        ]

        all_var = input_var + generation_var
        instruction, input, schema, system_prompt, \
            temperature, top_p, top_k, num_beams, max_new_tokens, repetition_penalty, stream_output = get_kwargs(kwargs, *all_var)

        generation_var_args = {}
        input_var_args = {}
        for cur_var in [(generation_var, generation_var_args), (input_var, input_var_args)]:
            _cur_var_list, _cur_arg = cur_var
            for var_name in _cur_var_list:
                if var_name in kwargs:
                    _cur_arg[var_name] = kwargs.pop(var_name)
        
        prompt = prompter.generate_prompt(**input_var_args)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            **generation_var_args
            # temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            # num_beams=num_beams,
            # repetition_penalty=repetition_penalty,
            # **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output[1:])   # # `1:` means that we need to filter bos token

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]  
        output = tokenizer.decode(s[1:])    # `1:` means that we need to filter bos token
        yield prompter.get_response(output)



    gr.Interface(
        fn=evaluate,
        inputs=web_config['components'],
        # inputs=[
        #     gr.components.Textbox(
        #         lines=2,
        #         label="Instruction",
        #         placeholder="<请在此输入你的问题>",
        #     ),
        #     gr.components.Textbox(
        #         lines=2, 
        #         label="Input", 
        #         placeholder="<可选参数>",
        #     ),
        #     gr.components.Slider(
        #         minimum=0, maximum=1, value=0.4, label="Temperature"
        #     ),
        #     gr.components.Slider(
        #         minimum=0, maximum=1, value=0.75, label="Top p"
        #     ),
        #     gr.components.Slider(
        #         minimum=0, maximum=100, step=1, value=40, label="Top k"
        #     ),
        #     gr.components.Slider(
        #         minimum=1, maximum=4, step=1, value=2, label="Beams"
        #     ),
        #     gr.components.Slider(
        #         minimum=1, maximum=2000, step=1, value=512, label="Max tokens"
        #     ),
        #     gr.components.Slider(
        #         minimum=1, maximum=2, step=0.1, value=1.3, label="Repetition Penalty"
        #     ),
        #     gr.components.Checkbox(label="Stream output"),
        # ],
        outputs=[
            gr.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title=web_config['title'],
        description=web_config['description'],
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
    """
    # multi-gpu
    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/generate_lora_web.py --base_model zjunlp/knowlm-13b-zhixi --multi_gpu --allocate [5,10,8,10]
    # single-gpu
    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/generate_lora_web.py --base_model zjunlp/knowlm-13b-zhixi

    CUDA_VISIBLE_DEVICES=0,1 python examples/generate_lora_web.py --base_model /disk/disk_20T/data/guihh/datasets/modelscope/OneKE --multi_gpu --allocate [16,16]
    """
