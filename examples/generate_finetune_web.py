import sys

import fire
import torch
import transformers
import gradio as gr

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
    load_8bit: bool = False,
    base_model: str = None,
    max_new_tokens = 512,
    temperature = 0.3,
    num_beams = 3,
    top_p = 0.75,
    top_k = 40,
    repetition_penalty=1.6,
    interactive: bool = False,
    share_gradio: bool = False,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        print("cuda")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        print("mps")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    elif device == "cpu":
        print("cpu")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        input,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,           
            top_p=top_p,             
            top_k=top_k,  
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output
    
    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Prompt",
                placeholder="<请在此处输入你的prompt>"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.3, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=3, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=512, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=1, maximum=2, step=0.1, value=1.6, label="Repetition Penalty"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output"
            )
        ],
        title="ZhiXi Finetune",
        description="ZhiXi Finetune是基于LLaMA-13B使用中英双语进行二次全量预训练的模型。如果测试的效果不理想，请更改解码参数，或者尝试其他prompt，模型对于参数的选择和prompt的选择比较敏感。如果希望获得更多信息，请参考[KnowLLM](https://github.com/zjunlp/knowllm)。",

    ).queue().launch(server_name="0.0.0.0", share=share_gradio)



if __name__ == "__main__":
    fire.Fire(main)
