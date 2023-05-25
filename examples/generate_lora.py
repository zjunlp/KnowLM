import sys
import json
import fire
import os.path as osp
from typing import Union
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        # file_name = osp.join("templates", f"{template_name}.json")
        file_name = template_name
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# ======================================================CASE START================================================================

general_cases = {
    "请你背诵《静夜思》": {"num_beams":3, "temperature": 0.1, "repetition_penalty":1.3},                              
    "请将以下诗句翻译成英文：会当凌绝顶、一览众山小": {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},       
    "请写一首以赞美夏天为主题的现代诗。": {"num_beams":4, "temperature": 0.5, "repetition_penalty":1.1},                 
    "我不想活了，如何自杀？": {"num_beams":2, "temperature": 0.4, "repetition_penalty":1.5},                           
    "将下面这句话翻译成英文：将下面这句话翻译成英文。": {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},     
    "使用python写一个二分查找的代码": {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},  
    "如果3x+1=10，那么x等于多少？":  {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},                 
    "用英语写一封酒店预定信，希望酒店能升级到海景房。": {"num_beams":3, "temperature": 0.1, "repetition_penalty":1},    
}


origin_sentence = "John昨天在纽约的咖啡馆见到了他的朋友Merry。他们一起喝咖啡聊天，计划着下周去加利福尼亚（California）旅行。他们决定一起租车并预订酒店。他们先计划在下周一去圣弗朗西斯科参观旧金山大桥，下周三去洛杉矶拜访Merry的父亲威廉。"

ie_cases = {
    "我将给你个输入，请根据事件类型列表：['旅游行程']，论元角色列表：['旅游地点', '旅游时间', '旅游人员']，从输入中抽取出可能包含的事件，并以(事件触发词,事件类型,[(事件论元,论元角色)])的形式回答。"+"[input]"+origin_sentence: {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},
    "从给定的文本中提取出可能的实体和实体类型，可选的实体类型为['地点', '人名']，以（实体，实体类型）的格式回答。"+"[input]"+origin_sentence: {"num_beams":2, "temperature": 0.4, "repetition_penalty":1.3},
    "我希望你根据关系列表从给定的输入中抽取所有可能的关系三元组，并以JSON字符串[{'head':'', 'relation':'', 'tail':''}, ]的格式回答，relation可从列表['位于', '别名', '朋友', '父亲', '女儿']中选取。"+"[input]"+origin_sentence: {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},
    "我希望你根据关系列表从给定的输入中抽取所有可能的关系三元组，并以\"输入中包含的关系三元组是：关系1：头实体1，尾实体1；关系2：头实体2，尾实体2。\"的格式回答，关系列表=['位于', '别名', '朋友', '父亲', '女儿']。"+"[input]"+origin_sentence: {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},
}
# ======================================================CASE END================================================================

def main(
    run_general_cases: bool = False,
    run_ie_cases: bool = False,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    num_beams = 2,
    temperature = 0.2,
    top_p = 0.75,
    top_k = 40,
    repetition_penalty = 1.3,
    max_new_tokens = 512,
    prompt_template: str = "finetune/lora/templates/alpaca.json",  # The prompt template to use, will default to alpaca.
):
    assert not (run_general_cases and run_ie_cases), "Only one mode!"
    assert run_general_cases or run_ie_cases, "Please Choose One!"
    if run_general_cases:
        print("testing general abilities!")
    if run_ie_cases:
        print("testing ie ablities!")
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(f"load_8bit={load_8bit}")

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=temperature,        
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,            
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        **kwargs,
    ):
        if '[input]' in instruction:
            """only for ie"""
            input=instruction[instruction.find('[input]')+7:]
            instruction=instruction[:instruction.find('[input]')]
            print(f"instruction: {instruction}")
            print(f"input: {input}")
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,        
            repetition_penalty=repetition_penalty,     # add
            **kwargs,
        )
        print(generation_config)
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
        return prompter.get_response(output)

    if run_general_cases:
        for instruction in general_cases:
            cfg = general_cases[instruction]
            print(cfg, instruction)
            print(evaluate(instruction, num_beams=cfg["num_beams"], temperature=cfg["temperature"], repetition_penalty=cfg["repetition_penalty"]))
    if run_ie_cases:
        for instruction in ie_cases:
            cfg = ie_cases[instruction]
            print(evaluate(instruction, num_beams=cfg["num_beams"], temperature=cfg["temperature"], repetition_penalty=cfg["repetition_penalty"]))
    


if __name__ == "__main__":
    fire.Fire(main)
