import sys

import fire
import torch
import transformers

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

cases = [
    "use python to write the bubble sort algorithm.",
    "自然语言处理是",
    "你对中国的大学了解的非常多。请评价一下浙江大学是一所怎样的学校。",
    "你很擅长将中文翻译成英文。将下面的句子翻译成英文：我们今天准备去西安看兵马俑。答案：",
    "你非常了解一些健康生活的习惯，请列举几个健康生活的建议：",
    "You are good at translating English into Chinese. Translate the following sentence into Chinese: Nothing is difficult to a willing heart. Answer:",
    "Here is the recommendation letter that I wrote for an application to a dragon feeder position at the Magic Unicorn Corporation:\nDear recruiter",
    "Can you help me write a formal email to a potential business partner proposing a joint venture? Your answer:",
    "using java to sort an unsorted array. Answer:",
    "这是我为我的学生申请浙江大学博士的推荐信：",
    "床前明月光，疑是地上霜。",
    "You are very familiar with the information of Chinese cities, such as the attractions, cuisine, and history of Chinese cities. Please introduce the city of Hangzhou. Hangzhou",
    "你阅读过李白的所有诗歌。李白的《将进酒》的原文是",
    "You are now a doctor. Here are some tips for a healthy life. 1.",
    "你非常了解中国的大学。请介绍一下浙江大学。",
    "你对中国的大学了解的非常多。请介绍一下浙江大学。答案：",
    "我爱你的英文是什么？",
    "Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\nAnswer: Roger started with 5 balls. 2 cans of 3 each is 6 tennis balls. 5 + 6 = 11. The answer is 11.\nQuestion: The cafeteria had 23 apples. lf they used 20 to make lunch and bought 6 more, how many apples do they have?\nAnswer: Cafeteria started with 23 apples.",
]

def main(
    load_8bit: bool = False,
    base_model: str = "../recover",
    max_new_tokens = 512,
    temperature = 0.3,
    num_beams = 3,
    top_p = 0.75,
    top_k = 40,
    repetition_penalty=1.6,
    interactive: bool = False,
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
    

    if not interactive:
        print(f"{'='*30}INFO{'='*31}")
        print("zhixi-13b loaded successfully, the next is case :)")
        print(f"{'='*30}START{'='*30}")
        for inputs in cases:
            print(f"Output: {evaluate(input=inputs)}")
    else:
        print(f"{'='*30}INFO{'='*31}")
        print("zhixi-13b loaded successfully, please input prompt :)")
        print("if you want to exit, please input exit :)")
        print(f"{'='*30}START{'='*30}")
        while True:
            prompt = input("Input: ")
            if prompt.strip().lower() == "exit":
                break
            print(f"Output: {evaluate(input=prompt)}")

if __name__ == "__main__":
    fire.Fire(main)
