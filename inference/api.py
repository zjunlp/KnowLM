import json
from fastapi import FastAPI, Request
import uvicorn
import datetime

import torch
from transformers import LlamaTokenizer
from transformers import GenerationConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from utils import Prompter


app = FastAPI()

@app.post("/")
async def complement(request: Request):
    global model, tokenizer, prompter

    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    instruction = json_post_list.get('instruction')
    input = json_post_list.get('input')
    prompt = prompter.generate_prompt(instruction=instruction, input=input)

    max_length = json_post_list.get('max_length', 16)
    top_p = json_post_list.get('top_p', 1.0)
    temperature = json_post_list.get('temperature', 1.0)
    repetition_penalty=json_post_list.get('repetition_penalty', 1.0)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_length,
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    response = prompter.get_response(output)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "generated_text": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", ' + response
    print(log)

    return answer


if __name__ == '__main__':

    tokenizer = LlamaTokenizer.from_pretrained('data/zhixi-13B-4bit')
    model = AutoGPTQForCausalLM.from_quantized('data/zhixi-13B-4bit', device='cuda:3')
    print(model.hf_device_map)
    print(model.device)
    prompter = Prompter('data/templates/alpaca.json')

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8800, workers=1)
