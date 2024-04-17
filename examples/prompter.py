"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


# class Prompter(object):
#     __slots__ = ("template", "_verbose")

#     def __init__(self, template_name: str = "", verbose: bool = False):
#         self._verbose = verbose
#         file_name = template_name
#         if not osp.exists(file_name):
#             raise ValueError(f"Can't read {file_name}")
#         with open(file_name) as fp:
#             self.template = json.load(fp)
#         if self._verbose:
#             print(
#                 f"Using prompt template {template_name}: {self.template['description']}"
#             )

#     def generate_prompt(
#         self,
#         instruction: str,
#         input: Union[None, str] = None,
#         label: Union[None, str] = None,
#     ) -> str:
#         # returns the full prompt from instruction and optional input
#         # if a label (=response, =output) is provided, it's also appended.
#         if input:
#             res = self.template["prompt_input"].format(
#                 instruction=instruction, input=input
#             )
#         else:
#             res = self.template["prompt_no_input"].format(
#                 instruction=instruction
#             )
#         if label:
#             res = f"{res}{label}"
#         if self._verbose:
#             print(res)
#         return res

#     def get_response(self, output: str) -> str:
#         return output.split(self.template["response_split"])[1].strip()


class Prompter:

    __SUPPORT_MODEL__ = ['oneke', 'zhixi']

    def __init__(self, model_name:str=None, prompt_template:str=None):
        assert prompt_template or model_name,\
            "You must specify `model_name` or `prompt_template`"
        if model_name:
            assert model_name in Prompter.__SUPPORT_MODEL__, \
                f"`{model_name}` is not support. You can change it by yourself."
        self.model_name = model_name
        self.prompt_template = prompt_template

        self.prompt_mapping = {
            'oneke': self.__oneke,
            'zhixi': self.__zhixi,
        }
        
        self.__cache_prompt = ""

    def __oneke(self, schema:str, input:str, instruction:str=None, system_prompt:str=None, model_prompt="[INST] {} [/INST]"):
        if instruction == None:
            instruction = "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string."
        if system_prompt == None:
            system_prompt = "You are a helpful assistant. 你是一个乐于助人的助手。"
        system_prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        json_prompt = f"""{{"instruction": "{instruction}", "schema": {schema}, "input": "{input}"}}"""

        final_prompt = model_prompt.format(
            system_prompt + json_prompt
        )
        return final_prompt
    
    def __zhixi(self, instruction:str, input:str=None):
        if instruction and input:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    def generate_prompt(self, *arg, **args) -> str:
        if self.model_name == None:
            if arg:
                final_prompt = self.prompt_template.format(arg[0])
            else:
                assert args
                final_prompt = self.prompt_template(**args)
        else:
            final_prompt = self.prompt_mapping[self.model_name](**args)
        self.__cache_prompt = final_prompt
        return final_prompt

    def get_response(self, output:str) -> str:
        return self.__cache_prompt.join(output.split(self.__cache_prompt)[1:])
        # return output[len(self.__cache_prompt):]

def test_prompter():
    prompter = Prompter(model_name='oneke')
    TEST_GT = "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"} [/INST]"
    prompt = prompter.generate_prompt(
        schema='["person", "organization", "else", "location"]',
        input="284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )"
    )
    assert prompt == TEST_GT

    prompter = Prompter(model_name='zhixi')
    prompt = prompter.generate_prompt(
        instruction='only instruction'
    )
    print(prompt)
    prompt = prompter.generate_prompt(
        instruction='instruction1',
        input='input1'
    )
    print(prompt)

    prompter = Prompter(prompt_template="[INST] {} [/INST]")
    prompt = prompter.generate_prompt(
        'input'
    )
    print(prompt)

if __name__ == '__main__':
    test_prompter()