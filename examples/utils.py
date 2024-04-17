import gradio as gr
from typing import Dict, Tuple, List

class Web:
    __SUPPORT_MODEL__ = ['zhixi', 'oneke']
    
    @classmethod
    def get_ui(cls, name:str) -> Dict:
        if name == 'zhixi':
            return cls.__zhixi_ui()
        elif name == 'oneke':
            return cls.__oneke_ui()
        else:
            assert False
        # return {
        #     'zhixi': Web.__zhixi_ui,
        #     'oneke': Web.__oneke_ui
        # }[name]()

    @classmethod
    def __zhixi_ui(cls) -> Dict:
        gradio_components: list = [
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="<请在此输入你的问题>",
            ),
            gr.components.Textbox(
                lines=2, 
                label="Input", 
                placeholder="<可选参数>",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.4, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=2, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=512, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=1, maximum=2, step=0.1, value=1.3, label="Repetition Penalty"
            ),
            gr.components.Checkbox(label="Stream output"),
        ]
        corresponding_name: list = [
            'instruction', 'input',
            'temperature', 'top_p', 'top_k',
            'num_beams', 'max_new_tokens', 'repetition_penalty',
            'stream_output'
        ]
        assert len(gradio_components) == len(corresponding_name)
        web_title = '智析'
        web_description = "智析（ZhiXi）是基于LLaMA-13B，先使用中英双语进行全量预训练，然后使用指令数据集进行LoRA微调（我们专门针对信息抽取进行优化）。如果希望获得更多信息，请参考[KnowLM](https://github.com/zjunlp/knowlm)。如果出现重复或者效果不佳，请调整repeatition_penalty、beams两个参数。"
        return {
            'components': gradio_components, 
            'var_name': corresponding_name,
            'title': web_title,
            'description': web_description
        }

    @classmethod
    def __oneke_ui(cls) -> Dict:
        corresponding_name = [
            'system_prompt', 'instruction', 'schema',
            'input', 'temperature', 'top_p', 'top_k', 
            'num_beams', 'max_new_tokens', 'repetition_penalty', 'stream_output'
        ]
        gradio_components = [
            gr.components.Textbox(
                lines=2,
                label="System Prompt",
                placeholder="<system prompt>",
                value="You are a helpful assistant. 你是一个乐于助人的助手。"
            ),
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="<instruction>",
                value="You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string."
            ),
            gr.components.Textbox(
                lines=2,
                label="Schema",
                placeholder="<schema>",
                value="""["person", "organization", "else", "location"]"""
            ),
            gr.components.Textbox(
                lines=2, 
                label="Input", 
                placeholder="<input>",
                value="284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )"
            ),
            
            gr.components.Slider(
                minimum=0, maximum=1, value=0.4, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=2, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=512, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=1, maximum=2, step=0.1, value=1.3, label="Repetition Penalty"
            ),
            gr.components.Checkbox(label="Stream output"),
        ]
        assert len(corresponding_name) == len(gradio_components)
        web_title = "OneKE"
        web_description = "<center>Model: https://huggingface.co/zjunlp/OneKE</center><br><center>Project: http://oneke.openkg.cn/</center>"
        return {
            'components': gradio_components, 
            'var_name': corresponding_name,
            'title': web_title,
            'description': web_description
        }