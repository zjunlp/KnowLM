import os

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from queue import Queue
from typing import List, Dict, Optional
import time
import asyncio
import argparse
import threading
from dataclasses import dataclass
from vllm import EngineArgs, SamplingParams

from utils import Prompter
from llm_engine import LLMEngine


app = FastAPI()


@dataclass
class GenerationInputs:
    req_id: int
    prompt: str
    sampling_config: dict


@dataclass
class GenerationOutput:
    req_id: int
    generated_text: str
    num_output_tokens: int
    error: str


class ModelThread:
    def __init__(self, vllm_args, model_ready_event, progress_call, loop):
        self.vllm_args = vllm_args
        self.model_ready_event = model_ready_event
        self.thread = None
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.progress_call = progress_call
        self.loop = loop

    def start_thread(self):
        self.thread = threading.Thread(target=self._thread, daemon=True)
        self.thread.start()

    def _thread(self):
        server = self.init_model(self.vllm_args)

        self.model_ready_event.set()

        while True:
            time.sleep(0.01)

            while not self.input_queue.empty():
                gen_input = self.input_queue.get_nowait()

                prompt = gen_input.prompt
                sampling_params = SamplingParams(
                    n=1,
                    top_p=gen_input.sampling_config.get('top_p', 1.0),
                    top_k=gen_input.sampling_config.get('top_k', -1),
                    max_tokens=gen_input.sampling_config.get('max_tokens', 16),
                    presence_penalty=gen_input.sampling_config.get('presence_penalty', 0.0),
                    frequency_penalty=gen_input.sampling_config.get('frequency_penalty', 0.0),
                    temperature=gen_input.sampling_config.get('temperature', 1.0),
                )

                req_id = gen_input.req_id

                server.add_request(
                    str(req_id),
                    prompt,
                    sampling_params,
                )

            vllm_outputs = server.step()

            needs_call_progress = False
            for vllm_output in vllm_outputs:
                if not vllm_output.finished:
                    continue

                needs_call_progress = True
                assert len(vllm_output.outputs) == 1
                req_id = int(vllm_output.request_id)
                generated_text = vllm_output.outputs[0].text
                num_output_tokens = len(vllm_output.outputs[0].token_ids)

                gen_output = GenerationOutput(
                    req_id=req_id,
                    generated_text=generated_text,
                    num_output_tokens=num_output_tokens,
                    error=None,
                )
                self.output_queue.put_nowait(gen_output)

            if needs_call_progress:
                asyncio.run_coroutine_threadsafe(self.progress_call(), loop)

    @staticmethod
    def init_model(vllm_args):
        print('Init model')
        server_args = EngineArgs.from_cli_args(vllm_args)
        server = LLMEngine.from_engine_args(server_args)
        print('Model ready')
        return server


class FastAPIServer:
    def __init__(self, loop, vllm_args):
        self.model_ready_event = asyncio.Event()

        self.requests = {}
        self.generations = {}
        self.request_queue = []
        self._next_req_id = 0

        self.loop = loop

        self.model_thread = ModelThread(
            vllm_args, self.model_ready_event, self.progress_async, self.loop)
        self.model_thread.start_thread()

    @property
    def next_req_id(self):
        rval = self._next_req_id
        self._next_req_id += 1
        return rval

    async def progress_async(self):
        return self.progress()

    def progress(self):
        sent_to_model = 0
        recv_from_model = 0

        for req_id in self.request_queue:
            prompt, sampling_config = self.requests[req_id]
            gen_inputs = GenerationInputs(
                req_id,
                prompt,
                sampling_config,
            )
            self.model_thread.input_queue.put_nowait(gen_inputs)
            sent_to_model += 1
        self.request_queue = []

        found_outputs = []
        while not self.model_thread.output_queue.empty():
            gen_output = self.model_thread.output_queue.get_nowait()
            found_outputs.append(gen_output)
            recv_from_model += 1

        for output in found_outputs:
            req_id = output.req_id
            ready_event, _, _, _ = self.generations[req_id]
            self.generations[req_id] = (
                ready_event, output.generated_text, output.num_output_tokens, output.error)
            ready_event.set()

        print(f'progress {sent_to_model=} {recv_from_model=}')

    async def is_ready(self):
        return self.model_ready_event.is_set()

    def add_request(self, prompt, sampling_config):
        req_id = self.next_req_id
        self.requests[req_id] = (prompt, sampling_config)
        self.request_queue.append(req_id)

        ready_event = asyncio.Event()
        self.generations[req_id] = (ready_event, None, None, None)
        return req_id

    async def get_generation(self, req_id):
        ready_event, _, _, _ = self.generations[req_id]
        await ready_event.wait()
        _, generation, num_output_tokens, error = self.generations[req_id]

        del self.generations[req_id]
        del self.requests[req_id]
        return generation, num_output_tokens, error

    async def generate(self, request_dict: Dict):
        global prompter

        instruction = request_dict['instruction']
        input = request_dict['input']
        prompt = prompter.generate_prompt(instruction=instruction, input=input)
        sampling_config = request_dict['parameters']

        req_id = self.add_request(prompt, sampling_config)
        self.progress()
        generation, num_output_tokens, error = await self.get_generation(req_id)

        return {
            'generated_text': generation,
            'num_output_tokens_cf': num_output_tokens,
            'error': error,
        }


@app.post("/generate")
async def generate_stream(request: Request):
    request_dict = await request.json()
    return await server.generate(request_dict)


@app.get("/is_ready")
async def is_ready(request: Request):
    return await server.is_ready()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--template_path', type=str, default='data/templates/alpaca.json')
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    vllm_args = EngineArgs.from_cli_args(args)
    
    prompter = Prompter(args.template_path)

    loop = asyncio.new_event_loop()
    server = FastAPIServer(loop, vllm_args)

    from uvicorn import Config, Server
    config = Config(app=app, loop=loop, host='0.0.0.0',
                    port=args.port, log_level="info")
    uvicorn_server = Server(config)

    loop.run_until_complete(uvicorn_server.serve())
