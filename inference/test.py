import json
import aiohttp
import argparse
from enum import Enum
import numpy as np
import asyncio
import time
from typing import List

from transformers import LlamaTokenizer, PreTrainedTokenizer

from utils import Prompter

np.random.seed(2023)


class GenerationBackend(str, Enum):
    HfTextGenerationInference = "HfTextGenerationInference"
    vLLM = "vLLM"
    NaiveHfPipeline = "NaiveHfPipeline"


def load_prompt(prompt_path, prompter: Prompter, num_examples=10000):

    with open(prompt_path, 'r') as f:
        records = json.load(f)

    random_sample_indices = np.random.permutation(len(records))[:num_examples]
    sampled_prompts = []
    for idx in random_sample_indices:
        record = records[idx]
        prompt = prompter.generate_prompt(instruction=record['instruction'], input=record['input'])
        sampled_prompts.append(prompt)

    return sampled_prompts


def get_prompt_lens(prompts, tokenizer: PreTrainedTokenizer):

    tokenized_prompts = tokenizer(prompts)
    tokenized_inputs = tokenized_prompts['input_ids']

    prompt_lens = [len(item) for item in tokenized_inputs]
    print(f'Prompt lens AVG {np.mean(prompt_lens):.3f}')
    print(f'Prompt lens MAX {np.max(prompt_lens)}')
    print(f'Prompt lens MEDIAN {np.median(prompt_lens)}')

    return prompt_lens


async def query_model_hf(prompt, verbose, tokenizer, total_requests, port):

    timeout = aiohttp.ClientTimeout(total=60*60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            inputs=prompt,
            max_length=256,
            top_p=0.7,
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))

            return (prompt, output)
        
        
async def query_model_vllm(prompt, verbose, tokenizer, total_requests, port):

    timeout = aiohttp.ClientTimeout(total=4*60*60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            inputs=prompt,
            parameters=dict(
                top_p=0.7,
                max_tokens=256
            ),
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}/generate', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            # necessary for latency calc
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))

            return (prompt, output)
        

def get_wait_time(mean_time_between_requests: float, distribution: str) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    else:
        return np.random.exponential(mean_time_between_requests)


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return
        

async def async_request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return
        

class MeasureLatency:
    def __init__(self):
        self._latencies = []
        self._per_token_latencies = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, output = await f(*args, **kwargs)

            # Do not record latency if request failed.
            if 'generated_text' in output:
                latency = time.time() - start
                self._latencies.append(latency)
                try:
                    num_output_tokens = int(len(output['generated_text'].split()) * 1.2)
                    self._per_token_latencies.append(
                        latency / num_output_tokens)
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass

            return prompt, output
        return measured
    

def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    # print(lens)
    return lens
    

def calculate_throughput(queries, dur_s, backend, tokenizer, median_token_latency, median_e2e_latency, all_e2e_latencies, all_per_token_latencies, results_filename, log_latencies):
    prompts = []
    responses = []
    naive_hf_lens = []
    ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        if 'generated_text' in response:
            prompts.append(prompt)
            responses.append(response['generated_text'])
        if 'naive_hf_lens' in response:
            naive_hf_lens.append(response['naive_hf_lens'])
        if 'ray_gen_len' in response:
            ray_gen_lens.append(response['ray_gen_len'])
        if 'num_output_tokens_cf' in response:
            cf_gen_lens.append(response['num_output_tokens_cf'])

    prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)['input_ids']]
    response_ids = [r for r in tokenizer.batch_encode_plus(responses)['input_ids']]

    print(
        f'check_len actual {list(sorted(len(response) for response in response_ids))}')
    print(f'   self-reported {list(sorted(cf_gen_lens))}')

    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #    print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    print(f'naive_hf_lens {list(sorted(naive_hf_lens))}')
    print(f'prompt_lens {list(sorted(prompt_lens))}')
    print(f'calc_throughput response_lens {list(sorted(response_lens))}')

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    if naive_hf_lens:
        # Manually count naive hf tok len
        total_resp_tokens = sum(
            [response_len for _, response_len in naive_hf_lens])
        total_prompt_tokens = sum(
            [prompt_len for prompt_len, _ in naive_hf_lens])

        response_token_count = total_prompt_tokens + total_resp_tokens

    if ray_gen_lens:
        response_token_count = sum(ray_gen_lens)

    if backend == GenerationBackend.NaiveHfPipeline:
        # It returns the prompt in the output.
        prompt_token_count = 0

    if cf_gen_lens:
        response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')

    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    # print(f'throughput_tok_s {throughput_tok_s:.02f}')

    qps = len(responses) / dur_s

    with open(results_filename, 'a') as f:
        msg = f'backend {backend} dur_s {dur_s:.02f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.02f} successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}, {median_token_latency=}, {median_e2e_latency=}'
        if log_latencies:
            msg += f' {all_e2e_latencies=} {all_per_token_latencies=}'
        print(msg, file=f)
        print(msg)
        

async def benchmark(
    backend: GenerationBackend,
    tokenizer,
    prompts: List[str],
    verbose: bool,
    results_filename: str,
    port: int,
    distribution: str,
    qps: float,
    log_latencies: bool,
):

    if backend == GenerationBackend.HfTextGenerationInference:
        query_model = query_model_hf
    elif backend == GenerationBackend.vLLM:
        query_model = query_model_vllm
    else:
        raise ValueError(f'unknown backend {backend}')

    m = MeasureLatency()

    query_model = m.measure(query_model)

    if distribution == "burst":
        qps = float('inf')

    print(
        f'Starting with backend={backend}, num_prompts={len(prompts)}')
    print(f'traffic distribution={distribution}, qps={qps}')

    total_requests = len(prompts)

    async_prompts = async_request_gen(
        iter(prompts), qps=qps, distribution=distribution)

    start_time = time.time()
    tasks = []
    async for prompt in async_prompts:
        tasks.append(asyncio.create_task(query_model(
            prompt, verbose, tokenizer, total_requests, port)))
    queries = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time

    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._latencies)

    calculate_throughput(queries, dur_s, backend, tokenizer, median_token_latency, median_e2e_latency,
                         m._latencies, m._per_token_latencies, results_filename, log_latencies)
    # calculate_cdf(m._latencies)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--prompt_file', type=str, default='data/training_data/alpaca_data_cleaned.json')
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--template_file', type=str, default='data/templates/alpaca.json')
    parser.add_argument('--results_filename', type=str, default='log')
    parser.add_argument(
        '--distribution', choices=["burst", "uniform", "poisson"], default="burst")
    parser.add_argument('--qps', type=float, default=5.0)
    parser.add_argument('--log_latencies', action="store_true",
                        help="Whether or not to write all latencies to the log file.")
    args = parser.parse_args()

    backend = GenerationBackend[args.backend]

    tokenizer = LlamaTokenizer.from_pretrained(
        'data/zhixi-13B',
        padding_side="left",
        truncation_side="left",
        bos_token='<s>', eos_token='</s>', add_bos_token=True, add_eos_token=False
    )
    tokenizer.pad_token_id = 0  # same as unk token id
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    prompter = Prompter(args.template_file)

    prompts = load_prompt(args.prompt_file, prompter=prompter, num_examples=args.num_examples)
    prompt_lens = get_prompt_lens(prompts=prompts, tokenizer=tokenizer)

    asyncio.run(benchmark(
        backend,
        tokenizer,
        prompts,
        args.verbose,
        args.results_filename,
        args.port,
        args.distribution,
        args.qps,
        args.log_latencies
    ))



if __name__ == '__main__':

    main()