"""download model and lora"""
import argparse
from huggingface_hub import snapshot_download
import time

def _print(message):
    print(f"[{time.ctime()}] {message}")

def add_argument():
    parser = argparse.ArgumentParser(description="download")
    parser.add_argument('--download_path', type=str, default='./CaMA', help="storage directory")
    parser.add_argument('--only_lora', action='store_true', default=False)
    parser.add_argument('--only_base', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--both', action='store_true', default=False)
    parser.add_argument('--specify', action='store_true', default=False)
    parser.add_argument('--repo_name', type=str, default="decapoda-research/llama-13b-hf")
    args = parser.parse_args()
    return args

def check_args(args):
    assert args.only_lora or args.only_base or args.both, \
        "Please select the file to download."
    assert (args.only_lora and not args.only_base and not args.both) \
           or (not args.only_lora and args.only_base and not args.both) \
           or (not args.only_lora and not args.only_base and args.both), \
        "args conflict."


if __name__ == '__main__':
    args = add_argument()
    if args.specify:
        _print(f"downloading {args.repo_name}......")
        snapshot_download(repo_id=args.repo_name, local_dir=args.download_path)
        _print("done!")
        exit()
    check_args(args)

    download = []
    if args.only_base or args.both:
        if args.fp16:
            download.append("ZhiXi-13B-Diff-fp16")
        else:
            download.append("ZhiXi-13B-Diff")
    if args.only_lora or args.both: download.append("ZhiXi-13B-LoRA")

    for file in download:
        _print(f"downloading {file}......")
        snapshot_download(repo_id=f"zjunlp/{file}", local_dir=args.download_path)
    _print("done!")
