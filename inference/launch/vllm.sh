max_num_batched_tokens=2048

CUDA_VISIBLE_DEVICES=0,1 python inference/launch_vllm.py \
    --port 12345 \
    --model data/zhixi-13B \
    --use-np-weights \
    --max-num-batched-tokens $max_num_batched_tokens \
    --dtype half \
    --tensor-parallel-size 2