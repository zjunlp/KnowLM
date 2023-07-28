MODEL_NAME=data/zhixi-13B

USE_FLASH_ATTENTION=true CUDA_VISIBLE_DEVICES=1 \
    text-generation-launcher \
    --model-id $MODEL_NAME \
    --quantize bitsandbytes \
    --num-shard 1 \
    --max-input-length 512 \
    --max-total-tokens 1024 \
    --max-batch-total-tokens 4000 \
    --max-waiting-tokens 7 \
    --waiting-served-ratio 1.2 \
    --port 8080