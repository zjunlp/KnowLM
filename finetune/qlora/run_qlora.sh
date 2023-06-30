# DATA
model_name=data/llama-7b
data_name=./data/training_data/alpaca_data_cleaned.json
output_dir=data/output_data/alpaca_lora

source_max_len=128
target_max_len=384

# MODEL
# lora
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.1

# TRAIN
# quant
bits=4
quant_type=nf4

# optim
max_steps=1000
warmup_ratio=0.05
optim=paged_adamw_32bit
lr_scheduler=constant
learning_rate=2e-4
per_device_train_batch_size=8
gradient_accumulation_steps=1
max_grad_norm=0.3


CUDA_VISIBLE_DEVICES=2 python train.py \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --dataset $data_name \
    --dataset_format alpaca \
    --eval_dataset_size 1024 \
    --source_max_len $source_max_len \
    --target_max_len $target_max_len \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --double_quant \
    --fp16 \
    --bits $bits \
    --quant_type $quant_type \
    --max_steps $max_steps \
    --warmup_ratio $warmup_ratio \
    --optim $optim \
    --weight_decay 0.0 \
    --max_grad_norm $max_grad_norm \
    --learning_rate $learning_rate \
    --lr_scheduler_type $lr_scheduler \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --logging_steps 10 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --per_device_eval_batch_size 1 \
    --save_steps 500 \
    --save_total_limit 40 \
    --dataloader_num_workers 4 \
    --seed 2023
