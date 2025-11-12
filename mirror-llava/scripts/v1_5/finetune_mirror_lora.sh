#!/bin/bash

epoch=3
question_file="./playground/data/train/mirror.json"
output_dir="./checkpoints/llava-v1.5-size-mirror-task-lora-epoch"$epoch
model_name_or_path="~/model/llava-v1.5-size"

if [[ $# -ge 3 ]]; then
    epoch=$1
    output_dir="${output_dir/epoch3/epoch$1}"
    output_dir="${output_dir/size/$2}"    
    model_name_or_path="${model_name_or_path/size/$2}"

    question_file="${question_file/mirror/mirror_$3}"
    output_dir="${output_dir/mirror/mirror_$3}"

elif [[ $# -ge 2 ]]; then
    epoch=$1
    output_dir="${output_dir/epoch3/epoch$1}"
    output_dir="${output_dir/size/$2}"    
    model_name_or_path="${model_name_or_path/size/$2}"
else
    echo "Usage: $0 <epoch> <size>"
    echo "Example: $0 50 7b"
    exit 1
fi

echo "Epoch: $epoch"
echo "Question File: $question_file"
echo "Output Directory: $output_dir"
echo "Model Path: $model_name_or_path"

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_name_or_path \
    --version v1 \
    --data_path $question_file \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
