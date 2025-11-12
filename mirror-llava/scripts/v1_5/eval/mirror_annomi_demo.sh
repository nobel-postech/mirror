
#!/bin/bash
epoch=3
input_data="~/data/AnnoMI/prompt_df.csv"
output_path="./playground/data/eval/answers/mirror-llava-v1.5-size_cot-epoch"$epoch
image_dir="~/data/AnnoMI/images"
model_path="./checkpoints/llava-v1.5-size-mirror-task-lora-epoch"$epoch
model_base="~/model/llava-v1.5-size"

if [[ $# -ge 3 ]]; then
    epoch=$1
    model_path="${model_path/epoch3/epoch$1}"
    model_path="${model_path/size/$2}"    
    model_base="${model_base/size/$2}"
    
    model_path="${model_path/mirror/mirror-$3}"
    output_path="${output_path/cot/$3}"
    output_path="${output_path/size/$2}"
    output_path="${output_path/epoch3/epoch$1}"
else
    true
fi
echo "Epoch: $epoch"
echo "Question File: $input_data"
echo "Output Path: $output_path"
echo "Model Path: $model_path"
echo "Base Model: $model_base"

python -m llava.chat.demo \
    --model-path $model_path \
    --model-base $model_base \
    --data-path $input_data \
    --annomi-dir $image_dir \
    --output-path $output_path \
    --temperature 0 \
    --conv-mode vicuna_v1 \
