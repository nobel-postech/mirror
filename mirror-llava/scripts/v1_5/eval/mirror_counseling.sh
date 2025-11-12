
#!/bin/bash
epoch=3
input_data="./playground/data/eval/test.csv"
output_dir="./playground/data/eval/answers"

model_path="./checkpoints/llava-v1.5-size-mirror-task-lora-epoch"$epoch
model_base="~/model/llava-v1.5-size"

image_dir="./playground/data/eval/mirror_images"


if [[ $# -ge 3 ]]; then
    epoch=$1
    model_path="${model_path/epoch3/epoch$1}"
    model_path="${model_path/size/$2}"    
    model_base="${model_base/size/$2}"
    image_dir="${image_dir/images/images_$2_$3}"
    
    model_path="${model_path/mirror/mirror-$3}"
else
    true
fi
echo "Epoch: $epoch"
echo "Question File: $input_data"
echo "Output Directory: $output_dir"
echo "Model Path: $model_path"
echo "Base Model: $model_base"

python -m llava.chat.model_chat \
    --model-path $model_path \
    --model-base $model_base \
    --input_data $input_data \
    --output_dir $output_dir \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --image_save_dir $image_dir
