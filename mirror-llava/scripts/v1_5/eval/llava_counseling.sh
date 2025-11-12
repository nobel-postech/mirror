
#!/bin/bash
input_data="./playground/data/eval/test.csv"
output_dir="./playground/data/eval/answers"

model_base="~/model/llava-v1.5-size"
image_dir="./playground/data/eval/llava_images"

if [[ $# -ge 1 ]]; then
    model_base="${model_base/size/$1}"
    image_dir="${image_dir/images/images_$1}"
else
    true
fi
echo "Question File: $input_data"
echo "Output Directory: $output_dir"
echo "Base Model: $model_base"

python -m llava.chat.model_chat \
    --model-path $model_base \
    --input_data $input_data \
    --output_dir $output_dir \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --image_save_dir $image_dir
