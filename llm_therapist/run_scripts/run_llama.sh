#!/bin/bash
export OPENAI_API_KEY=""

cd ..

CUDA_VISIBLE_DEVICES=0 python -m src.run \
    --client_model_name gpt-3.5-turbo \
    --counselor_model_path /home/model/Meta-Llama-3-8B-Instruct \
    --input_data ../../data/processed/test.csv