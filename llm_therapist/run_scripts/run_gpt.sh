#!/bin/bash
export OPENAI_API_KEY=""

cd ..

python -m src.run \
    --client_model_name gpt-3.5-turbo \
    --counselor_model_path gpt-3.5-turbo \
    --input_data ../../data/processed/test.csv

