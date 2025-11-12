#!/bin/bash

cd ..

export OPENAI_API_KEY=""
export IMG_VERIFICATION_URL= ""
export IMG_DESCRIPTION_URL= ""

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mirror_counseling.sh 5 7b base

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mirror_counseling.sh 5 7b planning

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mirror_counseling.sh 5 7b ec_planning