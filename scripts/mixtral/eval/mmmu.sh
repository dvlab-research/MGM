#!/bin/bash

CKPT="MGM/MGM-8x7B"
CONFIG="mgm/eval/MMMU/eval/configs/llava1.5.yaml"
output_file=./work_dirs/MMMU/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

python mgm/eval/MMMU/eval/run_llava.py \
        --data_path ./data/MGM-Eval/MMMU \
        --config_path $CONFIG \
        --model_path ./work_dirs/$CKPT \
        --answers-file $output_file \
        --split "validation" \
        --conv-mode mistral_instruct

python mgm/eval/MMMU/eval/eval.py --result_file $output_file --output_path ./work_dirs/MMMU/$CKPT/val.json