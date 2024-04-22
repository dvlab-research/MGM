#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="MGM/MGM-8x7B"

output_file=data/MGM-Eval/mm-vet/answers/$CKPT/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"

python -m mgm.eval.model_vqa \
    --model-path work_dirs/$CKPT \
    --question-file data/MGM-Eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder data/MGM-Eval/mm-vet/images \
    --answers-file $output_file \
    --temperature 0 \
    --conv-mode mistral_instruct

mkdir -p data/MGM-Eval/mm-vet/results/$CKPT

python scripts/convert_mmvet_for_eval.py \
    --src $output_file \
    --dst data/MGM-Eval/mm-vet/results/$CKPT/$CKPT.json

