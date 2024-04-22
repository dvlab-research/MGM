#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="MGM/MGM-34B"

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mgm.eval.model_vqa \
    --model-path work_dirs/$CKPT \
    --question-file data/MGM-Eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder data/MGM-Eval/mm-vet/images \
    --answers-file data/MGM-Eval/mm-vet/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode chatml_direct &
done

wait

output_file=data/MGM-Eval/mm-vet/answers/$CKPT/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/MGM-Eval/mm-vet/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p data/MGM-Eval/mm-vet/results/$CKPT

python scripts/convert_mmvet_for_eval.py \
    --src $output_file \
    --dst data/MGM-Eval/mm-vet/results/$CKPT/$CKPT.json

