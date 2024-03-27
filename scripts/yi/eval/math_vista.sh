#!/bin/bash
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="Mini-Gemini/Mini-Gemini-34B"
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m minigemini.eval.model_math_vista \
        --model-path work_dirs/$CKPT \
        --question-file data/MiniGemini-Eval/MathVista/testmini.json \
        --image-folder data/MiniGemini-Eval/MathVista \
        --answers-file data/MiniGemini-Eval/MathVista/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode chatml_direct &
done

wait

output_file=./data/MiniGemini-Eval/MathVista/answers/$CKPT/merge.jsonl
score_file=./data/MiniGemini-Eval/MathVista/answers/$CKPT/score.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/MiniGemini-Eval/MathVista/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python minigemini/eval/MathVista/extract_answer.py \
    --output_file $output_file \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE

python minigemini/eval/MathVista/calculate_score.py \
    --output_file $output_file \
    --score_file $score_file \
    --gt_file data/MiniGemini-Eval/MathVista/testmini.json
