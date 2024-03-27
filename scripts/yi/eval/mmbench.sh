#!/bin/bash

CKPT="Mini-Gemini/Mini-Gemini-34B"
SPLIT="mmbench_dev_20230712"

CUDA_VISIBLE_DEVICES=0 python -m minigemini.eval.model_vqa_mmbench \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/MiniGemini-Eval/mmbench/$SPLIT.tsv \
    --answers-file ./data/MiniGemini-Eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode chatml_direct 

mkdir -p ./data/MiniGemini-Eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./data/MiniGemini-Eval/mmbench/$SPLIT.tsv \
    --result-dir ./data/MiniGemini-Eval/mmbench/answers/$SPLIT \
    --upload-dir ./data/MiniGemini-Eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
