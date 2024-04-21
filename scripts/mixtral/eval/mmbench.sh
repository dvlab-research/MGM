#!/bin/bash

CKPT="MGM/MGM-8x7B"
SPLIT="mmbench_dev_20230712"

python -m mgm.eval.model_vqa_mmbench \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/MGM-Eval/mmbench/$SPLIT.tsv \
    --answers-file ./data/MGM-Eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode mistral_instruct 

mkdir -p ./data/MGM-Eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./data/MGM-Eval/mmbench/$SPLIT.tsv \
    --result-dir ./data/MGM-Eval/mmbench/answers/$SPLIT \
    --upload-dir ./data/MGM-Eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
