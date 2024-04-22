#!/bin/bash

CKPT="MGM/MGM-34B"

CUDA_VISIBLE_DEVICES=0 python -m mgm.eval.model_vqa_loader \
    --model-path work_dirs/$CKPT \
    --question-file data/MGM-Eval/MME/llava_mme.jsonl \
    --image-folder data/MGM-Eval/MME/MME_Benchmark_release_version \
    --answers-file data/MGM-Eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode chatml_direct

cd data/MGM-Eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
