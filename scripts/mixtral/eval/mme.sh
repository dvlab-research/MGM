#!/bin/bash
CKPT="MGM/MGM-8x7B"

python -m mgm.eval.model_vqa_loader \
    --model-path work_dirs/$CKPT \
    --question-file data/MGM-Eval/MME/llava_mme.jsonl \
    --image-folder data/MGM-Eval/MME/MME_Benchmark_release_version \
    --answers-file data/MGM-Eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode mistral_instruct


cd data/MGM-Eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
