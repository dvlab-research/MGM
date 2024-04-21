#!/bin/bash

CKPT="MGM/MGM-8x7B"
output_file=./work_dirs/textvqa/answers/$CKPT/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
 
python -m mgm.eval.model_vqa_loader \
        --model-path ./work_dirs/$CKPT \
        --question-file ./data/MGM-Eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./data/MGM-Eval/textvqa/train_images \
        --answers-file $output_file \
        --temperature 0 \
        --conv-mode mistral_instruct 

python -m mgm.eval.eval_textvqa \
    --annotation-file ./data/MGM-Eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file
