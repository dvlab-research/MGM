import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            if hasattr(model, "update_prompt"):
                model.update_prompt([[cur_prompt]])
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            if hasattr(model.config, 'image_size_aux'):
                if not hasattr(image_processor, 'image_size_raw'):
                    image_processor.image_size_raw = image_processor.crop_size.copy()
                image_processor.crop_size['height'] = model.config.image_size_aux
                image_processor.crop_size['width'] = model.config.image_size_aux
                image_processor.size['shortest_edge'] = model.config.image_size_aux

            image_tensor = process_images([image], image_processor, model.config)[0]
            image_grid = getattr(model.config, 'image_grid', 1)
            if hasattr(model.config, 'image_size_aux'):
                raw_shape = [image_processor.image_size_raw['height'] * image_grid, 
                            image_processor.image_size_raw['width'] * image_grid]
                image_tensor_aux = image_tensor
                image_tensor = torch.nn.functional.interpolate(image_tensor[None], 
                                                            size=raw_shape, 
                                                            mode='bilinear', 
                                                            align_corners=False)[0]
            else:
                image_tensor_aux = []

            if image_grid >= 2:            
                raw_image = image_tensor.reshape(3, 
                                                image_grid,
                                                image_processor.image_size_raw['height'],
                                                image_grid,
                                                image_processor.image_size_raw['width'])
                raw_image = raw_image.permute(1, 3, 0, 2, 4)
                raw_image = raw_image.reshape(-1, 3,
                                            image_processor.image_size_raw['height'],
                                            image_processor.image_size_raw['width'])
                
                if getattr(model.config, 'image_global', False):
                    global_image = image_tensor
                    if len(global_image.shape) == 3:
                        global_image = global_image[None]
                    global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[image_processor.image_size_raw['height'],
                                                                image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                    # [image_crops, image_global]
                    raw_image = torch.cat([raw_image, global_image], dim=0)
                image_tensor = raw_image.contiguous()
            
            images = image_tensor[None].to(dtype=model.dtype, device='cuda', non_blocking=True)
            images_aux = image_tensor_aux[None].to(dtype=model.dtype, device='cuda', non_blocking=True) if len(image_tensor_aux)>0 else None

            terminators = tokenizer.eos_token_id
            if "llama_3" in args.conv_mode:
                terminators = [terminators, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    images_aux=images_aux,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                    eos_token_id=terminators,  # End of sequence token
                    pad_token_id=tokenizer.pad_token_id,  # Pad token
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
