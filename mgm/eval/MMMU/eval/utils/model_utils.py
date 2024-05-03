from random import random
import torch

def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None):
    from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from mgm.conversation import conv_templates, SeparatorStyle

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end, ocr_tokens):
        if ocr_tokens is not None:
            qs = input_text + '\n' + ocr_tokens
        else:
            qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    ocr_tokens = sample.get('ocr', None)
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end, ocr_tokens)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    image_aux = sample['image_aux']
    if image_aux is not None:
        image_aux = image_aux.unsqueeze(0).half().cuda()
    
    terminators = tokenizer.eos_token_id
    if "llama_3" in args.conv_mode:
        terminators = [terminators, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    if image is not None:
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).half().cuda(),
            images_aux=image_aux,
            do_sample=True,
            temperature=1,
            num_beams=5,
            top_p=None,
            max_new_tokens=128,
            bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
            eos_token_id=terminators,  # End of sequence token
            pad_token_id=tokenizer.pad_token_id,  # Pad token
            use_cache=True)

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip('\n')
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response
