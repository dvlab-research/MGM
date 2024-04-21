#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import transformers
import safetensors
from transformers.deepspeed import is_deepspeed_zero3_enabled
import deepspeed

from .multimodal_encoder.builder import build_vision_tower, build_vision_tower_aux
from .multimodal_projector.builder import build_vision_projector

from mgm.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, 
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)

IS_NEW_TRANSFORMERS = transformers.__version__ >= "4.34.0"

class MGMMetaModel:

    def __init__(self, config):
        super(MGMMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_vision_tower_aux"):
            self.vision_tower_aux = build_vision_tower_aux(config, delay_load=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_tower_aux(self):
        vision_tower_aux = getattr(self, 'vision_tower_aux', None)
        if type(vision_tower_aux) is list:
            vision_tower_aux = vision_tower_aux[0]
        return vision_tower_aux

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        vision_tower_aux = model_args.vision_tower_aux
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_aux = vision_tower_aux

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if vision_tower_aux is not None:
            if self.get_vision_tower_aux() is None:
                vision_tower_aux = build_vision_tower_aux(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower_aux = [vision_tower_aux]
                else:
                    self.vision_tower_aux = vision_tower_aux
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower_aux = self.vision_tower_aux[0]
                else:
                    vision_tower_aux = self.vision_tower_aux
                vision_tower_aux.load_model()
            self.config.mm_hidden_size_aux = vision_tower_aux.hidden_size

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            if 'model' in mm_projector_weights.keys():
                mm_projector_weights = mm_projector_weights['model']
                if is_deepspeed_zero3_enabled():
                    if len(mm_projector_weights) > 0:
                        with deepspeed.zero.GatheredParameters(mm_projector_weights, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                self.mm_projector.load_state_dict(mm_projector_weights)
                else:
                    status = self.mm_projector.load_state_dict(mm_projector_weights, strict=False)
                    print('missing_keys:', status.missing_keys)
            else:
                if is_deepspeed_zero3_enabled():
                    named_parameters = get_w(mm_projector_weights, 'mm_projector')
                    if len(named_parameters) > 0:
                        with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                self.mm_projector.load_state_dict(named_parameters)
                else:
                    status = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
                    print('missing_keys:', status.missing_keys)
            self.mm_projector = self.mm_projector.to(device='cuda')

    def initialize_uni_modules(self, model_args, for_eval=False):  
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        self.config.image_size_aux = getattr(model_args, 'image_size_aux', 320)
        self.config.optimize_vision_tower = getattr(model_args, 'optimize_vision_tower', False)
        self.config.optimize_vision_tower_aux = getattr(model_args, 'optimize_vision_tower_aux', False)

        self.vlm_uni_query_projector  = nn.Sequential(nn.LayerNorm(self.config.mm_hidden_size), 
                                                      nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size))
        self.vlm_uni_aux_projector  = nn.Sequential(nn.LayerNorm(self.config.mm_hidden_size_aux),
                                                    nn.Linear(self.config.mm_hidden_size_aux, self.config.mm_hidden_size))
        self.vlm_uni_val_projector  = nn.Sequential(nn.LayerNorm(self.config.mm_hidden_size_aux),
                                                    nn.Linear(self.config.mm_hidden_size_aux, self.config.mm_hidden_size))
        
        if pretrain_mm_mlp_adapter is not None:
            projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            trainable_module = ['vlm_uni', 'vision_fpn', 'vision_stages']
            if hasattr(model_args, 'model_name_or_path'):
                model_save_path = model_args.model_name_or_path
            else:
                model_save_path = model_args.model_path
            model_idx_path = getattr(model_args, 'model_path', model_save_path)
            if IS_NEW_TRANSFORMERS:
                try:
                    weight_file = json.load(open(os.path.join(model_idx_path, 'model.safetensors.index.json'), 'r'))['weight_map']
                except:
                    weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
            else:
                weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
            model_path = set([weight_file[_key] for _key in weight_file if any([_module in _key for _module in trainable_module])])
            projector_weights = {}
            for _model in model_path:
                if not IS_NEW_TRANSFORMERS:
                    projector_weights.update(torch.load(os.path.join(model_idx_path, _model), map_location='cpu'))
                else:
                    with safetensors.safe_open(os.path.join(model_idx_path, _model), framework="pt", device='cpu') as f:
                        for _key in f.keys():
                            projector_weights.update({_key: f.get_tensor(_key)})
            if len(projector_weights) == 0:
                return

        def get_w(weights, keyword, main_module, sub_module):
            if getattr(main_module, sub_module, None) is None:
                return
            
            pretrain_weight = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}
            if len(pretrain_weight) == 0:
                return
            if is_deepspeed_zero3_enabled():
                named_parameters = [v for k, v in getattr(main_module, sub_module).named_parameters()]
                if len(named_parameters) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            getattr(main_module, sub_module).load_state_dict(pretrain_weight)
                    with deepspeed.zero.GatheredParameters(self.mm_projector[0].weight, modifier_rank=None):
                        weight_type = self.mm_projector[0].weight.dtype
                        device_type = self.mm_projector[0].weight.device
            else:
                weight_type = self.mm_projector[0].weight.dtype
                device_type = self.mm_projector[0].weight.device
                getattr(main_module, sub_module).load_state_dict(pretrain_weight)
            if weight_type == torch.uint8 or weight_type == torch.int8 or weight_type == torch.int16:
                weight_type = torch.float16
            getattr(main_module, sub_module).to(device=device_type, dtype=weight_type)
            print(f"Loading {sub_module} weights...")
        
        # load pretrained weights
        get_w(projector_weights, 'vision_tower.vision_tower', self.vision_tower, 'vision_tower')

        # load pretrained weights
        if self.config.optimize_vision_tower_aux:
            # not optimize vision stem, just used to check
            get_w(projector_weights, 'vision_tower_aux.vision_stem', self.vision_tower_aux, 'vision_stem')
            get_w(projector_weights, 'vision_tower_aux.vision_stages', self.vision_tower_aux, 'vision_stages')
        get_w(projector_weights, 'vlm_uni_query_projector', self, 'vlm_uni_query_projector')
        get_w(projector_weights, 'vlm_uni_aux_projector', self, 'vlm_uni_aux_projector')
        get_w(projector_weights, 'vlm_uni_val_projector', self, 'vlm_uni_val_projector')
    
class MGMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_tower_aux(self):
        return self.get_model().get_vision_tower_aux()

    def encode_images(self, images, images_aux=None, is_video=False):
        image_grid = getattr(self.config, 'image_grid', 1)
        image_global = getattr(self.config, 'image_global', False)
        if image_grid > 1:
            batch_size = images.shape[0]
            if image_global:
                global_images = images[:, -1:].flatten(0,1).contiguous()
                grid_images = images[:, :-1].flatten(0,1).contiguous()
                images = torch.cat([grid_images, global_images], dim=0)
            else:
                images = images.flatten(0,1).contiguous()
        
        image_features = self.get_model().get_vision_tower()(images)
        
        if image_global:
            image_feat_global = image_features[-len(global_images):]
            image_features = image_features[:len(grid_images)]
        
        if images_aux is not None:
            image_aux_features_raw = self.get_model().get_vision_tower_aux()(images_aux).to(
                dtype=image_features.dtype, device=image_features.device)
            
            if image_global:
                image_aux_features_global = F.interpolate(image_aux_features_raw.float(), 
                                                            scale_factor=1/image_grid, 
                                                            mode='bilinear', 
                                                            align_corners=False).to(dtype=image_aux_features_raw.dtype)
                image_feat_global, image_aux_feat_global = self.unified_resampler(image_feat_global, image_aux_features_global)

            if image_grid > 1:
                image_aux_features_raw = image_aux_features_raw.reshape(*image_aux_features_raw.shape[:2],
                                                                        image_grid,
                                                                        image_aux_features_raw.shape[-2]//image_grid,
                                                                        image_grid,
                                                                        image_aux_features_raw.shape[-1]//image_grid)
                image_aux_features_raw = image_aux_features_raw.permute(0, 2, 4, 1, 3, 5).flatten(1,2).flatten(0,1).contiguous()
            image_features, image_aux_features = self.unified_resampler(image_features, image_aux_features_raw)
            
            if image_grid > 1:
                image_features = image_features.reshape(batch_size, image_grid**2, *image_features.shape[1:])
                image_features = image_features.flatten(1,2).contiguous()
                image_aux_features = image_aux_features.reshape(batch_size, image_grid**2, *image_aux_features.shape[1:])
                image_aux_features = image_aux_features.flatten(1,2).contiguous()
            
            # add global features, [global, local]
            if image_global:
                image_features = torch.cat([image_feat_global, image_features], dim=1)
                image_aux_features = torch.cat([image_aux_feat_global, image_aux_features], dim=1)
            
            # token generation
            image_features = image_features + image_aux_features
        
        # process image features after token generation
        image_features = self.get_model().mm_projector(image_features)
        
        return image_features

    def unified_resampler(self, images, images_aux):
        # patchwise with square images
        patch_num = int(images.shape[1]**0.5)
        patch_size = images_aux.shape[-1]//patch_num
        # within patch attention
        images_aux = images_aux.permute(0,2,3,1)
        images_aux = images_aux.reshape(len(images_aux), patch_num, patch_size, patch_num, patch_size, images_aux.shape[-1])
        images_aux = images_aux.permute(0,1,3,2,4,5)
        images_aux = images_aux.reshape(len(images_aux), patch_num**2, patch_size**2, images_aux.shape[-1]).contiguous()

        # token attention
        embed_query = self.get_model().vlm_uni_query_projector(images)
        embed_aux = self.get_model().vlm_uni_aux_projector(images_aux)
        embed_value = self.get_model().vlm_uni_val_projector(images_aux) 
        embed_att = embed_query[:,:,None] @ (embed_aux.transpose(-1,-2) / (embed_aux.shape[-1]**0.5))
        embed_att = embed_att.nan_to_num()
        embed_feat = (embed_att.softmax(-1) @ embed_value).mean(2)
        
        return images, embed_feat

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images=None, images_aux=None,
    ):        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:                
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        if isinstance(images_aux, list):
            images_aux = torch.stack(images_aux, dim=0)

        image_features = self.encode_images(images, images_aux)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            
            max_pos_id = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                max_pos_id += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    max_pos_id += cur_image_features.shape[0]
            
            cur_new_input_embeds = [x.to(device=cur_input_embeds.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)  
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
