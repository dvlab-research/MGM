import os
from .clip_encoder import CLIPVisionTower
from .eva_encoder import EVAVisionTower
from .openclip_encoder import OpenCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "../processor/clip-patch14-224"))
    
    if not os.path.exists(vision_tower):
        raise ValueError(f'Not find vision tower: {vision_tower}')

    if "openai" in vision_tower.lower() or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTower(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_tower_aux(vision_tower_cfg, **kwargs):
    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux', getattr(vision_tower_cfg, 'vision_tower_aux', None))
    
    if not os.path.exists(vision_tower_aux):
        raise ValueError(f'Not find vision tower: {vision_tower_aux}')

    if "openclip" in vision_tower_aux.lower():
        return OpenCLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    elif "openai" in vision_tower_aux.lower():
        return CLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower_aux}')