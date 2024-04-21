from transformers import CLIPImageProcessor
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import get_resize_output_image_size

import torch
import torch.nn.functional as F

import numpy as np


class VideoFramesProcessor(CLIPImageProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, images, **kwargs):
        if not isinstance(images, np.ndarray):
            return super().preprocess(images=images, **kwargs)
        
        do_resize = kwargs.get('do_resize', self.do_resize)
        size = kwargs.get('size', self.size)
        size = get_size_dict(size, param_name="size", default_to_square=False)
        do_center_crop = kwargs.get('do_center_crop', self.do_center_crop)
        crop_size = kwargs.get('crop_size', self.crop_size)
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = kwargs.get('do_rescale', self.do_rescale)
        rescale_factor = kwargs.get('rescale_factor', self.rescale_factor)
        do_normalize = kwargs.get('do_normalize', self.do_normalize)
        image_mean = kwargs.get('image_mean', self.image_mean)
        image_std = kwargs.get('image_std', self.image_std)
        return_tensors = kwargs.get('return_tensors', None)

        def resize(images, output_size):
            images = images.permute((0, 3, 1, 2))
            images = F.interpolate(images, size=output_size, mode='bicubic')
            images = images.permute((0, 2, 3, 1))
            return images

        def center_crop(images, crop_size):
            crop_width, crop_height = crop_size["width"], crop_size["height"]
            img_width, img_height = images.shape[1:3]
            x = (img_width - crop_width) // 2
            y = (img_height - crop_height) // 2
            images = images[:, x:x+crop_width, y:y+crop_height]
            return images
        
        def rescale(images, rescale_factor):
            images = images * rescale_factor
            return images
        
        def normalize(images, mean, std):
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            images = (images - mean) / std
            return images

        images = torch.from_numpy(images).float()

        if do_resize:
            output_size = get_resize_output_image_size(images[0], size=size["shortest_edge"], default_to_square=False)
            images = resize(images, output_size)
        
        if do_center_crop:
            images = center_crop(images, crop_size)
        
        if do_rescale:
            images = rescale(images, rescale_factor)
        
        if do_normalize:
            images = normalize(images, image_mean, image_std)

        images = images.permute((0, 3, 1, 2))
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
