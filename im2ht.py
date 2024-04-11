import torch.nn.functional as F
import os
import math
import numpy as np
import warnings
from glob import glob

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

def load_custom_extension(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "custom_extension")
    tar_dir = os.path.join(src_dir, "build", ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob(f"{src_dir}/*.cu") + glob(f"{src_dir}/*.cpp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.utils.cpp_extension import load
        ext = load(
            name=ext_name,
            sources=srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[],
            build_directory=tar_dir,
        )
    return ext

# Defer calling load_custom_extension to make CUDA_VISIBLE_DEVICES happy
custom_extension = None


class IM2HTFunction(Function):
    @staticmethod
    def forward(ctx, input_image, ht_index, im_size, ht_size):
        ctx.im_size = _pair(im_size)
        ctx.ht_size = _pair(ht_size)
        ctx.save_for_backward(ht_index)

        # Ensure input_image tensor is contiguous and CUDA tensor
        if not input_image.is_contiguous():
            input_image = input_image.contiguous()
        if not input_image.is_cuda:
            raise ValueError("Input image tensor must be a CUDA tensor")

        # If input image dimensions do not match expected dimensions, adjust the dimensions
        if input_image.size(2) != ctx.im_size[0] or input_image.size(3) != ctx.im_size[1]:
            # Resize or pad the input image tensor to match the expected dimensions
            input_image = adjust_input_dimensions(input_image, ctx.im_size)

        # Pass the correct dimensions to the CUDA kernel
        output = custom_extension.im2ht_forward(
            input_image,
            ht_index,
            ctx.im_size[0],
            ctx.im_size[1],
            ctx.ht_size[0],
            ctx.ht_size[1]
        )
        print("Output finished=========================================================================================")
        return output

def adjust_input_dimensions(input_image, target_size):
   
    _, _, h, w = input_image.size()
    target_h, target_w = target_size
    print(target_size)
    # If input image is smaller than the target size, resize or pad it
    if h < target_h or w < target_w:
        # Resize the input image tensor to match the target size
        input_image = F.interpolate(input_image, size=(target_size[0].item(), target_size[1].item()), mode='nearest')
    elif h > target_h or w > target_w:
        # Pad the input image tensor to match the target size
        pad_h = max(0, h - target_h)
        pad_w = max(0, w - target_w)
        padding = (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2)
        input_image = F.pad(input_image, padding, mode='constant', value=0)

    return input_image



    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        ht_index = ctx.saved_tensors[0]  

        grad_input = custom_extension.im2ht_backward(
            grad_output,
            ht_index,
            ctx.im_size[0],
            ctx.im_size[1],
            ctx.ht_size[0],
            ctx.ht_size[1]
        )

        return (
            grad_input,  
            None, 
            None, 
            None  
        )


class IM2HT(nn.Module):
    def __init__(self, im_size, ht_size, vote_mapping):
        super(IM2HT, self).__init__()
        
        vote_mapping.requires_grad=False
        # self.register_buffer('vote_mapping', vote_mapping)
        self.register_buffer('vote_mapping', vote_mapping, persistent=False)
        
        global custom_extension
        print('#################### Custom extension compiling ############################')
        custom_extension = load_custom_extension("im2ht")
        print('#################### Done! ############################')

        self.im_size = _pair(im_size)
        self.ht_size = _pair(ht_size)

        # self.extra_repr()
        self.__repr__()

    def extra_repr(self):
        s = ('im_size={im_size}, ht_size={ht_size}')
        # return s.format(**self.__dict__)
        return print(s.format(**self.__dict__))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'im_size=' + str(self.im_size) + ', ht_size=' + str(self.ht_size) \
               + ', vote_mapping=' + str(self.vote_mapping.shape) +  ')'


    def forward(self, input):  
        print("Forward: Input size:", input.size(), "Expected size:", self.im_size)
        # print('IM2HT forward self.vote_index', self.vote_mapping.device, input.device)
        return IM2HTFunction.apply(
            input.contiguous(),
            self.vote_mapping,
            self.im_size,
            self.ht_size
        )
