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

def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "cpp_im2ht")
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

# defer calling load_cpp_ext to make CUDA_VISIBLE_DEVICES happy
im2ht = None


class IM2HTFunction(Function):
    @staticmethod
    def forward(ctx, input_im, ht_index, im_size, ht_size):
        ctx.im_size = _pair(im_size)
        ctx.ht_size = _pair(ht_size)
        ctx.save_for_backward(ht_index)

        # Ensure input_im tensor is contiguous and CUDA tensor
        if not input_im.is_contiguous():
            input_im = input_im.contiguous()
        if not input_im.is_cuda:
            raise ValueError("Input image tensor must be a CUDA tensor")

        # If input image dimensions do not match expected dimensions, adjust the dimensions
        if input_im.size(2) != ctx.im_size[0] or input_im.size(3) != ctx.im_size[1]:
            # Resize or pad the input image tensor to match the expected dimensions
            input_im = adjust_input_dimensions(input_im, ctx.im_size)

        # Pass the correct dimensions to the CUDA kernel
        output = im2ht.im2ht_forward(
            input_im,
            ht_index,
            ctx.im_size[0],
            ctx.im_size[1],
            ctx.ht_size[0],
            ctx.ht_size[1]
        )
        print("output finished=========================================================================================")
        return output

def adjust_input_dimensions(input_im, target_size):
    """
    Adjust input image dimensions to match the target size by resizing or padding.

    Parameters:
        input_im (torch.Tensor): Input image tensor.
        target_size (tuple): Target size (height, width) to adjust the image dimensions.

    Returns:
        torch.Tensor: Image tensor with adjusted dimensions.
    """
    _, _, h, w = input_im.size()
    target_h, target_w = target_size
    print(target_size)
    # If input image is smaller than the target size, resize or pad it
    if h < target_h or w < target_w:
        # Resize the input image tensor to match the target size
        input_im = F.interpolate(input_im, size=(target_size[0].item(), target_size[1].item()), mode='nearest')
    elif h > target_h or w > target_w:
        # Pad the input image tensor to match the target size
        pad_h = max(0, h - target_h)
        pad_w = max(0, w - target_w)
        padding = (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2)
        input_im = F.pad(input_im, padding, mode='constant', value=0)

    return input_im



    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        ht_index = ctx.saved_tensors[0]  # it is a list of length 1!

        grad_input = im2ht.im2ht_backward(
            grad_output,
            ht_index,
            ctx.im_size[0],
            ctx.im_size[1],
            ctx.ht_size[0],
            ctx.ht_size[1]
        )

        return (
            grad_input,  # input
            None, # ht_index
            None, # im_size
            None  # ht_size
        )


class IM2HT(nn.Module):
    def __init__(self, im_size, ht_size, vote_mapping):
        super(IM2HT, self).__init__()
        
        vote_mapping.requires_grad=False
        # self.register_buffer('vote_mapping', vote_mapping)
        self.register_buffer('vote_mapping', vote_mapping, persistent=False)
        
        global im2ht
        print('#################### im2ht compiling ############################')
        im2ht = load_cpp_ext("im2ht")
        print('#################### done! ############################')

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
