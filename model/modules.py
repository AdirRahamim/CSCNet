import torch
import numpy as np
from torch import nn
from torch.nn import functional
from collections import namedtuple
from model.utils import conv_power_method, calc_pad_sizes


class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out


ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride1','stride2','stride3', 'unfoldings'])


class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A_1=None, B_1=None, C_1=None, A_2=None, B_2=None, C_2=None,
                 A_3=None, B_3=None, C_3=None, threshold_1=1e-2, threshold_2=1e-2, threshold_3=1e-2):
        super(ConvLista_T, self).__init__()
        self.x = nn.Parameter(torch.ones(1))
        self.y = nn.Parameter(torch.ones(1))
        self.z = nn.Parameter(torch.ones(1))

        if A_1 is None:
            A_1 = torch.randn(params.num_filters, 1, params.kernel_size, params.kernel_size)
            l1 = conv_power_method(A_1, [512, 512], num_iters=200, stride=params.stride1)
            A_1 /= torch.sqrt(l1)
        if B_1 is None:
            B_1 = torch.clone(A_1)
        if C_1 is None:
            C_1 = torch.clone(A_1)
        self.apply_A1 = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride1, bias=False)
        self.apply_B1 = torch.nn.Conv2d(1, params.num_filters, kernel_size=params.kernel_size, stride=params.stride1, bias=False)
        self.apply_C1 = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride1, bias=False)
        self.apply_A1.weight.data = A_1
        self.apply_B1.weight.data = B_1
        self.apply_C1.weight.data = C_1
        self.soft_threshold1 = SoftThreshold(params.num_filters, threshold_1)

        if A_2 is None:
            A_2 = torch.randn(params.num_filters, 1, params.kernel_size, params.kernel_size)
            l2 = conv_power_method(A_2, [512, 512], num_iters=200, stride=params.stride2)
            A_2 /= torch.sqrt(l2)
        if B_2 is None:
            B_2 = torch.clone(A_2)
        if C_2 is None:
            C_2 = torch.clone(A_2)
        self.apply_A2 = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride2, bias=False)
        self.apply_B2 = torch.nn.Conv2d(1, params.num_filters, kernel_size=params.kernel_size, stride=params.stride2, bias=False)
        self.apply_C2 = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride2, bias=False)
        self.apply_A2.weight.data = A_2
        self.apply_B2.weight.data = B_2
        self.apply_C2.weight.data = C_2
        self.soft_threshold2 = SoftThreshold(params.num_filters, threshold_2)

        if A_3 is None:
            A_3 = torch.randn(params.num_filters, 1, params.kernel_size, params.kernel_size)
            l3 = conv_power_method(A_3, [512, 512], num_iters=200, stride=params.stride3)
            A_3 /= torch.sqrt(l3)
        if B_3 is None:
            B_3 = torch.clone(A_3)
        if C_3 is None:
            C_3 = torch.clone(A_3)
        self.apply_A3 = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride3, bias=False)
        self.apply_B3 = torch.nn.Conv2d(1, params.num_filters, kernel_size=params.kernel_size, stride=params.stride3, bias=False)
        self.apply_C3 = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride3, bias=False)
        self.apply_A3.weight.data = A_3
        self.apply_B3.weight.data = B_3
        self.apply_C3.weight.data = C_3
        self.soft_threshold3 = SoftThreshold(params.num_filters, threshold_3)

        self.params = params

    def _split_image(self, I, stride):
        if stride == 1:
            return I, torch.ones_like(I)
        left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(I, self.params.kernel_size, stride)
        I_batched_padded = torch.zeros(I.shape[0], stride ** 2, I.shape[1], top_pad + I.shape[2] + bot_pad,
                                       left_pad + I.shape[3] + right_pad).type_as(I)
        valids_batched = torch.zeros_like(I_batched_padded)
        for num, (row_shift, col_shift) in enumerate([(i, j) for i in range(stride) for j in range(stride)]):
            I_padded = functional.pad(I, pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
            valids = functional.pad(torch.ones_like(I), pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
            I_batched_padded[:, num, :, :, :] = I_padded
            valids_batched[:, num, :, :, :] = valids
        I_batched_padded = I_batched_padded.reshape(-1, *I_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return I_batched_padded, valids_batched

    def forward(self, I):
        I_batched_padded, valids_batched = self._split_image(I, self.params.stride1)
        conv_input = self.apply_B1(I_batched_padded)
        gamma_k = self.soft_threshold1(conv_input)
        for k in range(self.params.unfoldings - 1):
            x_k = self.apply_A1(gamma_k)
            r_k = self.apply_B1(x_k - I_batched_padded)
            gamma_k = self.soft_threshold1(gamma_k - r_k)
        output_all = self.apply_C1(gamma_k)
        output_cropped = torch.masked_select(output_all, valids_batched.byte()).reshape(I.shape[0], self.params.stride1 ** 2, *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output1 = output_cropped.mean(dim=1, keepdim=False)

        I_batched_padded, valids_batched = self._split_image(I, self.params.stride2)
        conv_input = self.apply_B2(I_batched_padded)
        gamma_k = self.soft_threshold2(conv_input)
        for k in range(self.params.unfoldings - 1):
            x_k = self.apply_A2(gamma_k)
            r_k = self.apply_B2(x_k - I_batched_padded)
            gamma_k = self.soft_threshold2(gamma_k - r_k)
        output_all = self.apply_C2(gamma_k)
        output_cropped = torch.masked_select(output_all, valids_batched.byte()).reshape(I.shape[0],
                                                                                        self.params.stride2 ** 2,
                                                                                        *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output2 = output_cropped.mean(dim=1, keepdim=False)

        I_batched_padded, valids_batched = self._split_image(I, self.params.stride3)
        conv_input = self.apply_B3(I_batched_padded)
        gamma_k = self.soft_threshold3(conv_input)
        for k in range(self.params.unfoldings - 1):
            x_k = self.apply_A3(gamma_k)
            r_k = self.apply_B3(x_k - I_batched_padded)
            gamma_k = self.soft_threshold3(gamma_k - r_k)
        output_all = self.apply_C3(gamma_k)
        output_cropped = torch.masked_select(output_all, valids_batched.byte()).reshape(I.shape[0],
                                                                                        self.params.stride3 ** 2,
                                                                                        *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output3 = output_cropped.mean(dim=1, keepdim=False)

        output = (self.x * output1 + self.y * output2 + self.z * output3) / (self.x + self.y + self.z)
        return output