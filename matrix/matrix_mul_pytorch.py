import os
import torch
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

# 编译并加载 CUDA 代码
source_cuda = "matrix_mul.cu"
module = load(name="matrix_mul", sources=[source_cuda], extra_cuda_cflags=["-arch=sm_70"], verbose=True)

class MatrixMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_A, input_B):
        output_C = torch.empty(input_A.size(0), input_B.size(1), device=input_A.device)
        module.matrix_mul_forward(input_A, input_B, output_C)
        ctx.save_for_backward(input_A, input_B)
        return output_C

    # 计算梯度
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input_A, input_B = ctx.saved_tensors
        grad_A = torch.mm(grad_output, input_B.transpose(0, 1))
        grad_B = torch.mm(input_A.transpose(0, 1), grad_output)
        return grad_A, grad_B

matrix_mul = MatrixMulFunction.apply
