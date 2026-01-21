import torch
from .FlyingConvFunction import flying_conv2d_fwd, flying_conv2d_input_bwd, flying_conv2d_weight_bwd, flying_conv2d_bias_bwd
# from FlyingConvFunction import flying_conv2d_fwd, flying_conv2d_input_bwd, flying_conv2d_weight_bwd, flying_conv2d_bias_bwd
import time
from tqdm import tqdm
import torch.nn.functional as F

class _triton_flying_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        scale: tuple = (4, 4),
    ):
        output = flying_conv2d_fwd(
            input,
            weight,
            bias,
            kernel_size[0],
            kernel_size[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            scale[0],
            scale[1],
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.params = kernel_size, stride, padding, dilation, scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        kernel_size, stride, padding, dilation, scale = ctx.params
        grad_input, grad_weight, grad_bias = None, None, None
        if input.requires_grad:
            grad_input = flying_conv2d_input_bwd(
                grad_output,
                input,
                weight,
                kernel_size[0],
                kernel_size[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                scale[0],
                scale[1],
            )
        if weight.requires_grad:
            grad_weight = flying_conv2d_weight_bwd(
                grad_output,
                input,
                weight,
                kernel_size[0],
                kernel_size[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                scale[0],
                scale[1],
            )
        if bias is not None and bias.requires_grad:
            grad_bias = flying_conv2d_bias_bwd(
                grad_output,
                bias,
                scale[0],
                scale[1],
            )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


triton_conv2d = _triton_flying_conv2d.apply

def _torch_flying_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, scale=4):
    B, C, H, W = input.shape
    _, _, H_down, W_down, K = weight.shape
    weight = weight.permute(0, 1, 4, 2, 3).contiguous()
    input_unfold = F.unfold(input, kernel_size=3, dilation=dilation, padding=padding, stride=stride).view(B, C, K, H, W)
    weight_up = F.interpolate(weight.view(B, C * K, H_down, W_down), scale_factor=scale, mode="bilinear").view(B, C, K, H, W)
    output = (input_unfold * weight_up).sum(dim=2)
    if bias is not None:
        bias_up = F.interpolate(bias, scale_factor=scale, mode="bilinear")
        output = output + bias_up
    return output


if __name__ == "__main__":
    triton_conv2d = _triton_flying_conv2d.apply
    B, C, H, W = 8, 64, 256, 256
    scale = 8
    k_size = 3
    input = torch.randn((B, C, H, W), device="cuda", requires_grad=True)
    weight = torch.randn((B, C, H // scale, W // scale, k_size**2), device="cuda", requires_grad=True)
    bias = torch.randn((B, C, H // scale, W // scale), device="cuda", requires_grad=True)
    # bias = None 
    triton_output = triton_conv2d(
        input,
        weight,
        bias,
        (k_size, k_size),
        (1, 1),
        (k_size // 2, k_size // 2),
        (1, 1),
        (scale, scale),
    )
    triton_output.mean((1,2,3)).sum().backward()
    start = time.time()
    for i in tqdm(range(3000)):
        triton_output = triton_conv2d(
            input,
            weight,
            bias,
            (k_size, k_size),
            (1, 1),
            (k_size // 2, k_size // 2),
            (1, 1),
            (scale, scale),
        )
        triton_output.mean((1,2,3)).sum().backward()
    print("FPS:", 3000 / (time.time() - start))
    print("GPU Memory Allocated:", torch.cuda.max_memory_allocated() / 1024**2)
    print("GPU Memory Reserved:", torch.cuda.max_memory_reserved() / 1024**2)
    from FlyingConvFunction import _flying_conv2d_fwd, _flying_conv2d_input_bwd, _flying_conv2d_weight_bwd, _flying_conv2d_bias_bwd
    print(_flying_conv2d_fwd.best_config)
    print(_flying_conv2d_input_bwd.best_config)
    print(_flying_conv2d_weight_bwd.best_config)
    print(_flying_conv2d_bias_bwd.best_config)