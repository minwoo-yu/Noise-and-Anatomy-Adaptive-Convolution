import os
from itertools import product

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from tqdm import tqdm
import torch.nn.functional as F
import time
# ruff: noqa: E731

BLOCK_C_FWD_ = [1, 2]
BLOCK_K_FWD_ = [2, 4]
TILE_H_FWD_ = [8, 16]
TILE_W_FWD_ = [8, 16]
WARPS_FWD_ = [1]
STAGES_FWD_ = [2]


def get_configs():
    configs = []
    for bc, bk, th, tw, nw, ns in product(BLOCK_C_FWD_, BLOCK_K_FWD_, TILE_H_FWD_, TILE_W_FWD_, WARPS_FWD_, STAGES_FWD_):
        num_elements = bc * bk * th * tw
        if num_elements > 4096:
            continue  # upper limit
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_C": bc, "BLOCK_SIZE_K": bk, "TILE_SIZE_H": th, "TILE_SIZE_W": tw},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=[
        "B",
        "C",
        "H_in",
        "W_in",
        "H_out",
        "W_out",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dil_h",
        "dil_w",
        "scale_h",
        "scale_w",
    ],
)
@triton.jit
def _flying_conv2d_fwd(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    B,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    scale_h,
    scale_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    """forward pass based on triton
    input: [B, C, H_in, W_in]
    weight: [B, C, H_weight, W_weight, K]
    output: [B, C, H_out, W_out]
    get down-scaled weight map, upsampling simulataneously! "on the flying"
    1. im2col input image data
    2. upsample weight map as target resolution (unlike convolution, weight varies different batch & location)
    3. element wise multiplication & sum along kernel index axis
    """
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    tile_h_idx = pid_hw // tl.cdiv(W_out, TILE_SIZE_W)
    tile_w_idx = pid_hw % tl.cdiv(W_out, TILE_SIZE_W)

    offs_h = tile_h_idx * TILE_SIZE_H + tl.arange(0, TILE_SIZE_H)
    offs_w = tile_w_idx * TILE_SIZE_W + tl.arange(0, TILE_SIZE_W)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    offs_b = pid_bc // num_c_blocks
    offs_c = (pid_bc % num_c_blocks) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    h_in_start = offs_h[:, None] * stride_h - pad_h  # [TILE_H, 1]
    w_in_start = offs_w[None, :] * stride_w - pad_w  # [1, TILE_W]

    H_w_down = H_out // scale_h
    W_w_down = W_out // scale_w
    h_curr = tl.maximum((offs_h[:, None].to(tl.float32) + 0.5) / scale_h - 0.5, 0)
    w_curr = tl.maximum((offs_w[None, :].to(tl.float32) + 0.5) / scale_w - 0.5, 0)
    h_down_0 = h_curr.to(tl.int32)
    w_down_0 = w_curr.to(tl.int32)
    alpha_h = h_curr - h_down_0.to(tl.float32)
    alpha_w = w_curr - w_down_0.to(tl.float32)

    neighbors = tl.arange(0, 4)
    n_h = neighbors // 2
    n_w = neighbors % 2

    # [TILE_H, TILE_W, 4]
    h_interps = tl.minimum(h_down_0[:, :, None] + n_h[None, None, :], H_w_down - 1)
    w_interps = tl.minimum(w_down_0[:, :, None] + n_w[None, None, :], W_w_down - 1)

    weight_h = (1.0 - n_h[None, None, :]) * (1.0 - alpha_h[:, :, None]) + n_h[None, None, :] * alpha_h[:, :, None]
    weight_w = (1.0 - n_w[None, None, :]) * (1.0 - alpha_w[:, :, None]) + n_w[None, None, :] * alpha_w[:, :, None]
    coeffs = weight_h * weight_w  # [TILE_H, TILE_W, 4]

    Kernel_total = kernel_h * kernel_w
    acc = tl.zeros((BLOCK_SIZE_C, TILE_SIZE_H, TILE_SIZE_W), dtype=tl.float32)
    bc_base_idx = offs_b * C * H_w_down * W_w_down + offs_c * H_w_down * W_w_down
    weight_base_idx = bc_base_idx[:, None, None, None] + (h_interps * W_w_down + w_interps)[None, :, :, :]  # [BLOCK_C, TILE_H, TILE_W, 4]
    for k_start in tl.range(0, Kernel_total, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        h_kernel_idx = offs_k // kernel_h * dil_h
        w_kernel_idx = offs_k % kernel_w * dil_w
        h_in_idx = h_in_start[None, :, :, None] + h_kernel_idx[None, None, None, :]  # [TILE_H, 1, 1, BLOCK_K]
        w_in_idx = w_in_start[:, None, :, None] + w_kernel_idx[None, None, None, :]  # [1, TILE_W, 1, BLOCK_K]

        # [BLOCK_C, 1, 1, BLOCK_K]
        mask_base = (offs_b < B) & (offs_c[:, None, None, None] < C) & (offs_k[None, None, None, :] < Kernel_total)
        input_idx = offs_b * C * H_in * W_in + offs_c[:, None, None, None] * H_in * W_in + h_in_idx * W_in + w_in_idx
        mask_input = mask_base & (h_in_idx >= 0) & (h_in_idx < H_in) & (w_in_idx >= 0) & (w_in_idx < W_in)
        input_vals = tl.load(input_ptr + input_idx, mask=mask_input, other=0.0)

        # [BLOCK_C, TILE_H, TILE_W, 4, BLOCK_K]
        weight_idx = weight_base_idx[:, :, :, :, None] * Kernel_total + offs_k
        weight_neighbors = tl.load(weight_ptr + weight_idx, mask=mask_base[:, :, :, None, :], other=0.0)
        weight_vals = tl.sum(weight_neighbors * coeffs[None, :, :, :, None], axis=3)
        acc += tl.sum(input_vals * weight_vals, axis=3)

    if bias_ptr is not None:
        bias_neighbors = tl.load(bias_ptr + weight_base_idx, mask=(offs_b < B) & (offs_c[:, None, None, None] < C), other=0.0)
        bias_vals = tl.sum(bias_neighbors * coeffs[None, :, :, :], axis=3)
        acc += bias_vals

    output_idx = offs_b * C * H_out * W_out + offs_c[:, None, None] * H_out * W_out + offs_h[:, None] * W_out + offs_w[None, :]
    mask_out = (offs_b < B) & (offs_c[:, None, None] < C) & (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)
    tl.store(output_ptr + output_idx, acc, mask=mask_out)


@triton_op("mylib::flying_conv2d_fwd", mutates_args={})
def flying_conv2d_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    str_h: int,
    str_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int,
    dil_w: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    B, C, H_in, W_in = input.shape
    _, _, H_weight, W_weight, K = weight.shape

    H_out = H_weight * scale_h
    W_out = W_weight * scale_w
    assert K == kernel_h * kernel_w
    assert H_out == (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // str_h + 1
    assert W_out == (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // str_w + 1
    assert isinstance(scale_h, int) and isinstance(scale_w, int)

    output = torch.zeros((B, C, H_out, W_out), dtype=input.dtype, device=input.device)  # [N, K, P, Q]
    grid = lambda meta: (
        triton.cdiv(H_out, meta["TILE_SIZE_H"]) * triton.cdiv(W_out, meta["TILE_SIZE_W"]),
        B * triton.cdiv(C, meta["BLOCK_SIZE_C"]),
    )
    wrap_triton(_flying_conv2d_fwd)[grid](
        output,
        input,
        weight,
        bias,
        B,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        kernel_h,
        kernel_w,
        str_h,
        str_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        scale_h,
        scale_w,
    )
    return output


# --------------------------------------------------------------------------------------------------------------
BLOCK_C_INP_ = [1, 2]
BLOCK_K_INP_ = [1, 2]
TILE_H_INP_ = [8, 16]
TILE_W_INP_ = [8, 16]
WARPS_INP_ = [1]
STAGES_INP_ = [2]


def get_configs():
    configs = []
    for bc, bk, th, tw, nw, ns in product(BLOCK_C_INP_, BLOCK_K_INP_, TILE_H_INP_, TILE_W_INP_, WARPS_INP_, STAGES_INP_):
        num_elements = bc * bk * th * tw
        if num_elements > 4096:
            continue  # upper limit
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_C": bc, "BLOCK_SIZE_K": bk, "TILE_SIZE_H": th, "TILE_SIZE_W": tw},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=[
        "B",
        "C",
        "H_in",
        "W_in",
        "H_out",
        "W_out",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dil_h",
        "dil_w",
        "scale_h",
        "scale_w",
    ],
)
@triton.jit
def _flying_conv2d_input_bwd(
    grad_inp_ptr,
    grad_out_ptr,
    weight_ptr,
    B,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    scale_h,
    scale_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    """backward pass for input data
    aL/aX = aL/aY * aY/aX = grad_output * interpolated_weight (chain rule)

    grad_input (to be calculated): [B, C, H_in, W_in]
    grad_output: [B, C, H_out, W_out]
    weight: [B, C, H_out // scale_h, W_out // scale_w, K]
    similar to the deconvolution (but on-the-flying calculation)
    1. im2col grad_output data to the same resolution as output (for transposed convolution)
    2. get upsampled weight for propagation & element wise multiplication
    3. sum along kernel axis
    """
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    tile_h_idx = pid_hw // tl.cdiv(W_in, TILE_SIZE_W)
    tile_w_idx = pid_hw % tl.cdiv(W_in, TILE_SIZE_W)

    offs_h = tile_h_idx * TILE_SIZE_H + tl.arange(0, TILE_SIZE_H)
    offs_w = tile_w_idx * TILE_SIZE_W + tl.arange(0, TILE_SIZE_W)
    num_c_blocks =  tl.cdiv(C, BLOCK_SIZE_C)
    offs_b = pid_bc // num_c_blocks
    offs_c = (pid_bc % num_c_blocks) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    h_in = offs_h[:, None] + pad_h  # [TILE_H, 1]
    w_in = offs_w[None, :] + pad_w  # [1, TILE_W]

    H_w_down = H_out // scale_h
    W_w_down = W_out // scale_w
    Kernel_total = kernel_h * kernel_w

    neighbors = tl.arange(0, 4)  # for linear interpolation
    n_h = neighbors // 2
    n_w = neighbors % 2
    n_h = n_h[None, None, None, None, :]
    n_w = n_w[None, None, None, None, :]

    mask_bc = (offs_b < B) & (offs_c < C)
    acc = tl.zeros((BLOCK_SIZE_C, TILE_SIZE_H, TILE_SIZE_W), dtype=tl.float32)
    weight_base_idx = offs_b * (C * H_w_down * W_w_down * Kernel_total) + offs_c * (H_w_down * W_w_down * Kernel_total)

    for k_start in tl.range(0, Kernel_total, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < Kernel_total  # [BLOCK_K]
        h_kernel_idx = offs_k // kernel_h * dil_h
        w_kernel_idx = offs_k % kernel_h * dil_w

        h_num = h_in[None, :, :, None] - h_kernel_idx[None, None, None, :]  # [1, TILE_H, 1, BLOCK_K]
        w_num = w_in[:, None, :, None] - w_kernel_idx[None, None, None, :]  # [1, 1, TILE_W, BLOCK_K]
        h_out_idx = h_num // stride_h
        w_out_idx = w_num // stride_w

        valid_spatial = (
            (h_num >= 0) & (h_num % stride_h == 0) & (w_num >= 0) & (w_num % stride_w == 0) & (h_out_idx < H_out) & (w_out_idx < W_out)
        )
        valid_out = valid_spatial & mask_k[None, None, None, :] & mask_bc[:, None, None, None]
        h_curr = (h_out_idx.to(tl.float32) + 0.5) / scale_h - 0.5
        w_curr = (w_out_idx.to(tl.float32) + 0.5) / scale_w - 0.5
        h_curr = tl.maximum(h_curr, 0.0)[:, :, :, :, None]
        w_curr = tl.maximum(w_curr, 0.0)[:, :, :, :, None]

        h_down_0 = h_curr.to(tl.int32)
        w_down_0 = w_curr.to(tl.int32)

        alpha_h = h_curr - h_down_0.to(tl.float32)
        alpha_w = w_curr - w_down_0.to(tl.float32)

        h_coord = tl.minimum(h_down_0 + n_h, H_w_down - 1)
        w_coord = tl.minimum(w_down_0 + n_w, W_w_down - 1)

        weight_val_h = 1.0 - tl.abs(alpha_h - n_h)
        weight_val_w = 1.0 - tl.abs(alpha_w - n_w)
        coeff = weight_val_h * weight_val_w

        # weight_idx: [BLOCK_C, TILE_H, TILE_W, BLOCK_K, 4]
        weight_idx = (
            weight_base_idx[:, None, None, None, None] + (h_coord * W_w_down + w_coord) * Kernel_total + offs_k[None, None, None, :, None]
        )
        mask_weight = valid_out[:, :, :, :, None]
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask_weight, other=0.0)

        # grad_output_idx: [BLOCK_C, TILE_H, TILE_W, BLOCK_K]
        grad_output_idx = offs_b * C * H_out * W_out + offs_c[:, None, None, None] * H_out * W_out + h_out_idx * W_out + w_out_idx
        grad_output_val = tl.load(grad_out_ptr + grad_output_idx, mask=valid_out, other=0.0)
        acc += tl.sum(grad_output_val * tl.sum(weight_val * coeff, axis=4), axis=3)  # interpolate and transpose conv

    grad_input_idx = offs_b * C * H_in * W_in + offs_c[:, None, None] * H_in * W_in + offs_h[:, None] * W_in + offs_w[None, :]
    mask_out = (offs_b < B) & (offs_c[:, None, None] < C) & (offs_h[:, None] < H_in) & (offs_w[None, :] < W_in)
    tl.store(grad_inp_ptr + grad_input_idx, acc, mask=mask_out)


@triton_op("mylib::flying_conv2d_input_bwd", mutates_args={})
def flying_conv2d_input_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    str_h: int,
    str_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int,
    dil_w: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    B, C, H_in, W_in = input.shape
    _, _, H_weight, W_weight, K = weight.shape

    H_out = H_weight * scale_h
    W_out = W_weight * scale_w

    grad_input = torch.zeros_like(input)
    grid = lambda meta: (
        triton.cdiv(H_in, meta["TILE_SIZE_H"]) * triton.cdiv(W_in, meta["TILE_SIZE_W"]),
        B * triton.cdiv(C, meta["BLOCK_SIZE_C"]),
    )
    wrap_triton(_flying_conv2d_input_bwd)[grid](
        grad_input,
        grad_output,
        weight,
        B,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        kernel_h,
        kernel_w,
        str_h,
        str_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        scale_h,
        scale_w,
    )
    return grad_input


#  ----------------------------------------------------------------------------------------------------------------------------

BLOCK_C_WEIGHT_ = [1, 2]
BLOCK_K_WEIGHT_ = [1, 2]
TILE_H_WEIGHT_ = [2, 4]
TILE_W_WEIGHT_ = [2, 4]
WARPS_WEIGHT_ = [1]
STAGES_WEIGHT_ = [2]


def get_configs():
    configs = []
    for bc, bk, th, tw, nw, ns in product(BLOCK_C_WEIGHT_, BLOCK_K_WEIGHT_, TILE_H_WEIGHT_, TILE_W_WEIGHT_, WARPS_WEIGHT_, STAGES_WEIGHT_):
        num_elements = bc * bk * th * tw
        if num_elements > 4096:
            continue  # upper limit
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_C": bc, "BLOCK_SIZE_K": bk, "TILE_SIZE_H": th, "TILE_SIZE_W": tw},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=[
        "B",
        "C",
        "H_in",
        "W_in",
        "H_out",
        "W_out",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dil_h",
        "dil_w",
        "SCALE_H",
        "SCALE_W",
    ],
)
@triton.jit
def _flying_conv2d_weight_bwd(
    grad_w_ptr,
    grad_out_ptr,
    input_ptr,
    B,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    """backward pass for weight data: gather version
    aL/aW = aL/aY * aY/aW = grad_output * input data (chain rule)

    grad_weight (to be calculated): [B, C, H_out // scale_h, W_out // scale_w, K]
    grad_output: [B, C, H_out, W_out]
    input: [B, C, H_in, W_in]
    1. im2col for output_grad to the same resolution as input data
    2. element wise multiplication (convolution)
    3. sum along neighborhood axis for inverse bilinear interpolation
    """
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)

    H_w_down = H_out // SCALE_H
    W_w_down = W_out // SCALE_W
    Kernel_total = kernel_h * kernel_w

    tile_h_idx = pid_hw // tl.cdiv(W_w_down, TILE_SIZE_W)
    tile_w_idx = pid_hw % tl.cdiv(W_w_down, TILE_SIZE_W)

    offs_h = tile_h_idx * TILE_SIZE_H + tl.arange(0, TILE_SIZE_H)
    offs_w = tile_w_idx * TILE_SIZE_W + tl.arange(0, TILE_SIZE_W)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    offs_b = pid_bc // num_c_blocks
    offs_c = (pid_bc % num_c_blocks) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    WIN_H_SIZE: tl.constexpr = 2 * SCALE_H
    WIN_W_SIZE: tl.constexpr = 2 * SCALE_W
    WIN_TOTAL: tl.constexpr = WIN_H_SIZE * WIN_W_SIZE
    offs_win = tl.arange(0, WIN_TOTAL)
    win_h_idx = offs_win // WIN_W_SIZE
    win_w_idx = offs_win % WIN_W_SIZE

    base_h_out = offs_h * SCALE_H - SCALE_H + (SCALE_H // 2)
    base_w_out = offs_w * SCALE_W - SCALE_W + (SCALE_W // 2)
    h_out_curr = base_h_out[:, None, None] + win_h_idx[None, None, :]  # [TILE_H, 1, WIN_H_SIZE]
    w_out_curr = base_w_out[None, :, None] + win_w_idx[None, None, :]  # [1, TILE_W, WIN_W_SIZE]

    h_curr = (h_out_curr.to(tl.float32) + 0.5) / SCALE_H - 0.5
    w_curr = (w_out_curr.to(tl.float32) + 0.5) / SCALE_W - 0.5
    h_curr = tl.maximum(h_curr, 0.0)
    w_curr = tl.maximum(w_curr, 0.0)

    h_down = h_curr.to(tl.int32)
    w_down = w_curr.to(tl.int32)
    alpha_h = h_curr - h_down.to(tl.float32)
    alpha_w = w_curr - w_down.to(tl.float32)

    n_h = offs_h[:, None, None] - h_down
    n_w = offs_w[None, :, None] - w_down
    weight_val_h = 1.0 - tl.abs(alpha_h - n_h)
    weight_val_w = 1.0 - tl.abs(alpha_w - n_w)
    weight_val_h += tl.where((h_down == H_w_down - 1) & (offs_h[:, None, None] == H_w_down - 1), alpha_h, 0.0)
    weight_val_w += tl.where((w_down == W_w_down - 1) & (offs_w[None, :, None] == W_w_down - 1), alpha_w, 0.0)
    coeffs = weight_val_h * weight_val_w  # [TILE_H, TILE_W, WIN_TOTAL]

    mask_bc = (offs_b < B) & (offs_c[:, None, None] < C)  # [BLOCK_C, 1, 1]
    mask_weight = (offs_h[:, None] < H_w_down) & (offs_w[None, :] < W_w_down)
    mask_win = (h_out_curr >= 0) & (h_out_curr < H_out) & (w_out_curr >= 0) & (w_out_curr < W_out) & (coeffs > 0.0)
    # grad_out_idx: [BLOCK_C, TILE_H, TILE_W, WIN_TOTAL]
    grad_out_idx = (
        offs_b * C * H_out * W_out
        + offs_c[:, None, None, None] * H_out * W_out
        + h_out_curr[None, :, :, :] * W_out
        + w_out_curr[None, :, :, :]
    )
    grad_out_val = tl.load(grad_out_ptr + grad_out_idx, mask=mask_win[None, :, :, :] & mask_bc[:, :, :, None], other=0.0)
    grad_out_coeff = grad_out_val * coeffs[None, :, :, :]

    grad_w_base_idx = (
        offs_b * (C * H_w_down * W_w_down * Kernel_total)
        + offs_c[:, None, None] * (H_w_down * W_w_down * Kernel_total)
        + (offs_h[None, :, None] * W_w_down + offs_w[None, None, :]) * Kernel_total
    )  # [BLOCK_C, TILE_H, TILE_W]
    for k_start in tl.range(0, Kernel_total, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < Kernel_total

        h_kernel_idx = offs_k // kernel_h
        w_kernel_idx = offs_k % kernel_w
        h_in_offs = h_kernel_idx * dil_h - pad_h
        w_in_offs = w_kernel_idx * dil_w - pad_w
        h_in_idx = h_out_curr[:, :, :, None] * stride_h + h_in_offs[None, None, None, :]  # [BLOCK_C, TILE_H, TILE_W, WIN_TOTAL]
        w_in_idx = w_out_curr[:, :, :, None] * stride_w + w_in_offs[None, None, None, :]  # [BLOCK_C, TILE_H, TILE_W, WIN_TOTAL]

        mask_input = (
            mask_win[:, :, :, None]
            & (h_in_idx >= 0)
            & (h_in_idx < H_in)
            & (w_in_idx >= 0)
            & (w_in_idx < W_in)
            & mask_k[None, None, None, :]
        )
        input_idx = offs_b * C * H_in * W_in + offs_c[:, None, None, None, None] * H_in * W_in + h_in_idx * W_in + w_in_idx
        input_val = tl.load(
            input_ptr + input_idx, mask=mask_input & mask_bc[:, :, :, None, None], other=0.0
        )  # [BLOCK_C, TILE_H, TILE_W, WIN_TOTAL, BLOCK_K]
        acc = tl.sum(input_val * grad_out_coeff[:, :, :, :, None], axis=3)  # [BLOCK_C, TILE_H, TILE_W, BLOCK_K]

        grad_w_idx = grad_w_base_idx[:, :, :, None] + offs_k[None, None, None, :]
        tl.store(
            grad_w_ptr + grad_w_idx,
            acc,
            mask=mask_weight[:, :, None] & mask_k[None, None, None] & mask_bc[:, :, :, None],
        )


@triton_op("mylib::flying_conv2d_weight_bwd", mutates_args={})
def flying_conv2d_weight_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    str_h: int,
    str_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int,
    dil_w: int,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    B, C, H_in, W_in = input.shape
    _, _, H_weight, W_weight, K = weight.shape

    H_out = H_weight * scale_h
    W_out = W_weight * scale_w

    grad_weight = torch.zeros_like(weight)
    grid = lambda meta: (
        triton.cdiv(H_weight, meta["TILE_SIZE_H"]) * triton.cdiv(W_weight, meta["TILE_SIZE_W"]),
        B * triton.cdiv(C, meta["BLOCK_SIZE_C"]),
    )
    wrap_triton(_flying_conv2d_weight_bwd)[grid](
        grad_weight,
        grad_output,
        input,
        B,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        kernel_h,
        kernel_w,
        str_h,
        str_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        scale_h,
        scale_w,
    )
    return grad_weight


# ----------------------------------------------------------------------------

BLOCK_C_BIAS_ = [2, 4]
TILE_H_BIAS_ = [4, 8]
TILE_W_BIAS_ = [4, 8]
WARPS_BIAS_ = [1]
STAGES_BIAS_ = [2]


def get_configs():
    configs = []
    for bc, th, tw, nw, ns in product(BLOCK_C_BIAS_, TILE_H_BIAS_, TILE_W_BIAS_, WARPS_BIAS_, STAGES_BIAS_):
        # num_elements = bc * th * tw
        # if num_elements > 4096:
        # continue  # upper limit
        configs.append(triton.Config({"BLOCK_SIZE_C": bc, "TILE_SIZE_H": th, "TILE_SIZE_W": tw}, num_warps=nw, num_stages=ns))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["B", "C", "H_out", "W_out", "SCALE_H", "SCALE_W"],
)
@triton.jit
def _flying_conv2d_bias_bwd(
    grad_bias_ptr,
    grad_out_ptr,
    B,
    C,
    H_out,
    W_out,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    """backward pass for bias data: gather version
    aL/aW = aL/aY * aY/aW = grad_output * input data (chain rule)

    grad_bias (to be calculated): [B, C, H_out // scale_h, W_out // scale_w]
    grad_output: [B, C, H_out, W_out]
    1. get four neighborhood pixels from grad_output data
    2. sum along neighborhood axis for inverse bilinear interpolation
    """
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)
    H_w_down = H_out // SCALE_H
    W_w_down = W_out // SCALE_W
    tile_h_idx = pid_hw // tl.cdiv(W_w_down, TILE_SIZE_W)
    tile_w_idx = pid_hw % tl.cdiv(W_w_down, TILE_SIZE_W)

    offs_h = tile_h_idx * TILE_SIZE_H + tl.arange(0, TILE_SIZE_H)
    offs_w = tile_w_idx * TILE_SIZE_W + tl.arange(0, TILE_SIZE_W)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    offs_b = pid_bc // num_c_blocks
    offs_c = (pid_bc % num_c_blocks) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    WIN_H_SIZE: tl.constexpr = 2 * SCALE_H
    WIN_W_SIZE: tl.constexpr = 2 * SCALE_W
    WIN_TOTAL: tl.constexpr = WIN_H_SIZE * WIN_W_SIZE
    offs_win = tl.arange(0, WIN_TOTAL)
    win_h_idx = offs_win // WIN_W_SIZE
    win_w_idx = offs_win % WIN_W_SIZE

    base_h_out = offs_h * SCALE_H - SCALE_H + (SCALE_H // 2)
    base_w_out = offs_w * SCALE_W - SCALE_W + (SCALE_W // 2)
    h_out_curr = base_h_out[:, None, None] + win_h_idx[None, None, :]  # [TILE_H, 1, WIN_H_SIZE]
    w_out_curr = base_w_out[None, :, None] + win_w_idx[None, None, :]  # [1, TILE_W, WIN_W_SIZE]

    h_curr = (h_out_curr.to(tl.float32) + 0.5) / SCALE_H - 0.5
    w_curr = (w_out_curr.to(tl.float32) + 0.5) / SCALE_W - 0.5
    h_curr = tl.maximum(h_curr, 0.0)
    w_curr = tl.maximum(w_curr, 0.0)

    h_down = h_curr.to(tl.int32)
    w_down = w_curr.to(tl.int32)
    alpha_h = h_curr - h_down.to(tl.float32)
    alpha_w = w_curr - w_down.to(tl.float32)

    n_h = offs_h[:, None, None] - h_down
    n_w = offs_w[None, :, None] - w_down
    weight_val_h = 1.0 - tl.abs(alpha_h - n_h)
    weight_val_w = 1.0 - tl.abs(alpha_w - n_w)
    weight_val_h += tl.where((h_down == H_w_down - 1) & (offs_h[:, None, None] == H_w_down - 1), alpha_h, 0.0)
    weight_val_w += tl.where((w_down == W_w_down - 1) & (offs_w[None, :, None] == W_w_down - 1), alpha_w, 0.0)
    coeffs = weight_val_h * weight_val_w  # [TILE_H, TILE_W, WIN_TOTAL]

    mask_bc = (offs_b < B) & (offs_c[:, None, None] < C)  # [BLOCK_C, 1, 1]
    mask_spatial = (offs_h[:, None] < H_w_down) & (offs_w[None, :] < W_w_down)  # [TILE_H, TILE_W]
    # mask_win: [TILE_H, TILE_W, WIN_TOTAL]
    mask_win = (h_out_curr >= 0) & (h_out_curr < H_out) & (w_out_curr >= 0) & (w_out_curr < W_out) & (coeffs > 0.0)
    # grad_out_idx: [BLOCK_C, TILE_H, TILE_W, WIN_TOTAL]
    grad_out_idx = (
        offs_b * C * H_out * W_out
        + offs_c[:, None, None, None] * H_out * W_out
        + h_out_curr[None, :, :, :] * W_out
        + w_out_curr[None, :, :, :]
    )
    grad_out_val = tl.load(grad_out_ptr + grad_out_idx, mask=mask_win[None, :, :, :] & mask_bc[:, :, :, None], other=0.0)

    grad_bias_val = tl.sum(grad_out_val * coeffs[None, :, :, :], axis=3)
    grad_bias_idx = (
        offs_b * C * H_w_down * W_w_down + offs_c[:, None, None] * H_w_down * W_w_down + offs_h[:, None] * W_w_down + offs_w[None, :]
    )
    tl.store(grad_bias_ptr + grad_bias_idx, grad_bias_val, mask=mask_spatial[None, :, :] & mask_bc)


@triton_op("mylib::flying_conv2d_bias_bwd", mutates_args={})
def flying_conv2d_bias_bwd(
    grad_output: torch.Tensor,
    bias: torch.Tensor,
    scale_h: int,
    scale_w: int,
) -> torch.Tensor:
    B, C, H_weight, W_weight = bias.shape
    H_out = H_weight * scale_h
    W_out = W_weight * scale_w

    grad_bias = torch.zeros_like(bias)
    grid = lambda meta: (
        triton.cdiv(H_weight, meta["TILE_SIZE_H"]) * triton.cdiv(W_weight, meta["TILE_SIZE_W"]),
        B * triton.cdiv(C, meta["BLOCK_SIZE_C"]),
    )
    wrap_triton(_flying_conv2d_bias_bwd)[grid](
        grad_bias,
        grad_output,
        B,
        C,
        H_out,
        W_out,
        scale_h,
        scale_w,
    )
    return grad_bias