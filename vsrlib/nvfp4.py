"""NVFP4 (E2M1, two-level scaled) DiT linears for Blackwell GPUs (sm100+).

Each value is stored as 4-bit E2M1 with an E4M3 scale per 16 values along K and
one fp32 scale per tensor (NVIDIA's two-level NVFP4 recipe). The tensor scale
puts the block scales into E4M3's normal range: the DiT weights' block scales
sit around 0.01, mostly subnormal in E4M3, so single-level scaling would round
them coarsely. Weights are quantized once; activations on every forward by a
Triton kernel (an absmax pass, then quantize + pack + swizzle). The GEMM is
cuBLASLt's native block-scaled FP4 through torch._scaled_mm_v2. Lossy.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

from .sm89_ops.fp8_quant import _absmax_kernel, _num_warps
from .sm89_ops.fp8_ffn import _gelu_absmax_kernel, _gelu_tanh

E2M1_MAX = 6.0
E4M3_MAX = 448.0
_GELU_C0 = 0.7978845608028654  # sqrt(2/pi), same constant as aten tanh-GELU

_BLOCK_SCALED = 2  # ScalingType.BlockWise1x16
_TENSORWISE = 0    # ScalingType.TensorWise
_SWIZZLE = 1       # SwizzleType.SWIZZLE_32_4_4
_NO_SWIZZLE = 0


@triton.jit
def _nvfp4_quant_kernel(x_ptr, gamax_ptr, q_ptr, s_ptr, M, K, NKB,
                        GELU: tl.constexpr, C0: tl.constexpr,
                        E2M1_MAX: tl.constexpr, E4M3_MAX: tl.constexpr):
    # One program = 128 rows x 64 columns = 128 x 4 scale blocks, which is
    # exactly one 512-byte tile of the SWIZZLE_32_4_4 scale layout.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * 128 + tl.arange(0, 128)
    cols = pid_k * 64 + tl.arange(0, 64)
    rmask = rows < M
    x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    if GELU:
        x = _gelu_tanh(x, C0)
    g = tl.maximum(tl.load(gamax_ptr), 1e-12) / (E2M1_MAX * E4M3_MAX)  # tensor scale
    xb = tl.reshape(x, [128, 4, 16])
    bamax = tl.max(tl.abs(xb), axis=2)
    s8 = tl.minimum(tl.div_rn(tl.div_rn(bamax, E2M1_MAX), g), E4M3_MAX).to(tl.float8e4nv)
    den = s8.to(tl.float32) * g
    z = tl.where(den[:, :, None] > 0, tl.div_rn(xb, den[:, :, None]), 0.0)
    a = tl.abs(z)
    # E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (codes 0-7); ties round to even.
    code = ((a > 0.25).to(tl.uint8) + (a >= 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8)
            + (a >= 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a >= 3.5).to(tl.uint8)
            + (a > 5.0).to(tl.uint8))
    code = code | ((z < 0).to(tl.uint8) * 8)
    lo, hi = tl.split(tl.reshape(code, [128, 32, 2]))
    packed = lo | (hi << 4)
    pcols = pid_k * 32 + tl.arange(0, 32)
    tl.store(q_ptr + rows[:, None] * (K // 2) + pcols[None, :], packed, mask=rmask[:, None])
    # Scale tile: offset (r % 32) * 16 + (r // 32) * 4 + c within the tile.
    r = tl.arange(0, 128)
    c = tl.arange(0, 4)
    soff = (pid_m * NKB + pid_k) * 512 + (r % 32)[:, None] * 16 + (r // 32)[:, None] * 4 + c[None, :]
    sbytes = tl.where(rmask[:, None], s8.to(tl.uint8, bitcast=True), 0)
    tl.store(s_ptr + soff, sbytes)


def quantize_nvfp4(x2, gelu=False):
    """x2: [M, K] contiguous bf16 (K % 64 == 0) -> (packed E2M1 [M, K/2] uint8,
    swizzled E4M3 block scales (uint8, flat), fp32 tensor scale [1]). With
    gelu=True, quantizes gelu_tanh(x2) without materializing it."""
    M, K = x2.shape
    if K % 64:
        raise ValueError(f"NVFP4 needs K % 64 == 0, got {K}")
    n = x2.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x2.device)
    BLOCK = 8192 if n >= 8192 else triton.next_power_of_2(max(n, 1))
    grid = (triton.cdiv(n, BLOCK),)
    if gelu:
        _gelu_absmax_kernel[grid](x2, amax, n, C0=_GELU_C0, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
    else:
        _absmax_kernel[grid](x2, amax, n, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
    mt, kt = triton.cdiv(M, 128), K // 64
    q = torch.empty((M, K // 2), dtype=torch.uint8, device=x2.device)
    s = torch.empty(mt * kt * 512, dtype=torch.uint8, device=x2.device)
    _nvfp4_quant_kernel[(mt, kt)](x2, amax, q, s, M, K, kt, GELU=gelu, C0=_GELU_C0,
                                  E2M1_MAX=E2M1_MAX, E4M3_MAX=E4M3_MAX, num_warps=8)
    return q, s, torch.clamp(amax, min=1e-12) / (E2M1_MAX * E4M3_MAX)


def nvfp4_mm(q, s, g, wq, ws, wg, bias):
    """(q, s, g) [M, K] activations x (wq, ws, wg) [N, K] weights -> bf16 [M, N]."""
    return torch._scaled_mm_v2(
        q.view(torch.float4_e2m1fn_x2), wq.view(torch.float4_e2m1fn_x2).t(),
        [s.view(torch.float8_e4m3fn), g], [_BLOCK_SCALED, _TENSORWISE], [_SWIZZLE, _NO_SWIZZLE],
        [ws.view(torch.float8_e4m3fn), wg], [_BLOCK_SCALED, _TENSORWISE], [_SWIZZLE, _NO_SWIZZLE],
        bias, torch.bfloat16,
    )


def dequantize_nvfp4(q, s, g, K):
    """Reference decode back to fp32 [M, K] (tests only)."""
    M = q.shape[0]
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=q.device)
    codes = torch.stack((q & 15, q >> 4), -1).reshape(M, K).long()
    vals = lut[codes]
    mt, kt = triton.cdiv(M, 128), K // 64
    sc = s.view(mt, kt, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(mt * 128, kt * 4)[:M]
    sc = sc.contiguous().view(torch.float8_e4m3fn).float()
    return vals.reshape(M, K // 16, 16) * sc.unsqueeze(-1) * g, sc


def probe(device):
    """Raise unless native NVFP4 GEMM works here (Blackwell + a torch that has it)."""
    if not hasattr(torch, "_scaled_mm_v2") or not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this PyTorch build lacks _scaled_mm_v2 / float4_e2m1fn_x2")
    cc = torch.cuda.get_device_capability(device)
    if cc < (10, 0):
        raise RuntimeError(f"needs a Blackwell GPU (sm100+), found sm{cc[0]}{cc[1]}")
    a = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
    b = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
    y = nvfp4_mm(*quantize_nvfp4(a), *quantize_nvfp4(b), None)
    torch.cuda.synchronize(device)
    if not torch.isfinite(y).all():
        raise RuntimeError("NVFP4 GEMM probe returned non-finite values")


class NVFP4Linear(nn.Module):
    """Inference-only Linear with static NVFP4 weights and dynamic NVFP4 activations."""

    def __init__(self, linear: nn.Linear, shared=None, last=False):
        super().__init__()
        self.shared, self.last = shared, last  # SharedInput; `last` clears it after use
        w = linear.weight.detach()
        if w.dtype != torch.bfloat16 or w.device.type != "cuda":
            raise ValueError("NVFP4 conversion needs a CUDA bf16 Linear")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if self.out_features % 16:
            raise ValueError("NVFP4 linear out_features must be a multiple of 16")
        wq, ws, wg = quantize_nvfp4(w.contiguous())
        self.register_buffer("wq", wq)
        self.register_buffer("ws", ws)
        self.register_buffer("wg", wg)
        self.bias = (nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
                     if linear.bias is not None else None)

    def _mm(self, q, s, g):
        return nvfp4_mm(q, s, g, self.wq, self.ws, self.wg, self.bias)

    def forward(self, x):
        shape = x.shape
        if self.shared is None:
            y = self._mm(*quantize_nvfp4(x.reshape(-1, shape[-1]).contiguous()))
        else:
            y = self._mm(*self.shared.get(x))
            if self.last:
                self.shared.clear()
        return y.reshape(*shape[:-1], self.out_features)


class NVFP4FFN(nn.Module):
    """Sequential(Linear, GELU(tanh), Linear) with the GELU fused into the
    quantization of the second GEMM's input (never materialized in bf16)."""

    def __init__(self, lin0: NVFP4Linear, lin2: NVFP4Linear):
        super().__init__()
        self.lin0 = lin0
        self.lin2 = lin2

    def forward(self, x):
        shape = x.shape
        h = self.lin0._mm(*quantize_nvfp4(x.reshape(-1, shape[-1]).contiguous()))
        out = self.lin2._mm(*quantize_nvfp4(h, gelu=True))
        return out.reshape(*shape[:-1], out.shape[-1])


class SharedInput:
    """Lets q/k/v of one attention layer quantize their common input once: the
    first caller quantizes, the others reuse it while `x` is the same tensor."""

    def __init__(self):
        self.key = None
        self.val = None

    def get(self, x):
        key = (x, x._version)
        if self.key is None or self.key[0] is not x or self.key[1] != x._version:
            self.val = quantize_nvfp4(x.reshape(-1, x.shape[-1]).contiguous())
            self.key = key
        return self.val

    def clear(self):
        self.key = self.val = None
