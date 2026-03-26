from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=13, suite="stage-b-kernel-benchmark-1-gpu-large")

MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0


@dataclass(frozen=True)
class CaseSpec:
    name: str
    batch_size: int
    num_tokens: int
    num_heads: int
    head_dim: int
    rope_dim: int
    is_neox: bool


BENCH_CASES = (
    CaseSpec("flux_1024", 1, 4096, 24, 128, 128, False),
    CaseSpec("qwen_image_1024", 1, 4096, 32, 128, 128, False),
    CaseSpec("qwen_image_partial", 1, 4096, 32, 128, 64, False),
    # Z-Image-Turbo default 1024x1024 config: dim=3840, num_heads=30 -> head_dim=128.
    CaseSpec("zimage_1024", 1, 4096, 30, 128, 128, False),
    CaseSpec("batch2_medium", 2, 2048, 24, 128, 128, False),
)
CASE_BY_NAME = {case.name: case for case in BENCH_CASES}


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEFAULT_DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEFAULT_DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def make_inputs(case: CaseSpec) -> dict[str, torch.Tensor | bool]:
    seed = (
        case.batch_size * 1_000_003
        + case.num_tokens * 8191
        + case.num_heads * 127
        + case.head_dim * 17
        + case.rope_dim
    )
    generator = torch.Generator(device=DEFAULT_DEVICE)
    generator.manual_seed(seed)
    return {
        "q": torch.randn(
            case.batch_size * case.num_tokens,
            case.num_heads,
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "k": torch.randn(
            case.batch_size * case.num_tokens,
            case.num_heads,
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "q_weight": torch.randn(
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "k_weight": torch.randn(
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "positions": torch.randint(
            0,
            MAX_SEQ_LEN,
            (case.batch_size * case.num_tokens,),
            device=DEFAULT_DEVICE,
            dtype=torch.int64,
            generator=generator,
        ),
        "cos_sin_cache": create_cos_sin_cache(case.rope_dim),
        "is_neox": case.is_neox,
    }


def clone_inputs(
    inputs: dict[str, torch.Tensor | bool],
) -> dict[str, torch.Tensor | bool]:
    out: dict[str, torch.Tensor | bool] = {}
    for key, value in inputs.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def split_qknorm_rope(inputs: dict[str, torch.Tensor | bool]) -> None:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    from sglang.jit_kernel.norm import fused_inplace_qknorm

    q = inputs["q"]
    k = inputs["k"]
    q_weight = inputs["q_weight"]
    k_weight = inputs["k_weight"]
    positions = inputs["positions"]
    cos_sin_cache = inputs["cos_sin_cache"]
    is_neox = bool(inputs["is_neox"])

    fused_inplace_qknorm(q, k, q_weight, k_weight)
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=q.shape[-1],
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def fused_qknorm_rope(inputs: dict[str, torch.Tensor | bool]) -> None:
    from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope

    fused_inplace_qknorm_rope(
        inputs["q"],
        inputs["k"],
        inputs["q_weight"],
        inputs["k_weight"],
        inputs["cos_sin_cache"],
        inputs["positions"],
        is_neox=bool(inputs["is_neox"]),
        rope_dim=inputs["cos_sin_cache"].shape[-1],
    )


def benchmark_case(
    case: CaseSpec, fn_builder: Callable[[dict[str, torch.Tensor | bool]], None]
) -> float:
    inputs = make_inputs(case)
    runtime_ms, _, _ = triton.testing.do_bench(
        lambda: fn_builder(inputs), quantiles=(0.5, 0.2, 0.8)
    )
    return float(runtime_ms)


def profile_case(case: CaseSpec, provider: str, warmup: int, iters: int) -> None:
    inputs = make_inputs(case)
    fn = split_qknorm_rope if provider == "split" else fused_qknorm_rope
    for _ in range(warmup):
        fn(inputs)
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iters):
        fn(inputs)
    ender.record()
    torch.cuda.synchronize()
    total_ms = starter.elapsed_time(ender)
    print(
        f"PROFILE {case.name} provider={provider} avg_ms={total_ms / iters:.6f} total_ms={total_ms:.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--provider", choices=["split", "fused"], default="fused")
    parser.add_argument(
        "--case", choices=sorted(CASE_BY_NAME), default="qwen_image_1024"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args, _ = parser.parse_known_args(argv)

    if args.profile:
        profile_case(CASE_BY_NAME[args.case], args.provider, args.warmup, args.iters)
        return 0

    weighted_split = 0.0
    weighted_fused = 0.0
    weight_sum = 0.0
    for case in BENCH_CASES:
        split_ms = benchmark_case(case, split_qknorm_rope)
        fused_ms = benchmark_case(case, fused_qknorm_rope)
        speedup = split_ms / fused_ms if fused_ms > 0 else math.inf
        print(
            f"CASE {case.name}: split_ms={split_ms:.6f} fused_ms={fused_ms:.6f} speedup={speedup:.4f}x"
        )
        weight = float(
            case.batch_size * case.num_tokens * case.num_heads * case.head_dim
        )
        weight_sum += weight
        weighted_split += weight * split_ms
        weighted_fused += weight * fused_ms

    avg_split = weighted_split / weight_sum
    avg_fused = weighted_fused / weight_sum
    print(f"SPLIT_MS: {avg_split:.6f}")
    print(f"FUSED_MS: {avg_fused:.6f}")
    print(f"SPEEDUP: {avg_split / avg_fused:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
