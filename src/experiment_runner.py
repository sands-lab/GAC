"""Experiment runner for night sweep experiments."""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
import numpy as np

from .config import BenchmarkConfig
from .measurement import benchmark_kernel, benchmark_gemm, compute_gemm_tflops, compute_gemm_bandwidth
from .utils import set_deterministic, get_dtype, allocate_tensors, compute_statistics
from .environment import collect_environment


def load_experiment_spec(spec_path: Path, exp_name: str) -> Dict[str, Any]:
    """Load experiment specification from YAML file."""
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    
    if exp_name not in spec.get("experiments", {}):
        raise ValueError(f"Experiment '{exp_name}' not found in spec file")
    
    return spec["experiments"][exp_name]


def resolve_dense_sweep_values(exp_spec: Dict[str, Any], axis: str) -> List[int]:
    """Resolve dense sweep values for an axis from either explicit values or a range."""
    axis = axis.upper()
    values_key = f"{axis}_values"
    range_key = f"{axis}_range"

    if values_key in exp_spec:
        values = exp_spec[values_key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Experiment spec key '{values_key}' must be a non-empty list")
        return [int(value) for value in values]

    if range_key not in exp_spec:
        raise ValueError(
            f"Experiment spec must define either '{values_key}' or '{range_key}'"
        )

    min_value, max_value = exp_spec[range_key]
    step1 = int(exp_spec.get(f"{axis}_step_1", 1))
    step2 = int(exp_spec.get(f"{axis}_step_2", step1))
    boundary = int(exp_spec.get(f"{axis}_boundary", 128))

    if min_value > max_value:
        raise ValueError(
            f"Experiment spec key '{range_key}' must be ordered as [min, max]"
        )

    values: List[int] = []
    low_end = min(max_value, boundary)

    if min_value <= low_end:
        values.extend(range(min_value, low_end + 1, step1))

    if max_value > boundary:
        if min_value <= boundary:
            high_start = boundary + step2
        else:
            high_start = min_value
        values.extend(range(high_start, max_value + 1, step2))

    return values


def run_multiple_trials(
    kernel_fn,
    warmup: int,
    measure: int,
    trials: int,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """Run multiple trials and aggregate results."""
    all_times_ms = []
    
    for trial in range(trials):
        stats = benchmark_kernel(kernel_fn, warmup, measure, device)
        all_times_ms.extend(stats["times_ms"])
    
    # Aggregate statistics across all trials
    agg_stats = compute_statistics(all_times_ms)
    
    return {
        "trials": trials,
        "total_measurements": len(all_times_ms),
        "timing": agg_stats,
        "timing_raw": all_times_ms,
    }


def run_s1_sdpa_dense_sweep(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """S1: SDPA dense sweep across head_dim range."""
    set_deterministic(seed)
    dtype_str = exp_spec["dtype"]
    dtype = get_dtype(dtype_str)
    shapes = exp_spec["shapes"]
    hd_min, hd_max = exp_spec["head_dim_range"]
    step1 = exp_spec.get("head_dim_step_1", 1)
    step2 = exp_spec.get("head_dim_step_2", 2)
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)
    
    # Generate head_dim list
    # step1 applies up to the boundary (default 128), step2 applies above
    boundary = exp_spec.get("head_dim_boundary", 128)
    head_dims = list(range(hd_min, min(boundary + 1, hd_max + 1), step1))
    if hd_max > boundary:
        head_dims.extend(range(boundary + step2, hd_max + 1, step2))
    
    results = {
        "experiment": "S1_sdpa_dense_sweep",
        "config": exp_spec,
        "measurements": [],
    }
    
    for shape in shapes:
        batch = shape["batch"]
        seq_len = shape["seq_len"]
        n_heads = shape["n_heads"]
        
        for head_dim in head_dims:
            torch.manual_seed(seed)
            query = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
            key = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
            value = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
            
            def kernel_fn():
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=False
                )
            
            try:
                trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                
                results["measurements"].append({
                    "shape": {"batch": batch, "seq_len": seq_len, "n_heads": n_heads, "head_dim": head_dim},
                    "dtype": dtype_str,
                    **trial_results,
                })
            except Exception as e:
                results["measurements"].append({
                    "shape": {"batch": batch, "seq_len": seq_len, "n_heads": n_heads, "head_dim": head_dim},
                    "dtype": dtype_str,
                    "error": str(e),
                })
            
            del query, key, value
            torch.cuda.empty_cache()
    
    return results


def run_s2_sdpa_backend_forced(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """S2: SDPA with forced backends."""
    set_deterministic(seed)
    dtype_str = exp_spec["dtype"]
    dtype = get_dtype(dtype_str)
    shape = exp_spec["shape"]
    batch = shape["batch"]
    seq_len = shape["seq_len"]
    n_heads = shape["n_heads"]
    head_dims = exp_spec["head_dims"]
    backends = exp_spec["backends"]
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)
    
    results = {
        "experiment": "S2_sdpa_backend_forced",
        "config": exp_spec,
        "measurements": [],
    }
    
    for head_dim in head_dims:
        torch.manual_seed(seed)
        query = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        
        for backend in backends:
            def make_kernel_fn(b):
                def kernel_fn():
                    if b == "AUTO":
                        return torch.nn.functional.scaled_dot_product_attention(
                            query, key, value, is_causal=False
                        )
                    elif b == "FLASH":
                        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                            return torch.nn.functional.scaled_dot_product_attention(
                                query, key, value, is_causal=False
                            )
                    elif b == "MEM_EFFICIENT":
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                            return torch.nn.functional.scaled_dot_product_attention(
                                query, key, value, is_causal=False
                            )
                    elif b == "MATH":
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                            return torch.nn.functional.scaled_dot_product_attention(
                                query, key, value, is_causal=False
                            )
                return kernel_fn
            
            kernel_fn = make_kernel_fn(backend)
            
            try:
                trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                
                results["measurements"].append({
                    "head_dim": head_dim,
                    "backend": backend,
                    "dtype": dtype_str,
                    **trial_results,
                })
            except Exception as e:
                results["measurements"].append({
                    "head_dim": head_dim,
                    "backend": backend,
                    "dtype": dtype_str,
                    "error": str(e),
                    "backend_unsupported": True,
                })
        
        del query, key, value
        torch.cuda.empty_cache()
    
    return results


def run_g3_gemm_k_dense(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """G3: GEMM with K dimension sweep."""
    set_deterministic(seed)
    dtypes = exp_spec["dtypes"]
    shape = exp_spec["shape"]
    M = shape["M"]
    N = shape["N"]
    K_values = resolve_dense_sweep_values(exp_spec, "K")
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)
    
    results = {
        "experiment": "G3_gemm_k_dense",
        "config": exp_spec,
        "measurements": [],
    }
    
    for dtype_str in dtypes:
        dtype = get_dtype(dtype_str)
        
        for K in K_values:
            a, b = allocate_tensors((M, K), (K, N), dtype=dtype, device=device, seed=seed)
            
            def kernel_fn():
                torch.matmul(a, b)
            
            try:
                trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                
                # Compute TFLOPs for mean
                mean_time_s = trial_results["timing"]["mean"] / 1000.0
                tflops = compute_gemm_tflops(M, N, K, mean_time_s)
                bandwidth = compute_gemm_bandwidth(M, N, K, dtype, mean_time_s)
                
                # Compute TFLOPs for all measurements
                tflops_list = [compute_gemm_tflops(M, N, K, t / 1000.0) for t in trial_results["timing_raw"]]
                
                results["measurements"].append({
                    "shape": {"M": M, "N": N, "K": K},
                    "dtype": dtype_str,
                    **trial_results,
                    "derived": {
                        "tflops_mean": tflops,
                        "tflops_stats": compute_statistics(tflops_list),
                        "bandwidth_gbs_mean": bandwidth,
                    },
                })
            except Exception as e:
                results["measurements"].append({
                    "shape": {"M": M, "N": N, "K": K},
                    "dtype": dtype_str,
                    "error": str(e),
                })
            
            del a, b
            torch.cuda.empty_cache()
    
    return results


def run_g4_gemm_n_dense(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """G4: GEMM with N dimension sweep (projection-like)."""
    set_deterministic(seed)
    dtypes = exp_spec["dtypes"]
    M_values = exp_spec["M_values"]
    K = exp_spec["K"]
    N_values = resolve_dense_sweep_values(exp_spec, "N")
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)
    
    results = {
        "experiment": "G4_gemm_n_dense_projectionlike",
        "config": exp_spec,
        "measurements": [],
    }
    
    for dtype_str in dtypes:
        dtype = get_dtype(dtype_str)
        
        for M in M_values:
            for N in N_values:
                a, b = allocate_tensors((M, K), (K, N), dtype=dtype, device=device, seed=seed)
                
                def kernel_fn():
                    torch.matmul(a, b)
                
                try:
                    trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                    
                    mean_time_s = trial_results["timing"]["mean"] / 1000.0
                    tflops = compute_gemm_tflops(M, N, K, mean_time_s)
                    bandwidth = compute_gemm_bandwidth(M, N, K, dtype, mean_time_s)
                    
                    tflops_list = [compute_gemm_tflops(M, N, K, t / 1000.0) for t in trial_results["timing_raw"]]
                    
                    results["measurements"].append({
                        "shape": {"M": M, "K": K, "N": N},
                        "dtype": dtype_str,
                        **trial_results,
                        "derived": {
                            "tflops_mean": tflops,
                            "tflops_stats": compute_statistics(tflops_list),
                            "bandwidth_gbs_mean": bandwidth,
                        },
                    })
                except Exception as e:
                    results["measurements"].append({
                        "shape": {"M": M, "K": K, "N": N},
                        "dtype": dtype_str,
                        "error": str(e),
                    })
                
                del a, b
                torch.cuda.empty_cache()
    
    return results


def run_g5_gemm_m_dense(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """G5: GEMM with M dimension sweep (sequence-like)."""
    set_deterministic(seed)
    dtypes = exp_spec["dtypes"]
    shape = exp_spec["shape"]
    K = shape["K"]
    N = shape["N"]
    M_values = resolve_dense_sweep_values(exp_spec, "M")
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)

    results = {
        "experiment": "G5_gemm_m_dense_sequence_like",
        "config": exp_spec,
        "measurements": [],
    }

    for dtype_str in dtypes:
        dtype = get_dtype(dtype_str)

        for M in M_values:
            a, b = allocate_tensors((M, K), (K, N), dtype=dtype, device=device, seed=seed)

            def kernel_fn():
                torch.matmul(a, b)

            try:
                trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)

                mean_time_s = trial_results["timing"]["mean"] / 1000.0
                tflops = compute_gemm_tflops(M, N, K, mean_time_s)
                bandwidth = compute_gemm_bandwidth(M, N, K, dtype, mean_time_s)

                tflops_list = [compute_gemm_tflops(M, N, K, t / 1000.0) for t in trial_results["timing_raw"]]

                results["measurements"].append({
                    "shape": {"M": M, "K": K, "N": N},
                    "dtype": dtype_str,
                    **trial_results,
                    "derived": {
                        "tflops_mean": tflops,
                        "tflops_stats": compute_statistics(tflops_list),
                        "bandwidth_gbs_mean": bandwidth,
                    },
                })
            except Exception as e:
                results["measurements"].append({
                    "shape": {"M": M, "K": K, "N": N},
                    "dtype": dtype_str,
                    "error": str(e),
                })

            del a, b
            torch.cuda.empty_cache()

    return results


def run_p1_padding_rescue(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """P1: Padding rescue comparison."""
    set_deterministic(seed)
    dtype_str = exp_spec["dtype"]
    dtype = get_dtype(dtype_str)
    logical_dim = exp_spec["logical_head_dim"]
    pad_options = exp_spec["pad_options"]
    sdpa_shape = exp_spec["sdpa_shape"]
    gemm_shapes = exp_spec["gemm_shapes"]
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)
    
    results = {
        "experiment": "P1_padding_rescue",
        "config": exp_spec,
        "measurements": [],
    }
    
    # SDPA tests
    batch = sdpa_shape["batch"]
    seq_len = sdpa_shape["seq_len"]
    n_heads = sdpa_shape["n_heads"]
    
    for physical_dim in pad_options:
        logical_dim_actual = min(logical_dim, physical_dim)
        
        torch.manual_seed(seed)
        query = torch.randn(batch, n_heads, seq_len, physical_dim, dtype=dtype, device=device)
        key = torch.randn(batch, n_heads, seq_len, physical_dim, dtype=dtype, device=device)
        value = torch.randn(batch, n_heads, seq_len, physical_dim, dtype=dtype, device=device)
        
        if physical_dim > logical_dim:
            # Zero out padding
            query[:, :, :, logical_dim:] = 0
            key[:, :, :, logical_dim:] = 0
            value[:, :, :, logical_dim:] = 0
        
        def kernel_fn():
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False
            )
            return out[:, :, :, :logical_dim_actual] if physical_dim > logical_dim_actual else out
        
        try:
            trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
            
            memory_overhead = ((physical_dim - logical_dim) / logical_dim * 100) if physical_dim > logical_dim else 0
            
            results["measurements"].append({
                "operation": "SDPA",
                "logical_dim": logical_dim_actual,
                "physical_dim": physical_dim,
                "memory_overhead_pct": memory_overhead,
                "dtype": dtype_str,
                **trial_results,
            })
        except Exception as e:
            results["measurements"].append({
                "operation": "SDPA",
                "logical_dim": logical_dim_actual,
                "physical_dim": physical_dim,
                "error": str(e),
            })
        
        del query, key, value
        torch.cuda.empty_cache()
    
    # GEMM tests
    for gemm_shape in gemm_shapes:
        M = gemm_shape["M"]
        if "K" in gemm_shape:
            # Reduction: (M, K) @ (K, N)
            K = gemm_shape["K"]
            N = gemm_shape["N"]
            for physical_dim in pad_options:
                logical_dim_actual = min(logical_dim, physical_dim)
                dim_to_pad = K if K == logical_dim else N
                
                if dim_to_pad == K:
                    a, b = allocate_tensors((M, physical_dim), (physical_dim, N), dtype=dtype, device=device, seed=seed)
                    if physical_dim > logical_dim:
                        a[:, logical_dim:] = 0
                        b[logical_dim:, :] = 0
                else:
                    a, b = allocate_tensors((M, K), (K, physical_dim), dtype=dtype, device=device, seed=seed)
                    if physical_dim > logical_dim:
                        b[:, logical_dim:] = 0
                
                def kernel_fn():
                    out = torch.matmul(a, b)
                    if dim_to_pad == N and physical_dim > logical_dim:
                        return out[:, :logical_dim]
                    return out
                
                try:
                    trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                    memory_overhead = ((physical_dim - logical_dim) / logical_dim * 100) if physical_dim > logical_dim else 0
                    
                    results["measurements"].append({
                        "operation": "GEMM",
                        "shape_type": "reduction" if K == logical_dim else "projection",
                        "logical_dim": logical_dim_actual,
                        "physical_dim": physical_dim,
                        "shape": {"M": M, "K": K if K != logical_dim else physical_dim, "N": N if N != logical_dim else physical_dim},
                        "memory_overhead_pct": memory_overhead,
                        "dtype": dtype_str,
                        **trial_results,
                    })
                except Exception as e:
                    results["measurements"].append({
                        "operation": "GEMM",
                        "shape_type": "reduction" if K == logical_dim else "projection",
                        "logical_dim": logical_dim_actual,
                        "physical_dim": physical_dim,
                        "error": str(e),
                    })
                
                del a, b
                torch.cuda.empty_cache()
    
    return results


def run_c21_backend_selection(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """C2.1: Verify PyTorch SDPA backend selection boundaries.

    This experiment verifies:
    1. Which backend is selected for each head_dim (AUTO mode)
    2. Backend availability for each head_dim
    3. Performance comparison across backends
    4. Focus on PaLU-typical dimensions (114-125 range)
    """
    set_deterministic(seed)
    dtype_str = exp_spec["dtype"]
    dtype = get_dtype(dtype_str)
    shapes = exp_spec["shapes"]
    head_dims = exp_spec["head_dims"]
    backends = exp_spec["backends"]
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)

    results = {
        "experiment": "C21_backend_selection",
        "config": exp_spec,
        "measurements": [],
        "backend_summary": {},  # Quick lookup: head_dim -> selected_backend
    }

    for shape in shapes:
        batch = shape["batch"]
        seq_len = shape["seq_len"]
        n_heads = shape["n_heads"]

        for head_dim in head_dims:
            torch.manual_seed(seed)
            query = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
            key = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
            value = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)

            backend_results = {}
            auto_timing = None

            for backend in backends:
                def make_kernel_fn(b):
                    def kernel_fn():
                        if b == "AUTO":
                            return torch.nn.functional.scaled_dot_product_attention(
                                query, key, value, is_causal=False
                            )
                        elif b == "FLASH":
                            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                                return torch.nn.functional.scaled_dot_product_attention(
                                    query, key, value, is_causal=False
                                )
                        elif b == "MEM_EFFICIENT":
                            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                                return torch.nn.functional.scaled_dot_product_attention(
                                    query, key, value, is_causal=False
                                )
                        elif b == "MATH":
                            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                                return torch.nn.functional.scaled_dot_product_attention(
                                    query, key, value, is_causal=False
                                )
                    return kernel_fn

                kernel_fn = make_kernel_fn(backend)

                try:
                    trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                    backend_results[backend] = {
                        "available": True,
                        "timing": trial_results["timing"],
                    }
                    if backend == "AUTO":
                        auto_timing = trial_results["timing"]["mean"]
                except Exception as e:
                    backend_results[backend] = {
                        "available": False,
                        "error": str(e),
                    }

            # Detect which backend AUTO is using by comparing timings
            detected_backend = "UNKNOWN"
            if auto_timing is not None:
                min_diff = float('inf')
                for b in ["FLASH", "MEM_EFFICIENT", "MATH"]:
                    if b in backend_results and backend_results[b].get("available", False):
                        diff = abs(backend_results[b]["timing"]["mean"] - auto_timing)
                        # Use relative difference
                        rel_diff = diff / auto_timing if auto_timing > 0 else float('inf')
                        if rel_diff < min_diff and rel_diff < 0.05:  # Within 5%
                            min_diff = rel_diff
                            detected_backend = b

            # Record alignment properties
            is_8_aligned = head_dim % 8 == 0
            is_16_aligned = head_dim % 16 == 0

            measurement = {
                "shape": {"batch": batch, "seq_len": seq_len, "n_heads": n_heads, "head_dim": head_dim},
                "dtype": dtype_str,
                "alignment": {
                    "mod_8": is_8_aligned,
                    "mod_16": is_16_aligned,
                },
                "detected_backend": detected_backend,
                "backend_results": backend_results,
            }
            results["measurements"].append(measurement)

            # Update summary
            key_str = f"{batch}x{seq_len}x{n_heads}x{head_dim}"
            results["backend_summary"][key_str] = {
                "head_dim": head_dim,
                "detected_backend": detected_backend,
                "is_8_aligned": is_8_aligned,
                "flash_available": backend_results.get("FLASH", {}).get("available", False),
                "mem_efficient_available": backend_results.get("MEM_EFFICIENT", {}).get("available", False),
            }

            del query, key, value
            torch.cuda.empty_cache()

    return results


def run_het1_hetero_batching(exp_spec: Dict, device: str = "cuda:0", seed: int = 42) -> Dict:
    """HET1: Heterogeneous head batching penalty."""
    set_deterministic(seed)
    dtype_str = exp_spec["dtype"]
    dtype = get_dtype(dtype_str)
    total_N = exp_spec["total_N"]
    H = exp_spec["H"]
    patterns = exp_spec["patterns"]
    warmup = exp_spec.get("warmup", 50)
    measure = exp_spec.get("measure", 200)
    trials = exp_spec.get("trials", 3)
    
    results = {
        "experiment": "HET1_head_hetero_batching_penalty",
        "config": exp_spec,
        "measurements": [],
    }
    
    M = total_N  # Use total_N as M for projection-like shape
    K = total_N
    
    for pattern_name, pattern in patterns.items():
        groups = pattern["groups"]
        
        # Uniform: single GEMM
        if pattern_name == "uniform":
            N = groups[0]["dim"] * groups[0]["count"]
            a, b = allocate_tensors((M, K), (K, N), dtype=dtype, device=device, seed=seed)
            
            def kernel_fn():
                return torch.matmul(a, b)
            
            try:
                trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                mean_time_s = trial_results["timing"]["mean"] / 1000.0
                tflops = compute_gemm_tflops(M, N, K, mean_time_s)
                
                results["measurements"].append({
                    "pattern": pattern_name,
                    "num_gemm_calls": 1,
                    "shape": {"M": M, "K": K, "N": N},
                    "dtype": dtype_str,
                    **trial_results,
                    "derived": {"tflops_mean": tflops},
                })
            except Exception as e:
                results["measurements"].append({
                    "pattern": pattern_name,
                    "error": str(e),
                })
            
            del a, b
            torch.cuda.empty_cache()
        
        else:
            # Hetero: multiple GEMMs
            total_latency = 0
            num_calls = 0
            total_flops = 0
            
            for group in groups:
                dim = group["dim"]
                count = group["count"]
                N_group = dim * count
                
                a, b = allocate_tensors((M, K), (K, N_group), dtype=dtype, device=device, seed=seed)
                
                def kernel_fn():
                    return torch.matmul(a, b)
                
                try:
                    trial_results = run_multiple_trials(kernel_fn, warmup, measure, trials, device)
                    mean_time_s = trial_results["timing"]["mean"] / 1000.0
                    tflops = compute_gemm_tflops(M, N_group, K, mean_time_s)
                    
                    total_latency += trial_results["timing"]["mean"]
                    num_calls += 1
                    total_flops += 2 * M * N_group * K
                except Exception as e:
                    results["measurements"].append({
                        "pattern": pattern_name,
                        "group": {"dim": dim, "count": count},
                        "error": str(e),
                    })
                
                del a, b
                torch.cuda.empty_cache()
            
            if num_calls > 0:
                effective_tflops = total_flops / (total_latency / 1000.0) / 1e12
                
                results["measurements"].append({
                    "pattern": pattern_name,
                    "num_gemm_calls": num_calls,
                    "total_latency_ms": total_latency,
                    "dtype": dtype_str,
                    "derived": {"effective_tflops": effective_tflops},
                })
    
    return results


EXPERIMENT_RUNNERS = {
    "sdpa_dense": run_s1_sdpa_dense_sweep,
    "sdpa_backend_forced": run_s2_sdpa_backend_forced,
    "sdpa_backend_selection": run_c21_backend_selection,
    "gemm_k_dense": run_g3_gemm_k_dense,
    "gemm_n_dense": run_g4_gemm_n_dense,
    "gemm_m_dense": run_g5_gemm_m_dense,
    "padding_rescue": run_p1_padding_rescue,
    "hetero_batching": run_het1_hetero_batching,
}


def run_experiment(
    exp_spec: Dict,
    device: str = "cuda:0",
    seed: int = 42
) -> Dict:
    """Run an experiment based on its specification."""
    exp_type = exp_spec["type"]
    
    if exp_type not in EXPERIMENT_RUNNERS:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    runner = EXPERIMENT_RUNNERS[exp_type]
    return runner(exp_spec, device=device, seed=seed)
