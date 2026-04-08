"""
GAC (GPU-Aligned Compression) Rank Allocation Algorithm.

Demonstrates the GAC framework:
1. Simulate fisher-proportional rank allocation (pre-rounding ideal targets)
2. Apply four strategies: unaligned, round-to-8, round-to-32, GAC DP
3. Compare: alignment statistics, parameter budget, estimated latency
4. Output rank configs for recompression experiments

Usage:
    python scripts/gac_rank_allocation.py \
        --scores results/rank_scores/llama3_8b.json \
        --palu-config /path/to/palu/config.json \
        --profile-csv results/alignment_sweep.csv \
        --output results/gac_allocation/
"""

import argparse
import json
import math
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.gcompress_bench.palu_loader import default_palu_base_dir


# ---------------------------------------------------------------------------
# Constants for Llama-3-8B with PaLU (group_size=4)
# ---------------------------------------------------------------------------
NUM_LAYERS = 32
NUM_GROUPS = 2          # 8 KV heads / group_size 4
FULL_RANK = 512         # head_dim(128) * group_size(4)
NUM_PROJECTIONS = 64    # 32 layers * 2 (k_proj, v_proj)
PROJ_NAMES = ["k_proj", "v_proj"]


def load_fisher_scores(scores_path: str) -> Dict[str, List[float]]:
    """Load fisher scores from rank_scores JSON."""
    with open(scores_path) as f:
        data = json.load(f)
    return {
        "k_proj": data["scores"]["fisher"]["k_proj"],
        "v_proj": data["scores"]["fisher"]["v_proj"],
    }


def load_palu_ranks(config_path: str) -> Dict[str, List[int]]:
    """Load existing PaLU per-layer ranks from checkpoint config."""
    with open(config_path) as f:
        cfg = json.load(f)
    ranks = {}
    for name, r in cfg.get("head_wise_ranks", {}).items():
        layer = int(name.split(".")[2])
        proj = name.split(".")[-1]
        ranks[(layer, proj)] = r  # list of per-group ranks
    return ranks


def simulate_fisher_allocation(
    fisher: Dict[str, List[float]], total_budget: int
) -> Dict[Tuple[int, str], float]:
    """
    Simulate fisher-proportional allocation with cap-and-redistribute.

    Returns per-(layer, proj) per-group float rank values.
    Total budget is matched by iteratively redistributing surplus from
    capped projections to uncapped ones.
    """
    all_scores = []
    keys = []
    for proj in PROJ_NAMES:
        for layer in range(NUM_LAYERS):
            all_scores.append(fisher[proj][layer])
            keys.append((layer, proj))

    fisher_sum = sum(all_scores)
    float_ranks = {}
    for i, key in enumerate(keys):
        per_group_float = (total_budget * all_scores[i] / fisher_sum) / NUM_GROUPS
        float_ranks[key] = per_group_float

    # Cap at FULL_RANK and redistribute surplus iteratively
    for _ in range(20):
        surplus = 0.0
        uncapped_fisher_sum = 0.0
        for i, key in enumerate(keys):
            if float_ranks[key] > FULL_RANK:
                surplus += (float_ranks[key] - FULL_RANK) * NUM_GROUPS
                float_ranks[key] = FULL_RANK
            elif float_ranks[key] < 8.0:
                float_ranks[key] = 8.0
            else:
                uncapped_fisher_sum += all_scores[i]

        if surplus < 1.0 or uncapped_fisher_sum < 1e-9:
            break

        for i, key in enumerate(keys):
            if float_ranks[key] < FULL_RANK:
                extra = (surplus * all_scores[i] / uncapped_fisher_sum) / NUM_GROUPS
                float_ranks[key] += extra

    # Clamp final values
    for key in float_ranks:
        float_ranks[key] = max(8.0, min(float(FULL_RANK), float_ranks[key]))

    return float_ranks


def strategy_unaligned(float_ranks: Dict, budget: int) -> Dict:
    """Floor to nearest integer (no alignment)."""
    ranks = {}
    for key, fr in float_ranks.items():
        ranks[key] = max(1, int(math.floor(fr)))
    # Greedy budget fill
    _greedy_fill(ranks, float_ranks, budget)
    return ranks


def strategy_round_to_n(float_ranks: Dict, fisher: Dict, budget: int, n: int) -> Dict:
    """Round each rank independently to nearest multiple of n, then fill budget."""
    ranks = {}
    for key, fr in float_ranks.items():
        rounded = max(n, round(fr / n) * n)
        rounded = min(FULL_RANK, rounded)
        ranks[key] = rounded

    total = _total_budget(ranks)

    # Sort by sensitivity (ascending) for removal, descending for addition
    def sensitivity(k):
        layer, proj = k
        return fisher[proj][layer]

    keys_by_sens_asc = sorted(float_ranks.keys(), key=sensitivity)
    keys_by_sens_desc = sorted(float_ranks.keys(), key=sensitivity, reverse=True)

    # Remove from least sensitive first if over budget
    while total > budget:
        removed = False
        for k in keys_by_sens_asc:
            if ranks[k] > n:
                ranks[k] -= n
                total -= n * NUM_GROUPS
                removed = True
                if total <= budget:
                    break
        if not removed:
            break

    # Add to most sensitive first if under budget
    while total < budget - n * NUM_GROUPS + 1:
        added = False
        for k in keys_by_sens_desc:
            if ranks[k] < FULL_RANK:
                ranks[k] += n
                total += n * NUM_GROUPS
                added = True
                if total >= budget:
                    break
        if not added:
            break

    return ranks


def strategy_gac_dp(
    float_ranks: Dict,
    fisher: Dict[str, List[float]],
    budget: int,
    align: int = 8,
    search_radius: int = 8,
) -> Dict:
    """
    GAC DP: Multi-choice knapsack for optimal aligned rank allocation.

    For each projection, generate candidate ranks (multiples of `align` near
    the ideal float rank). Use DP to maximize Fisher-weighted value under
    total budget constraint.

    Asymmetric formulation: value_i = f_i * (r_i - r*_i)
    - Round UP (r_i > r*_i): positive value (preserves information)
    - Round DOWN (r_i < r*_i): negative value (loses information)
    - High-Fisher layers get priority for rounding up

    The DP finds the allocation that:
    - Uses exactly the budget (or as close as possible)
    - Maximizes total Fisher-weighted value
    - All chosen ranks are multiples of `align`
    """
    # Build projection list with candidates
    projections = []
    for proj in PROJ_NAMES:
        for layer in range(NUM_LAYERS):
            key = (layer, proj)
            ideal = float_ranks[key]
            f_i = fisher[proj][layer]

            # Generate candidates: multiples of align near ideal
            ideal_aligned = round(ideal / align) * align
            candidates = []
            for offset in range(-search_radius, search_radius + 1):
                c = ideal_aligned + offset * align
                if c < align:
                    continue
                if c > FULL_RANK:
                    continue
                candidates.append(c)

            if not candidates:
                candidates = [max(align, min(FULL_RANK, ideal_aligned))]

            projections.append({
                "key": key,
                "ideal": ideal,
                "fisher": f_i,
                "candidates": candidates,
            })

    # DP: maximize total Fisher-weighted value under budget constraint
    budget_unit = align * NUM_GROUPS  # smallest budget quantum
    B = budget // budget_unit + 1

    n = len(projections)
    NEG_INF = float("-inf")

    # dp[b] = max value using exactly b budget units for first i projections
    dp = [NEG_INF] * (B + 1)
    dp[0] = 0.0
    choice = [[None] * (B + 1) for _ in range(n)]

    for i, proj in enumerate(projections):
        new_dp = [NEG_INF] * (B + 1)
        for c in proj["candidates"]:
            value_c = proj["fisher"] * (c - proj["ideal"])
            c_units = (c * NUM_GROUPS) // budget_unit
            for b in range(c_units, B + 1):
                prev_b = b - c_units
                if dp[prev_b] > NEG_INF and dp[prev_b] + value_c > new_dp[b]:
                    new_dp[b] = dp[prev_b] + value_c
                    choice[i][b] = c
        dp = new_dp

    # Find best feasible solution: prefer using full budget, but accept close
    max_b = budget // budget_unit
    best_b = None
    best_value = NEG_INF

    # First try exact budget match
    for b in range(max_b, max(0, max_b - 3), -1):
        if dp[b] > NEG_INF:
            best_b = b
            best_value = dp[b]
            break

    # If no exact match, search wider
    if best_b is None:
        for b in range(max_b, -1, -1):
            if dp[b] > NEG_INF:
                best_b = b
                best_value = dp[b]
                break

    if best_b is None:
        return strategy_round_to_n(float_ranks, fisher, budget, align)

    # Backtrack
    ranks = {}
    b = best_b
    for i in range(n - 1, -1, -1):
        c = choice[i][b]
        if c is None:
            # Should not happen if DP is correct
            c = projections[i]["candidates"][len(projections[i]["candidates"]) // 2]
        ranks[projections[i]["key"]] = c
        b -= (c * NUM_GROUPS) // budget_unit

    return ranks


def _greedy_fill(ranks: Dict, float_ranks: Dict, budget: int):
    """Greedily add +1 to ranks to fill budget gap."""
    total = _total_budget(ranks)
    if total >= budget:
        return
    # Sort by fractional part (descending) - add to those closest to rounding up
    keys_sorted = sorted(
        float_ranks.keys(),
        key=lambda k: -(float_ranks[k] - ranks[k]),
    )
    diff = budget - total
    idx = 0
    while diff > 0 and idx < len(keys_sorted) * FULL_RANK:
        k = keys_sorted[idx % len(keys_sorted)]
        if ranks[k] < FULL_RANK:
            ranks[k] += 1
            diff -= NUM_GROUPS
        idx += 1


def _total_budget(ranks: Dict) -> int:
    """Compute total budget: sum of per_group_rank * NUM_GROUPS."""
    return sum(r * NUM_GROUPS for r in ranks.values())


def load_profile_table(csv_path: str) -> Dict:
    """Load alignment profiling data for latency estimation."""
    table = {"K": {}, "N": {}, "M": {}}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim = row["dim_name"]
            val = int(row["dim_value"])
            table[dim][val] = {
                "time_us": float(row["time_us"]),
                "tflops": float(row["tflops"]),
                "kernel": row["kernel"],
            }
    return table


def estimate_alignment_penalty(rank: int, profile_table: Dict) -> float:
    """
    Estimate alignment penalty ratio for a given rank dimension.

    Uses the K-dimension profiling data pattern:
    - mod 8 == 0: no penalty (1.0)
    - mod 2 == 0 but mod 8 != 0: ~20% penalty (CUTLASS align2)
    - odd: ~35% penalty (CUTLASS align1)

    If exact value is in profile table, use it directly.
    """
    # Check if we have direct profiling data
    if rank in profile_table.get("K", {}):
        aligned_ref = None
        # Find nearest aligned reference
        for delta in range(0, 9):
            ref = (rank // 8) * 8
            if ref in profile_table["K"]:
                aligned_ref = profile_table["K"][ref]["time_us"]
                break
            ref = ((rank // 8) + 1) * 8
            if ref in profile_table["K"]:
                aligned_ref = profile_table["K"][ref]["time_us"]
                break
        if aligned_ref:
            return profile_table["K"][rank]["time_us"] / aligned_ref
        return 1.0

    # Use empirical pattern from profiling data
    if rank % 8 == 0:
        return 1.0  # aligned, no penalty
    elif rank % 2 == 0:
        return 1.20  # CUTLASS align2 penalty (~20%)
    else:
        return 1.35  # CUTLASS align1 penalty (~35%)


def analyze_strategy(
    name: str,
    ranks: Dict,
    float_ranks: Dict,
    fisher: Dict[str, List[float]],
    profile_table: Dict,
) -> Dict:
    """Compute statistics for a rank allocation strategy."""
    total_budget = _total_budget(ranks)

    # Alignment statistics
    n_aligned_8 = sum(1 for r in ranks.values() if r % 8 == 0)
    n_aligned_32 = sum(1 for r in ranks.values() if r % 32 == 0)
    n_misaligned = sum(1 for r in ranks.values() if r % 8 != 0)
    n_odd = sum(1 for r in ranks.values() if r % 2 != 0)

    # Fisher-weighted value (asymmetric: positive = round up, negative = round down)
    fisher_value = 0.0
    weighted_abs_dev = 0.0
    for proj in PROJ_NAMES:
        for layer in range(NUM_LAYERS):
            key = (layer, proj)
            ideal = float_ranks[key]
            actual = ranks[key]
            f_i = fisher[proj][layer]
            fisher_value += f_i * (actual - ideal)
            weighted_abs_dev += f_i * abs(actual - ideal)

    # Estimated latency penalty (sum of per-projection penalties)
    total_penalty = 0.0
    for key, r in ranks.items():
        penalty = estimate_alignment_penalty(r, profile_table)
        total_penalty += penalty

    avg_penalty = total_penalty / len(ranks)

    # Rank statistics
    all_ranks = list(ranks.values())

    return {
        "strategy": name,
        "total_budget": total_budget,
        "n_projections": len(ranks),
        "n_aligned_mod8": n_aligned_8,
        "n_aligned_mod32": n_aligned_32,
        "n_misaligned_mod8": n_misaligned,
        "n_odd": n_odd,
        "pct_aligned_mod8": 100.0 * n_aligned_8 / len(ranks),
        "fisher_value": fisher_value,
        "weighted_abs_deviation": weighted_abs_dev,
        "avg_latency_penalty": avg_penalty,
        "rank_mean": np.mean(all_ranks),
        "rank_min": min(all_ranks),
        "rank_max": max(all_ranks),
    }


def ranks_to_palu_config(ranks: Dict) -> Dict[str, List[int]]:
    """Convert (layer, proj) -> rank dict to PaLU head_wise_ranks format."""
    config = {}
    for (layer, proj), rank in sorted(ranks.items()):
        name = f"model.layers.{layer}.self_attn.{proj}"
        config[name] = [rank] * NUM_GROUPS
    return config


def main():
    parser = argparse.ArgumentParser(description="GAC Rank Allocation")
    parser.add_argument("--scores", default="results/rank_scores/llama3_8b.json")
    parser.add_argument("--palu-config",
                        default=str(
                            default_palu_base_dir()
                            / "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-svd"
                            / "config.json"
                        ))
    parser.add_argument("--profile-csv", default="results/alignment_sweep.csv")
    parser.add_argument("--output", default="results/gac_allocation")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    fisher = load_fisher_scores(args.scores)
    palu_ranks = load_palu_ranks(args.palu_config)
    profile_table = load_profile_table(args.profile_csv)

    # Compute total budget from existing PaLU checkpoint
    total_budget = sum(sum(r) for r in palu_ranks.values())
    print(f"Total budget from PaLU checkpoint: {total_budget}")
    print(f"Compression ratio: {total_budget / (NUM_PROJECTIONS * NUM_GROUPS * FULL_RANK):.3f}")

    # Step 1: Simulate fisher-proportional allocation (ideal float ranks)
    float_ranks = simulate_fisher_allocation(fisher, total_budget)

    # Save ideal float ranks for reference
    float_ranks_list = []
    for proj in PROJ_NAMES:
        for layer in range(NUM_LAYERS):
            float_ranks_list.append({
                "layer": layer,
                "proj": proj,
                "ideal_rank": float_ranks[(layer, proj)],
                "fisher_score": fisher[proj][layer],
            })

    with open(out_dir / "ideal_float_ranks.json", "w") as f:
        json.dump(float_ranks_list, f, indent=2)

    # Step 2: Apply four strategies
    strategies = {}

    # a) Unaligned (floor to int)
    strategies["unaligned"] = strategy_unaligned(float_ranks, total_budget)

    # b) Round to nearest 8
    strategies["round8"] = strategy_round_to_n(float_ranks, fisher, total_budget, 8)

    # c) Round to nearest 32 (PaLU default)
    strategies["round32"] = strategy_round_to_n(float_ranks, fisher, total_budget, 32)

    # c') PaLU actual (from checkpoint)
    palu_actual = {}
    for (layer, proj), r in palu_ranks.items():
        palu_actual[(layer, proj)] = r[0]  # per-group rank
    strategies["palu_actual"] = palu_actual

    # d) GAC DP (8-aligned, globally optimal)
    strategies["gac_dp"] = strategy_gac_dp(float_ranks, fisher, total_budget,
                                            align=8, search_radius=5)

    # Step 3: Analyze each strategy
    print("\n" + "=" * 100)
    print(f"{'Strategy':<16} {'Budget':>8} {'Aligned/8':>10} {'Aligned/32':>11} "
          f"{'Misaligned':>10} {'Odd':>5} {'F.Value':>10} {'Abs.Dev':>10} {'Avg Penalty':>12}")
    print("=" * 110)

    results = []
    for name, ranks in strategies.items():
        stats = analyze_strategy(name, ranks, float_ranks, fisher, profile_table)
        results.append(stats)
        print(f"{stats['strategy']:<16} {stats['total_budget']:>8} "
              f"{stats['n_aligned_mod8']:>6}/{stats['n_projections']:<4}"
              f"{stats['n_aligned_mod32']:>7}/{stats['n_projections']:<4}"
              f"{stats['n_misaligned_mod8']:>8}  {stats['n_odd']:>5} "
              f"{stats['fisher_value']:>10.1f} {stats['weighted_abs_deviation']:>10.1f} "
              f"{stats['avg_latency_penalty']:>10.3f}x")

    print("=" * 100)

    # Step 4: Save results
    with open(out_dir / "strategy_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save rank configs for recompression
    for name, ranks in strategies.items():
        if name == "palu_actual":
            continue  # already exists as checkpoint
        config = ranks_to_palu_config(ranks)
        with open(out_dir / f"ranks_{name}.json", "w") as f:
            json.dump(config, f, indent=2)

    # Step 5: Print per-layer comparison
    print("\n\nPer-layer rank comparison (per-group ranks):")
    print(f"{'Layer':<6} {'Proj':<8} {'Ideal':>8} {'Unaligned':>10} "
          f"{'Round8':>8} {'Round32':>8} {'PaLU':>8} {'GAC_DP':>8}")
    print("-" * 76)
    for proj in PROJ_NAMES:
        for layer in range(NUM_LAYERS):
            key = (layer, proj)
            ideal = float_ranks[key]
            print(f"{layer:<6} {proj:<8} {ideal:>8.1f} "
                  f"{strategies['unaligned'][key]:>10} "
                  f"{strategies['round8'][key]:>8} "
                  f"{strategies['round32'][key]:>8} "
                  f"{strategies['palu_actual'][key]:>8} "
                  f"{strategies['gac_dp'][key]:>8}")

    # Step 6: Detailed tradeoff analysis
    print("\n\n=== Key Findings ===")

    # Compare unaligned vs GAC DP
    unaligned_stats = results[0]
    gac_stats = results[-1]
    palu_stats = [r for r in results if r["strategy"] == "palu_actual"][0]

    print(f"\n1. Unaligned allocation: {unaligned_stats['n_misaligned_mod8']}/{unaligned_stats['n_projections']} "
          f"projections misaligned ({100*unaligned_stats['n_misaligned_mod8']/unaligned_stats['n_projections']:.0f}%)")
    print(f"   → Average latency penalty: {unaligned_stats['avg_latency_penalty']:.3f}x")

    print(f"\n2. GAC DP: {gac_stats['n_misaligned_mod8']}/{gac_stats['n_projections']} misaligned")
    print(f"   → Average latency penalty: {gac_stats['avg_latency_penalty']:.3f}x")
    print(f"   → Fisher value: {gac_stats['fisher_value']:.1f} "
          f"(vs PaLU {palu_stats['fisher_value']:.1f})")

    speedup = unaligned_stats['avg_latency_penalty'] / gac_stats['avg_latency_penalty']
    print(f"\n3. Estimated speedup from alignment (unaligned → GAC): {speedup:.2f}x")

    print(f"   Fisher value GAC: {gac_stats['fisher_value']:.1f}, PaLU: {palu_stats['fisher_value']:.1f}")

    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
