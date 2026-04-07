"""
Evaluation: perplexity and lm-eval harness for baseline/palu variants.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment import collect_environment
from .metrics import compute_stats
from .palu_loader import load_palu_model
from .dimension_repair import DimensionRepairer

DEFAULT_BASELINE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_model(variant: str, device: str, dtype_str: str = "float16", baseline_model_id: str = DEFAULT_BASELINE_MODEL_ID):
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    if variant == "baseline":
        tokenizer = AutoTokenizer.from_pretrained(baseline_model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            baseline_model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        palu_dir = None
    elif variant == "palu":
        model, tokenizer, palu_dir = load_palu_model(device=device, torch_dtype=torch_dtype)
    elif variant == "palu_repair":
        model, tokenizer, palu_dir = load_palu_model(device=device, torch_dtype=torch_dtype)
        repairer = DimensionRepairer(strategy="minimal")
        model, _ = repairer.repair_model(model, inplace=False)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, palu_dir


def load_text_corpus() -> str:
    tiny_path = Path("data/tiny_corpus.txt")
    if tiny_path.exists():
        return tiny_path.read_text()
    tiny_path.parent.mkdir(parents=True, exist_ok=True)
    tiny_text = "Artificial intelligence is transforming systems. Performance depends on alignment."
    tiny_path.write_text(tiny_text)
    return tiny_text


def get_wikitext(allow_fallback: bool = False):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return "\n".join(ds["text"])
    except Exception as e:
        if allow_fallback:
            return load_text_corpus()
        raise RuntimeError(
            "Failed to load WikiText-2 validation split. "
            "Pass --allow-fallback-corpus only for smoke/debug runs."
        ) from e


def compute_ppl(model, tokenizer, text: str, device: str, block_size: int = 512):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    nlls = []
    for i in range(0, input_ids.size(1), block_size):
        chunk = input_ids[:, i : i + block_size]
        if chunk.size(1) < 2:
            continue
        with torch.inference_mode():
            out = model(chunk, labels=chunk)
            nll = out.loss * chunk.size(1)
            nlls.append(nll)
    if not nlls:
        return {"ppl": None, "nll": None, "tokens": 0}
    nll_sum = torch.stack(nlls).sum()
    tok_count = input_ids.size(1)
    ppl = torch.exp(nll_sum / tok_count).item()
    return {"ppl": ppl, "nll": nll_sum.item(), "tokens": tok_count}


def run_ppl(model, tokenizer, device, allow_fallback: bool = False):
    text = get_wikitext(allow_fallback=allow_fallback)
    return compute_ppl(model, tokenizer, text, device)


def _score_text(model, tokenizer, device: str, text: str) -> float:
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(ids)
    logits = out.logits[0, :-1]
    targets = ids[0, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(1, targets.unsqueeze(1)).sum().item()


def run_tasks(model, tokenizer, tasks: str, limit: int, device: str):
    from datasets import load_dataset

    model.eval()
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    scores = {}
    raw = {}

    for task_name in task_list:
        correct = 0
        total = 0

        if task_name == "piqa":
            ds = load_dataset("piqa", split="validation", trust_remote_code=True)
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            for ex in ds:
                goal = ex["goal"]
                choices = [ex["sol1"], ex["sol2"]]
                label = ex["label"]
                pred = max(
                    range(len(choices)),
                    key=lambda idx: _score_text(model, tokenizer, device, f"{goal} {choices[idx]}")
                )
                correct += int(pred == label)
                total += 1

        elif task_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            for ex in ds:
                ctx = ex["ctx"]
                endings = ex["endings"]
                label = int(ex["label"])
                pred = max(
                    range(len(endings)),
                    key=lambda idx: _score_text(model, tokenizer, device, f"{ctx} {endings[idx]}")
                )
                correct += int(pred == label)
                total += 1

        else:
            raw[task_name] = {"error": f"unsupported task: {task_name}"}
            continue

        score = correct / total if total else 0.0
        scores[task_name] = score
        raw[task_name] = {"correct": correct, "total": total, "accuracy": score}

    return {"raw": raw, "scores": scores}


def save_results(run_dir: Path, config: dict, raw: dict, summary: dict, run_summary: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    env = collect_environment()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    (run_dir / "raw.json").write_text(json.dumps(raw, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "run_summary.md").write_text(run_summary)


def main():
    parser = argparse.ArgumentParser(description="LLM eval (ppl / lm-eval)")
    parser.add_argument("--variant", choices=["baseline", "palu", "palu_repair"], required=True)
    parser.add_argument("--suite", choices=["ppl", "tasks", "all"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--baseline-model-id", default=DEFAULT_BASELINE_MODEL_ID)
    parser.add_argument("--tasks", default="piqa,hellaswag")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--allow-fallback-corpus", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.variant}_{args.suite}"
    run_dir = args.out / run_id

    model, tokenizer, palu_dir = load_model(args.variant, args.device, args.dtype, args.baseline_model_id)
    config = {
        "variant": args.variant,
        "suite": args.suite,
        "tasks": args.tasks,
        "limit": args.limit,
        "device": args.device,
        "dtype": args.dtype,
        "baseline_model_id": args.baseline_model_id,
        "palu_dir": str(palu_dir) if palu_dir else None,
    }

    raw = {}
    summary = {}
    if args.suite in {"ppl", "all"}:
        res = run_ppl(model, tokenizer, args.device, allow_fallback=args.allow_fallback_corpus)
        raw["ppl"] = res
        summary["ppl"] = res["ppl"]
        summary["tokens"] = res["tokens"]
    if args.suite in {"tasks", "all"}:
        res = run_tasks(model, tokenizer, args.tasks, args.limit, args.device)
        raw["tasks"] = res
        summary["scores"] = res.get("scores", {})

    run_summary = f"# Run summary\\n\\nVariant: {args.variant}\\nSuite: {args.suite}\\nRun ID: {run_id}\\n"
    save_results(run_dir, config, raw, summary, run_summary)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
