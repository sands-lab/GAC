from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "third_party/LLM-Pruner/lm-evaluation-harness/lm_eval/tasks/quac.py"
REGISTRY_PATH = ROOT / "third_party/LLM-Pruner/lm-evaluation-harness/lm_eval/tasks/__init__.py"


def _load_quac_module():
    spec = importlib.util.spec_from_file_location("issue46_quac", TASK_PATH)
    quac = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    rf_stub = types.SimpleNamespace(greedy_until=lambda ctx, spec: ("rf", ctx, spec))

    base_module = types.ModuleType("lm_eval.base")
    base_module.Task = type("Task", (), {})
    base_module.rf = rf_stub
    base_module.mean = lambda items: sum(items) / len(items)

    lm_eval_module = types.ModuleType("lm_eval")
    datasets_module = types.ModuleType("lm_eval.datasets")
    quac_dataset_package = types.ModuleType("lm_eval.datasets.quac")
    quac_dataset_module = types.ModuleType("lm_eval.datasets.quac.quac")
    quac_dataset_module.__file__ = str(TASK_PATH)

    lm_eval_module.base = base_module
    lm_eval_module.datasets = datasets_module
    datasets_module.quac = quac_dataset_package
    quac_dataset_package.quac = quac_dataset_module

    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "lm_eval",
            "lm_eval.base",
            "lm_eval.datasets",
            "lm_eval.datasets.quac",
            "lm_eval.datasets.quac.quac",
        )
    }

    sys.modules["lm_eval"] = lm_eval_module
    sys.modules["lm_eval.base"] = base_module
    sys.modules["lm_eval.datasets"] = datasets_module
    sys.modules["lm_eval.datasets.quac"] = quac_dataset_package
    sys.modules["lm_eval.datasets.quac.quac"] = quac_dataset_module

    try:
        spec.loader.exec_module(quac)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return quac


def main() -> None:
    task_text = TASK_PATH.read_text()
    registry_text = REGISTRY_PATH.read_text()
    errors: list[str] = []

    if "NotImplementedError(\"Evaluation not implemented\")" in task_text:
        errors.append("quac task still raises NotImplementedError for evaluation methods")
    if "# TODO: implement evaluation." in task_text:
        errors.append("quac task still contains TODO placeholders for evaluation")
    if '"quac": quac.QuAC' not in registry_text:
        errors.append("task registry does not expose quac.QuAC")
    if '# "quac": quac.QuAC' in registry_text:
        errors.append("task registry still comments out quac.QuAC")

    quac = _load_quac_module()
    task = object.__new__(quac.QuAC)
    raw_doc = {
        "title": "France",
        "section_title": "Geography",
        "paragraph": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answer": "Paris",
    }
    processed = task._process_doc(dict(raw_doc))
    context = task.doc_to_text(processed)

    if processed["title"] != "France - Geography":
        errors.append(
            f"_process_doc did not combine title and section_title: {processed['title']!r}"
        )

    for snippet in (
        "TITLE: France - Geography",
        "PARAGRAPH: Paris is the capital of France.",
        "Q: What is the capital of France?",
    ):
        if snippet not in context:
            errors.append(f"doc_to_text missing snippet {snippet!r}")

    if not context.rstrip().endswith("A:"):
        errors.append("doc_to_text must end with an answer prompt")

    if task.doc_to_target(processed).strip() != "Paris":
        errors.append("doc_to_target did not return the gold answer")

    sentinel = object()
    original_greedy_until = quac.rf.greedy_until
    quac.rf.greedy_until = lambda ctx, spec: (sentinel, ctx, spec)
    try:
        request = task.construct_requests(processed, "CTX")
    finally:
        quac.rf.greedy_until = original_greedy_until

    if request != (sentinel, "CTX", {"until": ["\n"]}):
        errors.append(f"construct_requests returned unexpected request payload: {request!r}")

    perfect = task.process_results(processed, [" Paris\nQ: ignored"])
    wrong = task.process_results(processed, ["Lyon"])

    for metric_name in ("f1", "em"):
        if metric_name not in perfect:
            errors.append(f"process_results missing metric {metric_name!r}")

    if not math.isclose(perfect.get("f1", -1.0), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"expected perfect f1 == 1.0, got {perfect.get('f1')!r}")
    if not math.isclose(perfect.get("em", -1.0), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"expected perfect em == 1.0, got {perfect.get('em')!r}")
    if wrong.get("f1", 1.0) != 0.0 or wrong.get("em", 1.0) != 0.0:
        errors.append(f"expected incorrect answer metrics to be 0.0, got {wrong!r}")

    aggregation = task.aggregation()
    higher_is_better = task.higher_is_better()

    if set(aggregation) != {"f1", "em"}:
        errors.append(f"aggregation returned unexpected metrics: {sorted(aggregation)}")
    else:
        if not math.isclose(aggregation["f1"]([1.0, 0.0]), 0.5, rel_tol=0.0, abs_tol=1e-9):
            errors.append("aggregation['f1'] did not behave like mean")
        if not math.isclose(aggregation["em"]([1.0, 0.0]), 0.5, rel_tol=0.0, abs_tol=1e-9):
            errors.append("aggregation['em'] did not behave like mean")

    if higher_is_better != {"f1": True, "em": True}:
        errors.append(f"higher_is_better returned unexpected payload: {higher_is_better!r}")

    if errors:
        for error in errors:
            print(f"FAIL [46-llmpruner-quac-eval]: {error}")
        raise SystemExit(1)

    print(
        "PASS [46-llmpruner-quac-eval]: QuAC task is registered and the deterministic request/metric contract passes."
    )


if __name__ == "__main__":
    main()
