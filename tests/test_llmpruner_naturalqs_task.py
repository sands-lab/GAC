from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "third_party/LLM-Pruner/lm-evaluation-harness/lm_eval/tasks/naturalqs.py"
REGISTRY_PATH = ROOT / "third_party/LLM-Pruner/lm-evaluation-harness/lm_eval/tasks/__init__.py"


def _load_naturalqs_module():
    spec = importlib.util.spec_from_file_location("issue48_naturalqs", TASK_PATH)
    naturalqs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    load_calls: list[dict[str, object]] = []
    validation_dataset = [
        {
            "question": {"text": "What is the capital of France?"},
            "document": {
                "tokens": {
                    "token": [
                        "<P>",
                        "Paris",
                        "is",
                        "the",
                        "capital",
                        "of",
                        "France",
                    ],
                    "is_html": [True, False, False, False, False, False, False],
                }
            },
            "annotations": {
                "long_answer": [{"start_token": 1, "end_token": 7}],
                "short_answers": [[{"start_token": 1, "end_token": 2}]],
            },
        }
    ]

    rf_stub = types.SimpleNamespace(greedy_until=lambda ctx, spec: ("rf", ctx, spec))

    class Task:
        def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
            self.download(data_dir, cache_dir, download_mode)
            self._training_docs = None
            self._fewshot_docs = None

    base_module = types.ModuleType("lm_eval.base")
    base_module.Task = Task
    base_module.rf = rf_stub
    base_module.mean = lambda items: sum(items) / len(items)

    metrics_module = types.ModuleType("lm_eval.metrics")
    metrics_module.metric_max_over_ground_truths = (
        lambda metric_fn, prediction, ground_truths: max(
            metric_fn(prediction, ground_truth) for ground_truth in ground_truths
        )
    )

    lm_eval_module = types.ModuleType("lm_eval")
    lm_eval_module.base = base_module
    lm_eval_module.metrics = metrics_module

    datasets_module = types.ModuleType("datasets")

    def load_dataset(**kwargs):
        load_calls.append(kwargs)
        return validation_dataset

    datasets_module.load_dataset = load_dataset

    original_modules = {
        name: sys.modules.get(name)
        for name in ("datasets", "lm_eval", "lm_eval.base", "lm_eval.metrics")
    }

    sys.modules["datasets"] = datasets_module
    sys.modules["lm_eval"] = lm_eval_module
    sys.modules["lm_eval.base"] = base_module
    sys.modules["lm_eval.metrics"] = metrics_module

    try:
        spec.loader.exec_module(naturalqs)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return naturalqs, validation_dataset, load_calls


def main() -> None:
    task_text = TASK_PATH.read_text()
    registry_text = REGISTRY_PATH.read_text()
    errors: list[str] = []

    for forbidden_snippet, message in (
        ("NotImplementedError(\"Evaluation not implemented\")", "naturalqs task still raises NotImplementedError"),
        ("# TODO: implement evaluation.", "naturalqs task still contains evaluation TODO placeholders"),
        ("TODO: NaturalQS has a *really* large train set", "naturalqs task still contains the train-download TODO instead of an implemented guard"),
    ):
        if forbidden_snippet in task_text:
            errors.append(message)

    if '"naturalqs": naturalqs.NaturalQs' not in registry_text:
        errors.append("task registry does not expose naturalqs.NaturalQs")
    if '# "naturalqs": naturalqs.NaturalQs' in registry_text:
        errors.append("task registry still comments out naturalqs.NaturalQs")

    naturalqs, validation_dataset, load_calls = _load_naturalqs_module()
    task = naturalqs.NaturalQs(data_dir="DATA", cache_dir="CACHE", download_mode="MODE")

    if len(load_calls) != 1:
        errors.append(f"expected one validation-only dataset load, got {len(load_calls)} calls")
    else:
        expected_call = {
            "path": "natural_questions",
            "name": None,
            "data_dir": "DATA",
            "cache_dir": "CACHE",
            "download_mode": "MODE",
            "split": "validation",
        }
        if load_calls[0] != expected_call:
            errors.append(f"unexpected load_dataset call: {load_calls[0]!r}")

    if task.has_training_docs():
        errors.append("has_training_docs should be False to avoid the giant train split")
    if not task.has_validation_docs():
        errors.append("has_validation_docs should remain True")
    if task.has_test_docs():
        errors.append("has_test_docs should remain False")
    if task.training_docs() != []:
        errors.append(f"training_docs should be empty, got {task.training_docs()!r}")
    if task.validation_docs() is not validation_dataset:
        errors.append("validation_docs did not return the downloaded validation split")

    doc = validation_dataset[0]
    target = task.doc_to_target(doc)
    answers = task.answer_texts(doc)
    context = task.doc_to_text(doc)

    if answers != ["Paris", "Paris is the capital of France"]:
        errors.append(f"unexpected answer extraction payload: {answers!r}")
    if target != " Paris":
        errors.append(f"doc_to_target returned unexpected value: {target!r}")
    if "What is the capital of France?" not in context or not context.rstrip().endswith("A:"):
        errors.append(f"doc_to_text returned unexpected prompt: {context!r}")
    if task.doc_to_decontamination_query(doc) != "What is the capital of France?":
        errors.append("doc_to_decontamination_query did not reuse the question text")

    sentinel = object()
    original_greedy_until = naturalqs.rf.greedy_until
    naturalqs.rf.greedy_until = lambda ctx, spec: (sentinel, ctx, spec)
    try:
        request = task.construct_requests(doc, "CTX")
    finally:
        naturalqs.rf.greedy_until = original_greedy_until

    if request != (sentinel, "CTX", {"until": ["\n"]}):
        errors.append(f"construct_requests returned unexpected payload: {request!r}")

    perfect_short = task.process_results(doc, [" Paris\nQ: ignored"])
    perfect_long = task.process_results(doc, ["Paris is the capital of France"])
    wrong = task.process_results(doc, ["Lyon"])

    for metric_name in ("f1", "em"):
        if metric_name not in perfect_short:
            errors.append(f"process_results missing metric {metric_name!r}")

    if not math.isclose(perfect_short.get("f1", -1.0), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"expected perfect short-answer f1 == 1.0, got {perfect_short.get('f1')!r}")
    if not math.isclose(perfect_short.get("em", -1.0), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"expected perfect short-answer em == 1.0, got {perfect_short.get('em')!r}")
    if not math.isclose(perfect_long.get("f1", -1.0), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"expected perfect long-answer f1 == 1.0, got {perfect_long.get('f1')!r}")
    if not math.isclose(perfect_long.get("em", -1.0), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"expected perfect long-answer em == 1.0, got {perfect_long.get('em')!r}")
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
            print(f"FAIL [48-llmpruner-naturalqs-eval]: {error}")
        raise SystemExit(1)

    print(
        "PASS [48-llmpruner-naturalqs-eval]: NaturalQS task uses a validation-only load, exposes deterministic QA metrics, and stays on the repo-tracked patch path."
    )


if __name__ == "__main__":
    main()
