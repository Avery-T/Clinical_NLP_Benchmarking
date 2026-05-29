#!/usr/bin/env python3
"""Benchmark a small local LLM on the prepared clinical classification tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openai import OpenAI
from sklearn.metrics import roc_auc_score


DEFAULT_BASE_URL = "http://localhost:1233/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "google/gemma-4-e4b"
DEFAULT_PREPARED_DIR = "benchmark_runs/prepared"
DEFAULT_OUTPUT_DIR = "benchmark_runs/small_llm"

TASK_DESCRIPTIONS = {
    "diabetes": "whether this synthetic patient has diabetes",
    "hypertension": "whether this synthetic patient has hypertension",
    "coronary_heart_disease": "whether this synthetic patient has coronary heart disease",
    "stroke": "whether this synthetic patient has a history of stroke",
    "asthma": "whether this synthetic patient has asthma",
    "copd": "whether this synthetic patient has COPD or emphysema",
}

SYSTEM_PROMPT = """You are a clinical binary classifier.

Return only valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.
Do not write a thinking process or chain-of-thought.
Write the final JSON object immediately.

Use exactly this schema:
{
  "label": 0 or 1,
  "confidence": number between 0 and 1,
  "reason": "brief clinical rationale",
  "evidence": ["short evidence item", "short evidence item"]
}
"""


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int | None
    timeout: float


class ClassificationFailure(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def task_description(task_name: str) -> str:
    return TASK_DESCRIPTIONS.get(
        task_name,
        f"whether this synthetic patient is positive for {task_name.replace('_', ' ')}",
    )


def load_split(task_dir: Path, split: str) -> pd.DataFrame:
    path = task_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared split: {path}")

    df = pd.read_csv(path)
    required = {"patient_id", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def discover_task_dirs(prepared_dir: Path, tasks: list[str] | None) -> list[Path]:
    if tasks:
        task_dirs = [prepared_dir / task for task in tasks]
    else:
        task_dirs = [
            path for path in sorted(prepared_dir.iterdir())
            if path.is_dir() and (path / "test.csv").exists()
        ]

    missing = [path for path in task_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared task directories: "
            + ", ".join(str(path) for path in missing)
        )
    if not task_dirs:
        raise FileNotFoundError(f"No prepared task directories found in {prepared_dir}")
    return task_dirs


def validate_result(data: dict[str, Any]) -> dict[str, Any]:
    required = ["label", "confidence", "reason", "evidence"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

    if data["label"] not in [0, 1]:
        raise ValueError("label must be 0 or 1")

    if not isinstance(data["confidence"], (int, float)):
        raise ValueError("confidence must be a number")
    if not 0 <= float(data["confidence"]) <= 1:
        raise ValueError("confidence must be between 0 and 1")

    if not isinstance(data["reason"], str):
        raise ValueError("reason must be a string")

    if not isinstance(data["evidence"], list):
        raise ValueError("evidence must be a list")
    if not all(isinstance(item, str) for item in data["evidence"]):
        raise ValueError("all evidence items must be strings")

    return {
        "label": int(data["label"]),
        "confidence": float(data["confidence"]),
        "reason": data["reason"],
        "evidence": data["evidence"],
    }


def parse_model_json(raw_output: str) -> dict[str, Any]:
    raw_output = raw_output.strip()
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw_output}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON output must be an object")
    return validate_result(parsed)


def truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def select_one_shot_examples(train_df: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in [1, 0]:
        matches = train_df[train_df["label"] == label]
        if matches.empty:
            raise ValueError(f"Could not find one-shot example with label {label}")
        row = matches.iloc[0]
        examples.append(
            {
                "patient_id": row["patient_id"],
                "label": int(row["label"]),
                "text": row["text"],
            }
        )
    return examples


def build_zero_shot_prompt(task_name: str, patient_text: str, max_chars: int) -> str:
    description = task_description(task_name)
    return f"""Classify this patient for the task: {description}.

Label meanings:
- 1 = positive for the task.
- 0 = negative for the task.

Use only the provided patient text. Return JSON only.

Patient text:
\"\"\"
{truncate_text(patient_text, max_chars)}
\"\"\"
"""


def build_one_shot_prompt(
    task_name: str,
    patient_text: str,
    examples: list[dict[str, Any]],
    max_chars: int,
) -> str:
    description = task_description(task_name)
    formatted_examples = []
    for index, example in enumerate(examples, start=1):
        formatted_examples.append(
            f"""Example {index}
Patient text:
\"\"\"
{truncate_text(example["text"], max_chars)}
\"\"\"
Correct JSON:
{{"label": {example["label"]}, "confidence": 1.0, "reason": "Labeled training example.", "evidence": ["training example label"]}}"""
        )

    return f"""Classify this patient for the task: {description}.

Label meanings:
- 1 = positive for the task.
- 0 = negative for the task.

Use the examples to learn the output format and label meaning. Then classify the final patient.

{chr(10).join(formatted_examples)}

Final patient text:
\"\"\"
{truncate_text(patient_text, max_chars)}
\"\"\"

Return JSON only for the final patient.
"""


def create_prompt(
    task_name: str,
    patient_text: str,
    shot: str,
    examples: list[dict[str, Any]],
    max_chars: int,
) -> str:
    if shot == "zero":
        return build_zero_shot_prompt(task_name, patient_text, max_chars)
    if shot == "one":
        return build_one_shot_prompt(task_name, patient_text, examples, max_chars)
    raise ValueError(f"Unknown shot mode: {shot}")


def classify_patient(
    client: OpenAI,
    config: LlmConfig,
    prompt: str,
) -> tuple[dict[str, Any], str]:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    raw_output = message.content or ""
    try:
        return parse_model_json(raw_output), raw_output
    except ValueError as error:
        if not raw_output.strip():
            reasoning = getattr(message, "reasoning_content", "") or ""
            detail = (
                f"Model returned empty content "
                f"(finish_reason={choice.finish_reason}, reasoning_chars={len(reasoning)})"
            )
            raise ClassificationFailure(detail, reasoning) from error
        raise ClassificationFailure(str(error), raw_output) from error


def classify_patient_with_retry(
    client: OpenAI,
    config: LlmConfig,
    prompt: str,
    max_retries: int,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    last_raw = ""
    for attempt in range(max_retries + 1):
        try:
            return classify_patient(client, config, prompt)
        except Exception as error:
            last_error = error
            last_raw = getattr(error, "raw_output", "")
            print(f"Attempt {attempt + 1} failed: {error}")
    raise ClassificationFailure(
        f"Could not get valid classification: {last_error}",
        last_raw,
    ) from last_error


def build_summary(
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
    failed: int,
    total: int,
) -> dict[str, Any]:
    classified = true_positive + true_negative + false_positive + false_negative
    accuracy = safe_divide(true_positive + true_negative, classified)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    macro_f1_negative_precision = safe_divide(true_negative, true_negative + false_negative)
    macro_f1_negative_recall = specificity
    negative_f1 = safe_divide(
        2 * macro_f1_negative_precision * macro_f1_negative_recall,
        macro_f1_negative_precision + macro_f1_negative_recall,
    )

    return {
        "total_attempted": total,
        "classified": classified,
        "failed": failed,
        "correct": true_positive + true_negative,
        "incorrect": false_positive + false_negative,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "positive_support": true_positive + false_negative,
        "negative_support": true_negative + false_positive,
        "support": classified,
    }


def update_confusion_counts(
    actual_label: int,
    predicted_label: int,
    counts: dict[str, int],
) -> None:
    if actual_label == 1 and predicted_label == 1:
        counts["true_positive"] += 1
    elif actual_label == 0 and predicted_label == 0:
        counts["true_negative"] += 1
    elif actual_label == 0 and predicted_label == 1:
        counts["false_positive"] += 1
    elif actual_label == 1 and predicted_label == 0:
        counts["false_negative"] += 1


def result_fieldnames() -> list[str]:
    return [
        "task",
        "shot",
        "split",
        "source_row",
        "patient_id",
        "actual_label",
        "predicted_label",
        "correct",
        "confidence",
        "reason",
        "evidence",
        "error",
        "raw_output",
        "patient_text",
    ]


def evaluate_task_shot(
    client: OpenAI,
    config: LlmConfig,
    task_dir: Path,
    split: str,
    shot: str,
    output_dir: Path,
    limit: int | None,
    max_retries: int,
    progress_every: int,
    max_chars: int,
) -> dict[str, Any]:
    task_name = task_dir.name
    eval_df = load_split(task_dir, split)
    if limit is not None:
        eval_df = eval_df.head(limit).copy()

    examples = []
    if shot == "one":
        examples = select_one_shot_examples(load_split(task_dir, "train"))

    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = task_output_dir / f"{shot}_shot_predictions.csv"
    metrics_path = task_output_dir / f"{shot}_shot_metrics.json"

    counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "failed": 0,
    }
    actual_labels: list[int] = []
    positive_scores: list[float] = []
    total = 0

    with predictions_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=result_fieldnames())
        writer.writeheader()

        for source_row, row in enumerate(eval_df.itertuples(index=False), start=1):
            total += 1
            if progress_every > 0 and (total == 1 or total % progress_every == 0):
                print(f"Classifying {task_name} {shot}-shot row {total}/{len(eval_df)}")

            prompt = create_prompt(
                task_name=task_name,
                patient_text=row.text,
                shot=shot,
                examples=examples,
                max_chars=max_chars,
            )
            record = {
                "task": task_name,
                "shot": f"{shot}_shot",
                "split": split,
                "source_row": source_row,
                "patient_id": row.patient_id,
                "actual_label": int(row.label),
                "predicted_label": "",
                "correct": "",
                "confidence": "",
                "reason": "",
                "evidence": "",
                "error": "",
                "raw_output": "",
                "patient_text": row.text,
            }

            try:
                prediction, raw_output = classify_patient_with_retry(
                    client=client,
                    config=config,
                    prompt=prompt,
                    max_retries=max_retries,
                )
                predicted_label = int(prediction["label"])
                actual_label = int(row.label)
                correct = predicted_label == actual_label
                confidence = float(prediction["confidence"])
                positive_score = confidence if predicted_label == 1 else 1.0 - confidence
                update_confusion_counts(actual_label, predicted_label, counts)
                actual_labels.append(actual_label)
                positive_scores.append(positive_score)
                record.update(
                    {
                        "predicted_label": predicted_label,
                        "correct": int(correct),
                        "confidence": confidence,
                        "reason": prediction["reason"],
                        "evidence": json.dumps(prediction["evidence"]),
                        "raw_output": raw_output,
                    }
                )
            except Exception as error:
                counts["failed"] += 1
                record["error"] = str(error)
                record["raw_output"] = getattr(error, "raw_output", "")

            writer.writerow(record)

    summary = build_summary(
        true_positive=counts["true_positive"],
        true_negative=counts["true_negative"],
        false_positive=counts["false_positive"],
        false_negative=counts["false_negative"],
        failed=counts["failed"],
        total=total,
    )
    if len(set(actual_labels)) == 2:
        summary["roc_auc"] = float(roc_auc_score(actual_labels, positive_scores))
    summary.update(
        {
            "task": task_name,
            "shot": f"{shot}_shot",
            "split": split,
            "model": config.model,
            "base_url": config.base_url,
            "predictions_file": str(predictions_path),
            "created_at": utc_now(),
        }
    )

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"{task_name} {shot}-shot: "
        f"accuracy={summary['accuracy']:.3f}, f1={summary['f1']:.3f}, "
        f"failed={summary['failed']}"
    )
    return summary


def shot_choices(values: Iterable[str]) -> list[str]:
    normalized = []
    for value in values:
        value = value.strip().lower().replace("-shot", "")
        if value in {"zero", "0"}:
            normalized.append("zero")
        elif value in {"one", "1"}:
            normalized.append("one")
        else:
            raise ValueError(f"Unknown shot mode: {value}")
    if not normalized:
        raise ValueError("At least one shot mode is required.")
    return normalized


def aggregate_key(task_name: str, shot: str) -> str:
    return f"{task_name}_{shot}_shot"


def split_aggregate_task_name(task_key: str) -> tuple[str, str]:
    for suffix, label in [
        ("_zero_shot", "zero-shot"),
        ("_one_shot", "one-shot"),
    ]:
        if task_key.endswith(suffix):
            return task_key[: -len(suffix)], label
    return task_key, "small LLM"


def save_score_plot(
    aggregate: dict[str, Any],
    output_dir: Path,
    metrics: list[str],
    dpi: int,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_key, task_metrics in aggregate["tasks"].items():
        task_name, shot_name = split_aggregate_task_name(task_key)
        for metric_name in metrics:
            if metric_name in task_metrics:
                rows.append(
                    {
                        "task": task_name,
                        "shot": shot_name,
                        "metric": metric_name,
                        "score": float(task_metrics[metric_name]),
                    }
                )

    if not rows:
        raise ValueError("No small LLM metric scores available for plotting.")

    summary_path = plot_dir / "score_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    tasks = sorted({row["task"] for row in rows})
    shots = []
    for row in rows:
        if row["shot"] not in shots:
            shots.append(row["shot"])

    lookup = {
        (row["task"], row["shot"], row["metric"]): row["score"]
        for row in rows
    }
    metric_titles = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1": "F1",
        "roc_auc": "ROC AUC",
    }

    fig_width = max(8.0, 2.2 * len(tasks))
    fig_height = 3.0 * len(metrics)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_width, fig_height), squeeze=False)

    x_positions = np.arange(len(tasks))
    bar_width = min(0.8 / max(len(shots), 1), 0.32)
    offsets = (np.arange(len(shots)) - (len(shots) - 1) / 2) * bar_width
    colors = ["#36688d", "#c06c2d", "#5a8f64"]

    for metric_index, metric_name in enumerate(metrics):
        ax = axes[metric_index][0]
        for shot_index, shot_name in enumerate(shots):
            scores = [
                lookup.get((task_name, shot_name, metric_name), np.nan)
                for task_name in tasks
            ]
            bars = ax.bar(
                x_positions + offsets[shot_index],
                scores,
                width=bar_width,
                label=shot_name,
                color=colors[shot_index % len(colors)],
            )
            for bar, score in zip(bars, scores):
                if np.isnan(score):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(score + 0.015, 1.03),
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(metric_titles.get(metric_name, metric_name), fontsize=12, pad=10)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([task.replace("_", " ").title() for task in tasks])
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0][0].legend(loc="lower right")
    fig.suptitle("Small LLM Clinical Classification Scores", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    plot_path = plot_dir / "scores.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Small LLM plot written to {plot_path}")
    print(f"Small LLM plot summary written to {summary_path}")
    return plot_path


def evaluate_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = LlmConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)
    task_dirs = discover_task_dirs(prepared_dir, args.tasks)
    shots = shot_choices(args.shots)
    limit = args.limit if args.limit and args.limit > 0 else None

    aggregate: dict[str, Any] = {
        "created_at": utc_now(),
        "model_name": config.model,
        "base_url": config.base_url,
        "split": args.split,
        "shots": [f"{shot}_shot" for shot in shots],
        "tasks": {},
    }

    for task_dir in task_dirs:
        for shot in shots:
            summary = evaluate_task_shot(
                client=client,
                config=config,
                task_dir=task_dir,
                split=args.split,
                shot=shot,
                output_dir=output_dir,
                limit=limit,
                max_retries=args.max_retries,
                progress_every=args.progress_every,
                max_chars=args.max_chars,
            )
            aggregate["tasks"][aggregate_key(task_dir.name, shot)] = summary

    aggregate_path = output_dir / "metrics.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(f"Aggregate metrics written to {aggregate_path}")
    if not args.no_plot:
        save_score_plot(
            aggregate=aggregate,
            output_dir=output_dir,
            metrics=args.plot_metrics,
            dpi=args.dpi,
        )
    return aggregate


def classify_single(args: argparse.Namespace) -> dict[str, Any]:
    config = LlmConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)
    prompt = build_zero_shot_prompt(args.task, args.single_text, args.max_chars)
    prediction, raw_output = classify_patient_with_retry(
        client=client,
        config=config,
        prompt=prompt,
        max_retries=args.max_retries,
    )
    result = {"prediction": prediction, "raw_output": raw_output}
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a small LLM on prepared clinical classification tasks."
    )
    parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=["diabetes", "hypertension"])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--shots", nargs="*", default=["zero", "one"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-metrics", nargs="*", default=["accuracy", "f1", "roc_auc"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--single-text", default=None)
    parser.add_argument("--task", default="diabetes")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.single_text:
            classify_single(args)
        else:
            evaluate_benchmark(args)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
