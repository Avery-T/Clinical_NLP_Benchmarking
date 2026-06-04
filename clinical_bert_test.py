#!/usr/bin/env python3
"""Build and run ClinicalBERT classification benchmarks on Synthea CSV data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_OUTPUT_DIR = "benchmark_runs"
REWARD_RUN_NAME = "clinicalbert_reward"
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    positive_regex: str
    description: str = ""
    label_source: str = "conditions"


DEFAULT_TASKS: dict[str, TaskSpec] = {
    "diabetes": TaskSpec(
        name="diabetes",
        positive_regex=r"\bdiabetes\b",
        description="Patient has a diabetes condition in conditions.csv.",
    ),
    "hypertension": TaskSpec(
        name="hypertension",
        positive_regex=r"\bhypertension\b",
        description="Patient has a hypertension condition in conditions.csv.",
    ),
    "coronary_heart_disease": TaskSpec(
        name="coronary_heart_disease",
        positive_regex=r"\bcoronary heart disease\b|\bmyocardial infarction\b",
        description="Patient has coronary heart disease or myocardial infarction.",
    ),
    "stroke": TaskSpec(
        name="stroke",
        positive_regex=r"\bstroke\b",
        description="Patient has a stroke condition in conditions.csv.",
    ),
    "asthma": TaskSpec(
        name="asthma",
        positive_regex=r"\basthma\b",
        description="Patient has asthma in conditions.csv.",
    ),
    "copd": TaskSpec(
        name="copd",
        positive_regex=r"\bchronic obstructive\b|\bpulmonary emphysema\b|\bcopd\b",
        description="Patient has COPD-like disease in conditions.csv.",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def info(message: str) -> None:
    print(message, flush=True)


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, index_col=False).columns)


def read_csv_chunks(
    path: Path,
    usecols: Iterable[str] | None = None,
    chunksize: int = 100_000,
) -> Iterable[pd.DataFrame]:
    kwargs: dict[str, Any] = {
        "chunksize": chunksize,
        "dtype": str,
        "index_col": False,
        "on_bad_lines": "skip",
        "low_memory": False,
    }
    if usecols is not None:
        header = set(read_header(path))
        existing = [column for column in usecols if column in header]
        if not existing:
            return
        kwargs["usecols"] = existing
    yield from pd.read_csv(path, **kwargs)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def clean_list(values: Iterable[Any], limit: int = 60) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_text(value)
        key = item.lower()
        if item and key not in seen:
            items.append(item)
            seen.add(key)
        if len(items) >= limit:
            break
    return "; ".join(items)


def safe_task_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("_") or "task"


def load_task_specs(config_path: Path | None = None) -> dict[str, TaskSpec]:
    tasks = dict(DEFAULT_TASKS)
    if config_path is None:
        return tasks

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    entries = raw.get("tasks", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("Task config must be a list or an object with a 'tasks' list.")

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every task config entry must be an object.")
        spec = TaskSpec(
            name=entry["name"],
            positive_regex=entry["positive_regex"],
            description=entry.get("description", ""),
            label_source=entry.get("label_source", "conditions"),
        )
        if spec.label_source != "conditions":
            raise ValueError(
                f"Task {spec.name!r} uses label_source={spec.label_source!r}; "
                "only 'conditions' is supported right now."
            )
        tasks[spec.name] = spec
    return tasks


def choose_tasks(all_tasks: dict[str, TaskSpec], names: list[str] | None) -> list[TaskSpec]:
    if not names:
        return list(all_tasks.values())

    missing = [name for name in names if name not in all_tasks]
    if missing:
        available = ", ".join(sorted(all_tasks))
        raise ValueError(f"Unknown task(s): {', '.join(missing)}. Available tasks: {available}")
    return [all_tasks[name] for name in names]


def discover_csv_dirs(data_roots: list[Path]) -> list[Path]:
    csv_dirs: set[Path] = set()
    for root in data_roots:
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        if root.is_dir() and (root / "conditions.csv").exists():
            csv_dirs.add(root)
        for child in root.rglob("csv"):
            if child.is_dir() and any(child.glob("*.csv")):
                csv_dirs.add(child)
    return sorted(csv_dirs)


def valid_patient_ids(values: pd.Series) -> list[str]:
    text = values.dropna().astype(str).str.strip()
    return [value for value in text if UUID_RE.match(value)]


def collect_patient_ids(csv_dirs: list[Path], chunksize: int) -> set[str]:
    patient_ids: set[str] = set()
    patient_files = ["patients.csv", "conditions.csv", "encounters.csv", "medications.csv"]

    for csv_dir in csv_dirs:
        for file_name in patient_files:
            path = csv_dir / file_name
            if not path.exists():
                continue
            column = "ID" if file_name == "patients.csv" else "PATIENT"
            for chunk in read_csv_chunks(path, usecols=[column], chunksize=chunksize):
                if column in chunk:
                    patient_ids.update(valid_patient_ids(chunk[column]))

    return patient_ids


def collect_condition_labels(
    csv_dirs: list[Path],
    tasks: list[TaskSpec],
    chunksize: int,
) -> tuple[dict[str, set[str]], set[str]]:
    compiled = {
        task.name: re.compile(task.positive_regex, flags=re.IGNORECASE)
        for task in tasks
    }
    positive_patients = {task.name: set() for task in tasks}
    condition_patients: set[str] = set()

    for csv_dir in csv_dirs:
        path = csv_dir / "conditions.csv"
        if not path.exists():
            continue
        info(f"Reading labels from {path}")
        for chunk in read_csv_chunks(
            path,
            usecols=["PATIENT", "DESCRIPTION"],
            chunksize=chunksize,
        ):
            if "PATIENT" not in chunk or "DESCRIPTION" not in chunk:
                continue
            patient_series = chunk["PATIENT"].dropna().astype(str).str.strip()
            condition_patients.update(value for value in patient_series if UUID_RE.match(value))
            descriptions = chunk["DESCRIPTION"].fillna("").astype(str)
            patients = chunk["PATIENT"].fillna("").astype(str).str.strip()

            for task_name, pattern in compiled.items():
                matches = descriptions.str.contains(pattern, na=False, regex=True)
                positive_patients[task_name].update(
                    patient
                    for patient in patients[matches]
                    if UUID_RE.match(patient)
                )

    return positive_patients, condition_patients


def pick_patient_sample(
    all_patients: set[str],
    positives: set[str],
    max_patients: int,
    balance: bool,
    rng: random.Random,
) -> dict[str, int]:
    positives = positives & all_patients
    negatives = all_patients - positives

    if not positives:
        raise ValueError("No positive patients found for this task.")
    if not negatives:
        raise ValueError("No negative patients found for this task.")

    positive_list = sorted(positives)
    negative_list = sorted(negatives)
    rng.shuffle(positive_list)
    rng.shuffle(negative_list)

    if balance:
        n_pos = min(len(positive_list), max_patients // 2)
        n_neg = min(len(negative_list), max_patients - n_pos)
        if n_neg > n_pos:
            n_neg = min(n_neg, n_pos)
        if n_pos < max_patients // 2:
            n_neg = min(len(negative_list), max_patients - n_pos)
    else:
        population = [(patient_id, 1) for patient_id in positive_list]
        population.extend((patient_id, 0) for patient_id in negative_list)
        rng.shuffle(population)
        return dict(population[:max_patients])

    selected: dict[str, int] = {patient_id: 1 for patient_id in positive_list[:n_pos]}
    selected.update({patient_id: 0 for patient_id in negative_list[:n_neg]})
    return selected


def add_phrase(
    note_parts: dict[str, dict[str, list[str]]],
    seen: dict[str, dict[str, set[str]]],
    patient_id: str,
    section: str,
    phrase: str,
    max_events: int,
) -> None:
    phrase = clean_text(phrase)
    if not phrase:
        return
    key = phrase.lower()
    patient_seen = seen[patient_id][section]
    patient_parts = note_parts[patient_id][section]
    if key in patient_seen or len(patient_parts) >= max_events:
        return
    patient_seen.add(key)
    patient_parts.append(phrase)


def phrase_description(row: pd.Series) -> str:
    return clean_text(row.get("DESCRIPTION", ""))


def phrase_observation(row: pd.Series) -> str:
    description = clean_text(row.get("DESCRIPTION", ""))
    value = clean_text(row.get("VALUE", ""))
    units = clean_text(row.get("UNITS", ""))
    if not description:
        return ""
    if value and units:
        return f"{description}: {value} {units}"
    if value:
        return f"{description}: {value}"
    return description


def phrase_demographics(row: pd.Series) -> str:
    parts: list[str] = []
    birthdate = clean_text(row.get("BIRTHDATE", ""))
    gender = clean_text(row.get("GENDER", ""))
    race = clean_text(row.get("RACE", ""))
    ethnicity = clean_text(row.get("ETHNICITY", ""))
    if birthdate:
        parts.append(f"born {birthdate}")
    if gender:
        parts.append(f"gender {gender}")
    if race:
        parts.append(f"race {race}")
    if ethnicity:
        parts.append(f"ethnicity {ethnicity}")
    return ", ".join(parts)


def add_table_to_notes(
    csv_dirs: list[Path],
    selected_patients: set[str],
    note_parts: dict[str, dict[str, list[str]]],
    seen: dict[str, dict[str, set[str]]],
    file_name: str,
    section: str,
    usecols: list[str],
    phrase_fn: Callable[[pd.Series], str],
    chunksize: int,
    max_events: int,
    patient_column: str = "PATIENT",
) -> None:
    for csv_dir in csv_dirs:
        path = csv_dir / file_name
        if not path.exists():
            continue
        for chunk in read_csv_chunks(path, usecols=usecols, chunksize=chunksize):
            if patient_column not in chunk:
                continue
            chunk[patient_column] = chunk[patient_column].fillna("").astype(str).str.strip()
            filtered = chunk[chunk[patient_column].isin(selected_patients)]
            for _, row in filtered.iterrows():
                add_phrase(
                    note_parts,
                    seen,
                    row[patient_column],
                    section,
                    phrase_fn(row),
                    max_events,
                )


def build_patient_notes(
    csv_dirs: list[Path],
    selected_patients: set[str],
    chunksize: int,
    max_events_per_section: int,
    include_conditions: bool,
    include_careplans: bool,
) -> dict[str, str]:
    note_parts: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    info(f"Building patient text for {len(selected_patients):,} selected patients")
    add_table_to_notes(
        csv_dirs,
        selected_patients,
        note_parts,
        seen,
        file_name="patients.csv",
        section="demographics",
        usecols=["ID", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"],
        phrase_fn=phrase_demographics,
        chunksize=chunksize,
        max_events=1,
        patient_column="ID",
    )

    table_specs = [
        ("encounters.csv", "encounters", ["PATIENT", "DESCRIPTION"], phrase_description),
        ("medications.csv", "medications", ["PATIENT", "DESCRIPTION"], phrase_description),
        ("procedures.csv", "procedures", ["PATIENT", "DESCRIPTION"], phrase_description),
        ("observations.csv", "observations", ["PATIENT", "DESCRIPTION", "VALUE", "UNITS"], phrase_observation),
        ("allergies.csv", "allergies", ["PATIENT", "DESCRIPTION"], phrase_description),
        ("immunizations.csv", "immunizations", ["PATIENT", "DESCRIPTION"], phrase_description),
    ]
    for file_name, section, usecols, phrase_fn in table_specs:
        info(f"Adding {section} text")
        add_table_to_notes(
            csv_dirs,
            selected_patients,
            note_parts,
            seen,
            file_name=file_name,
            section=section,
            usecols=usecols,
            phrase_fn=phrase_fn,
            chunksize=chunksize,
            max_events=max_events_per_section,
        )

    if include_careplans:
        info("Adding careplan text")
        add_table_to_notes(
            csv_dirs,
            selected_patients,
            note_parts,
            seen,
            file_name="careplans.csv",
            section="careplans",
            usecols=["PATIENT", "DESCRIPTION"],
            phrase_fn=phrase_description,
            chunksize=chunksize,
            max_events=max_events_per_section,
        )

    if include_conditions:
        info("Adding condition text")
        add_table_to_notes(
            csv_dirs,
            selected_patients,
            note_parts,
            seen,
            file_name="conditions.csv",
            section="conditions",
            usecols=["PATIENT", "DESCRIPTION"],
            phrase_fn=phrase_description,
            chunksize=chunksize,
            max_events=max_events_per_section,
        )

    section_order = [
        "demographics",
        "encounters",
        "medications",
        "procedures",
        "observations",
        "allergies",
        "immunizations",
        "careplans",
        "conditions",
    ]
    notes: dict[str, str] = {}
    for patient_id in selected_patients:
        sections: list[str] = []
        for section in section_order:
            values = note_parts[patient_id].get(section, [])
            if values:
                title = section.replace("_", " ").title()
                sections.append(f"{title}: {clean_list(values, max_events_per_section)}.")
        notes[patient_id] = " ".join(sections) or (
            "Synthetic patient record with no selected non-diagnostic events available."
        )
    return notes


def stratify_or_none(rows: list[dict[str, Any]]) -> list[int] | None:
    labels = [int(row["label"]) for row in rows]
    counts = Counter(labels)
    if len(counts) != 2 or min(counts.values()) < 2:
        return None
    return labels


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = train_size + val_size + test_size
    train_size = train_size / total
    val_size = val_size / total
    test_size = test_size / total

    train_rows, temp_rows = train_test_split(
        rows,
        train_size=train_size,
        random_state=seed,
        stratify=stratify_or_none(rows),
    )
    relative_test = test_size / (val_size + test_size)
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=relative_test,
        random_state=seed,
        stratify=stratify_or_none(temp_rows),
    )
    return list(train_rows), list(val_rows), list(test_rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def prepare_benchmark(args: argparse.Namespace) -> Path:
    data_roots = [Path(root) for root in args.data_root]
    output_dir = Path(args.output_dir)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    all_task_specs = load_task_specs(Path(args.task_config) if args.task_config else None)
    tasks = choose_tasks(all_task_specs, args.tasks)
    csv_dirs = discover_csv_dirs(data_roots)
    if not csv_dirs:
        raise FileNotFoundError(f"No extracted Synthea csv directories found under {data_roots}")

    info(f"Found {len(csv_dirs)} extracted csv folder(s)")
    for csv_dir in csv_dirs:
        info(f"  - {csv_dir}")

    positives_by_task, condition_patients = collect_condition_labels(
        csv_dirs,
        tasks,
        chunksize=args.chunksize,
    )
    all_patients = collect_patient_ids(csv_dirs, chunksize=args.chunksize)
    all_patients.update(condition_patients)
    if not all_patients:
        raise ValueError("No patient IDs found in extracted CSV data.")
    info(f"Found {len(all_patients):,} unique patient IDs")

    rng = random.Random(args.seed)
    selected_by_task: dict[str, dict[str, int]] = {}
    for task in tasks:
        selected = pick_patient_sample(
            all_patients=all_patients,
            positives=positives_by_task[task.name],
            max_patients=args.max_patients,
            balance=args.balance,
            rng=rng,
        )
        selected_by_task[task.name] = selected
        counts = Counter(selected.values())
        info(
            f"Task {task.name}: selected {len(selected):,} patients "
            f"({counts.get(1, 0):,} positive, {counts.get(0, 0):,} negative)"
        )

    selected_patient_ids = {
        patient_id
        for labels in selected_by_task.values()
        for patient_id in labels
    }
    notes = build_patient_notes(
        csv_dirs=csv_dirs,
        selected_patients=selected_patient_ids,
        chunksize=args.chunksize,
        max_events_per_section=args.max_events_per_section,
        include_conditions=args.include_conditions_in_text,
        include_careplans=args.include_careplans_in_text,
    )

    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "data_roots": [str(path) for path in data_roots],
        "csv_dirs": [str(path) for path in csv_dirs],
        "max_patients": args.max_patients,
        "balance": args.balance,
        "include_conditions_in_text": args.include_conditions_in_text,
        "include_careplans_in_text": args.include_careplans_in_text,
        "max_events_per_section": args.max_events_per_section,
        "seed": args.seed,
        "tasks": {},
    }

    for task in tasks:
        task_dir = prepared_dir / safe_task_name(task.name)
        rows = [
            {
                "patient_id": patient_id,
                "text": notes[patient_id],
                "label": label,
            }
            for patient_id, label in selected_by_task[task.name].items()
        ]
        rng.shuffle(rows)
        train_rows, val_rows, test_rows = split_rows(
            rows,
            seed=args.seed,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
        )
        write_rows(task_dir / "train.csv", train_rows)
        write_rows(task_dir / "val.csv", val_rows)
        write_rows(task_dir / "test.csv", test_rows)

        split_counts = {
            "train": dict(Counter(row["label"] for row in train_rows)),
            "val": dict(Counter(row["label"] for row in val_rows)),
            "test": dict(Counter(row["label"] for row in test_rows)),
        }
        task_manifest = {
            **asdict(task),
            "selected_patients": len(rows),
            "positive_patients_available": len(positives_by_task[task.name]),
            "split_counts": split_counts,
            "files": {
                "train": str(task_dir / "train.csv"),
                "val": str(task_dir / "val.csv"),
                "test": str(task_dir / "test.csv"),
            },
        }
        with (task_dir / "task_info.json").open("w", encoding="utf-8") as handle:
            json.dump(task_manifest, handle, indent=2)
        manifest["tasks"][task.name] = task_manifest

    with (prepared_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    info(f"Prepared benchmark data in {prepared_dir}")
    return prepared_dir


def discover_prepared_tasks(prepared_dir: Path, requested: list[str] | None = None) -> list[Path]:
    if (prepared_dir / "train.csv").exists() and (prepared_dir / "test.csv").exists():
        return [prepared_dir]

    task_dirs = [
        path for path in sorted(prepared_dir.iterdir())
        if path.is_dir() and (path / "train.csv").exists() and (path / "test.csv").exists()
    ]
    if requested:
        requested_safe = {safe_task_name(name) for name in requested}
        task_dirs = [path for path in task_dirs if path.name in requested_safe]
    if not task_dirs:
        raise FileNotFoundError(f"No prepared task directories found in {prepared_dir}")
    return task_dirs


def choose_column(columns: Iterable[str], preferred: list[str], kind: str, path: Path) -> str:
    """Find a text or label column even if a teammate used a slightly different name."""
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in preferred:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(
        f"{path} is missing a {kind} column. Tried: {', '.join(preferred)}. "
        f"Available columns: {', '.join(columns)}"
    )


def load_split(task_dir: Path, split: str) -> pd.DataFrame:
    path = task_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)

    text_col = choose_column(
        df.columns,
        ["text", "patient_text", "note", "notes", "clinical_text", "input_text"],
        "text",
        path,
    )
    label_col = choose_column(
        df.columns,
        ["label", "labels", "target", "class", "class_id", "diagnosis_label"],
        "label",
        path,
    )
    patient_col = None
    try:
        patient_col = choose_column(
            df.columns,
            ["patient_id", "patient", "id", "subject_id"],
            "patient id",
            path,
        )
    except ValueError:
        patient_col = None

    normalized = pd.DataFrame()
    normalized["patient_id"] = (
        df[patient_col].astype(str)
        if patient_col is not None
        else [f"{task_dir.name}_{split}_{index}" for index in range(len(df))]
    )
    normalized["text"] = df[text_col].fillna("").astype(str)
    normalized["label"] = pd.to_numeric(df[label_col], errors="raise").astype(int)
    return normalized


def load_split_or_all(task_dir: Path, split: str) -> pd.DataFrame:
    if split == "all":
        split_frames = []
        for split_name in ["train", "val", "test"]:
            split_df = load_split(task_dir, split_name)
            split_df["split"] = split_name
            split_frames.append(split_df)
        return pd.concat(split_frames, ignore_index=True)

    df = load_split(task_dir, split)
    df["split"] = split
    return df


def infer_num_labels(*frames: pd.DataFrame) -> int:
    labels = sorted(
        {
            int(label)
            for frame in frames
            for label in frame["label"].dropna().astype(int).tolist()
        }
    )
    if not labels:
        raise ValueError("Could not infer labels because no labels were found.")
    expected = list(range(max(labels) + 1))
    if labels != expected:
        raise ValueError(
            f"Labels must be zero-based contiguous class ids for cross entropy. "
            f"Found labels {labels}, expected {expected}."
        )
    return max(labels) + 1


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    label_ids = sorted({int(value) for value in np.concatenate([labels, predictions])})
    average = "binary" if label_ids == [0, 1] else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_ids).tolist(),
        "label_ids": label_ids,
        "num_labels": len(label_ids),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "support": int(len(labels)),
    }
    for label_id in label_ids:
        metrics[f"class_{label_id}_support"] = int(np.sum(labels == label_id))
    if label_ids == [0, 1]:
        metrics["positive_support"] = int(np.sum(labels == 1))
        metrics["negative_support"] = int(np.sum(labels == 0))
        if probabilities is not None:
            positive_scores = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
            metrics["roc_auc"] = float(roc_auc_score(labels, positive_scores))
    return metrics


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    return classification_metrics(labels, predictions, probabilities)


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def add_probability_columns(prediction_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    if probabilities.ndim == 1:
        prediction_df["probability"] = probabilities
        return prediction_df
    if probabilities.shape[1] == 2:
        prediction_df["probability"] = probabilities[:, 1]
    for label_id in range(probabilities.shape[1]):
        prediction_df[f"probability_class_{label_id}"] = probabilities[:, label_id]
    prediction_df["predicted_probability"] = probabilities.max(axis=1)
    return prediction_df


def reward_output_dir(output_root: Path) -> Path:
    if output_root.name == REWARD_RUN_NAME:
        return output_root
    return output_root / REWARD_RUN_NAME


def save_confusion_matrix_image(
    matrix: list[list[int]],
    labels: list[int],
    output_path: Path,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir) / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"created_at": utc_now(), "tasks": {}}

    for task_dir in discover_prepared_tasks(prepared_dir, args.tasks):
        info(f"Running TF-IDF baseline for {task_dir.name}")
        train_df = load_split(task_dir, "train")
        val_df = load_split(task_dir, "val")
        test_df = load_split(task_dir, "test")
        train_all = pd.concat([train_df, val_df], ignore_index=True)

        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        max_features=args.tfidf_max_features,
                        min_df=args.tfidf_min_df,
                        ngram_range=(1, args.tfidf_max_ngram),
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=args.max_iter,
                        class_weight="balanced",
                        n_jobs=args.n_jobs,
                    ),
                ),
            ]
        )
        model.fit(train_all["text"], train_all["label"])
        probabilities = model.predict_proba(test_df["text"])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        labels = test_df["label"].to_numpy()
        metrics = binary_metrics(labels, predictions, probabilities)
        metrics["classification_report"] = classification_report(
            labels,
            predictions,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        results["tasks"][task_dir.name] = metrics

        task_output = output_dir / task_dir.name
        task_output.mkdir(parents=True, exist_ok=True)
        prediction_df = test_df[["patient_id", "label"]].copy()
        prediction_df["probability"] = probabilities
        prediction_df["prediction"] = predictions
        prediction_df.to_csv(task_output / "predictions.csv", index=False)
        with (task_output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        info(
            f"Baseline {task_dir.name}: "
            f"accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}, "
            f"roc_auc={metrics.get('roc_auc', float('nan')):.3f}"
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    info(f"Baseline results written to {output_dir}")
    return results


def require_clinicalbert_dependencies() -> tuple[Any, Any, Any, Any]:
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ClinicalBERT benchmarking needs torch and transformers. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return torch, DataLoader, Dataset, (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )


def require_encoder_dependencies() -> tuple[Any, Any, Any]:
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ClinicalBERT encoding needs torch and transformers. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def make_torch_dataset_class(torch: Any, Dataset: Any) -> type:
    class TextDataset(Dataset):  # type: ignore[misc, valid-type]
        def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {
                key: torch.tensor(value[idx])
                for key, value in self.encodings.items()
            }
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    return TextDataset


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    probabilities = softmax_probabilities(logits)
    if probabilities.shape[1] < 2:
        raise ValueError("Positive-class probability requires at least two labels.")
    return probabilities[:, 1]


def evaluate_torch_model(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = int(batch["labels"].size(0))
            total_loss += float(outputs.loss.item()) * batch_size
            total_count += batch_size
            all_logits.append(outputs.logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probabilities = softmax_probabilities(logits)
    predictions = np.argmax(logits, axis=1)
    metrics = classification_metrics(labels, predictions, probabilities)
    return total_loss / max(total_count, 1), metrics, labels, predictions, probabilities


def reward_weighted_classification_loss(
    torch: Any,
    logits: Any,
    labels: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    """Return cross-entropy loss scaled by a simple correctness/confidence reward.

    This is reinforcement-learning-inspired, not full RL. The model still learns
    from supervised labels, but each sample's loss is weighted by a reward signal
    derived from the current prediction and confidence.
    """
    per_sample_loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="none",
    )

    # Rewards are computed from detached probabilities so the reward calculation
    # does not create a second gradient path through the model's own prediction.
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = torch.max(probabilities, dim=1)
        correct = predictions.eq(labels)

        correct_reward = args.correct_reward + args.confidence_bonus * confidence
        wrong_reward = args.wrong_reward - args.confidence_penalty * confidence
        rewards = torch.where(correct, correct_reward, wrong_reward)

        # Loss weights must be non-negative. Positive rewards reinforce correct
        # examples, while negative rewards are treated as penalty magnitudes that
        # make wrong predictions, especially confident ones, count more.
        weight_signal = torch.where(rewards >= 0, rewards, torch.abs(rewards))
        weights = 1.0 + args.reward_scale * weight_signal
        weights = torch.clamp(
            weights,
            min=args.min_reward_weight,
            max=args.max_reward_weight,
        )

    weighted_loss = (per_sample_loss * weights).mean()
    stats = {
        "mean_reward": float(rewards.mean().detach().cpu().item()),
        "mean_reward_weight": float(weights.mean().detach().cpu().item()),
        "mean_confidence": float(confidence.mean().detach().cpu().item()),
        "batch_accuracy": float(correct.float().mean().detach().cpu().item()),
    }
    return weighted_loss, stats


def iter_batch_indices(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def embedding_slice(values: np.ndarray, dims: int) -> list[float]:
    if dims <= 0:
        dims = values.shape[0]
    dims = min(dims, values.shape[0])
    return [float(value) for value in values[:dims]]


def run_encode(args: argparse.Namespace) -> dict[str, Any]:
    torch, AutoModel, AutoTokenizer = require_encoder_dependencies()
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir) / "clinicalbert_raw_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    info(f"Using device: {device}")
    info("Encoding inputs only; no training, optimizer, epochs, or classifier head.")

    results: dict[str, Any] = {
        "created_at": utc_now(),
        "model_name": args.model_name,
        "device": str(device),
        "split": args.split,
        "max_length": args.max_length,
        "embedding_dims_written": args.embedding_dims,
        "tasks": {},
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()

    for task_dir in discover_prepared_tasks(prepared_dir, args.tasks):
        df = load_split_or_all(task_dir, args.split)
        if args.limit > 0:
            df = df.head(args.limit).copy()
        df = df.reset_index(drop=True)
        task_output = output_dir / task_dir.name
        task_output.mkdir(parents=True, exist_ok=True)
        output_path = task_output / f"{args.split}_outputs.jsonl"

        info(f"Encoding {len(df):,} {args.split} rows for {task_dir.name}")
        row_count = 0
        embedding_width = None
        with output_path.open("w", encoding="utf-8") as handle:
            for start, end in iter_batch_indices(len(df), args.batch_size):
                batch_df = df.iloc[start:end]
                encoded = tokenizer(
                    batch_df["text"].tolist(),
                    truncation=True,
                    padding=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                with torch.no_grad():
                    outputs = model(**encoded)
                    last_hidden = outputs.last_hidden_state
                    attention = encoded["attention_mask"].unsqueeze(-1)
                    cls_embeddings = last_hidden[:, 0, :]
                    mean_embeddings = (
                        (last_hidden * attention).sum(dim=1)
                        / attention.sum(dim=1).clamp(min=1)
                    )

                cls_np = cls_embeddings.detach().cpu().numpy()
                mean_np = mean_embeddings.detach().cpu().numpy()
                token_counts = encoded["attention_mask"].sum(dim=1).detach().cpu().numpy()
                embedding_width = int(cls_np.shape[1])

                pooler = getattr(outputs, "pooler_output", None)
                pooler_np = pooler.detach().cpu().numpy() if pooler is not None else None

                for row_offset, (_, row) in enumerate(batch_df.iterrows()):
                    record: dict[str, Any] = {
                        "patient_id": row["patient_id"],
                        "split": row["split"],
                        "label": int(row["label"]),
                        "token_count": int(token_counts[row_offset]),
                        "cls_norm": float(np.linalg.norm(cls_np[row_offset])),
                        "mean_norm": float(np.linalg.norm(mean_np[row_offset])),
                        "cls_embedding": embedding_slice(
                            cls_np[row_offset],
                            args.embedding_dims,
                        ),
                        "mean_embedding": embedding_slice(
                            mean_np[row_offset],
                            args.embedding_dims,
                        ),
                    }
                    if pooler_np is not None:
                        record["pooler_norm"] = float(np.linalg.norm(pooler_np[row_offset]))
                        record["pooler_embedding"] = embedding_slice(
                            pooler_np[row_offset],
                            args.embedding_dims,
                        )
                    if args.include_text:
                        record["text"] = row["text"]
                    handle.write(json.dumps(record) + "\n")
                    row_count += 1

        task_summary = {
            "rows_encoded": row_count,
            "output_file": str(output_path),
            "embedding_width": embedding_width,
        }
        results["tasks"][task_dir.name] = task_summary
        info(f"Raw outputs for {task_dir.name} written to {output_path}")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def missing_classification_head_keys(loading_info: dict[str, Any]) -> list[str]:
    head_markers = ("classifier", "pre_classifier", "score")
    missing_keys = loading_info.get("missing_keys", []) or []
    return [
        key for key in missing_keys
        if any(marker in key for marker in head_markers)
    ]


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    torch, DataLoader, Dataset, transformers_objects = require_clinicalbert_dependencies()
    AutoModelForSequenceClassification, AutoTokenizer, _ = transformers_objects
    TextDataset = make_torch_dataset_class(torch, Dataset)

    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir) / "clinicalbert_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    info(f"Using device: {device}")
    info("Running inference only; no training, optimizer, or epochs.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model_result = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        use_safetensors=False,
        output_loading_info=True,
    )
    if isinstance(model_result, tuple):
        model, loading_info = model_result
    else:
        model = model_result
        loading_info = {}

    missing_head_keys = missing_classification_head_keys(loading_info)
    if missing_head_keys and not args.allow_random_head:
        missing_preview = ", ".join(missing_head_keys[:6])
        raise RuntimeError(
            "This checkpoint does not include a trained sequence-classification head "
            f"({missing_preview}). Base ClinicalBERT cannot classify your labels by itself. "
            "Use --model-name with a checkpoint already fine-tuned for this exact task, "
            "or pass --allow-random-head only for debugging random outputs."
        )

    model.to(device)
    model.eval()

    results: dict[str, Any] = {
        "created_at": utc_now(),
        "model_name": args.model_name,
        "device": str(device),
        "split": args.split,
        "max_length": args.max_length,
        "tasks": {},
    }

    for task_dir in discover_prepared_tasks(prepared_dir, args.tasks):
        df = load_split_or_all(task_dir, args.split)
        if args.limit > 0:
            df = df.head(args.limit).copy()
        df = df.reset_index(drop=True)

        info(f"Classifying {len(df):,} {args.split} rows for {task_dir.name}")
        dataset = TextDataset(
            df["text"].tolist(),
            df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        loss, metrics, labels, predictions, probabilities = evaluate_torch_model(
            torch,
            model,
            loader,
            device,
        )
        metrics["loss"] = loss
        results["tasks"][task_dir.name] = metrics

        task_output = output_dir / task_dir.name
        task_output.mkdir(parents=True, exist_ok=True)
        prediction_df = df[["patient_id", "split", "label"]].copy()
        prediction_df["prediction"] = predictions
        prediction_df = add_probability_columns(prediction_df, probabilities)
        prediction_df.to_csv(task_output / f"{args.split}_predictions.csv", index=False)
        with (task_output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        info(
            f"Inference {task_dir.name}: "
            f"accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}, "
            f"roc_auc={metrics.get('roc_auc', float('nan')):.3f}"
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    info(f"Inference-only results written to {output_dir}")
    return results


def display_run_name(name: str) -> str:
    names = {
        "baseline": "TF-IDF baseline",
        "clinicalbert": "ClinicalBERT fine-tuned",
        "clinicalbert_reward": "ClinicalBERT reward-weighted",
        "clinicalbert_reward_weighted": "ClinicalBERT reward-weighted",
        "clinicalbert_inference": "ClinicalBERT inference",
        "small_llm": "Small LLM",
    }
    return names.get(name, name.replace("_", " ").title())


def normalize_plot_record(run_name: str, task_name: str) -> tuple[str, str]:
    if run_name == "small_llm":
        for suffix, label in [
            ("_zero_shot", "Small LLM zero-shot"),
            ("_one_shot", "Small LLM one-shot"),
        ]:
            if task_name.endswith(suffix):
                return task_name[: -len(suffix)], label
    return task_name, display_run_name(run_name)


def load_score_records(
    output_dir: Path,
    run_names: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_name in run_names:
        metrics_path = output_dir / run_name / "metrics.json"
        if not metrics_path.exists():
            info(f"Skipping missing metrics file: {metrics_path}")
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for task_name, task_metrics in data.get("tasks", {}).items():
            plot_task_name, model_name = normalize_plot_record(run_name, task_name)
            for metric_name in metrics:
                value = task_metrics.get(metric_name)
                if value is None:
                    continue
                records.append(
                    {
                        "run": run_name,
                        "model": model_name,
                        "task": plot_task_name,
                        "metric": metric_name,
                        "score": float(value),
                    }
                )
    if not records:
        raise ValueError("No metric scores found to plot.")
    return records


def write_score_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = sorted(records, key=lambda row: (row["task"], row["metric"], row["run"]))
    pd.DataFrame(rows).to_csv(path, index=False)


def run_plot(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    run_names = args.runs or ["baseline", "clinicalbert", "clinicalbert_inference"]
    plot_metrics = [metric for metric in args.metrics if metric != "weighted_f1"]
    if len(plot_metrics) != len(args.metrics):
        info("Skipping weighted_f1 in plots; it remains available in metrics JSON.")
    if not plot_metrics:
        raise ValueError("No plottable metrics requested.")

    records = load_score_records(output_dir, run_names, plot_metrics)
    summary_csv = plot_dir / "score_summary.csv"
    write_score_csv(summary_csv, records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({row["task"] for row in records})
    models = []
    for row in records:
        if row["model"] not in models:
            models.append(row["model"])

    score_lookup = {
        (row["task"], row["model"], row["metric"]): row["score"]
        for row in records
    }
    metric_titles = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1": "F1",
        "macro_f1": "Macro F1",
        "roc_auc": "ROC AUC",
    }
    colors = ["#2f6f8f", "#c76f3a", "#5b8f5a", "#7b5ea7", "#b84a62"]

    fig_width = max(9.0, 2.2 * len(tasks))
    fig_height = 3.2 * len(plot_metrics)
    fig, axes = plt.subplots(
        len(plot_metrics),
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    x_positions = np.arange(len(tasks))
    bar_width = min(0.8 / max(len(models), 1), 0.28)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_width

    for metric_index, metric_name in enumerate(plot_metrics):
        ax = axes[metric_index][0]
        for model_index, model_name in enumerate(models):
            scores = [
                score_lookup.get((task, model_name, metric_name), np.nan)
                for task in tasks
            ]
            bars = ax.bar(
                x_positions + offsets[model_index],
                scores,
                width=bar_width,
                label=model_name,
                color=colors[model_index % len(colors)],
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
    fig.suptitle("Clinical NLP Classification Scores", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path = Path(args.output_file) if args.output_file else plot_dir / "scores.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    info(f"Saved score plot to {output_path}")
    info(f"Saved score summary to {summary_csv}")
    for row in sorted(records, key=lambda item: (item["task"], item["model"], item["metric"])):
        info(f"{row['model']} {row['task']} {row['metric']}: {row['score']:.4f}")
    return output_path


def run_clinicalbert(args: argparse.Namespace) -> dict[str, Any]:
    torch, DataLoader, Dataset, transformers_objects = require_clinicalbert_dependencies()
    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup = (
        transformers_objects
    )
    TextDataset = make_torch_dataset_class(torch, Dataset)

    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir) / "clinicalbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    info(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results: dict[str, Any] = {
        "created_at": utc_now(),
        "model_name": args.model_name,
        "device": str(device),
        "tasks": {},
    }

    for task_dir in discover_prepared_tasks(prepared_dir, args.tasks):
        info(f"Fine-tuning ClinicalBERT for {task_dir.name}")
        train_df = load_split(task_dir, "train")
        val_df = load_split(task_dir, "val")
        test_df = load_split(task_dir, "test")
        num_labels = infer_num_labels(train_df, val_df, test_df)

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            local_files_only=args.local_files_only,
            use_safetensors=False,
        )
        model.to(device)

        train_dataset = TextDataset(
            train_df["text"].tolist(),
            train_df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )
        val_dataset = TextDataset(
            val_df["text"].tolist(),
            val_df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )
        test_dataset = TextDataset(
            test_df["text"].tolist(),
            test_df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = -1.0
        best_state: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_count = 0
            for step, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_size = int(batch["labels"].size(0))
                train_loss += float(loss.item()) * batch_size
                train_count += batch_size
                if args.logging_steps and step % args.logging_steps == 0:
                    info(
                        f"  epoch {epoch} step {step}/{len(train_loader)} "
                        f"loss={train_loss / max(train_count, 1):.4f}"
                    )

            val_loss, val_metrics, _, _, _ = evaluate_torch_model(
                torch,
                model,
                val_loader,
                device,
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
            history.append(epoch_record)
            info(
                f"  epoch {epoch}: train_loss={epoch_record['train_loss']:.4f}, "
                f"val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.3f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_metrics, labels, predictions, probabilities = evaluate_torch_model(
            torch,
            model,
            test_loader,
            device,
        )
        test_metrics["test_loss"] = test_loss
        test_metrics["history"] = history
        results["tasks"][task_dir.name] = test_metrics

        task_output = output_dir / task_dir.name
        task_output.mkdir(parents=True, exist_ok=True)
        prediction_df = test_df[["patient_id", "label"]].copy()
        prediction_df["prediction"] = predictions
        prediction_df = add_probability_columns(prediction_df, probabilities)
        prediction_df.to_csv(task_output / "predictions.csv", index=False)
        with (task_output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)

        if args.save_model:
            model.save_pretrained(task_output / "model")
            tokenizer.save_pretrained(task_output / "model")

        info(
            f"ClinicalBERT {task_dir.name}: "
            f"accuracy={test_metrics['accuracy']:.3f}, f1={test_metrics['f1']:.3f}, "
            f"roc_auc={test_metrics.get('roc_auc', float('nan')):.3f}"
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    info(f"ClinicalBERT results written to {output_dir}")
    return results


def run_reward_weighted_clinicalbert(args: argparse.Namespace) -> dict[str, Any]:
    torch, DataLoader, Dataset, transformers_objects = require_clinicalbert_dependencies()
    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup = (
        transformers_objects
    )
    TextDataset = make_torch_dataset_class(torch, Dataset)

    prepared_dir = Path(args.prepared_dir)
    output_dir = reward_output_dir(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    info(f"Using device: {device}")
    info(
        "Running reward-weighted ClinicalBERT fine-tuning. This is supervised "
        "classification with reward-based loss weights, not full reinforcement learning."
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    reward_config = {
        "correct_reward": args.correct_reward,
        "wrong_reward": args.wrong_reward,
        "confidence_bonus": args.confidence_bonus,
        "confidence_penalty": args.confidence_penalty,
        "reward_scale": args.reward_scale,
        "min_reward_weight": args.min_reward_weight,
        "max_reward_weight": args.max_reward_weight,
    }
    results: dict[str, Any] = {
        "created_at": utc_now(),
        "model_name": args.model_name,
        "device": str(device),
        "reward_config": reward_config,
        "tasks": {},
    }

    for task_dir in discover_prepared_tasks(prepared_dir, args.tasks):
        info(f"Reward-weighted fine-tuning ClinicalBERT for {task_dir.name}")
        train_df = load_split(task_dir, "train")
        val_df = load_split(task_dir, "val")
        test_df = load_split(task_dir, "test")
        num_labels = infer_num_labels(train_df, val_df, test_df)
        info(f"  detected {num_labels} label(s) from prepared CSV files")

        # Use the same tokenizer/checkpoint and prepared train/val/test CSVs as
        # the standard ClinicalBERT benchmark so metrics are directly comparable.
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            local_files_only=args.local_files_only,
            use_safetensors=False,
        )
        model.to(device)

        train_dataset = TextDataset(
            train_df["text"].tolist(),
            train_df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )
        val_dataset = TextDataset(
            val_df["text"].tolist(),
            val_df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )
        test_dataset = TextDataset(
            test_df["text"].tolist(),
            test_df["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = -1.0
        best_state: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_count = 0
            reward_totals = {
                "mean_reward": 0.0,
                "mean_reward_weight": 0.0,
                "mean_confidence": 0.0,
                "batch_accuracy": 0.0,
            }

            for step, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch["labels"]
                model_inputs = {
                    key: value
                    for key, value in batch.items()
                    if key != "labels"
                }

                # Forward pass followed by reward-weighted cross entropy.
                outputs = model(**model_inputs)
                loss, reward_stats = reward_weighted_classification_loss(
                    torch,
                    outputs.logits,
                    labels,
                    args,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_size = int(labels.size(0))
                train_loss += float(loss.item()) * batch_size
                train_count += batch_size
                for key, value in reward_stats.items():
                    reward_totals[key] += value * batch_size

                if args.logging_steps and step % args.logging_steps == 0:
                    info(
                        f"  epoch {epoch} step {step}/{len(train_loader)} "
                        f"loss={train_loss / max(train_count, 1):.4f}, "
                        f"reward={reward_totals['mean_reward'] / max(train_count, 1):.3f}, "
                        f"weight={reward_totals['mean_reward_weight'] / max(train_count, 1):.3f}"
                    )

            val_loss, val_metrics, _, _, _ = evaluate_torch_model(
                torch,
                model,
                val_loader,
                device,
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "mean_reward": reward_totals["mean_reward"] / max(train_count, 1),
                "mean_reward_weight": (
                    reward_totals["mean_reward_weight"] / max(train_count, 1)
                ),
                "mean_confidence": reward_totals["mean_confidence"] / max(train_count, 1),
                "train_batch_accuracy": reward_totals["batch_accuracy"] / max(train_count, 1),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
            history.append(epoch_record)
            info(
                f"  epoch {epoch}: train_loss={epoch_record['train_loss']:.4f}, "
                f"mean_reward={epoch_record['mean_reward']:.3f}, "
                f"val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.3f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_metrics, labels, predictions, probabilities = evaluate_torch_model(
            torch,
            model,
            test_loader,
            device,
        )
        test_metrics["test_loss"] = test_loss
        test_metrics["history"] = history
        test_metrics["reward_config"] = reward_config
        results["tasks"][task_dir.name] = test_metrics

        task_output = output_dir / task_dir.name
        task_output.mkdir(parents=True, exist_ok=True)
        prediction_df = test_df[["patient_id", "label"]].copy()
        prediction_df["prediction"] = predictions
        prediction_df = add_probability_columns(prediction_df, probabilities)
        prediction_df.to_csv(task_output / "predictions.csv", index=False)
        with (task_output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
        save_confusion_matrix_image(
            matrix=test_metrics["confusion_matrix"],
            labels=test_metrics["label_ids"],
            output_path=task_output / "confusion_matrix.png",
            title=f"{task_dir.name} Reward-Weighted ClinicalBERT",
        )

        # Save the trained checkpoint so it can be reloaded later with the
        # regular `infer` command or compared in a final report.
        if args.save_model:
            model.save_pretrained(task_output / "model")
            tokenizer.save_pretrained(task_output / "model")

        info(
            f"Reward-weighted ClinicalBERT {task_dir.name}: "
            f"accuracy={test_metrics['accuracy']:.3f}, f1={test_metrics['f1']:.3f}, "
            f"roc_auc={test_metrics.get('roc_auc', float('nan')):.3f}"
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    info(f"Reward-weighted ClinicalBERT results written to {output_dir}")
    return results


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Root containing extracted Synthea csv folders. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--task-config", default=None, help="Optional JSON file with extra tasks.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task names to prepare/run.")
    parser.add_argument("--max-patients", type=int, default=2_000)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-events-per-section", type=int, default=40)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--include-conditions-in-text",
        action="store_true",
        help="Include condition descriptions in model text. Useful for debugging; leaks labels for condition tasks.",
    )
    parser.add_argument(
        "--include-careplans-in-text",
        action="store_true",
        help="Include careplan descriptions in model text. Useful for debugging; may leak condition labels.",
    )
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument("--balance", dest="balance", action="store_true", default=True)
    balance_group.add_argument("--no-balance", dest="balance", action="store_false")


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-dir", default=f"{DEFAULT_OUTPUT_DIR}/prepared")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--tfidf-max-features", type=int, default=50_000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-max-ngram", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=None)


def add_clinicalbert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-dir", default=f"{DEFAULT_OUTPUT_DIR}/prepared")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-model", action="store_true")


def add_reward_weighted_args(parser: argparse.ArgumentParser) -> None:
    add_clinicalbert_args(parser)
    parser.set_defaults(
        save_model=True,
        prepared_dir=f"{DEFAULT_OUTPUT_DIR}/all_tasks/prepared",
        output_dir=f"{DEFAULT_OUTPUT_DIR}/all_tasks",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="prepared_dir",
        help="Alias for --prepared-dir; points to prepared train/val/test CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Alias for --output-dir.",
    )
    parser.add_argument(
        "--no-save-model",
        dest="save_model",
        action="store_false",
        help="Skip saving the trained reward-weighted checkpoint.",
    )
    parser.add_argument(
        "--correct-reward",
        type=float,
        default=1.0,
        help="Base reward when the current prediction matches the true label.",
    )
    parser.add_argument(
        "--wrong-reward",
        type=float,
        default=-0.5,
        help="Base reward/penalty when the current prediction is wrong.",
    )
    parser.add_argument(
        "--confidence-bonus",
        type=float,
        default=0.5,
        help="Additional reward scaled by confidence for correct predictions.",
    )
    parser.add_argument(
        "--confidence-penalty",
        type=float,
        default=0.5,
        help="Additional penalty scaled by confidence for wrong predictions.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.5,
        help="How strongly rewards change the per-sample loss weight.",
    )
    parser.add_argument(
        "--min-reward-weight",
        type=float,
        default=0.25,
        help="Lower bound for reward-derived loss weights.",
    )
    parser.add_argument(
        "--max-reward-weight",
        type=float,
        default=2.0,
        help="Upper bound for reward-derived loss weights.",
    )


def add_encode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-dir", default=f"{DEFAULT_OUTPUT_DIR}/prepared")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=10, help="Rows per task to encode. Use 0 for all rows.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=16,
        help="Number of embedding dimensions to write. Use 0 for all dimensions.",
    )
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--cpu", action="store_true")


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-dir", default=f"{DEFAULT_OUTPUT_DIR}/prepared")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=0, help="Rows per task to classify. Use 0 for all rows.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--allow-random-head",
        action="store_true",
        help="Allow inference when the checkpoint has no trained classifier head. Debug only.",
    )


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Metric run directories to include, for example baseline clinicalbert "
            "clinicalbert_reward small_llm."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["accuracy", "f1", "roc_auc"],
        help="Metric names to plot.",
    )
    parser.add_argument("--dpi", type=int, default=180)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ClinicalBERT on synthetic Synthea classification tasks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build task CSVs from Synthea data.")
    add_prepare_args(prepare_parser)

    baseline_parser = subparsers.add_parser("baseline", help="Run TF-IDF logistic baseline.")
    add_baseline_args(baseline_parser)

    clinicalbert_parser = subparsers.add_parser(
        "clinicalbert",
        help="Fine-tune a ClinicalBERT classifier head on prepared tasks.",
    )
    add_clinicalbert_args(clinicalbert_parser)

    reward_parser = subparsers.add_parser(
        "clinicalbert-reward",
        help="Fine-tune ClinicalBERT with reward-weighted supervised loss.",
    )
    add_reward_weighted_args(reward_parser)

    encode_parser = subparsers.add_parser(
        "encode",
        help="Run base ClinicalBERT forward passes and save raw embeddings; no training.",
    )
    add_encode_args(encode_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference with an already-trained sequence-classification checkpoint; no training.",
    )
    add_infer_args(infer_parser)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Save a Matplotlib graph of benchmark scores.",
    )
    add_plot_args(plot_parser)

    all_parser = subparsers.add_parser("all", help="Prepare data, baseline, and ClinicalBERT.")
    add_prepare_args(all_parser)
    all_parser.add_argument("--skip-baseline", action="store_true")
    all_parser.add_argument("--skip-clinicalbert", action="store_true")
    all_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    all_parser.add_argument("--local-files-only", action="store_true")
    all_parser.add_argument("--epochs", type=int, default=2)
    all_parser.add_argument("--batch-size", type=int, default=8)
    all_parser.add_argument("--eval-batch-size", type=int, default=16)
    all_parser.add_argument("--max-length", type=int, default=256)
    all_parser.add_argument("--learning-rate", type=float, default=2e-5)
    all_parser.add_argument("--weight-decay", type=float, default=0.01)
    all_parser.add_argument("--warmup-ratio", type=float, default=0.06)
    all_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    all_parser.add_argument("--logging-steps", type=int, default=50)
    all_parser.add_argument("--num-workers", type=int, default=0)
    all_parser.add_argument("--cpu", action="store_true")
    all_parser.add_argument("--save-model", action="store_true")
    all_parser.add_argument("--tfidf-max-features", type=int, default=50_000)
    all_parser.add_argument("--tfidf-min-df", type=int, default=2)
    all_parser.add_argument("--tfidf-max-ngram", type=int, default=2)
    all_parser.add_argument("--max-iter", type=int, default=1_000)
    all_parser.add_argument("--n-jobs", type=int, default=None)

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if hasattr(args, "data_root") and args.data_root is None:
        args.data_root = ["Data"]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv))

    try:
        if args.command == "prepare":
            prepare_benchmark(args)
        elif args.command == "baseline":
            run_baseline(args)
        elif args.command == "clinicalbert":
            run_clinicalbert(args)
        elif args.command == "clinicalbert-reward":
            run_reward_weighted_clinicalbert(args)
        elif args.command == "encode":
            run_encode(args)
        elif args.command == "infer":
            run_infer(args)
        elif args.command == "plot":
            run_plot(args)
        elif args.command == "all":
            prepared_dir = prepare_benchmark(args)
            args.prepared_dir = str(prepared_dir)
            if not args.skip_baseline:
                run_baseline(args)
            if not args.skip_clinicalbert:
                run_clinicalbert(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
