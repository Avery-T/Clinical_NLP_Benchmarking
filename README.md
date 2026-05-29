# Data is downloaded from
https://synthea.mitre.org/downloads?utm_source=chatgpt.com
# Clinical NLP Benchmarking

Benchmark ClinicalBERT on synthetic Synthea patient classification tasks.

The script only scans extracted `csv/` folders under `Data/`. To add more data later,
extract another Synthea output anywhere under `Data/` and rerun the same command.

## Setup

```bash
pip install -r requirements.txt
```

`torch` and `transformers` are only needed for the ClinicalBERT run. Dataset
preparation and the TF-IDF baseline work with `pandas` and `scikit-learn`.

## Quick Smoke Test

This prepares two balanced tasks and runs a fast baseline:

```bash
python clinical_bert_test.py all \
  --tasks diabetes hypertension \
  --max-patients 1000 \
  --skip-clinicalbert
```

Outputs are written under `benchmark_runs/`.

## Plot Scores

Save a Matplotlib chart from the saved benchmark metrics:

```bash
python clinical_bert_test.py plot \
  --output-dir benchmark_runs \
  --runs baseline clinicalbert
```

The chart is written to `benchmark_runs/plots/scores.png`, and the plotted
values are also written to `benchmark_runs/plots/score_summary.csv`.

## Raw ClinicalBERT Output

ClinicalBERT by itself does not output disease classes. It outputs vectors for
the input text. To see those raw outputs with no training:

```bash
python clinical_bert_test.py encode \
  --prepared-dir benchmark_runs/prepared \
  --tasks diabetes \
  --split all \
  --limit 0 \
  --include-text
```

The output JSONL is written under `benchmark_runs/clinicalbert_raw_outputs/`.

## No-Training Classification

You can benchmark classification without training only if `--model-name` points
to a checkpoint that already has a trained sequence-classification head for the
same labels in your prepared data.

```bash
python clinical_bert_test.py infer \
  --prepared-dir benchmark_runs/prepared \
  --tasks diabetes \
  --model-name path/or/huggingface-id/of-trained-classifier \
  --split test
```

The default base Bio ClinicalBERT checkpoint is not a diabetes or hypertension
classifier. The `infer` command will refuse to benchmark it as a classifier
unless you pass `--allow-random-head`, which is only useful for debugging.

## ClinicalBERT Fine-Tuning Benchmark

After installing the model dependencies:

```bash
python clinical_bert_test.py clinicalbert \
  --prepared-dir benchmark_runs/prepared \
  --tasks diabetes hypertension \
  --epochs 2 \
  --batch-size 8
```

This command trains a small classification head on top of ClinicalBERT, which is
why it uses epochs. Use `encode` instead when you only want model outputs for
inputs.

The default model is `emilyalsentzer/Bio_ClinicalBERT`. Use `--model-name` to
switch models, for example:

```bash
python clinical_bert_test.py clinicalbert \
  --model-name medicalai/ClinicalBERT
```

## Default Tasks

The built-in condition-label tasks are:

- `diabetes`
- `hypertension`
- `coronary_heart_disease`
- `stroke`
- `asthma`
- `copd`

Labels come from `conditions.csv`. By default, condition and careplan
descriptions are not included in the model text, so the benchmark does not leak
the answer directly into the input. Add `--include-conditions-in-text` or
`--include-careplans-in-text` only for debugging.

## Custom Tasks

Add tasks with a JSON file:

```json
{
  "tasks": [
    {
      "name": "lung_cancer",
      "positive_regex": "\\blung cancer\\b",
      "description": "Patient has lung cancer in conditions.csv."
    }
  ]
}
```

Then run:

```bash
python clinical_bert_test.py prepare --task-config tasks.json --tasks lung_cancer
```
