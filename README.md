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

Save a Matplotlib comparison chart from multiple saved benchmark metrics:

```bash
python clinical_bert_test.py plot \
  --output-dir benchmark_runs \
  --runs baseline clinicalbert small_llm
```

The chart is written to `benchmark_runs/plots/scores.png`, and the plotted
values are also written to `benchmark_runs/plots/score_summary.csv`.

## Small LLM Prompting Benchmark

Use the same prepared patient `test.csv` files with a local OpenAI-compatible
small LLM server, such as LM Studio:

```bash
python small_llm_benchmarking.py \
  --prepared-dir benchmark_runs/prepared \
  --tasks diabetes hypertension \
  --split test \
  --shots zero one \
  --output-dir benchmark_runs/small_llm
```

Defaults:

- `--base-url http://localhost:1233/v1`
- `--api-key lm-studio`
- `--model google/gemma-4-e4b`

Run a quick smoke test first:

```bash
python small_llm_benchmarking.py \
  --tasks diabetes \
  --shots zero \
  --limit 3 \
  --output-dir benchmark_runs/small_llm
```

Outputs:

- `benchmark_runs/small_llm/<task>/<shot>_shot_predictions.csv`
- `benchmark_runs/small_llm/<task>/<shot>_shot_metrics.json`
- `benchmark_runs/small_llm/metrics.json`
- `benchmark_runs/small_llm/plots/scores.png`

Small LLM runs create their own plot by default. Use `--no-plot` to skip it.
Only use `clinical_bert_test.py plot --runs ... small_llm` when you explicitly
want a comparison chart against ClinicalBERT or the TF-IDF baseline.

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

## Reward-Weighted ClinicalBERT Experiment

Part 4 uses a reinforcement-learning-inspired ClinicalBERT training experiment:
the model is still trained as a supervised classifier, but each example's
cross-entropy loss is weighted by a reward computed from the model's current
prediction. The script automatically detects the prepared CSV text column,
label column, and number of labels for each task. In the current uploaded
5000-sample project data, `benchmark_runs/all_tasks/prepared` contains six
binary disease tasks: diabetes, hypertension, coronary heart disease, stroke,
asthma, and COPD.

This is not full reinforcement learning. There is no environment, trajectory,
policy rollout, or delayed reward. ClinicalBERT is a sequence-classification
model, so this project uses a manageable reward-based loss weighting approach
that can be compared fairly against the standard fine-tuning benchmark.

Reward calculation during each training batch:

- Correct prediction: `correct_reward + confidence_bonus * confidence`
- Wrong prediction: `wrong_reward - confidence_penalty * confidence`
- Loss weight: `1 + reward_scale * abs(reward_or_penalty)`, clipped between
  `min_reward_weight` and `max_reward_weight`. Positive rewards reinforce
  correct examples; negative rewards become penalty magnitudes that make
  confident mistakes count more.

With the defaults, correct and confident predictions receive larger loss
weights, correct but uncertain predictions receive medium weights, wrong and
confident predictions receive stronger loss penalties, and wrong but uncertain
predictions receive smaller loss penalties.

Run reward-weighted training on the same prepared splits:

```bash
python clinical_bert_test.py clinicalbert-reward \
  --prepared-dir benchmark_runs/all_tasks/prepared \
  --output-dir benchmark_runs/all_tasks \
  --epochs 2 \
  --batch-size 8
```

With the current repository layout, this shorter command uses those same
defaults:

```bash
python clinical_bert_test.py clinicalbert-reward
```

You can also use the standalone wrapper:

```bash
python reward_weighted_clinicalbert.py \
  --data_dir benchmark_runs/all_tasks/prepared \
  --output_dir benchmark_runs/all_tasks \
  --epochs 2 \
  --batch-size 8
```

Training automatically evaluates the best validation checkpoint on the test
split using the same benchmark metrics: accuracy, precision, recall, F1,
macro F1, weighted F1, ROC AUC when available, and confusion matrix.

Outputs are saved under:

- `benchmark_runs/all_tasks/clinicalbert_reward/<task>/predictions.csv`
- `benchmark_runs/all_tasks/clinicalbert_reward/<task>/metrics.json`
- `benchmark_runs/all_tasks/clinicalbert_reward/<task>/confusion_matrix.png`
- `benchmark_runs/all_tasks/clinicalbert_reward/<task>/model/`
- `benchmark_runs/all_tasks/clinicalbert_reward/metrics.json`

If you prefer the root-level folder requested in the project checklist, pass
`--output-dir benchmark_runs`; that writes to `benchmark_runs/clinicalbert_reward/`.

To rerun evaluation later from a saved task checkpoint:

```bash
python clinical_bert_test.py infer \
  --prepared-dir benchmark_runs/all_tasks/prepared \
  --tasks diabetes \
  --model-name benchmark_runs/all_tasks/clinicalbert_reward/diabetes/model \
  --split test
```

Include it in the final comparison plot/table:

```bash
python clinical_bert_test.py plot \
  --output-dir benchmark_runs/all_tasks \
  --runs clinicalbert clinicalbert_reward small_llm \
  --metrics accuracy f1 macro_f1 weighted_f1 roc_auc
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
