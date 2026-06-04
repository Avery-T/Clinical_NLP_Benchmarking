#!/usr/bin/env python3
"""Run the reward-weighted ClinicalBERT fine-tuning experiment.

This small wrapper keeps Part 4 visible as its own training script while reusing
the same dataset loading, preprocessing, metrics, and output format implemented
in clinical_bert_test.py.
"""

from __future__ import annotations

import sys

from clinical_bert_test import main


if __name__ == "__main__":
    raise SystemExit(main(["clinicalbert-reward", *sys.argv[1:]]))
