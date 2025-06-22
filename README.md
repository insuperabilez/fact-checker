# Fact-Checking ML Pipeline

This repository contains a machine learning pipeline for training binary classification models for fact-checking tasks. The system processes multiple authoritative datasets through standardized preprocessing pipelines, consolidates them into a unified training dataset, and fine-tunes a ModernBERT model for fact verification.

## System Overview

The fact-checker implements a comprehensive ETL pipeline that:
1. Processes heterogeneous fact-checking datasets (FELM, FEVER, HaluEval, LIAR)
2. Standardizes data into a common schema
3. Trains and evaluates a ModernBERT-based classification model

### Key Features
- Modular dataset processors for each source
- Standardized CSV output format
- Configurable training pipeline
- Model evaluation with accuracy and F1 metrics

###Model

https://huggingface.co/insuperabile/modernbert-factcheck

## Datasets Processed

| Dataset      | Processor | Input Format | Label Mapping |
|--------------|-----------|--------------|---------------|
| FELM         | `create_felm_csv.py` | Hugging Face Dataset | Binary 0/1 |
| FEVER        | `create_fever_csv.py` | JSONL | SUPPORTS→1, REFUTES→0 |
| HaluEval     | `create_halueval_dataset.py` | JSON | hallucination→0, correct→1 |
| LIAR         | `create_liar_dataset.py` | TSV | true→0, false/pants-fire→1 |

## Training Configuration

### Hyperparameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Model | ModernBERT-base | Base architecture |
| Batch size | 32 | Training efficiency |
| Learning rate | 5e-5 | Optimization |
| Epochs | 1 | Training duration |
| Max length | 512 | Sequence truncation |


