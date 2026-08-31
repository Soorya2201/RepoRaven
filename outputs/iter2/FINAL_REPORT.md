# RepoAudit Report — iter2

**Findings are advisory triage, not adjudication.** This system is measurably imperfect (precision 43% on our own eval set — roughly 1 in 4-5 findings is wrong). Every finding below cites an exact file:line and the paper sentence it's checked against, so you can verify it in under a minute. **Do not forward any finding to an author, editor, or committee without checking the citation yourself first.** Full context: README.md.

Papers audited: 9  ·  Micro-avg F1: 0.273  ·  Precision: 43%  ·  Recall: 20%

---
## Proto2Proto: Can you recognize the car, the way I do?

**Paper:** arXiv:2204.11830  
**Repo:** https://github.com/archmaester/proto2proto @ `2fdc4fb29d`

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
---

## G-Mixup: Graph Data Augmentation for Graph Classification

**Paper:** arXiv:2202.07179  
**Repo:** https://github.com/ahxt/g-mixup @ `0813bb3231`

**1 finding**, ordered by how much they're worth your time:

### 1. [MEDIUM] single_seed_reporting (confidence: 95%)

> **Paper says:** we report the test accuracy on ten runs.

**Code:** `run_gmixup.sh:2`

The shell script loops over only eight seed values, not ten as claimed.

*Verify this yourself before acting on it — see the human-review note in README.md.*

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 1 unconfirmed, 5 missed. F1 = 0.0.</sub>

---

## The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Paper:** arXiv:1803.03635  
**Repo:** https://github.com/google-research/lottery-ticket-hypothesis @ `1f17279d28`

> ⚠️ **This run hit an error and may be incomplete:** Tool loop ended with no final text response.

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 2 missed. F1 = 0.0.</sub>

---

## Network Deconvolution

**Paper:** arXiv:1905.11926  
**Repo:** https://github.com/yechengxi/deconvolution @ `d221c221b7`

> ⚠️ **This run hit an error and may be incomplete:** Parse failed even after repair pass. Original error: Could not parse JSON list (first 300 chars): 'We need to extract claims of categories: hyperparameter, seed_protocol, metric, data_split_protocol, eval_protocol. Must be actual statements in the paper text. Provide exact quote and description. Nee

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---

## BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning

**Paper:** arXiv:2305.04769  
**Repo:** https://github.com/NeurAI-Lab/BiRT @ `1340a97bff`

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 2 missed. F1 = 0.0.</sub>

---

## Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations

**Paper:** arXiv:2110.12088  
**Repo:** https://github.com/UCSC-REAL/cifar-10-100n @ `49df7d8a69`

**1 finding**, ordered by how much they're worth your time:

### 1. [MEDIUM] other (confidence: 95%)

> **Paper says:** We randomly split the training dataset of CIFAR-10 without replacement into ten batches.

**Code:** `data/datasets.py:30-45`

The input_dataset function creates a CIFAR10 train_dataset without any splitting into ten batches; it loads the full training set directly.

*Verify this yourself before acting on it — see the human-review note in README.md.*

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 1 unconfirmed, 0 missed. F1 = 0.0.</sub>

---

## [SYNTHETIC FIXTURE] Not a real paper

**Paper:** synthetic fixture, not a real paper  
**Repo:** local:data/synthetic/synthetic_multi_defect (constructed by RepoAudit, no external URL) @ `n/a (not a`

**3 findings**, ordered by how much they're worth your time:

### 1. [HIGH] metric_misimplementation (confidence: 100%)

> **Paper says:** We report the macro-averaged F1 score as our primary metric, computed on the held-out test set only, after training is complete.

**Code:** `train.py:60-64`

The evaluate() function calls f1_score(y, preds) without specifying the 'average' argument, so sklearn defaults to binary averaging, contradicting the claimed macro-averaged F1.

*Verify this yourself before acting on it — see the human-review note in README.md.*

### 2. [HIGH] test_set_contamination (confidence: 100%)

> **Paper says:** All hyperparameter selection is performed on a validation split; the test set is touched exactly once, for final reporting.

**Code:** `train.py:46-50`

train_with_early_stopping selects the best max_iter based on accuracy on X_test/y_test, i.e., the test set, violating the claim of using a validation split only.

*Verify this yourself before acting on it — see the human-review note in README.md.*

### 3. [HIGH] eval_on_train (confidence: 100%)

> **Paper says:** computed on the held-out test set only

**Code:** `train.py:73-78`

The final "test" evaluation mistakenly passes X_train and y_train to evaluate(), so reported test metrics are actually computed on the training set.

*Verify this yourself before acting on it — see the human-review note in README.md.*

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 3 confirmed against known ground truth, 0 unconfirmed, 0 missed. F1 = 1.0.</sub>

---

## Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints

**Paper:** arXiv:2311.08675  
**Repo:** https://github.com/xiaoboxia/LBCS @ `338a31dc5a`

> ⚠️ **This run hit an error and may be incomplete:** Tool loop ended with no final text response.

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---

## Sample-specific Masks for Visual Reprogramming

**Paper:** arXiv:2406.03150  
**Repo:** https://github.com/tmlr-group/SMM @ `eeac33afc4`

> ⚠️ **This run hit an error and may be incomplete:** Dropped 1 ungrounded finding(s) (paper_evidence didn't match any extracted quote).

**2 findings**, ordered by how much they're worth your time:

### 1. [HIGH] single_seed_reporting (confidence: 99%)

> **Paper says:** Experiments are run with three seeds on a single A100 GPU and the averaged test accuracy is reported.

**Code:** `instancewise_vp.py:20, readme.md:53-55`

The code defines a single '--seed' argument with default 0 and the README shows example commands using only one seed; there is no loop or script to run three seeds and average results.

*Verify this yourself before acting on it — see the human-review note in README.md.*

### 2. [LOW] other (confidence: 95%)

> **Paper says:** We follow Chen et al. (2023) to split the datasets.

**Code:** `data/prepare_data.py:17-24`

Dataset splitting is performed using the standard torchvision train/test splits (train=True/False) rather than any procedure from Chen et al. (2023).

*Verify this yourself before acting on it — see the human-review note in README.md.*

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 2 unconfirmed, 1 missed. F1 = 0.0.</sub>

---
