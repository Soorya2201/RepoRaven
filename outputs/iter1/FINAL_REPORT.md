# RepoRaven Report — iter1

**Findings are advisory triage, not adjudication.** This system is measurably imperfect (precision 75% on our own eval set — roughly 1 in 4-5 findings is wrong). Every finding below cites an exact file:line and the paper sentence it's checked against, so you can verify it in under a minute. **Do not forward any finding to an author, editor, or committee without checking the citation yourself first.** Full context: README.md.

Papers audited: 9  ·  Micro-avg F1: 0.667  ·  Precision: 75%  ·  Recall: 60%

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

<sub>ℹ️ A formatting slip in the model's raw output was automatically repaired before scoring; findings below are unaffected.</sub>

**5 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** The batch size is set to 128.

**Code:** `src/gmixup.py`

The code defines the batch_size argument with a default value of 32, not 128.

### 2. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** λ ∈ [ 0.1 , 0.2 ]

**Code:** `src/gmixup.py`

The lam_range argument defaults to "[0.005, 0.01]", which does not match the claimed range of [0.1, 0.2].

### 3. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** we generate 20% more graph for training graph.

**Code:** `src/gmixup.py`

The aug_ratio argument defaults to 0.15 (15%), indicating 15% more graphs rather than the claimed 20%.

### 4. [HIGH] metric_misimplementation (confidence: 100%)

> **Paper says:** The metric is AUROC.

**Code:** `src/gmixup.py`

The test function computes classification accuracy; no AUROC calculation is present in the code.

### 5. [MEDIUM] single_seed_reporting (confidence: 100%)

> **Paper says:** we report the test accuracy on ten runs.

**Code:** `src/gmixup.py`

The code sets a single seed and runs training once; there is no loop or aggregation over ten runs as claimed.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 3 confirmed against known ground truth, 2 unconfirmed, 2 missed. F1 = 0.6.</sub>

---

## The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Paper:** arXiv:1803.03635  
**Repo:** https://github.com/google-research/lottery-ticket-hypothesis @ `1f17279d28`

<sub>ℹ️ A formatting slip in the model's raw output was automatically repaired before scoring; findings below are unaffected.</sub>

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** We use Adam at a learning rate of 0.0012.

**Code:** `mnist_fc/constants.py:33`

The paper claims the use of Adam optimizer with learning rate 0.0012, but the code uses GradientDescentOptimizer with learning rate 0.1.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 1 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.6667.</sub>

---

## Network Deconvolution

**Paper:** arXiv:1905.11926  
**Repo:** https://github.com/yechengxi/deconvolution @ `d221c221b7`

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---

## BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning

**Paper:** arXiv:2305.04769  
**Repo:** https://github.com/NeurAI-Lab/BiRT @ `1340a97bff`

**2 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** weight decay of 1e-6

**Code:** `main.py: parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')`

The code sets the default weight decay to 0.05 (5e-2), which contradicts the claimed value of 1e-6.

### 2. [MEDIUM] single_seed_reporting (confidence: 100%)

> **Paper says:** averaged across three random seeds

**Code:** `main.py: parser.add_argument('--seed', default=0, type=int)`

The code only accepts a single seed via the --seed argument and shows no mechanism (e.g., loop over seeds, wrapper script, or aggregation step) to run and average results across multiple seeds.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 2 confirmed against known ground truth, 0 unconfirmed, 0 missed. F1 = 1.0.</sub>

---

## Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations

**Paper:** arXiv:2110.12088  
**Repo:** https://github.com/UCSC-REAL/cifar-10-100n @ `49df7d8a69`

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [MEDIUM] other (confidence: 100%)

> **Paper says:** We randomly split the training dataset of CIFAR-10 without replacement into ten batches.

**Code:** `data/cifar.py`

The code loads the CIFAR-10 training data from 5 predefined batches (data_batch_1 to data_batch_5) and concatenates them, without any random split into ten batches as claimed.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 1 unconfirmed, 0 missed. F1 = 0.0.</sub>

---

## [SYNTHETIC FIXTURE] Not a real paper

**Paper:** synthetic fixture, not a real paper  
**Repo:** local:data/synthetic/synthetic_multi_defect (constructed by RepoRaven, no external URL)

**3 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] metric_misimplementation (confidence: 100%)

> **Paper says:** We report the macro-averaged F1 score as our primary metric, computed on the held-out test set only, after training is complete.

**Code:** `train.py:65`

The code calls f1_score(y, preds) without specifying the average parameter, which defaults to 'binary' instead of macro-averaged as claimed.

### 2. [HIGH] test_set_contamination (confidence: 100%)

> **Paper says:** All hyperparameter selection is performed on a validation split; the test set is touched exactly once, for final reporting.

**Code:** `train.py:52`

The test set is used to select the best max_iter hyperparameter inside the training loop (line 52), meaning the test set is touched multiple times for hyperparameter selection, not just once for final reporting.

### 3. [HIGH] eval_on_train (confidence: 100%)

> **Paper says:** computed on the held-out test set only, after training is complete.

**Code:** `train.py:78`

The code evaluates the model on the training set (X_train, y_train) but labels it as test set evaluation, so the reported test numbers are actually from the training set.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 3 confirmed against known ground truth, 0 unconfirmed, 0 missed. F1 = 1.0.</sub>

---

## Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints

**Paper:** arXiv:2311.08675  
**Repo:** https://github.com/xiaoboxia/LBCS @ `338a31dc5a`

> ⚠️ **This run hit an error and may be incomplete:** Verifier LLM call failed: ReadTimeout: HTTPSConnectionPool(host='integrate.api.nvidia.com', port=443): Read timed out. (read timeout=240.0)

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---

## Sample-specific Masks for Visual Reprogramming

**Paper:** arXiv:2406.03150  
**Repo:** https://github.com/tmlr-group/SMM @ `eeac33afc4`

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---
