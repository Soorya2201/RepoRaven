# RepoAudit Report — iter3

**Findings are advisory triage, not adjudication.** This system is measurably imperfect (precision 69% on our own eval set — roughly 1 in 4-5 findings is wrong). Every finding below cites an exact file:line and the paper sentence it's checked against, so you can verify it in under a minute. **Do not forward any finding to an author, editor, or committee without checking the citation yourself first.** Full context: README.md.

Papers audited: 9  ·  Micro-avg F1: 0.643  ·  Precision: 69%  ·  Recall: 60%

---
## Proto2Proto: Can you recognize the car, the way I do?

**Paper:** arXiv:2204.11830  
**Repo:** https://github.com/archmaester/proto2proto @ `2fdc4fb29d`

<sub>ℹ️ A formatting slip in the model's raw output was automatically repaired before scoring; findings below are unaffected.</sub>

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
---

## G-Mixup: Graph Data Augmentation for Graph Classification

**Paper:** arXiv:2202.07179  
**Repo:** https://github.com/ahxt/g-mixup @ `0813bb3231`

<sub>ℹ️ A formatting slip in the model's raw output was automatically repaired before scoring; findings below are unaffected.</sub>

**4 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** The batch size is set to size is set to 128.

**Code:** `src/gmixup.py:128`

The evidence shows --batch_size default=32, which does not match the claimed value of 128.

### 2. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** mixup ratio λ ∈ [0.1, 0.2]

**Code:** `src/gmixup.py:132`

The evidence shows the default lambda range is "[0.005, 0.01]", which does not match the claimed range [0.1, 0.2].

### 3. [HIGH] data_leakage_global_preprocessing (confidence: 90%)

> **Paper says:** We split the dataset into train/val/test data by 7 : 1 : 2.

**Code:** `src/gmixup.py:49`

The preprocessing call deg.mean() at line 49 computes statistics on deg (derived from the full dataset) before the train/val/test split at line 250 (train_dataset = dataset[:train_nums]), causing global preprocessing leakage.

### 4. [LOW] single_seed_reporting (confidence: 95%)

> **Paper says:** we report the test accuracy on ten runs.

**Code:** `run_gmixup.sh:2`

The script loops over 8 seed values (1314,11314,21314,31314,41314,51314,61314,71314), which is fewer than the ten runs claimed.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 4 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.8889.</sub>

---

## The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Paper:** arXiv:1803.03635  
**Repo:** https://github.com/google-research/lottery-ticket-hypothesis @ `1f17279d28`

<sub>ℹ️ A formatting slip in the model's raw output was automatically repaired before scoring; findings below are unaffected.</sub>

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 2 missed. F1 = 0.0.</sub>

---

## Network Deconvolution

**Paper:** arXiv:1905.11926  
**Repo:** https://github.com/yechengxi/deconvolution @ `d221c221b7`

**4 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** we set B = 64 before calculating the covariance matrix.

**Code:** `main_imagenet.py:51`

The evidence shows the batch-size flag (-b/--batch-size) defaults to 256, not the claimed value of 64.

### 2. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** we set the batch size to 128 and the weight decay to 0.001.

**Code:** `main_imagenet.py:51`

The evidence shows the default batch size is 256 (flag -b/--batch-size default=256), which contradicts the claimed batch size of 128.

### 3. [HIGH] hyperparam_drift (confidence: 100%)

> **Paper says:** we can use the optimal learning rate 1.0 and get high accuracy as well.

**Code:** `main_imagenet.py:56`

The repository's default learning rate is 0.1 (flag --lr/--learning-rate default=0.1), which does not match the claimed value of 1.0.

### 4. [MEDIUM] hyperparam_drift (confidence: 100%)

> **Paper says:** the final accuracy actually improves with increased weight decay (.0005 to .005).

**Code:** `main_imagenet.py:67`

Evidence E12 shows the default weight decay is 1e-4 (0.0001), which is outside the claimed range of .0005 to .005.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 1 confirmed against known ground truth, 3 unconfirmed, 0 missed. F1 = 0.4.</sub>

---

## BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning

**Paper:** arXiv:2305.04769  
**Repo:** https://github.com/NeurAI-Lab/BiRT @ `1340a97bff`

> ⚠️ **This run hit an error and may be incomplete:** Parse failed even after repair pass. Original error: Could not parse JSON list (first 300 chars): 'We need to extract claims of categories: hyperparameter, seed_protocol, metric, data_split_protocol, eval_protocol. Must be actual statements in the paper text. Provide exact quote and description. Ret

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 2 missed. F1 = 0.0.</sub>

---

## Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations

**Paper:** arXiv:2110.12088  
**Repo:** https://github.com/UCSC-REAL/cifar-10-100n @ `49df7d8a69`

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] single_seed_reporting (confidence: 95%)

> **Paper says:** mean ± standard deviation of 5 runs

**Code:** `main.py:17`

The code shows a single seed argument with default value 0 and no loop over multiple seeds, indicating only one run was performed, contradicting the claim of averaging over 5 runs.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 1 unconfirmed, 0 missed. F1 = 0.0.</sub>

---

## [SYNTHETIC FIXTURE] Not a real paper

**Paper:** synthetic fixture, not a real paper  
**Repo:** local:data/synthetic/synthetic_multi_defect (constructed by RepoAudit, no external URL)

**3 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] metric_misimplementation (confidence: 95%)

> **Paper says:** We report the macro-averaged F1 score as our primary metric, computed on the held-out test set only, after training is complete.

**Code:** `train.py:64`

The evidence shows a call to f1_score(y, preds) without an average parameter, which defaults to binary averaging in sklearn, not macro-averaged as claimed.

### 2. [HIGH] test_set_contamination (confidence: 95%)

> **Paper says:** All hyperparameter selection is performed on a validation split; the test set is touched exactly once, for final reporting.

**Code:** `train.py:51`

Evidence E6 shows test accuracy being computed on X_test/y_test inside the training loop, indicating the test set is used for model selection rather than only once for final reporting.

### 3. [HIGH] eval_on_train (confidence: 95%)

> **Paper says:** computed on the held-out test set only, after training is complete.

**Code:** `train.py:77`

The final evaluate call in train.py (line 77, evidence E2) is invoked with X_train and y_train, indicating evaluation on training data rather than the held-out test set.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 3 confirmed against known ground truth, 0 unconfirmed, 0 missed. F1 = 1.0.</sub>

---

## Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints

**Paper:** arXiv:2311.08675  
**Repo:** https://github.com/xiaoboxia/LBCS @ `338a31dc5a`

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---

## Sample-specific Masks for Visual Reprogramming

**Paper:** arXiv:2406.03150  
**Repo:** https://github.com/tmlr-group/SMM @ `eeac33afc4`

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] single_seed_reporting (confidence: 95%)

> **Paper says:** Experiments are run with three seeds on a single A100 GPU

**Code:** `readme.md:53`

The evidence shows a hardcoded seed value of 0 with no loop over multiple seeds, indicating only a single seed was used.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 1 confirmed against known ground truth, 0 unconfirmed, 0 missed. F1 = 1.0.</sub>

---
