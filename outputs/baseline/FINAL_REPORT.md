# RepoRaven Report — baseline

**Findings are advisory triage, not adjudication.** This system is measurably imperfect (precision 29% on our own eval set — roughly 1 in 4-5 findings is wrong). Every finding below cites an exact file:line and the paper sentence it's checked against, so you can verify it in under a minute. **Do not forward any finding to an author, editor, or committee without checking the citation yourself first.** Full context: README.md.

Papers audited: 9  ·  Micro-avg F1: 0.182  ·  Precision: 29%  ·  Recall: 13%

---
## Proto2Proto: Can you recognize the car, the way I do?

**Paper:** arXiv:2204.11830  
**Repo:** https://github.com/archmaester/proto2proto @ `2fdc4fb29d`

**2 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] metric_misimplementation (confidence: ?)

> **Paper says:** Section 3.1: Definitions of Average number of Active Patches (AAP), Average Jaccard Similarity of Active Patches with Teacher (AJS), and Prototype Matching Score (PMS)

**Code:** `src/services/_evaluate/protopnet_evaluate.py: evaluate method (when calc_pm=False)`

The paper defines AAP and AJS in terms of active patches (local representations), but the code computes them in terms of active prototypes (counting prototypes with at least one active patch per image and Jaccard similarity of active prototype sets), leading to a different metric.

### 2. [MEDIUM] single_seed_reporting (confidence: ?)

> **Paper says:** Experimental results in Tables 2-5 report single numbers without mention of multiple runs or variance.

**Code:** `main.py: lines 20-25 (setting seed from config) and lack of multiple seeds in paper.`

The paper reports results from a single random seed without mentioning multiple runs or variance, making it impossible to assess the robustness of the results.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 2 unconfirmed, 0 missed. F1 = 0.0.</sub>

---

## G-Mixup: Graph Data Augmentation for Graph Classification

**Paper:** arXiv:2202.07179  
**Repo:** https://github.com/ahxt/g-mixup @ `0813bb3231`

**2 findings**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] other (confidence: ?)

> **Paper says:** Appendix F.1: Note that best test epoch is selected on a validation set, and we report the test accuracy on ten runs.

**Code:** `src/gmixup.py: absence of validation loop during training (only training loss is computed and optimizer stepped)`

The paper states that the best epoch is selected using the validation set, but the provided code does not evaluate on the validation set during training to pick the best model. Instead, it trains for a fixed number of epochs and evaluates on the test set at the last epoch, which may lead to overfitting and inflated test performance.

### 2. [MEDIUM] other (confidence: ?)

> **Paper says:** Appendix F.1: ... we report the test accuracy on ten runs.

**Code:** `src/gmixup.py: torch.manual_seed(seed); random.seed(seed)`

The paper reports results averaged over ten runs, but the code only executes a single run per invocation. To reproduce the paper's results, the user must run the code ten times with different seeds and average the results.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 2 unconfirmed, 5 missed. F1 = 0.0.</sub>

---

## The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Paper:** arXiv:1803.03635  
**Repo:** https://github.com/google-research/lottery-ticket-hypothesis @ `1f17279d28`

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [HIGH] test_set_contamination (confidence: ?)

> **Paper says:** Section 2: 'We randomly sampled a 5,000-example validation set from the training set and used the remaining 55,000 training examples as our training set for the rest of the paper (including Section 2).'

**Code:** `mnist_fc/lottery_experiment.py
 datasets/dataset_mnist.py`

The paper states that a validation set is used to determine the iteration of minimum validation loss for early stopping, but the code does not create or use a validation set. Instead, the code logs test set performance at regular intervals, which would be used to select the best iteration, leading to test set contamination.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 1 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.6667.</sub>

---

## Network Deconvolution

**Paper:** arXiv:1905.11926  
**Repo:** https://github.com/yechengxi/deconvolution @ `d221c221b7`

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [MEDIUM] single_seed_reporting (confidence: ?)

> **Paper says:** The paper does not mention the random seed used for experiments or multiple runs to account for variance.

**Code:** `arg_parser.py`

The paper reports results from a single run without specifying the seed or conducting multiple runs, making it difficult to assess the variability of the results.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 1 unconfirmed, 1 missed. F1 = 0.0.</sub>

---

## BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning

**Paper:** arXiv:2305.04769  
**Repo:** https://github.com/NeurAI-Lab/BiRT @ `1340a97bff`

> ⚠️ **This run hit an error and may be incomplete:** LLM call failed: ReadTimeout: HTTPSConnectionPool(host='integrate.api.nvidia.com', port=443): Read timed out. (read timeout=240.0)

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 2 missed. F1 = 0.0.</sub>

---

## Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations

**Paper:** arXiv:2110.12088  
**Repo:** https://github.com/UCSC-REAL/cifar-10-100n @ `49df7d8a69`

> ⚠️ **This run hit an error and may be incomplete:** Could not parse JSON from model response (first 300 chars): '[\n  {\n    "failure_mode": "other",\n    "severity": "medium",\n    "paper_evidence": "Top 3 performances are highlighted in bold (mean ± standard deviation of 5 runs).",\n    "code_evidence": "main.py",\n    "explanation": The paper repo

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
---

## [SYNTHETIC FIXTURE] Not a real paper

**Paper:** synthetic fixture, not a real paper  
**Repo:** local:data/synthetic/synthetic_multi_defect (constructed by RepoRaven, no external URL)

> ⚠️ **This run hit an error and may be incomplete:** Could not parse JSON from model response (first 300 chars): '[\n  {\n    "failure_mode": "test_set_contamination",\n    "severity": "high",\n    "paper_evidence": "All hyperparameter selection is performed on a validation split; the test set is touched exactly once, for final reporting.",\n    "code

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 3 missed. F1 = 0.0.</sub>

---

## Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints

**Paper:** arXiv:2311.08675  
**Repo:** https://github.com/xiaoboxia/LBCS @ `338a31dc5a`

**1 finding**, ordered by how much they're worth your time:

*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*

### 1. [MEDIUM] single_seed_reporting (confidence: ?)

> **Paper says:** Section 5 Experiments: 'All experiments are repeated 20 times on NVIDIA GTX3090 GPUs with PyTorch.' (Table 1) and 'All experiments are repeated ten times on NVIDIA GTX3090 GPUs with PyTorch.' (Tables 2 and 3).

**Code:** `run_fashion.py: no loop over seeds; only a single call to optimize() after parsing arguments.`

The paper claims results are averaged over multiple runs (10-20 times), but the provided code executes a single optimization run without averaging over multiple seeds, potentially hiding variability and undermining reproducibility.

<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 1 confirmed against known ground truth, 0 unconfirmed, 0 missed. F1 = 1.0.</sub>

---

## Sample-specific Masks for Visual Reprogramming

**Paper:** arXiv:2406.03150  
**Repo:** https://github.com/tmlr-group/SMM @ `eeac33afc4`

> ⚠️ **This run hit an error and may be incomplete:** LLM call failed: Exception: [503] {'message': 'Service temporarily overloaded', 'type': 'Service Unavailable', 'code': 503}
{'error': {'message': 'Service temporarily overloaded', 'type': 'Service Unavailable', 'code': 503}, 'status': 503}

**No findings.** Nothing in this repo contradicted a claim we could check against the paper.

This is not the same as a guarantee of correctness — it means our specific checks (hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. See the README's human-review note before treating this as a clean bill of health.
<sub>Scoring (for eval purposes, not shown to end users in a real deployment): 0 confirmed against known ground truth, 0 unconfirmed, 1 missed. F1 = 0.0.</sub>

---
