# Synthetic Multi-Defect Fixture

**This is not a real paper or repository.** It was constructed by the RepoRaven
project itself (not scraped, not from a real author) to provide ground-truth
positive test cases for failure modes that a time-boxed search of public
papers + ReScience C reports + GitHub issues could not source real citable
examples for: `metric_misimplementation`, `test_set_contamination`, and
`eval_on_train`.

(A fourth mode, `single_seed_reporting`, was originally planted here too, but
was retired once real instances were found by hand in two actual ICML 2024
papers sourced from PaperBench's paper list — see `ground_truth.json`'s
`lbcs` and `sample_specific_masks` entries. This script now genuinely loops
over 5 seeds and reports mean +/- std, so it no longer exhibits that bug.)

See `data/papers/synthetic_multi_defect.html` for the paired synthetic "paper
excerpt" whose claims this code deliberately contradicts. Every bug here is
intentional and documented in `data/ground_truth.json` under
`"synthetic_multi_defect"` with `"synthetic": true`.

Do not treat this as a real training pipeline — it uses `sklearn`'s built-in
iris/wine datasets purely as a vehicle to demonstrate each bug pattern in
runnable, syntactically valid Python that a static-analysis checker (AST/grep
tools built in later phases) can actually parse and flag.
