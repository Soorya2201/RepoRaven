"""Synthetic training script — deliberately contains 3 planted bugs.

See README.md: this fixture is constructed by RepoRaven for eval coverage,
not a real paper's code. Each bug below is cited by exact line number in
data/ground_truth.json.

Note: an earlier version of this fixture also planted a single_seed_reporting
bug (one hardcoded seed despite a paper claim of multi-seed averaging). That
finding was retired once real-world instances were found and verified by hand
in two actual ICML 2024 papers (LBCS, sample_specific_masks) — see
ground_truth.json's "lbcs" and "sample_specific_masks" entries. This script
now genuinely loops over multiple seeds and reports mean +/- std, so it no
longer exhibits that bug (keeping it would create a false positive: a correct
checker would flag it, but ground_truth would no longer list it as expected).
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

SEEDS = [42, 43, 44, 45, 46]  # genuinely averaged over 5 seeds, see main()


def load_split(seed):
    X, y = load_wine(return_X_y=True)
    # Reduced to a 2-class subset so sklearn's f1_score default ('binary')
    # runs without raising -- it silently scores only the positive class
    # instead of the macro-average the paper claims, rather than crashing.
    mask = y != 2
    X, y = X[mask], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def train_with_early_stopping(X_train, y_train, X_test, y_test, seed):
    """Select the best training epoch by watching held-out performance."""
    best_test_acc = -1.0
    best_model = None
    for max_iter in [10, 50, 100, 200, 500]:
        model = LogisticRegression(max_iter=max_iter, random_state=seed)
        model.fit(X_train, y_train)
        # BUG (test_set_contamination): model selection (choosing max_iter)
        # is driven by accuracy on X_test/y_test inside the training loop,
        # not a held-out validation split. The test set influences which
        # model configuration is kept, so the final reported test score is
        # no longer an unbiased estimate.
        test_acc = accuracy_score(y_test, model.predict(X_test))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model
    return best_model


def evaluate(model, X, y):
    preds = model.predict(X)
    # BUG (metric_misimplementation): paper claims "macro-averaged F1 score"
    # as the reported metric; f1_score() is called here with no `average`
    # argument, so sklearn defaults to average='binary', which is wrong for
    # this 3-class problem (raises/misreports rather than computing F1-macro).
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    return {"f1": f1, "accuracy": acc}


def run_once(seed):
    X_train, X_test, y_train, y_test = load_split(seed)
    model = train_with_early_stopping(X_train, y_train, X_test, y_test, seed)

    train_metrics = evaluate(model, X_train, y_train)
    # BUG (eval_on_train): the line below is labeled "Final test set
    # evaluation" but passes X_train/y_train instead of X_test/y_test, so the
    # reported "test" numbers are actually train-set numbers.
    test_metrics = evaluate(model, X_train, y_train)  # should be X_test, y_test
    return train_metrics, test_metrics


def main():
    all_test_f1 = []
    all_test_acc = []
    for seed in SEEDS:
        _, test_metrics = run_once(seed)
        all_test_f1.append(test_metrics["f1"])
        all_test_acc.append(test_metrics["accuracy"])

    print(f"Test F1 (mean +/- std over {len(SEEDS)} seeds): "
          f"{np.mean(all_test_f1):.4f} +/- {np.std(all_test_f1):.4f}")
    print(f"Test accuracy (mean +/- std over {len(SEEDS)} seeds): "
          f"{np.mean(all_test_acc):.4f} +/- {np.std(all_test_acc):.4f}")


if __name__ == "__main__":
    main()
