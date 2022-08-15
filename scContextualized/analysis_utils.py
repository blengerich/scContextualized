"""
Utilities for post-hoc analysis of trained Contextualized models.
"""

import numpy as np
from sklearn.metrics import roc_auc_score as roc


def print_acc_by_covars(
    Y_train,
    train_preds,
    Y_test,
    test_preds,
    covar_df,
    train_idx,
    test_idx,
    covar_stds=None,
    covar_means=None,
    covar_encoders=None,
):
    """

    :param Y_train:
    :param train_preds:
    :param Y_test:
    :param test_preds:
    :param covar_df:
    :param train_idx:
    :param test_idx:
    :param covar_stds:  (Default value = None)
    :param covar_means:  (Default value = None)
    :param covar_encoders:  (Default value = None)

    """
    for i, covar in enumerate(covar_df.columns):
        my_labels = covar_df.values[:, i]
        if covar_stds is not None:
            my_labels *= covar_stds[i]
        if covar_means is not None:
            my_labels += covar_means[i]
        try:
            my_labels = covar_encoders[i].inverse_transform(my_labels.astype(int))
        except:
            pass
        if len(set(my_labels)) > 20:
            continue
        print("=" * 20)
        print(covar)
        print("-" * 10)
        for label in sorted(set(my_labels)):
            try:
                train_roc = roc(
                    Y_train[my_labels[train_idx] == label],
                    np.squeeze(train_preds)[my_labels[train_idx] == label],
                )
            except (IndexError, ValueError):
                train_roc = np.nan
            try:
                test_roc = roc(
                    Y_test[my_labels[test_idx] == label],
                    np.squeeze(test_preds)[my_labels[test_idx] == label],
                )
            except (IndexError, ValueError):
                test_roc = np.nan
            print(
                "{}:\t Train ROC: {:.2f}, Test ROC: {:.2f}".format(
                    label, train_roc, test_roc
                )
            )
        print("=" * 20)


"""
def add_simulated_effect():
    """ """
    #


def test_recovery_of_context_effects():
    """ """
    # TODO: test the recovery of simulated effects

    C

def test_recovery_of_homogeneous_effects():
    """ """
    # For n iterations,
    # For each effect size
    # Make plot of recoveries and return streng
    # TODO: Recovery depends on strength of effect?


def test_recovery_of_heterogeneous_effects():
    """ """
    # TODO: Recovery deends on strength of effect and distribution of C? C, X must be similar.
"""
