"""
Utilities for plotting learned Contextualized models.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scContextualized import utils


def plot_embedding_for_all_covars(
    reps, covars_df, covars_stds=None, covars_means=None, covars_encoders=None
):
    """

    :param reps:
    :param covars_df:
    :param covars_stds:  (Default value = None)
    :param covars_means:  (Default value = None)
    :param covars_encoders:  (Default value = None)

    """
    for i, covar in enumerate(covars_df.columns):
        my_labels = covars_df.iloc[:, i].values
        if covars_stds is not None:
            my_labels *= covars_stds
        if covars_means is not None:
            my_labels += covars_means
        if covars_encoders is not None:
            my_labels = covars_encoders[i].inverse_transform(my_labels.astype(int))
        try:
            plot_lowdim_rep(reps[:, :2], my_labels, cbar_label=covar, min_samples=0)
        except:
            print("Error with covar {}".format(covar))


def make_grid_mat(ar, n_vis):
    """

    :param ar:
    :param n_vis:

    """
    ar_vis = np.zeros((n_vis, ar.shape[1]))
    for j in range(ar.shape[1]):
        ar_vis[:, j] = np.linspace(np.min(ar[:, j]), np.max(ar[:, j]), n_vis)
    return ar_vis


def make_C_vis(C, n_vis):
    """

    :param C:
    :param n_vis:

    """
    return make_grid_mat(C.values, n_vis)


def simple_plot(
    xs,
    ys,
    x_label,
    y_label,
    y_lowers=None,
    y_uppers=None,
    x_ticks=None,
    x_ticklabels=None,
    y_ticks=None,
    y_ticklabels=None,
):
    """

    :param xs:
    :param ys:
    :param x_label:
    :param y_label:
    :param y_lowers:  (Default value = None)
    :param y_uppers:  (Default value = None)
    :param x_ticks:  (Default value = None)
    :param x_ticklabels:  (Default value = None)
    :param y_ticks:  (Default value = None)
    :param y_ticklabels:  (Default value = None)

    """
    fig = plt.figure()
    if y_lowers is not None and y_uppers is not None:
        plt.fill_between(xs, np.squeeze(y_lowers), np.squeeze(y_uppers), alpha=0.2)
    plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_ticks is not None:
        plt.xticks(x_ticks, x_ticklabels)
    if y_ticks is not None:
        plt.yticks(y_ticks, y_ticklabels)
    plt.show()


def plot_homogeneous_context(
    predict_params,
    C,
    encoders,
    C_means,
    C_stds,
    ylabel="Odds Ratio of Outcome",
    C_vis=None,
    n_vis=1000,
    min_effect_size=1.1,
):
    """

    :param predict_params:
    :param C:
    :param encoders:
    :param C_means:
    :param C_stds:
    :param ylabel:  (Default value = "Odds Ratio of Outcome")
    :param C_vis:  (Default value = None)
    :param n_vis:  (Default value = 1000)
    :param min_effect_size:  (Default value = 1.1)

    """
    print("Estimating Homogeneous Contextual Effects.")
    if C_vis is None:
        print(
            """Generating visualizing datapoints by assuming the encoder is
            an additive model and thus doesn't require sampling on a manifold.
            If the encoder has interactions, please supply C_vis so that we
            can visualize these effects on the correct data manifold."""
        )
        C_vis = make_C_vis(C, n_vis)
    for j in range(C.shape[1]):
        vals_to_plot = np.linspace(np.min(C.values[:, j]), np.max(C.values[:, j]), 1000)
        C_j = C_vis.copy()
        C_j[:, :j] = 0.0
        C_j[:, j + 1 :] = 0.0
        try:
            (_, mus) = predict_params(utils.prepend_zero(C_j), individual_preds=True)
            means = np.squeeze(np.mean(mus[:, 1:]))  # Homogeneous Effects
            lowers = np.percentile(mus[:, 1:], 2.5, axis=0)
            uppers = np.percentile(mus[:, 1:], 97.5, axis=0)
        except:
            (_, mus) = predict_params(
                utils.prepend_zero(C_j),
            )
            means = np.squeeze(mus[1:])  # Homogeneous Effects
            lowers, uppers = None, None
        effect = np.exp(means - np.min(means))
        try:
            lowers -= np.min(means)
            uppers -= np.min(means)
        except:
            pass
        x_classes = encoders[j].classes_
        x_ticks = (np.array(list(range(len(x_classes)))) - C_means[j]) / C_stds[j]

        if np.max(effect) > min_effect_size:
            simple_plot(
                vals_to_plot,
                effect,
                x_label=C.columns.tolist()[j],
                y_label=ylabel,
                y_lowers=lowers,
                y_uppers=uppers,
                x_ticks=x_ticks,
                x_ticklabels=x_classes,
            )


def plot_homogeneous_tx(
    predict_params, C, X, X_names, ylabel="Odds Ratio of Outcome", min_effect_size=1.1
):
    """

    :param predict_params:
    :param C:
    :param X:
    :param X_names:
    :param ylabel:  (Default value = "Odds Ratio of Outcome")
    :param min_effect_size:  (Default value = 1.1)

    """
    # TODO: Barchart?
    C_vis = np.zeros_like(C.values)
    X_vis = make_grid_mat(X, 1000)
    (models, mus) = predict_params(
        utils.prepend_zero(C_vis),
    )
    models = np.squeeze(models[1:])  # Heterogeneous Effects
    homogeneous_tx_effects = np.mean(models, axis=0)
    try:
        (models, mus) = predict_params(utils.prepend_zero(C_vis), individual_preds=True)
        models = np.squeeze(models[:, 1:])  # Heterogeneous Effects
        lowers = np.percentile(np.mean(models, axis=1), 2.5, axis=0)
        uppers = np.percentile(np.mean(models, axis=1), 97.5, axis=0)
    except:
        pass
    effects = []
    for k in range(models.shape[-1]):
        effect = homogeneous_tx_effects[k] * X_vis[:, k]
        effect -= np.min(effect)
        effects.append(np.max(effect))
    for (k, _) in reversed(
        sorted(enumerate(effects), key=lambda x: x[1])
    ):  # order by decreasing impact
        effect = homogeneous_tx_effects[k] * X_vis[:, k]
        my_min = np.min(effect)
        effect -= my_min
        effect = np.exp(effect)
        try:
            my_lowers = lowers[k] * X_vis[:, k]
            my_uppers = uppers[k] * X_vis[:, k]
            my_lowers = np.exp(my_lowers - my_min)
            my_uppers = np.exp(my_uppers - my_min)
        except:
            my_lowers, my_uppers = None, None
        if np.max(effect) > min_effect_size:
            simple_plot(
                X_vis[:, k],
                effect,
                x_label="Expression of {}".format(X_names[k]),
                y_label=ylabel,
                y_lowers=my_lowers,
                y_uppers=my_uppers,
            )


def plot_heterogeneous(
    predict_params,
    C,
    X,
    encoders,
    C_means,
    C_stds,
    X_names,
    ylabel="Influence of ",
    min_effect_size=0.003,
    n_vis=1000,
    max_classes_for_discrete=10,
):
    """

    :param predict_params:
    :param C:
    :param X:
    :param encoders:
    :param C_means:
    :param C_stds:
    :param X_names:
    :param ylabel:  (Default value = "Influence of ")
    :param min_effect_size:  (Default value = 0.003)
    :param n_vis:  (Default value = 1000)
    :param max_classes_for_discrete:  (Default value = 10)

    """

    C_vis = make_C_vis(C, n_vis)
    for j in range(C.shape[1]):
        C_j = C_vis.copy()
        C_j[:, :j] = 0.0
        C_j[:, j + 1 :] = 0.0
        (models, mus) = predict_params(
            utils.prepend_zero(C_j),
        )
        models = np.squeeze(models[1:])  # Heterogeneous Effects
        heterogeneous_effects = models - np.mean(models, axis=0)

        try:
            (models, mus) = predict_params(
                utils.prepend_zero(C_j), individual_preds=True
            )
            models = np.squeeze(models[:, 1:])  # Heterogeneous Effects
            means = np.mean(models, axis=1)
            my_lowers = np.percentile(models - means, 2.5, axis=0)
            my_uppers = np.percentile(models - means, 97.5, axis=0)
        except:
            my_lowers, my_uppers = None, None

        # TODO: Fix for continuous-valued C.
        x_ticks = None
        x_ticklabels = None
        try:
            x_classes = encoders[j].classes_
            if len(x_classes) <= max_classes_for_discrete:
                x_ticks = (np.array(list(range(len(x_classes)))) - C_means[j]) / C_stds[
                    j
                ]
                x_ticklabels = x_classes
        except:
            pass
        for k in range(heterogeneous_effects.shape[1]):
            try:
                my_lowers, my_uppers = lowers[:, k], uppers[:, k]
            except:
                my_lowers, my_uppers = None, None
            if np.max(heterogeneous_effects[:, k]) > min_effect_size:
                simple_plot(
                    C_vis[:, j],
                    heterogeneous_effects[:, k],
                    x_label=C.columns.tolist()[j],
                    y_label="{}{}".format(ylabel, X_names[k]),
                    y_lowers=my_lowers,
                    y_uppers=my_uppers,
                    x_ticks=x_ticks,
                    x_ticklabels=x_ticklabels,
                )


def plot_hallucinations(
    predict_y, X, C, Y, models, mus, compressor, target_probs=np.linspace(0.1, 0.9, 9)
):
    """

    :param predict_y:
    :param X:
    :param C:
    :param Y:
    :param models:
    :param mus:
    :param compressor:
    :param target_probs:  (Default value = np.linspace(0.1)
    :param 0.9:
    :param 9):

    """
    plt.figure(figsize=(10, 10))
    hallucinated_all = []

    for target in target_probs:
        t = utils.to_lodds(target)
        inner_prods = np.array([np.dot(X[i], models[i]) for i in range(len(models))])
        preds_lodds = inner_prods + np.squeeze(mus)
        preds = utils.sigmoid(preds_lodds)
        sensitivities = np.array([np.dot(m, m) for m in models])
        diffs = preds_lodds - t
        deltas = -diffs / sensitivities
        deltas = np.expand_dims(deltas, 1)
        hallucinated = X + deltas * models
        hallucinated_preds = predict_y(
            utils.prepend_zero(C),
            utils.prepend_zero(hallucinated),
            np.squeeze(utils.prepend_zero(Y)),
        )[1:, 0]
        dists = np.abs(
            [preds[i] - hallucinated_preds[i] for i in range(len(hallucinated_preds))]
        )
        hallucinated_small = compressor.transform(hallucinated)
        min_pred = np.min(hallucinated_preds)
        max_pred = np.max(hallucinated_preds)
        pred_range = max_pred - min_pred
        colors = np.array(
            [
                (
                    pred,
                    0,
                    1 - pred,
                )  # (pred - min_pred) / pred_range, 0, (max_pred - pred)/ pred_range)
                for pred in hallucinated_preds
            ]
        )  # red-blue scale
        alphas = 0.8 * np.array([max(0, x) for x in np.array(1 - dists)])
        hallucinated_all.append(hallucinated_small)
        plt.scatter(
            hallucinated_small[:, 0],
            hallucinated_small[:, 1],
            alpha=alphas,
            c=colors,
            s=16,
        )
    hallucinated_all = np.array(hallucinated_all)
    for i in range(hallucinated_all.shape[1]):
        plt.plot(
            hallucinated_all[:, i, 0],
            hallucinated_all[:, i, 1],
            alpha=0.2,
            color="gray",
            linestyle="--",
        )
    X_small = compressor.transform(X)
    plt.scatter(X_small[:, 0], X_small[:, 1], marker="*", alpha=1.0, s=32)
    plt.xlabel("Expression PC 1", fontsize=36)
    plt.ylabel("Expression PC 2", fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()


def plot_lowdim_rep(
    low_dim,
    labels,
    xlabel="Expression PC 1",
    ylabel="Expression PC 2",
    min_samples=100,
    figname=None,
    cbar_label=None,
    discrete=False,
    title="",
):
    """

    :param low_dim:
    :param labels:
    :param xlabel:  (Default value = "Expression PC 1")
    :param ylabel:  (Default value = "Expression PC 2")
    :param min_samples:  (Default value = 100)
    :param figname:  (Default value = None)
    :param cbar_label:  (Default value = None)
    :param discrete:  (Default value = False)
    :param title:  (Default value = "")

    """

    if len(set(labels)) < 40:  # discrete labels
        discrete = True
        cmap = plt.cm.jet  # define the colormap
    else:
        discrete = False
        tag = labels
        norm = None
        cmap = plt.cm.coolwarm
    if discrete:
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N
        )
        tag, tag_names = utils.convert_to_one_hot(labels)
        order = np.argsort(tag_names)
        tag_names = np.array(tag_names)[order]
        tag = np.array([list(order).index(int(x)) for x in tag])
        good_tags = [np.sum(tag == i) > min_samples for i in range(len(tag_names))]
        tag_names = np.array(tag_names)[good_tags]
        good_idxs = np.array([good_tags[int(tag[i])] for i in range(len(tag))])
        tag = tag[good_idxs]
        tag, _ = utils.convert_to_one_hot(tag)
        bounds = np.linspace(0, len(tag_names), len(tag_names) + 1)
        try:
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        except:
            print(
                "Not enough values for a colorbar (needs at least 2 values), quitting."
            )
            return
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # setup the plot
    if discrete:
        scat = ax.scatter(
            low_dim[good_idxs, 0],
            low_dim[good_idxs, 1],
            c=tag,
            alpha=1.0,
            s=100,
            cmap=cmap,
            norm=norm,
        )
    else:
        scat = ax.scatter(
            low_dim[:, 0], low_dim[:, 1], c=labels, alpha=1.0, s=100, cmap=cmap
        )
    plt.xlabel(xlabel, fontsize=48)
    plt.ylabel(ylabel, fontsize=48)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=52)

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    if discrete:
        cb = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm,
            spacing="proportional",
            ticks=bounds[:-1] + 0.5,  # boundaries=bounds,
            format="%1i",
        )
        try:
            cb.ax.set(
                yticks=bounds[:-1] + 0.5, yticklabels=np.round(tag_names)
            )  # , fontsize=24)
        except:
            cb.ax.set(yticks=bounds[:-1] + 0.5, yticklabels=tag_names)  # , fontsize=24)
    else:
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, format="%.1f")
        # cb.ax.set_yticklabels(fontsize=24)
    if cbar_label is not None:
        cb.ax.set_ylabel(cbar_label, fontsize=32)
    if figname is not None:
        plt.savefig("results/{}.pdf".format(figname), dpi=300, bbox_inches="tight")
