"""
Utilities for plotting learned Contextualized models.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scContextualized import utils


def simple_plot(
    x_vals,
    y_vals,
    **kwargs,
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
    plt.figure(figsize=kwargs.get('figsize', (8, 8)))
    if 'y_lowers' in kwargs and 'y_uppers' in kwargs:
        plt.fill_between(
            x_vals,
            np.squeeze(kwargs['y_lowers']),
            np.squeeze(kwargs['y_uppers']),
            alpha=kwargs.get("fill_alpha", 0.2),
            color=kwargs.get("fill_color", "blue"),
        )
    plt.plot(x_vals, y_vals)
    plt.xlabel(kwargs.get('x_label', 'X'))
    plt.ylabel(kwargs.get('y_label', 'Y'))
    if 'x_ticks' in kwargs and 'x_ticklabels' in kwargs:
        plt.xticks(kwargs['x_ticks'], kwargs['x_ticklabels'])
    if 'y_ticks' in kwargs and 'y_ticklabels' in kwargs:
        plt.yticks(kwargs['y_ticks'], kwargs['y_ticklabels'])
    plt.show()


def plot_embedding_for_all_covars(
    reps, covars_df, covars_stds=None, covars_means=None, covars_encoders=None, **kwargs
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
        if kwargs.get("dithering_pct", 0.0) > 0:
            reps[:, 0] += np.random.normal(
                0, kwargs["dithering_pct"] * np.std(reps[:, 0]), size=reps[:, 0].shape
            )
            reps[:, 1] += np.random.normal(
                0, kwargs["dithering_pct"] * np.std(reps[:, 1]), size=reps[:, 1].shape
            )
        try:
            plot_lowdim_rep(
                reps[:, :2],
                my_labels,
                cbar_label=covar,
                min_samples=kwargs.get("min_samples", 0),
                **kwargs,
            )
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
    classification=True,
    verbose=True,
    **kwargs,
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
    if verbose:
        print("Estimating Homogeneous Contextual Effects.")
        if classification:
            print(
                """Assuming classification and exponentiating odds ratios.
            If this is wrong, use classification=False parameter."""
            )
    if C_vis is None:
        if verbose:
            print(
                """Generating visualizing datapoints by assuming the encoder is
            an additive model and thus doesn't require sampling on a manifold.
            If the encoder has interactions, please supply C_vis so that we
            can visualize these effects on the correct data manifold."""
            )
        C_vis = make_C_vis(C, n_vis)
    for j in range(C.shape[1]):
        vals_to_plot = np.linspace(
            np.min(C.values[:, j]), np.max(C.values[:, j]), n_vis
        )
        C_j = C_vis.copy()
        C_j[:, :j] = 0.0
        C_j[:, j + 1 :] = 0.0
        try:
            (_, mus) = predict_params(utils.prepend_zero(C_j), individual_preds=True)
            mus = np.squeeze(mus[:, 1:])
            means = np.mean(mus, axis=0)  # Homogeneous Effects
            lowers = np.percentile(mus, kwargs.get("lower_pct", 2.5), axis=0)
            uppers = np.percentile(mus, kwargs.get("upper_pct", 97.5), axis=0)
        except:
            (_, mus) = predict_params(
                utils.prepend_zero(C_j),
            )
            means = np.squeeze(mus[1:])  # Homogeneous Effects
            lowers, uppers = None, None

        effect = means - np.min(means)
        if lowers is not None and uppers is not None:
            lowers -= np.min(means)
            uppers -= np.min(means)
        if classification:
            effect = np.exp(effect)
            if lowers is not None and uppers is not None:
                lowers = np.exp(lowers)
                uppers = np.exp(uppers)
        try:
            x_classes = encoders[j].classes_
            # Line up class values with centered values.
            x_ticks = (np.array(list(range(len(x_classes)))) - C_means[j]) / C_stds[j]
        except:
            x_classes = None
            x_ticks = None

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
    predict_params,
    C,
    X,
    X_names,
    ylabel="Odds Ratio of Outcome",
    min_effect_size=1.1,
    classification=True,
    **kwargs,
):
    """

    :param predict_params:
    :param C:
    :param X:
    :param X_names:
    :param ylabel:  (Default value = "Odds Ratio of Outcome")
    :param min_effect_size:  (Default value = 1.1)

    """
    C_vis = np.zeros_like(C.values)
    X_vis = make_grid_mat(X, 1000)
    (betas, _) = predict_params(
        C_vis, individual_preds=True
    )  # boostraps x C_vis x outcomes x predictors
    homogeneous_betas = np.mean(betas, axis=1)  # bootstraps x outcomes x predictors
    for outcome in range(homogeneous_betas.shape[1]):
        betas = homogeneous_betas[:, outcome, :]  # bootstraps x predictors
        my_avg_betas = np.mean(betas, axis=0)
        lowers = np.percentile(betas, kwargs.get("lower_pct", 2.5), axis=0)
        uppers = np.percentile(betas, kwargs.get("upper_pct", 97.5), axis=0)
        max_impacts = []
        # Calculate the max impact of each effect.
        for k in range(my_avg_betas.shape[0]):
            effect = my_avg_betas[k] * X_vis[:, k]
            effect_range = np.max(effect) - np.min(effect)
            if classification:
                effect_range = np.exp(effect_range)
            max_impacts.append(effect_range)
        effects_by_desc_impact = np.argsort(max_impacts)[::-1]

        # Plot Booleans all together.
        # TODO: Share this code with the context plotting.
        boolean_vars = [j for j in range(X.shape[-1]) if len(set(X[:, j])) == 2]
        if len(boolean_vars) > 0:
            impacts = [max_impacts[j] for j in boolean_vars]
            names = [X_names[j] for j in boolean_vars]
            plt.figure(figsize=kwargs.get('figsize', (12, 8)))
            sorted_i = np.argsort(impacts)
            upper_impact = np.max([0, np.max(uppers[i]) - impacts[i]])
            if classification:
                upper_impact = np.exp(upper_impact)
            for counter, i in enumerate(sorted_i):
                plt.bar(
                    counter,
                    impacts[i],
                    width=0.5,
                    color="blue",
                    edgecolor="black",
                    yerr=upper_impact,
                )  # Assume symmetric error.
            plt.xticks(
                range(len(names)),
                np.array(names)[sorted_i],
                rotation=60,
                fontsize=kwargs.get('boolean_x_ticksize', 18),
                ha="right",
            )
            plt.ylabel(
                kwargs.get("ylabel", "Odds Ratio of Outcome"),
                fontsize=kwargs.get("ylabel_fontsize", 32),
            )
            plt.yticks(fontsize=kwargs.get("ytick_fontsize", 18))
            if kwargs.get("bool_figname", None) is not None:
                plt.savefig(kwargs.get("bool_figname"), dpi=300, bbox_inches="tight")
            else:
                plt.show()

        for k in effects_by_desc_impact:
            if k in boolean_vars:
                continue
            effect = my_avg_betas[k] * X_vis[:, k]
            my_min = np.min(effect)
            effect -= my_min
            if classification:
                effect = np.exp(effect)
            my_lowers = lowers[k] * X_vis[:, k] - my_min
            my_uppers = uppers[k] * X_vis[:, k] - my_min
            if classification:
                my_lowers = np.exp(my_lowers)
                my_uppers = np.exp(my_uppers)
            if np.max(effect) > min_effect_size:
                simple_plot(
                    X_vis[:, k],
                    effect,
                    x_label=f"{kwargs.get('xlabel_prefix', 'Expression of')} {X_names[k]}",
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
    min_effect_size=0.003,
    n_vis=1000,
    max_classes_for_discrete=10,
    **kwargs,
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
    C_names = C.columns.tolist()
    for j in range(C.shape[1]):
        C_j = C_vis.copy()
        C_j[:, :j] = 0.0
        C_j[:, j + 1 :] = 0.0
        (models, mus) = predict_params(
            C_j, individual_preds=True
        )  # n_bootstraps x n_vis x outcomes x predictors
        homogeneous_effects = np.mean(
            models, axis=1
        )  # n_bootstraps x outcomes x predictors
        heterogeneous_effects = models.copy()
        for i in range(n_vis):
            heterogeneous_effects[:, i] -= homogeneous_effects
        # n_bootstraps x n_vis x outcomes x predictors

        for outcome in range(heterogeneous_effects.shape[2]):
            my_effects = heterogeneous_effects[
                :, :, outcome, :
            ]  # n_bootstraps x n_vis x predictors
            means = np.mean(my_effects, axis=0)  # n_vis x predictors
            my_lowers = np.percentile(my_effects, kwargs.get("lower_pct", 2.5), axis=0)
            my_uppers = np.percentile(my_effects, kwargs.get("upper_pct", 97.5), axis=0)

            x_ticks = None
            x_ticklabels = None
            try:
                x_classes = encoders[j].classes_
                if len(x_classes) <= max_classes_for_discrete:
                    x_ticks = (
                        np.array(list(range(len(x_classes)))) - C_means[j]
                    ) / C_stds[j]
                    x_ticklabels = x_classes
            except:
                pass
            for k in range(my_effects.shape[-1]):
                if np.max(heterogeneous_effects[:, k]) > min_effect_size:
                    simple_plot(
                        C_vis[:, j],
                        means[:, k],
                        x_label=C_names[j],
                        y_label=f"{kwargs.get('y_prefix', 'Influence of')} {X_names[k]}",
                        y_lowers=my_lowers[:, k],
                        y_uppers=my_uppers[:, k],
                        x_ticks=x_ticks,
                        x_ticklabels=x_ticklabels,
                        **kwargs,
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
    **kwargs,
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

    if len(set(labels)) < kwargs.get("max_classes_for_discrete", 10):  # discrete labels
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
        ax.scatter(
            low_dim[good_idxs, 0],
            low_dim[good_idxs, 1],
            c=tag,
            alpha=kwargs.get("alpha", 1.0),
            s=100,
            cmap=cmap,
            norm=norm,
        )
    else:
        ax.scatter(
            low_dim[:, 0],
            low_dim[:, 1],
            c=labels,
            alpha=kwargs.get("alpha", 1.0),
            s=100,
            cmap=cmap,
        )
    plt.xlabel(xlabel, fontsize=kwargs.get("xlabel_fontsize", 48))
    plt.ylabel(ylabel, fontsize=kwargs.get("ylabel_fontsize", 48))
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=kwargs.get("title_fontsize", 52))

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    if discrete:
        color_bar = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm,
            spacing="proportional",
            ticks=bounds[:-1] + 0.5,  # boundaries=bounds,
            format="%1i",
        )
        try:
            color_bar.ax.set(
                yticks=bounds[:-1] + 0.5, yticklabels=np.round(tag_names)
            )  # , fontsize=24)
        except:
            color_bar.ax.set(yticks=bounds[:-1] + 0.5, yticklabels=tag_names)
    else:
        color_bar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, format="%.1f")
        # cb.ax.set_yticklabels(fontsize=24)
    if cbar_label is not None:
        color_bar.ax.set_ylabel(cbar_label, fontsize=kwargs.get('cbar_fontsize', 32))
    if figname is not None:
        plt.savefig("results/{}.pdf".format(figname), dpi=300, bbox_inches="tight")
