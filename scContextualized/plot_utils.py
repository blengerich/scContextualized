
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from scContextualized import utils


# Assume NGAM encoder.
def plot_homogeneous_context(C, X, Y, encoders, C_means, C_stds, ylabel="Odds Ratio of Outcome"):
    for j in range(C.shape[1]):
        vals_to_plot = np.linspace(np.min(C.values[:, j]), np.max(C.values[:, j]), 1000)
        C_vis = np.zeros((1000, C.shape[1]))
        C_vis[:, j] = vals_to_plot

        context_factor = C.columns.tolist()[j]
        all_dataloader = ncr.dataloader(utils.prepend_zero(C_vis),
                                        utils.prepend_zero(np.zeros((len(C_vis), X.shape[1]))),
                                        utils.prepend_zero(np.zeros((len(C_vis), Y.shape[1]))),
                                        batch_size=16)
        (models, mus) = trainer.predict_params(ncr, all_dataloader)
        models = np.squeeze(models[1:]) # Heterogeneous Effects
        mus = np.squeeze(mus[1:]) # Homogeneous Effects
        x_classes = encoders[j].classes_
        x_ticks = (np.array(list(range(len(x_classes)))) - C_means[j]) / C_stds[j]

        homogeneous_context_effects = mus - np.min(mus)
        if np.max(homogeneous_context_effects) > 0.01:
            fig = plt.figure()
            plt.plot(vals_to_plot, np.exp(homogeneous_context_effects))
            plt.xlabel(context_factor)
            plt.xticks(x_ticks, x_classes)
            plt.ylabel(ylabel)
            plt.show()

def plot_homogeneous_tx(C, X, Y, selected_transcript_names, ylabel="Odds Ratio of Outcome"):
    vals_to_plot = np.linspace(np.min(C.values[:, 0]), np.max(C.values[:, 0]), 1000)
    C_vis = np.zeros((1000, C.shape[1]))
    C_vis[:, 0] = vals_to_plot
    context_factor = C.columns.tolist()[0]
    all_dataloader = ncr.dataloader(utils.prepend_zero(C_vis),
                                    utils.prepend_zero(np.zeros((len(C_vis), X.shape[1]))),
                                    utils.prepend_zero(np.zeros((len(C_vis), Y.shape[1]))),
                                    batch_size=16)
    (models, mus) = trainer.predict_params(ncr, all_dataloader)
    models = np.squeeze(models[1:]) # Heterogeneous Effects
    homogeneous_tx_effects = np.mean(models, axis=0)
    effects = []
    for k in range(models.shape[1]): #, key=lambda x: homogeneous_tx_effects):
        vals_to_plot = np.linspace(np.min(X[:, k]), np.max(X[:, k]), 1000)
        my_effect = homogeneous_tx_effects[k]*vals_to_plot
        my_effect -= np.mean(my_effect)
        effects.append(np.max(my_effect))
    for (k, _) in reversed(sorted(enumerate(effects), key=lambda x: x[1])):
        vals_to_plot = np.linspace(np.min(X[:, k]), np.max(X[:, k]), 1000)
        my_effect = homogeneous_tx_effects[k]*vals_to_plot
        my_effect -= np.mean(my_effect)
        my_effect = np.exp(my_effect)
        if np.max(my_effect) > 1.1:
            fig = plt.figure()
            plt.plot(vals_to_plot, my_effect)
            plt.xlabel("Expression of {}".format(selected_transcripts_names[k]), fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.show()

def plot_heterogeneous(C, X, Y, encoders, C_means, C_stds, selected_transcripts_names, ylabel="Influence of "):
    for j in range(C.shape[1]):
        vals_to_plot = np.linspace(np.min(C.values[:, j]), np.max(C.values[:, j]), 1000)
        C_vis = np.zeros((1000, C.shape[1]))
        C_vis[:, j] = vals_to_plot

        context_factor = C.columns.tolist()[j]
        all_dataloader = ncr.dataloader(utils.prepend_zero(C_vis),
                                        utils.prepend_zero(np.zeros((len(C_vis), X.shape[1]))),
                                        utils.prepend_zero(np.zeros((len(C_vis), Y.shape[1]))),
                                        batch_size=16)
        (models, mus) = trainer.predict_params(ncr, all_dataloader)
        models = np.squeeze(models[1:]) # Heterogeneous Effects
        heterogeneous_effects = models - np.mean(models, axis=0)
        print(np.max(heterogeneous_effects))

        x_classes = encoders[j].classes_
        x_ticks = (np.array(list(range(len(x_classes)))) - C_means[j]) / C_stds[j]

        for k in range(models.shape[1]):
            if np.max(heterogeneous_effects[:, k]) > 0.003:
                fig = plt.figure()
                gene = selected_transcripts_names[k]
                plt.plot(vals_to_plot, heterogeneous_effects[:, k])
                plt.xlabel(context_factor)
                plt.xticks(x_ticks, x_classes)
                plt.ylabel("{}{}".format(ylabel, gene))
                plt.show()



def plot_hallucinations(X, C, Y, models, mus, compressor):
    plt.figure(figsize=(10, 10))
    hallucinated_all = []
    target_probs = np.linspace(0.1, 0.9, 9)

    for target in target_probs:
        t = utils.to_lodds(target)
        inner_prods = np.array([np.dot(X[i], models[i]) for i in range(len(models))])
        preds_lodds = inner_prods + np.squeeze(mus)
        sensitivities = np.array([np.dot(m, m) for m in models])
        diffs = preds_lodds - t
        deltas = -diffs / sensitivities
        deltas = np.expand_dims(deltas, 1)
        hallucinated = X + deltas*models
        # TODO: pass in predict function
        hallucinated_dataloader = ncr.dataloader(prepend_zero(C),
                                                 prepend_zero(hallucinated),
                                                 np.squeeze(prepend_zero(Y)),
                                                 batch_size=16)
        hallucinated_preds = trainer.predict_y(ncr, hallucinated_dataloader)[1:, 0]

        dists = np.abs([preds[i] - hallucinated_preds[i] for i in range(len(hallucinated_preds))])
        hallucinated_small = compressor.transform(hallucinated)
        min_pred = np.min(hallucinated_preds)
        max_pred = np.max(hallucinated_preds)
        pred_range = max_pred - min_pred
        colors = np.array([(pred, 0, 1-pred) #(pred - min_pred) / pred_range, 0, (max_pred - pred)/ pred_range)
                           for pred in hallucinated_preds]) # red-blue scale
        alphas = 0.8*np.array([max(0, x) for x in np.array(1 - dists)])
        hallucinated_all.append(hallucinated_small)
        plt.scatter(hallucinated_small[:, 0], hallucinated_small[:, 1],
                    alpha=alphas, c=colors, s=16)
    hallucinated_all = np.array(hallucinated_all)
    for i in range(hallucinated_all.shape[1]):
        plt.plot(hallucinated_all[:, i, 0], hallucinated_all[:, i, 1],
                 alpha=0.2, color='gray', linestyle='--')
    X_small = compressor.transform(X)
    plt.scatter(X_small[:, 0], X_small[:, 1], marker='*', alpha=1.0, s=32)
    plt.xlabel("Expression PC 1", fontsize=36)
    plt.ylabel("Expression PC 2", fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()


def plot_lowdim_rep(low_dim, labels, xlabel="Expression PC 1", ylabel="Expression PC 2",
    min_samples=100, figname=None, cbar_label=None, discrete=False):

    if len(set(labels)) < 40: # discrete labels
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
            'Custom cmap', cmaplist, cmap.N)
        tag, tag_names = utils.convert_to_one_hot(labels)
        order = np.argsort(tag_names)
        tag_names = np.array(tag_names)[order]
        tag = np.array([list(order).index(int(x)) for x in tag])
        good_tags = [np.sum(tag == i) > min_samples for i in range(len(tag_names))]
        tag_names = np.array(tag_names)[good_tags]
        good_idxs = np.array([good_tags[int(tag[i])] for i in range(len(tag))])
        tag = tag[good_idxs]
        tag, _ = utils.convert_to_one_hot(tag)
        bounds = np.linspace(0, len(tag_names), len(tag_names)+1)
        try:
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        except:
            print("Not enough values for a colorbar (needs at least 2 values), quitting.")
            return
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # setup the plot
    if discrete:
        scat = ax.scatter(low_dim[good_idxs, 0], low_dim[good_idxs, 1],
                      c=tag, alpha=1.0, s=100,
                      cmap=cmap, norm=norm)
    else:
        scat = ax.scatter(low_dim[:, 0], low_dim[:, 1],
                      c=labels, alpha=1.0, s=100,
                      cmap=cmap)
    plt.xlabel(xlabel, fontsize=48)
    plt.ylabel(ylabel, fontsize=48)
    plt.xticks([])
    plt.yticks([])

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    if discrete:
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
            spacing='proportional', ticks=bounds+0.5,#boundaries=bounds,
                                       format='%1i')
        #print(np.round(tag_names))
        try:
            cb.ax.set_yticklabels(np.round(tag_names), fontsize=24)
        except:
            cb.ax.set_yticklabels(tag_names, fontsize=24)
    else:
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, format='%.1f')
        #cb.ax.set_yticklabels(fontsize=24)
    if cbar_label is not None:
        cb.ax.set_ylabel(cbar_label, fontsize=32)
    if figname is not None:
        plt.savefig("results/{}.pdf".format(figname), dpi=300, bbox_inches='tight')