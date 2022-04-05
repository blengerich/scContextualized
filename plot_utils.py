
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

import math
def sigmoid(x, slope=1):
    return 1. / (1. + np.exp(-slope*x))
def inv_sigmoid(x, slope=1):
    return (1./slope)*np.log(x / (1-x))
def to_lodds(x):
    return -np.log(1./x - 1)

def plot_hallucinations(X, C, Y, models, mus, compressor):
    plt.figure(figsize=(10, 10))
    hallucinated_all = []
    target_probs = np.linspace(0.1, 0.9, 9)

    for target in target_probs:
        t = to_lodds(target)
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