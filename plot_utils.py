
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

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