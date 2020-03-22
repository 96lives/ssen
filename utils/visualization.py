import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import io
import torch
from torchvision.transforms import ToTensor
from MulticoreTSNE import MulticoreTSNE as TSNE


def get_cmap():
    return np.array(list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()))


def plt_to_tensor(plt, clear=True):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    if clear: plt.clf()
    buf.seek(0)
    img = PIL.Image.open(buf)
    return ToTensor()(img)


def visualize_embs(embs, labels, remove_bg, max_sample, num_view, axis_range=None):
    '''
    Args
        embs:
            (cpu, float) Tensor of N x 3
        labels:
            (cpu, long) Tensor of N
        remove_bg:
            whether to remove 0 labels
        max_sample:
            maximum number of samples
        num_view:
            number of views to visualize in plot
    :return:
        imgs: list of img (in tensor)
        axes_range: [x_lim, y_lim, z_lim]
            x_lim: range of img's x axes (tuple of min max)
            y_lim: range of img's y axes (tuple of min max)
            z_lim: range of img's z axes (tuple of min max)
    '''

    if remove_bg:
        not_bg_idxs = (labels != 0) \
            .nonzero().squeeze(dim=1)
        embs = embs[not_bg_idxs]
        labels = labels[not_bg_idxs]

    # if not sampling, set max_sample to -1
    if max_sample > 0:
        sample = np.random.RandomState(0)\
            .permutation(embs.shape[0])[:max_sample]
        embs = embs[sample]
        labels = labels[sample]

    colors = get_cmap()
    color = colors[np.squeeze(labels)]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if axis_range is not None:
        ax.set_xlim(axis_range[0])
        ax.set_ylim(axis_range[1])
        ax.set_zlim(axis_range[2])

    ax.scatter(
        xs=embs[:, 0], ys=embs[:, 1], zs=embs[:, 2],
        color=color, alpha=0.3, marker='o', linewidths=0
    )

    imgs = []
    for angle in range(0, 360, int(360 / num_view)):
        ax.view_init(elev=None, azim=angle)
        imgs.append(plt_to_tensor(plt, clear=False))
    axis_range = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    plt.close('all')

    return imgs, axis_range


def visualize_tsne(embs, labels, config, save=False):
    '''
    inputs:
        embs: tensor of N x D
        labels: tensor of N x 1
        max_sample: int

    returns:
        img: np.array
    '''
    not_bg_idxs = (labels != 0) \
        .nonzero().squeeze(dim=1)
    embs = embs[not_bg_idxs]
    labels = labels[not_bg_idxs]

    # sample point cloud
    n = embs.shape[0]
    sample = np.random.RandomState(0)\
        .permutation(n)[:config['max_sample']]
    embs = embs[sample]
    labels = labels[sample]

    # tsne
    tsne = TSNE(n_jobs=config['num_core'], n_iter=config['num_iter'])
    tsne_embs = tsne.fit_transform(embs.cpu().numpy())
    colors = get_cmap()
    color = colors[np.squeeze(labels)]
    plt.scatter(
        x=tsne_embs[:, 0], y=tsne_embs[:, 1],
        color=color, alpha=0.3, marker='o', linewidths=0
    )
    if save:
        return plt
    ret = plt_to_tensor(plt)
    plt.close('all')
    return ret


def build_cmap(cmap, num_colors, fixed=False):
    # Get color map
    cmap = plt.cm.get_cmap(cmap, num_colors)
    if not fixed:
        rand_perm = torch.randperm(num_colors)
        cmap = [cmap(rand_perm[i]) for i in range(num_colors)]
    else:
        cmap = [cmap(i) for i in range(num_colors)]
    return torch.tensor(cmap)[:, :3]

