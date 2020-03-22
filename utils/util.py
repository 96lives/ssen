import torch
import numpy as np
# code from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065


def pairwise_distance(x: torch.Tensor, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def build_block_matrix(block_partitions):
    '''
    Input:
        partition_idxs: list block partitions
    out:
        tensor of sum(block_partitions) x sum(block_partitions)

    Ex)
    >> build_one_block_matrix([1, 2, 3])
       torch.tensor(
           [1, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
        )
    '''
    n = sum(block_partitions)
    out = torch.zeros([n, n])

    start = 0
    for partition in block_partitions:
        end = start + partition
        out[start: end, start:end] += torch.ones([partition, partition])
        start = end

    return out


def compute_ious(seeds, embs, inst_idxs, thres):
    '''
    Input:
        seeds: List of tuple (emb, inst)
        embs: Embedding vectors of Tensor of N x D
        inst_idxs: List of tensors with indices of each instances
        thres: embedding threshold
    Output:
        ious: List of floats
    '''
    try:
        ious = []
        for (emb, inst) in seeds:
            diff = (emb.unsqueeze(dim=0) - embs).norm(dim=1)
            valid_thres_idxs = (diff < thres).nonzero().cpu()
            single_inst_idxs = inst_idxs[inst].cpu()
            intersection = len(np.intersect1d(valid_thres_idxs, single_inst_idxs))
            union = len(np.union1d(valid_thres_idxs, single_inst_idxs))
            iou = float(intersection) / union
            ious.append(iou)
        return ious
    except IndexError:
        __import__('pdb').set_trace()

if __name__ == '__main__':
    out = build_block_matrix([1, 2, 3])
    print(out)
