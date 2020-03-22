import torch
import math


def get_rot_mat(axis, angle):
    if axis == 'x':
        return torch.tensor([
            [1., 0., 0.],
            [0., torch.cos(angle), -torch.sin(angle)],
            [0., torch.sin(angle), torch.cos(angle)]
        ])
    elif axis == 'y':
        return torch.tensor([
            [torch.cos(angle), 0., -torch.sin(angle)],
            [0., 1., 0.],
            [torch.sin(angle), 0., torch.cos(angle)]
        ])
    elif axis == 'z':
        return torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0.],
            [torch.sin(angle), torch.cos(angle), 0.],
            [0., 0., 1.],
        ])
    else:
        raise ValueError('Argument has to be either x or y or z')



class Compose:
    '''
    Compose transforms
    Input:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    Output:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coords, feats):
        for t in self.transforms:
            coords, feats = t(coords, feats)
        return coords, feats

######################
# Color augmentation #
######################


class ChromaticJitter:
    '''
    Randomly jitter feat (per point operation) with Gaussian distribution
    Input:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    Output:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    '''

    def __init__(self, scale, clip):
        assert clip >= 0
        self.scale = scale
        self.clip = clip

    def __call__(self, coords, feats):
        if torch.rand(1) < 0.9:
            jitter = self.scale * torch.randn_like(feats).to(feats.device)
            jitter.clamp(-self.clip, self.clip)
            feats = feats + jitter
        return coords, feats


class ChromaticTranslation:
    '''
    Randomly translate feats
    Input:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    Output:
        coords: Tensor of N x 3
    '''
    def __init__(self, scale):
        assert scale >= 0
        self.scale = scale

    def __call__(self, coords, feats):
        if torch.rand(1) < 0.9:
            trans = (torch.rand(1, 3) - 0.5) * self.scale
            feats = feats + trans.view(1, -1).to(feats.device)
        return coords, feats

######################
# Coord Augmentation #
######################

class CoordScaleRotate:
    '''
    Randomly scale and rotate points
    uniform sample with z axis and normal sample around x axis
    theta_z_max: rotate around z-axis
    Input:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    Output:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    '''

    def __init__(self, scale_max, rot_x_scale, rot_x_clip, auto_rotate_axis):
        assert rot_x_clip >= 0.
        assert (scale_max <= 1.) and (scale_max >= 0.)
        self.scale_max = scale_max
        self.rot_x_scale = rot_x_scale
        self.rot_x_clip = rot_x_clip
        self.auto_rotate_axis = auto_rotate_axis

    def __call__(self, coords, feats):
        rot_mat = torch.eye(3)
        if self.auto_rotate_axis:
            # Sample rotation matrix w.r.t. y axis
            theta = torch.empty(1).uniform_(-1, 1)
            theta = theta * math.pi
            rot_mat = rot_mat.__matmul__(
                get_rot_mat(self.auto_rotate_axis, theta)
            )

        # Sample rotation matrix w.r.t. x axis
        theta_x = self.rot_x_scale * torch.randn(1)
        theta_x = theta_x.clamp(-self.rot_x_clip, self.rot_x_clip)
        theta_x = theta_x * math.pi
        rot_x = get_rot_mat('x', theta_x)
        rot_mat = rot_mat.__matmul__(rot_x)

        # Sample scale
        scale = torch.empty(1)\
            .uniform_(1 - self.scale_max, 1 + self.scale_max)
        rot_mat = rot_mat * scale
        coords = coords.__matmul__(rot_mat.to(coords.device))
        return coords, feats


class CoordFlip:
    '''
    Randomly flip with x or y axis
    '''
    def __init__(self):
        pass

    def __call__(self, coords, feats):
        for curr_ax in [0, 1]:
            if torch.rand(1) < 0.5:
                coords_max = torch.max(coords[:, curr_ax])
                coords[:, curr_ax] = coords_max - coords[:, curr_ax]
        return coords, feats


class CoordTranslation:
    '''
    translate point cloud
    Input:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    Output:
        coords: Tensor of N x 3
        Feats: Tensor of N x 3
    '''

    def __init__(self, max):
        assert max >= 0
        self.max = max

    def __call__(self, coords, feats):
        # Sample translation value
        translation = torch.zeros([1, 3]).uniform_(0, self.max)
        coords = coords + translation.to(coords.device)
        return coords, feats

