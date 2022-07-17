from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch


@functional_transform('remove_p_offset')
class PressureOffset(BaseTransform):

    def __call__(self, data):
        means = data.y[:, 0].mean()
        data.y[:, 0] -= means

        return data


@functional_transform('dim_less')
class DimLess(BaseTransform):

    def __init__(self, vel_upstream):
        '''input magnitude of upstream velocity |v|'''
        self.v = vel_upstream

    def __call__(self, data):

        data.x[:, 0] /= 2      ## norm x coordinate
        data.x[:, 1] /= 2      ## norm y coordinate

        data.x[:, 2] /= self.v  ## v_x
        data.x[:, 3] /= self.v  ## v_y

        data.y[:, 0] /= 1000 * self.v ** 2  ## Pressre dimless
        data.y[:, 1] /= self.v  ## u_x
        data.y[:, 2] /= self.v  ## u_y

        return data


def min_max(g):
    min_v_x = torch.min(g.x[:, 2])
    max_v_x = torch.max(g.x[:, 2])

    min_v_y = torch.min(g.x[:, 3])
    max_v_y = torch.max(g.x[:, 3])

    min_P = torch.min(g.y[:, 0])
    max_P = torch.max(g.y[:, 0])

    min_u_x = torch.min(g.y[:, 1])
    max_u_x = torch.max(g.y[:, 1])

    min_u_y = torch.min(g.y[:, 2])
    max_u_y = torch.max(g.y[:, 2])

    return torch.tensor([min_v_x, max_v_x, min_v_y, max_v_y, min_P, max_P, min_u_x, max_u_x, min_u_y, max_u_y])


def scale(x, min_x, max_x):
    return 2 * (x - min_x) / (max_x - min_x) - 1


@functional_transform('Normalize')
class Normalize(BaseTransform):
    def __init__(self, min_max_vector):
        '''input min_max of data'''
        self.min_max_vector = min_max_vector

    def __call__(self, data):
        data.x[:, 2] = scale(data.x[:, 2], self.min_max_vector[0], self.min_max_vector[1])
        data.x[:, 3] = scale(data.x[:, 3], self.min_max_vector[2], self.min_max_vector[3])

        data.y[:, 0] = scale(data.y[:, 0], self.min_max_vector[4], self.min_max_vector[5])
        data.y[:, 1] = scale(data.y[:, 1], self.min_max_vector[6], self.min_max_vector[7])
        data.y[:, 2] = scale(data.y[:, 2], self.min_max_vector[8], self.min_max_vector[9])

        return data
