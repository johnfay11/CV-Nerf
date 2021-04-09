import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as func


"""
From paper:

We encourage the representation to be multiview consistent by restricting
the network to predict the volume density σ as a function of only the location
x, while allowing the RGB color c to be predicted as a function of both location
and viewing direction. To accomplish this, the MLP FΘ first processes the input
3D coordinate x with 8 fully-connected layers (using ReLU activations and 256
channels per layer), and outputs σ and a 256-dimensional feature vector. This
feature vector is then concatenated with the camera ray’s viewing direction and
passed to one additional fully-connected layer (using a ReLU activation and 128
channels) that output the view-dependent RGB color.
See Fig. 3 for an example of how our method uses the input viewing direction
to represent non-Lambertian effects. As shown in Fig. 4, a model trained without
view dependence (only x as input) has difficulty representing specularities.
"""


class DensityModel(nn.Module):
    def __init__(self):
        super(DensityModel, self).__init__()

        l1 = nn.Linear(3, 256 * 3)
        l2 = nn.Linear(3 * 256, 256 * 3)
        l3 = nn.Linear(3 * 256, 256 * 3)
        l4 = nn.Linear(3 * 256, 256 * 3)
        l5 = nn.Linear(3 * 256, 256 * 3)
        l6 = nn.Linear(3 * 256, 256 * 3)
        l7 = nn.Linear(3 * 256, 256 * 3)
        feature_vector = nn.Linear(3 * 256, 256)
        density = nn.Linear(3*256, 1)

    def call(self, input):
        """
        :param input: (x,y,z) position coordinate
        :return: feature_vector, density: feature vector is a 256-D vector which is passed into
        the main network. Density is the volume density at (x,y,z). Density is a 1-D vector
        """

        out = func.relu(self.l1(input))
        out = func.relu(self.l2(out))
        out = func.relu(self.l3(out))
        out = func.relu(self.l4(out))
        out = func.relu(self.l5(out))
        out = func.relu(self.l6(out))
        out = func.relu(self.l7(out))
        ## TODO: not sure what should feed into density layer (l7 or feature vec?)
        feat_vec = func.relu(self.feature_vector(out))
        density = func.relu(self.density(out))

        return feat_vec, density


class OutputModel(nn.Module):
    def __init__(self):
        super(OutputModel, self).__init__()

        layer = nn.Linear((256+2)*128,3)

    def call(self,feat_vec,theta_x,theta_y):
        """

        :param feat_vec: from DensityModel (256-D torch tensor)
        :param theta_x: 1-D torch tensor
        :param theta_y: 1-D torch tensor
        :return: 3-D torch tensor (r,g,b)
        """

        in_vec = torch.cat((feat_vec,theta_x,theta_y))
        rgb = func.relu(self.layer(in_vec))

        return rgb


def loss_func(gt_rgb,gt_density,rgb,density):
    pass


def train():
    pass













if __name__ == '__main__':
    print('test')