import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as func
from math import sin, cos, pi

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


class Model(nn.Module):
    def __init__(self, xyz_L=10, angle_L=4):
        super(Model, self).__init__()
        self.xyz_L = xyz_L
        self.angle_L = angle_L

        self.l1 = nn.Linear(Model._encoding_dim(3, self.xyz_L), 256)
        torch.nn.init.xavier_uniform_(self.l1.weight)

        self.l2 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l2.weight)

        self.l3 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l3.weight)

        self.l4 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l4.weight)

        self.l5 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l5.weight)

        self.l6 = nn.Linear(256 + Model._encoding_dim(3, self.xyz_L), 256)
        torch.nn.init.xavier_uniform_(self.l6.weight)

        self.l7 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l7.weight)

        self.l8 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l8.weight)

        self.l9 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.l9.weight)

        self.l_alpha = nn.Linear(256, 1)
        torch.nn.init.xavier_uniform_(self.l_alpha.weight)

        self.l10 = nn.Linear(256 + Model._encoding_dim(3, self.angle_L), 128)
        torch.nn.init.xavier_uniform_(self.l10.weight)

        self.l11 = nn.Linear(128, 3)
        torch.nn.init.xavier_uniform_(self.l11.weight)

        N_1 = 10
        N_2 = 4

        freq_bands = 2. ** torch.linspace(0., N_1 - 1, N_1)
        freq_bands_2 = 2. ** torch.linspace(0., N_2 - 1, N_2)

        embed_fns = []
        embed_fns_2 = []

        for freq in freq_bands:
            for p_fn in [torch.cos, torch.sin]:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))

        for freq in freq_bands_2:
            for p_fn in [torch.cos, torch.sin]:
                embed_fns_2.append(lambda x, p_fn=p_fn,
                                          freq=freq: p_fn(x * freq))

        self.embed_fns = embed_fns
        self.embed_fns_2 = embed_fns_2

    @staticmethod
    def _encoding_dim(num_comp, L):
        return 2 * num_comp * L

    @staticmethod
    def _pos_encoding(pos, out, L):
        """
        :param pos: pos positions, or angle unit vector (assumed dtype is python list)
        :param L: defines number of terms in encoding (2*L terms)
        :return: FloatTensor input for network
        """
        # for i in range(pos.shape[0]):
        #    for j in range(L):
        #        out[(i * L) + j] = (sin((2 ** j) * pi * pos[i]))
        #        out[(i * L) + j + 1] = (cos((2 ** j) * pi * pos[i]))
        # return out

    def forward(self, xyz, view_angle):
        """
        :param xyz: (x,y,z) position coordinate
        :param view_angle: view angle unit vector
            (both params assumed to be python lists)
        """
        X = torch.cat([fn(xyz) for fn in self.embed_fns], -1)
        X_ang = torch.cat([fn(view_angle) for fn in self.embed_fns_2], -1)
        # size = list(xyz.size()[:-1]) + [Model._encoding_dim(3, self.xyz_L)]

        # X = torch.zeros(size).cuda()
        # X_ang = torch.zeros(size).cuda()

        # for ray in range(X.shape[0]):
        #  for pt in range(X.shape[1]):
        #    X[ray, pt] = Model._pos_encoding(xyz[ray, pt], X[ray, pt], self.xyz_L)

        out = func.relu(self.l1(X))
        out = func.relu(self.l2(out))
        out = func.relu(self.l3(out))
        out = func.relu(self.l4(out))
        out = func.relu(self.l5(out))

        # skip connection that concatenates X to the fifth layer’s activation
        out = torch.cat((X, out), -1)

        out = func.relu(self.l6(out))
        out = func.relu(self.l7(out))
        out = func.relu(self.l8(out))

        density = self.l_alpha(out)
        out = self.l9(out)

        # this is a subtle optimization
        # for ray in range(X.shape[0]):
        #    X_ang[ray, :] = Model._pos_encoding(view_angle, X_ang[ray, 0], self.angle_L)

        out = torch.cat([out, X_ang], -1)
        out = func.relu(self.l10(out))
        rgb = func.sigmoid(self.l11(out))
        return torch.cat([rgb, density], -1)
