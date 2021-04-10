import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as func
from math import sin,cos,pi


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
    def __init__(self):
        super(Model, self).__init__()

        self.xyz_L = 10
        self.angle_L = 4

        l1 = nn.Linear(60, 256)
        l2 = nn.Linear(256, 256)
        l3 = nn.Linear(256, 256)
        l4 = nn.Linear(256, 256)
        l5 = nn.Linear(256+60, 256)

        l6 = nn.Linear(256, 256)
        l7 = nn.Linear(256, 256)
        l8 = nn.Linear(256,257)

        l9 = nn.Linear(256+24,128)
        l10 = nn.Linear(128,3)

    def pos_encoding(self,pos,L):
        """
        :param pos: pos positions, or angle unit vector (assumed dtype is python list)
        :param L: defines number of terms in encoding (2*L terms)
        :return: FloatTensor input for network
        """
        out = []
        for i in range(len(pos)):
            for j in range(L):
                out.append(sin((2**j)*pi*pos[i]))
                out.append(cos((2**j)*pi*pos[i]))

        out = torch.FloatTensor(out)

        return out

    def call(self, xyz,view_angle):
        """
        :param xyz: (x,y,z) position coordinate
        :param view_angle: view angle unit vector
            (both params assumed to be python lists)
        """
        input = self.pos_encoding(xyz,self.xyz_L)
        out = func.relu(self.l1(input))
        out = func.relu(self.l2(out))
        out = func.relu(self.l3(out))
        out = func.relu(self.l4(out))
        out = torch.cat((input,out))
        out = func.relu(self.l5(out))
        out = func.relu(self.l6(out))
        out = func.relu(self.l7(out))
        out = func.relu(self.l8(out))

        density = out[0]
        feat_vec = out[1:]

        out = feat_vec

        angle_encoding = self.pos_encoding(view_angle,self.angle_L)

        out = torch.cat((out,angle_encoding))
        out = func.relu(self.l9(out))
        out = self.l10(out)

        return out



def loss_func(gt_rgb,gt_density,rgb,density):
    pass


def train():
    pass













if __name__ == '__main__':
    print('test')