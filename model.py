import torch as torch
import torch.nn as nn
import torch.nn.functional as F

STD_CHUNK_SIZE = 65536


def model_forward(xyz, angles, fn, freq_xyz_fn, freq_angle_fn, amort_chunk=STD_CHUNK_SIZE):
    def custom_reshape(fn, ck):
        if ck is None:
            return fn

        def ret(inputs):
            return torch.cat([fn(inputs[i:i + ck]) for i in range(0, inputs.shape[0], ck)], 0)

        return ret

    x_vec = torch.reshape(xyz, [-1, xyz.shape[-1]])
    embedded = freq_xyz_fn(x_vec)

    if angles is not None:
        dirs = angles[:, None].expand(xyz.shape)
        dirs_vec = torch.reshape(dirs, [-1, dirs.shape[-1]])
        embedded_dirs = freq_angle_fn(dirs_vec)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = custom_reshape(fn, amort_chunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(xyz.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


class FreqEmbedding:
    def __init__(self, freqs, dim=3):
        embed_fns = []
        d = dim
        out_dim = 0

        embed_fns.append(lambda x: x)
        out_dim += d

        # pre-compute high frequency input encoding (see paper)
        components = 2. ** torch.linspace(0., freqs - 1, steps=freqs)
        for freq in components:
            embed_fns.append(lambda x, p_fn=torch.sin, freq=freq: p_fn(x * freq))
            out_dim += d

            embed_fns.append(lambda x, p_fn=torch.cos, freq=freq: p_fn(x * freq))
            out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


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

        self.l1 = nn.Linear(Model._encoding_dim(3, self.xyz_L) + 3, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)

        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256 + Model._encoding_dim(3, self.xyz_L) + 3, 256)

        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 256)
        self.l9 = nn.Linear(256, 256)

        self.l_alpha = nn.Linear(256, 1)
        self.l10 = nn.Linear(256 + Model._encoding_dim(3, self.angle_L) + 3, 128)
        self.l11 = nn.Linear(128, 3)

    @staticmethod
    def _encoding_dim(num_comp, L):
        return 2 * num_comp * L

    def forward(self, x):
        """
        :param xyz: (x,y,z) position coordinate
        :param view_angle: view angle unit vector
            (both params assumed to be python lists)
        """
        xyz, ang = torch.split(x,
                               [3 + Model._encoding_dim(3, self.xyz_L), 3 + Model._encoding_dim(3, self.angle_L)],
                               dim=-1)

        out = F.relu(self.l1(xyz))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        out = F.relu(self.l5(out))

        # skip connection that concatenates X to the fifth layer’s activation
        out = torch.cat((xyz, out), -1)

        out = F.relu(self.l6(out))
        out = F.relu(self.l7(out))
        out = F.relu(self.l8(out))

        density = self.l_alpha(out)
        out = self.l9(out)

        out = torch.cat([out, ang], -1)
        out = F.relu(self.l10(out))
        rgb = self.l11(out)

        return torch.cat([rgb, density], -1)
