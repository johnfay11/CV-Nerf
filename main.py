import configargparse
import torch
import numpy as np

from data_helpers import load_blender_data
from model import OutputModel


def parse_settings():
    # use standard config parser
    parser = configargparse.ArgumentParser('args')

    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--name', type=str)
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--data_dir', type=str)

    parser.add_argument("--dtype", type=str, default='llff',
                        help='llff or blender')

    parser.add_argument('--n_rays', type=int, default=4096)
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--n_fine_samples', type=int, default=0)

    # position encoding dimensions
    parser.add_argument('--L_pos', type=int, default=10)
    parser.add_argument('--L_ang', type=int, default=4)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=250, help='learning rate decay')

    parser.add_argument('--iter', type=int, default=100000)

    # we might not need this parameter (could set to size of dataset)
    parser.add_argument("--testskip", type=int, default=8)

    # this is useful for reducing the size of blender models
    parser.add_argument('--half_res', action='store_true')
    parser.add_argument('--debug', type=bool, default=False)

    # white background is needed to accurately reproduce the synthetic examples
    parser.add_argument('--white_bkg', type=bool, default=False)
    return parser.parse_args()


def load_dataset(args):
    if args.dtype != 'llff' and args.dtype != 'blender':
        raise ValueError('Invalid data type. Must be one of llff or blender.')

    # here we can probably plug in the LLFF library and blender extensions.
    # TODO: return images, poses, poses to render, camera intrinsics, maybe other information?

    if args.dtype == 'blender':
        return load_blender_data(args.data_dir, half_res=args.half_res, testskip=args.testskip)
    else:
        raise NotImplementedError("please implement me! (LLFF)")


def compute_rays(h, w, f, pose):
    # see: https://graphics.cs.wisc.edu/WP/cs559-fall2016/files/2016/12/shirley_chapter_4.pdf and
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays

    # spans from 0 to h - 1, row-wise
    y_grid = torch.arange(0, h, dtype=torch.float32)

    # spans from 0 to w - 1, column-wise
    x_grid = torch.arange(0, w, dtype=torch.float32)

    # discretize image into hxw grid; note that meshgrid is implemented poorly, so we have to use numpy
    x, y = np.meshgrid(x_grid, y_grid, indexing='xy')
    x, y = torch.Tensor(x), torch.Tensor(y)

    # compute direction (3D vector) of each ray using virtual pinhole model
    d_x = (x - w * .5) / f
    d_y = -(y - h * .5) / f
    dirs = torch.stack([d_x, d_y, -torch.ones_like(x)], -1)

    # now, we project each ray direction into world space using the camera pose
    proj = dirs[..., np.newaxis, :] * pose[:3, :3]
    dirs = torch.sum(proj, -1)

    # last column of projection matrix contains origin of all rays
    origins = pose[:3, -1].expand(dirs.shape)
    return origins, dirs


def process_volume_info(raw_rgba, t_samples, r_dirs, noise=0.0, bkg=False):
    INF_DIST = 1e10
    EPS = 1e-10 # for numerical stability (see: https://effectivemachinelearning.com/PyTorch/7._Numerical_stability_in_PyTorch)

    # generate noise for volume channel (see p. 19) for more information
    _noise = 0.0
    if noise > 0.0:
        _noise = torch.random.normal(raw_rgba[..., 3].shape) * noise

    # compute distances between samples (denotes as deltas in equation (3))
    deltas = t_samples[..., 1:] - t_samples[..., :-1]
    last_delta = torch.Tensor([INF_DIST]).expand(deltas[..., :1].shape)
    deltas = torch.cat([deltas, last_delta], -1) * torch.norm(r_dirs[..., None, :], dim=-1)

    # alpha compositing; ensure that relu has already been applied!!
    alpha = 1.0 - torch.exp(-(raw_rgba[..., 3] + _noise) * deltas)

    # apply sigmoid activation to colors (see architecture diagram); ensure that sigmoid has already been applied!!
    colors = raw_rgba[..., :3]

    # compute T_i(s) using alpha values (see eq. (3))
    T_weights = torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + EPS], -1)

    # have to use a roundabout way of computing cumprod because of the way it's not exclusive in pytorch
    T_weights = alpha * torch.cumprod(T_weights, -1)[:, :-1]

    # extract ray-wise RGB values by summing along the approximated path integral (eq. (3))
    rgb_map = torch.sum(T_weights[..., None] * colors, -2)

    # add background if specified
    if bkg:
        # in RGB space, 1.0 is white
        rgb_map += (1.0 - torch.sum(T_weights, -1)[..., None])

    return rgb_map, T_weights


def inv_transform_sampling(pts, weights, n):
    """
    Get of sampled points and corresponding weights, we perform inverse transform sampling:
    https://en.wikipedia.org/wiki/Inverse_transform_sampling. In essence, this technique allows us
    to sample n points from the weight pdf.
    """

    # numerical stability
    EPS = 1e-6
    weights += weights + EPS

    # map weights to a probability distribution
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # construct cdf from pdf
    cdf = [torch.zeros_like(pdf[..., :1]), pdf]
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(cdf, -1)
    return None


# TODO: add option to render from fundamental matrix
def render(cam_intrs, rays, model, model_fine, bounds, args):
    r_origins, r_dirs = rays

    # represents theta and phi, as specified in the paper
    d_vec = r_dirs / torch.norm(r_dirs, dim=-1, keepdim=True)
    d_vec = torch.reshape(d_vec, [-1, 3]).float()

    # partition [0, 1] using n points and rescale into [t_n, t_f]
    t_samples = torch.linspace(0., 1., steps=args.n_samples)
    t_samples = bounds[0] * (1. - t_samples) + bounds[1] * t_samples
    t_samples = t_samples.expand([args.n_rays, args.n_samples])

    # get n_rays * n_samples random samples in [0, 1]
    t_r = torch.rand(t_samples.shape)

    # rescale by computing the middle of each interval
    midpoints = .5 * (t_samples[..., 1:] + t_samples[..., :-1])

    # add back lowest and highest sample
    u = torch.cat([midpoints, t_samples[..., -1:]], -1)
    l = torch.cat([t_samples[..., :1], midpoints], -1)
    t_samples = l + (u - l) * t_r

    # compute the (x, y, z) coords of each timestep using ray origins and directions
    # coords: (n_rays, 3) * (n_rays, n_samples) -> (n_rays, n_samples, 3)
    coords = r_dirs[..., None, :] * t_samples[..., :, None]
    coords = r_origins[..., None, :] + coords

    # TODO: add model, which ingests coords and d_vec
    model = None
    model_fine = None

    # Returned from model; tensor of size n_rays x n_samples x 4
    rgba = torch.rand(coords.shape)

    # tensor of size n_rays x 3
    rgb, weights = process_volume_info(rgba, t_samples, r_dirs, noise=1.0, bkg=args.white_bkg)

    midpoints = .5 * (t_samples[..., 1:] + t_samples[..., :-1])
    _weights = weights[..., 1:-1] # remove weights not used for hierarchical sampling

    with torch.no_grad():
        fine_samples = inv_transform_sampling(midpoints, _weights, args.n_fine_samples)


def main():
    args = parse_settings()

    '''
    images: numpy array of shape (total_ims, im_height, im_width, rgba)
    poses: numpy array of shape (total_ims, 4, 4)
    cam_params: list of the form [H, W, f]
    i_split: list of the form [train_indices, val_indices, test_indices]
    bounds: list of the form [near bound, far bound]
    '''
    images, poses, render_poses, cam_params, i_split, bounds = load_dataset(args)
    train_idx, val_idx, test_idx = i_split

    # unpack camera intrinsics
    height, width, f = cam_params
    height, width = int(height), int(width)

    model = None
    model_fine = None
    if torch.cuda.is_available():
        model = model.cuda()
        model_fine = model_fine.cuda()
        images = torch.Tensor(images).cuda()
        poses = torch.Tensor(poses).cuda()
        render_poses = torch.Tensor(render_poses).cuda()
    elif not args.debug:
        raise ValueError("CUDA is not available")

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    steps = args.iter
    step = 0

    # training loop
    for i in range(steps):
        # select random image
        im_idx = np.random.choice(train_idx)
        im = images[im_idx]

        # extract projection matrix:
        # (https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera)
        pose = poses[im_idx, :3, :4]

        # both origins and orientations are needed to determine a ray
        r_origins, r_dirs = compute_rays(height, width, f, torch.Tensor(pose))

        # select a random subset of rays from a H x W grid
        h_grid = torch.linspace(0, height - 1, height)
        w_grid = torch.linspace(0, width - 1, width)
        grid = torch.meshgrid(h_grid, w_grid)
        grid = torch.stack(grid, -1)

        # (H x W, 2) tensor containing all possible pixels
        grid = torch.reshape(grid, [-1, 2])
        batch_indices = np.random.choice(grid.shape[0], size=[args.n_rays], replace=False)
        batch_pixels = grid[batch_indices].long()

        r_origins = r_origins[batch_pixels[:, 0], batch_pixels[:, 1]]
        r_dirs = r_dirs[batch_pixels[:, 0], batch_pixels[:, 1]]
        batch_rays = torch.stack([r_origins, r_dirs], 0)
        batch_pixels = im[batch_pixels[:, 0], batch_pixels[:, 1]]

        # renders rays into RGB values
        rgb_c, rgb_f = render([height, width, f], batch_rays, model, bounds, args)

        #optimizer.zero_grad()
        # TODO: update learning rate, backprop, loss


if __name__ == '__main__':
    main()
