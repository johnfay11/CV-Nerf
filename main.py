import os, sys
import numpy as np
import imageio
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

import configargparse

from model import *
from utils import *
from data_helpers import load_blender_data, load_llff_data, get_ndc

# fix pytorch related constants
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def _batcher(ray_vec, b_size=32768, **kwargs):
    res = {}
    for i in range(0, ray_vec.shape[0], b_size):
        ret = render_rays(ray_vec[i: i + b_size], **kwargs)
        for k in ret:
            if k not in res:
                res[k] = []
            res[k].append(ret[k])
    return {k: torch.cat(res[k], 0) for k in res}


def compute_rays(h, w, f, pose):
    # see: https://graphics.cs.wisc.edu/WP/cs559-fall2016/files/2016/12/shirley_chapter_4.pdf and
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays

    # spans from 0 to h - 1, row-wise
    # y_grid = torch.linspace(0, h - 1, h).cuda()
    y_grid = torch.linspace(0, h - 1, h)

    # spans from 0 to w - 1, column-wise
    x_grid = torch.linspace(0, w - 1, w)

    # discretize image into hxw grid; note that meshgrid is implemented poorly, so we have to use numpy
    x, y = torch.meshgrid(x_grid, y_grid)
    x = x.t()
    y = y.t()

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


def render(h, w, f, chunk=32768, rays=None, pose=None, ndc=True, near=0.0, far=1.0, **kwargs):
    if pose is not None:
        # this case is only for rendering videos (i.e. inference)
        r_origins, r_dirs = compute_rays(h, w, f, pose)
    else:
        # use provided ray batch
        r_origins, r_dirs = rays

    # normalize and reformat angle data
    d_vec = r_dirs / torch.norm(r_dirs, dim=-1, keepdim=True)
    d_vec = torch.reshape(d_vec, [-1, 3]).float()

    # save current shape for later
    local_shape = r_dirs.shape

    # project into ndc format if working with real imagery
    if ndc:
        # for forward facing scenes
        r_origins, r_dirs = get_ndc(h, w, f, 1.0, r_origins, r_dirs)

    r_origins = torch.reshape(r_origins, [-1, 3]).float()
    r_dirs = torch.reshape(r_dirs, [-1, 3]).float()

    # prepare for batching
    t_n, t_f = near * torch.ones_like(r_dirs[..., :1]), far * torch.ones_like(r_dirs[..., :1])
    rays = torch.cat([r_origins, r_dirs, t_n, t_f], -1)
    rays = torch.cat([rays, d_vec], -1)

    # render via mini-batching
    res = _batcher(rays, chunk, **kwargs)
    for i in res:
        shape = list(local_shape[:-1]) + list(res[i].shape[1:])
        res[i] = torch.reshape(res[i], shape)

    # unpack and return rgb data
    ret_list = [res[k] for k in ['rgb_map']]
    ret_dict = {k: res[k] for k in res if k not in  ['rgb_map']}
    return ret_list + [ret_dict]


def render_full(render_poses, cam_params, ck, render_kwargs, save_dir=None, factor=0):
    height, width, focal = cam_params

    # downsample if we're constrained for resources
    if factor != 0:
        height = height // factor
        width = width // factor
        focal = focal / factor

    frames = []

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, _ = render(height, width, focal, chunk=ck, pose=c2w[:3, :4], **render_kwargs)
        frames.append(rgb.cpu().numpy())

        if save_dir is not None:
            rgb8 = cont_to_byte8_im(frames[-1])
            filename = os.path.join(save_dir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(frames, 0)
    return rgbs


def create_model(args):
    # hierarchical position encoding (see paper)
    xyz_embedder = FreqEmbedding(10)
    ang_embedder = FreqEmbedding(4)

    coarse_model = Model().to(device)
    grad_vars = list(coarse_model.parameters())

    fine_model = Model().to(device)
    grad_vars += list(fine_model.parameters())

    forward_fn = lambda inputs, viewdirs, network_fn: model_forward(inputs, viewdirs, network_fn,
                                                                        freq_xyz_fn=lambda v: xyz_embedder.embed(v),
                                                                        freq_angle_fn=lambda v: ang_embedder.embed(v),
                                                                        amort_chunk=args.amort_size)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr, betas=(0.9, 0.999))

    init = 0
    render_kwargs_train = {
        'forward': forward_fn,
        'fine_model': fine_model,
        'coarse_model': coarse_model,
        'n_fine_samples': args.n_fine_samples,
        'n_coarse_samples': args.n_coarse_samples,
        'white_bkg': args.white_bkg,
        'perturb': args.perturb,
        'noise': args.noise,
    }

    # copy training settings and remove training-only parameters
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['noise'] = 0.
    render_kwargs_test['perturb'] = False

    if args.dtype != 'llff' or args.no_ndc:
        # ndc word positioning not necessary for synthetic data
        render_kwargs_train['ndc'] = False
    return render_kwargs_train, render_kwargs_test, init, grad_vars, optimizer

# helper function for computing alpha values in process_volume_info
def _alpha_composite(vals, deltas):
    return 1.0 - torch.exp(deltas * -F.relu(vals))

def process_volume_info(raw_rgba, t_samples, r_dirs, noise=0.0, bkg=False):
    INF_DIST = 1e10

    # compute distances between samples (denotes as deltas in equation (3))
    deltas = t_samples[..., 1:] - t_samples[..., :-1]
    deltas = torch.cat([deltas, torch.tensor([INF_DIST]).expand(deltas[..., :1].shape)], -1)
    deltas = deltas * torch.norm(r_dirs[..., None, :], dim=-1)

    # extract last RGB layer
    rgb = torch.sigmoid(raw_rgba[..., :3])

    # add random noise to aid training
    _noise = 0.
    if noise > 0.:
        _noise = torch.randn(raw_rgba[..., 3].shape) * noise

    # alpha compositing; ensure that relu has already been applied!!
    alpha = _alpha_composite(raw_rgba[..., 3] + _noise, deltas)

    # compute T_i(s) using alpha values (see eq. (3)); 1e-10 added for numerical stability
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]

    # extract ray-wise RGB values by summing along the approximated path integral (eq. (3))
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    mp = torch.sum(weights, -1)

    # add background if specified
    if bkg:
        rgb_map = rgb_map + (1. - mp[..., None])
    return rgb_map, weights


def render_rays(ray_batch, coarse_model, forward, n_coarse_samples, perturb=0., n_fine_samples=0,
                                                                                fine_model=None,
                                                                                white_bkg=False,
                                                                                noise=0.):
    # extract dirs and origins from mini-batch
    n_rays = ray_batch.shape[0]
    r_origins, r_dirs = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # sample coarse points
    t_samples = torch.linspace(0., 1., steps=n_coarse_samples)
    s_vals = near * (1. - t_samples) + far * (t_samples)
    s_vals = s_vals.expand([n_rays, n_coarse_samples])

    # random perturbation of points tends to improve training
    if perturb > 0.:
        midpoints = .5 * (s_vals[..., 1:] + s_vals[..., :-1])
        l = torch.cat([s_vals[..., :1], midpoints], -1)
        u = torch.cat([midpoints, s_vals[..., -1:]], -1)

        # rescale coarse points
        t_rand = torch.rand(s_vals.shape)
        s_vals = l + (u - l) * t_rand

    pts = r_origins[..., None, :] + r_dirs[..., None, :] * s_vals[..., :, None]

    # generate coarse rgb vals
    raw = forward(pts, viewdirs, coarse_model)
    rgb_map, weights = process_volume_info(raw, s_vals, r_dirs, noise, white_bkg)
    rgb_c = rgb_map

    # perform inverse sampling to get fine points
    midpoints = .5 * (s_vals[..., 1:] + s_vals[..., :-1])
    f_samples = inv_transform_sampling(midpoints, weights[..., 1:-1], n_fine_samples)

    f_samples = f_samples.detach()
    s_vals, _ = torch.sort(torch.cat([s_vals, f_samples], -1), -1)
    pts = r_origins[..., None, :] + r_dirs[..., None, :] * s_vals[..., :, None]

    # get fine rgb predictions
    run_fn = coarse_model if fine_model is None else fine_model
    raw = forward(pts, viewdirs, run_fn)

    rgb, _ = process_volume_info(raw, s_vals, r_dirs, noise, white_bkg)

    ret = {'rgb_map': rgb}
    ret['rgb_c'] = rgb_c
    return ret


def load_dataset(args):
    if args.dtype != 'llff' and args.dtype != 'blender':
        raise ValueError('Invalid data type. Must be one of llff or blender.')

    # here we can probably plug in the LLFF library and blender extensions.
    # TODO: return images, poses, poses to render, camera intrinsics, maybe other information?

    if args.dtype == 'blender':
        print('Loading blender data.')
        return load_blender_data(args.data_dir, half_res=args.half_res, testskip=args.testskip, bkg=args.white_bkg)
    else:
        print('Loading LLFF data.')
        return load_llff_data(args.data_dir)


def decayed_learning_rate(step, decay_steps, initial_lr, decay_rate=0.1):
    return initial_lr * (decay_rate ** (step / decay_steps))


def main():
    parser = config_parser()
    args = parser.parse_args()

    images, poses, render_poses, cam_params, i_split, bounds = load_dataset(args)

    if args.dtype == 'llff':
        test_idx = [i_split]
        test_idx = np.arange(images.shape[0])[::8]

        val_idx = test_idx
        train_idx = []
        for i in np.arange(int(images.shape[0])):
            if i not in test_idx and i not in val_idx:
                train_idx.append(i)
        train_idx = np.array(train_idx)

        if not args.ndc:
            b = np.array(bounds).flatten()
            near = np.min(b) * .9
            far = np.max(b) * 1.
        else:
            near = 0.
            far = 1.

        bounds = [near, far]

    else:
        train_idx, val_idx, test_idx = i_split

    # Cast intrinsics to right types
    height, width, focal = cam_params
    height, width = int(height), int(width)
    # hwf = [H, W, focal]

    near, far = bounds

    if args.render_test:
        render_poses = np.array(poses[test_idx])

    # Create log dir and copy the config file
    basedir = args.base_dir
    name = args.name
    os.makedirs(os.path.join(basedir, name), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_model(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_poses = torch.Tensor(render_poses).to(device)

    n_rays = args.n_rays
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    iters = 100000

    start = start + 1
    for i in trange(start, iters):
        time0 = time.time()

        # Random from one image
        im_idx = np.random.choice(train_idx)
        im = images[im_idx]
        pose = poses[im_idx, :3, :4]

        r_origins, r_dirs = compute_rays(height, width, focal, torch.tensor(pose))  # (H, W, 3), (H, W, 3)

        grid = None

        # crop as warm-up; tends to improve rendering performance and optimization
        if i < args.precrop_iters:
            delta_h = int(height // 2 * args.precrop_frac)
            delta_w = int(width // 2 * args.precrop_frac)
            grid = torch.stack(
                torch.meshgrid(
                    torch.linspace(height // 2 - delta_h, height // 2 + delta_h - 1, 2 * delta_h),
                    torch.linspace(width // 2 - delta_w, width // 2 + delta_w - 1, 2 * delta_w)
                ), -1)
        else:
            h_grid = torch.linspace(0, height - 1, height)
            w_grid = torch.linspace(0, width - 1, width)
            grid = torch.stack(torch.meshgrid(h_grid, w_grid), -1)

        # (H x W, 2) tensor containing all possible pixels
        grid = torch.reshape(grid, [-1, 2])
        batch_indices = np.random.choice(grid.shape[0], size=[n_rays], replace=False)
        batch_pixels = grid[batch_indices].long()

        r_origins = r_origins[batch_pixels[:, 0], batch_pixels[:, 1]]
        r_dirs = r_dirs[batch_pixels[:, 0], batch_pixels[:, 1]]
        batch_rays = torch.stack([r_origins, r_dirs], 0)
        batch_pixels = im[batch_pixels[:, 0], batch_pixels[:, 1]]

        # renders rays into RGB values
        rgb, extras = render(height, width, focal, chunk=args.b_size, rays=batch_rays, **render_kwargs_train)

        # get coarse rgb data
        rgb_c = extras['rgb_c']
        optimizer.zero_grad()
        loss = torch.mean((rgb - batch_pixels) ** 2)
        loss = loss + torch.mean((rgb_c - batch_pixels) ** 2)

        # backprop (~0.4 seconds per step)!
        loss.backward()
        optimizer.step()

        # constants for exponential learning rate decay
        DECAY_RATE = 0.1
        DECAY_SIZE = 1000

        # update learning rate: https://tinyurl.com/48makybr using exponential decay
        lr = decayed_learning_rate(i, DECAY_SIZE * args.lr_decay, args.lr, decay_rate=DECAY_RATE)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if i % args.save_freq == 0:
            path = os.path.join(basedir, name, '{:06d}.tar'.format(i))
            torch.save({
                'step': i,
                'fine_model_state_dict': render_kwargs_train['fine_model'].state_dict(),
                'coarse_model_state_dict': render_kwargs_train['coarse_model'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

        if i % args.video_freq == 0 and i > 0:
            with torch.no_grad():
                frames = render_full(render_poses, [height, width, focal], args.b_size, render_kwargs_test, args.save_dir)
            moviebase = os.path.join(args.save_dir, name, '{}_spiral_{:06d}_'.format(name, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', cont_to_byte8_im(frames), fps=30, quality=8)

        if i % args.print_freq == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--name", type=str,
                        help='experiment name')
    parser.add_argument("--base_dir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--data_dir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument('--save_dir', type=str, default='./logs')

    parser.add_argument("--n_rays", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_decay", type=int, default=250)

    parser.add_argument("--b_size", type=int, default=1024 * 32)
    parser.add_argument("--amort_size", type=int, default=1024 * 64,)
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--n_coarse_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--n_fine_samples", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.)
    parser.add_argument("--noise", type=float, default=0.)

    parser.add_argument("--precrop_iters", type=int, default=0,)
    parser.add_argument("--precrop_frac", type=float, default=.5)

    parser.add_argument("--testskip", type=int, default=8)

    parser.add_argument("--white_bkg", action='store_true')
    parser.add_argument("--half_res", action='store_true')

    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--render_only", action='store_true')
    parser.add_argument("--render_test", action='store_true',)
    parser.add_argument("--render_factor", type=int, default=0)

    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--video_freq", type=int, default=5000)

    parser.add_argument("--dtype", type=str, default='llff',
                        help='llff or blender')

    return parser


if __name__ == '__main__':
    main()
