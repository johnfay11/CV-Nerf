import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import configargparse

from model import *
from utils import *
from data_helpers import load_blender_data, load_llff_data, get_ndc

np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


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


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,

           **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = compute_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # provide ray directions as input
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_full(render_poses, hwf, chunk, render_kwargs, save_dir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())

        if save_dir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(save_dir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    return rgbs


def create_model(args):
    # hierarchical position encoding (see paper)
    xyz_embedder = FreqEmbedding(10)
    ang_embedder = FreqEmbedding(4)

    output_ch = 5
    skips = [4]
    coarse_model = Model().to(device)
    grad_vars = list(coarse_model.parameters())

    fine_model = Model().to(device)

    grad_vars += list(fine_model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=lambda v: xyz_embedder.embed(v),
                                                                        embeddirs_fn=lambda v: ang_embedder.embed(v),
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr, betas=(0.9, 0.999))

    start = 0

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'n_fine_samples': args.n_fine_samples,
        'fine_model': fine_model,
        'n_coarse_samples': args.n_coarse_samples,
        'coarse_model': coarse_model,
        'white_bkg': args.white_bkg,
        'noise': args.noise,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dtype != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['noise'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


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


def render_rays(ray_batch,
                coarse_model,
                network_query_fn,
                n_coarse_samples,
                perturb=0.,
                n_fine_samples=0,
                fine_model=None,
                white_bkg=False,
                noise=0.):
    n_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=n_coarse_samples)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([n_rays, n_coarse_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_query_fn(pts, viewdirs, coarse_model)
    rgb_map, weights = process_volume_info(raw, z_vals, rays_d, noise, white_bkg)

    rgb_map_0 = rgb_map

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = inv_transform_sampling(z_vals_mid, weights[..., 1:-1], n_fine_samples)
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

    run_fn = coarse_model if fine_model is None else fine_model
    raw = network_query_fn(pts, viewdirs, run_fn)

    rgb_map, weights = process_volume_info(raw, z_vals, rays_d, noise, white_bkg)

    ret = {'rgb_map': rgb_map}
    ret['rgb0'] = rgb_map_0
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

    iters = 200000 + 1

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

        grid = torch.reshape(grid, [-1, 2])
        batch_indices = np.random.choice(grid.shape[0], size=[n_rays], replace=False)
        batch_pixels = grid[batch_indices].long()

        r_origins = r_origins[batch_pixels[:, 0], batch_pixels[:, 1]]
        r_dirs = r_dirs[batch_pixels[:, 0], batch_pixels[:, 1]]
        batch_rays = torch.stack([r_origins, r_dirs], 0)
        batch_pixels = im[batch_pixels[:, 0], batch_pixels[:, 1]]

        rgb, extras = render(height, width, focal, chunk=args.chunk, rays=batch_rays,
                             **render_kwargs_train)

        optimizer.zero_grad()
        loss = torch.mean((rgb - batch_pixels) ** 2)

        if 'rgb0' in extras:
            loss = loss + torch.mean((extras['rgb0'] - batch_pixels) ** 2)

        loss.backward()
        optimizer.step()

        DECAY_RATE = 0.1
        DECAY_SIZE = 1000

        # update learning rate: https://tinyurl.com/48makybr using exponential decay
        lr = decayed_learning_rate(i, DECAY_SIZE * args.lr_decay, args.lr, decay_rate=DECAY_RATE)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if i % args.i_weights == 0:
            path = os.path.join(basedir, name, '{:06d}.tar'.format(i))
            torch.save({
                'step': i,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs = render_full(render_poses, [height, width, focal], args.chunk, render_kwargs_test, args.save_dir)
                print('Writing video at', os.path.join(args.save_dir, 'test_vid_{:d}.mp4'.format(step)))

            print('Done, saving', rgbs.shape)
            moviebase = os.path.join(args.save_dir, name, '{}_spiral_{:06d}_'.format(name, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

        #if i % args.i_testset == 0 and i > 0:
        #    testsavedir = os.path.join(basedir, name, 'testset_{:06d}'.format(i))
        #    os.makedirs(testsavedir, exist_ok=True)
        #    print('test poses shape', poses[test_idx].shape)
        #    with torch.no_grad():
        #        render_full(torch.Tensor(poses[test_idx]).to(device), [height, width, focal], args.chunk,
        #                    render_kwargs_test,
        #                    gt_imgs=images[test_idx], savedir=testsavedir)
        #    print('Saved test set')

        if i % args.i_print == 0:
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

    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
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

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--white_bkg", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=500)

    parser.add_argument("--dtype", type=str, default='llff',
                        help='llff or blender')

    return parser


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
