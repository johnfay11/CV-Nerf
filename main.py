import configargparse
import torch
import torch.nn.functional as func
import numpy as np
import copy

import os
import imageio
import time

from data_helpers import load_blender_data
from model import Model


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
    parser.add_argument('--noise', type=float, default=1.0)

    parser.add_argument('--save_freq', type=int, default=2500)
    parser.add_argument('--video_freq', type=int, default=1)
    parser.add_argument('--update_freq', type=int, default=50)
    return parser.parse_args()


def load_dataset(args):
    if args.dtype != 'llff' and args.dtype != 'blender':
        raise ValueError('Invalid data type. Must be one of llff or blender.')

    # here we can probably plug in the LLFF library and blender extensions.
    # TODO: return images, poses, poses to render, camera intrinsics, maybe other information?

    if args.dtype == 'blender':
        return load_blender_data(args.data_dir, half_res=args.half_res, testskip=args.testskip, bkg=args.white_bkg)
    else:
        raise NotImplementedError("please implement me! (LLFF)")


def compute_rays(h, w, f, pose):
    print('compute_rays!!')
    h = torch.tensor(h)
    w = torch.tensor(w)
    f = torch.tensor(f)
    pose = torch.tensor(pose)
    h = h.cuda()
    w = w.cuda()
    f = f.cuda()
    pose = pose.cuda()
    # see: https://graphics.cs.wisc.edu/WP/cs559-fall2016/files/2016/12/shirley_chapter_4.pdf and
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays

    # spans from 0 to h - 1, row-wise
    y_grid = torch.arange(0, h, dtype=torch.float32).cuda()

    # spans from 0 to w - 1, column-wise
    x_grid = torch.arange(0, w, dtype=torch.float32).cuda()

    # discretize image into hxw grid; note that meshgrid is implemented poorly, so we have to use numpy
    x, y = torch.meshgrid(x_grid, y_grid)
    x = x.t()
    y = y.t()

    # h = torch.from_numpy(h)
    # w = torch.from_numpy(w)
    # f = torch.from_numpy(f)

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
    INF_DIST = 110
    EPS = 1e-10  # for numerical stability (see: https://effectivemachinelearning.com/PyTorch/7._Numerical_stability_in_PyTorch)

    # generate noise for volume channel (see p. 19) for more information
    _noise = 0.0
    if noise > 0.0:
        _noise = torch.randn(raw_rgba[..., 3].shape) * noise

    _noise = _noise.cuda()

    # compute distances between samples (denotes as deltas in equation (3))
    deltas = t_samples[..., 1:] - t_samples[..., :-1]
    last_delta = torch.Tensor([INF_DIST]).expand(deltas[..., :1].shape).cuda()
    deltas = torch.cat([deltas, last_delta], -1) * torch.norm(r_dirs[..., None, :], dim=-1)

    # alpha compositing; ensure that relu has already been applied!!
    alpha = 1.0 - torch.exp(-func.relu((raw_rgba[..., 3] + _noise) * deltas))

    # apply sigmoid activation to colors (see architecture diagram); ensure that sigmoid has already been applied!!
    colors = raw_rgba[..., :3]

    # compute T_i(s) using alpha values (see eq. (3))
    T_weights = torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1. - alpha + EPS], -1)

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
    EPS = 1e-5
    weights += EPS

    # map weights to a probability distribution
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # construct cdf (denote F) from pdf
    cdf = torch.cumsum(pdf, -1)
    cdf = [torch.zeros_like(cdf[..., :1]).cuda(), cdf]
    cdf = torch.cat(cdf, -1)

    # uniformly sample points
    unif_samp = torch.rand(list(cdf.shape[:-1]) + [n]).cuda()

    """
    Invert the CDF and compute F^{-1}(U). This reduces to a searching for domain values of F that contain the 
    values of U and then rescaling. See the following source for more information:
    - http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf 
    - https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html 
    - http://www.cse.psu.edu/~rtc12/CSE586/lectures/cse586samplingPreMCMC.pdf
    """

    unif_samp = unif_samp.contiguous()
    # luckily, searchsorted implements this searching functionality!
    i = torch.searchsorted(cdf, unif_samp, right=True)

    # upper and lower bounds of U w.r.t. F
    upper = torch.min((cdf.shape[-1] - 1) * torch.ones_like(i), i)
    lower = torch.max(torch.zeros_like(i - 1), i - 1)
    indices = torch.stack([lower, upper], -1)

    # rescale input parameters to match shape of uniformly chosen points
    _new_shape = [indices.shape[0], indices.shape[1], cdf.shape[-1]]
    cdf = cdf.unsqueeze(1).expand(_new_shape)
    cdf = torch.gather(cdf, 2, indices)
    pts = pts.unsqueeze(1).expand(_new_shape)
    pts = torch.gather(pts, 2, indices)

    # rescale into [t_n, t_f] domain
    scale = cdf[..., 1] - cdf[..., 0]
    # need this for numerical stability
    scale = torch.where(scale < EPS, torch.ones_like(scale), scale)
    return (pts[..., 1] - pts[..., 0]) * ((unif_samp - cdf[..., 0]) / scale) + cdf[..., 0]


def render(rays, coarse_model, fine_model, bounds, args):
    inference = coarse_model is None

    with torch.no_grad():
        r_origins, r_dirs = rays

        # represents theta and phi, as specified in the paper
        d_vec = r_dirs / torch.norm(r_dirs, dim=-1, keepdim=True)
        d_vec = torch.reshape(d_vec, [-1, 3]).float()

        # partition [0, 1] using n points and rescale into [t_n, t_f]
        t_samples = torch.linspace(0., 1., steps=args.n_samples).cuda()
        t_samples = bounds[0] * (1. - t_samples) + bounds[1] * t_samples
        t_samples = t_samples.expand([args.n_rays, args.n_samples])

        # get n_rays * n_samples random samples in [0, 1]
        t_r = torch.rand(t_samples.shape).cuda()

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

    # duplicate angle for each point
    _d_vec = d_vec[..., None, :].expand(coords.shape)

    # Returned from model; tensor of size n_rays x n_samples x 4
    rgba = coarse_model(coords, _d_vec) if coarse_model is not None else None

    # tensor of size n_rays x 3
    rgb, weights = process_volume_info(rgba, t_samples, r_dirs, noise=args.noise, bkg=args.white_bkg)

    # remove weights not used for hierarchical sampling
    avg_pts = (t_samples[..., 1:] + t_samples[..., :-1]) / 2.0
    _weights = weights[..., 1:-1]

    fine_samples = inv_transform_sampling(avg_pts, _weights, args.n_fine_samples)
    # Combine coarse and fine samples; TODO: determine if this is right.
    t_samples = torch.cat([t_samples, fine_samples], -1)

    # compute coordinates, as shown above
    coords = r_origins[..., None, :] + r_dirs[..., None, :] * t_samples[..., :, None]

    _d_vec = d_vec[..., None, :].expand(coords.shape)

    # Returned from model; tensor of size n_rays x n_samples x 4
    rgba_f = fine_model(coords, _d_vec)

    # tensor of size n_rays x 3
    rgb_f, weights_f = process_volume_info(rgba_f, t_samples, r_dirs, noise=args.noise, bkg=args.white_bkg)
    return rgb, rgb_f


def render_full(render_poses, cam_params, save_dir, coarse_mode, fine_model, bounds, args):
    height, width, f = cam_params

    args = copy.deepcopy(args)
    args.noise = 0.0

    pred_ims = []
    for i, pose_mat in enumerate(render_poses):
        print('reached!')
        r_origins, r_dirs = compute_rays(height, width, f, torch.Tensor(pose_mat))
        r_origins = r_origins.cpu()
        r_dirs = r_dirs.cpu()

        h_grid = torch.linspace(0, height - 1, height).cuda()
        w_grid = torch.linspace(0, width - 1, width).cuda()
        grid = torch.meshgrid(h_grid, w_grid)
        grid = torch.stack(grid, -1)

        grid = torch.reshape(grid, [-1, 2])

        batch_size = args.n_rays
        n_batches = int(np.ceil((height * width) / batch_size))

        # batching so that we don't saturate GPU memory
        _d_seen_indices = []  # TODO: remove after testing
        for j in range(n_batches):
            batch_indices = np.arange((j * batch_size), min((j + 1) * batch_size), height * width)

            if args.debug:
                _d_seen_indices.extend(list(batch_indices))

            batch_pixels = grid[batch_indices].long()

            r_origins = r_origins[batch_pixels[:, 0], batch_pixels[:, 1]]
            r_dirs = r_dirs[batch_pixels[:, 0], batch_pixels[:, 1]]
            batch_rays = torch.stack([r_origins, r_dirs], 0)

            _, rgb_f = render(batch_rays, coarse_mode, fine_model, bounds, args)
            pred_ims.append(pred_ims)

        if args.debug:
            assert _d_seen_indices == list(range(height * width))

        # convert to 8bytes
        im_8 = (255 * np.clip(pred_ims[-1], 0, 1)).astype(np.uint8)
        filename = os.path.join(save_dir, '{:04d}.png'.format(i))
        imageio.imwrite(filename, im_8)

    pred_ims = [x.cpu().numpy() for x in pred_ims]
    pred_ims = np.stack(pred_ims, 0)
    return pred_ims


def decayed_learning_rate(step, decay_steps, initial_lr, decay_rate=0.1):
    return initial_lr * (decay_rate ** (step / decay_steps))


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

    coarse_model = Model(xyz_L=args.L_pos, angle_L=args.L_ang)
    fine_model = Model(xyz_L=args.L_pos, angle_L=args.L_ang)

    params = list(coarse_model.parameters()) + list(fine_model.parameters())

    optimizer = torch.optim.Adam(params=params, lr=args.lr, betas=(0.9, 0.999))
    steps = args.iter
    step = 0

    if torch.cuda.is_available():
        coarse_model = coarse_model.cuda()
        fine_model = fine_model.cuda()
        images = torch.Tensor(images).cuda()
        poses = torch.Tensor(poses).cuda()
    elif not args.debug:
        raise ValueError("CUDA is not available")

    # training loop
    for i in range(steps):
        step += 1
        print('========')
        print('Step: ' + str(step))

        step_time = 0.0
        step_time = time.time()

        # don't collect gradients during preprocessing
        # select random image
        im_idx = np.random.choice(train_idx)
        im = images[im_idx]

        print(im.shape)

        # extract projection matrix:
        # (https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera)
        pose = poses[im_idx, :3, :4]

        # both origins and orientations are needed to determine a ray
        r_origins, r_dirs = compute_rays(height, width, f, pose)

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

        if torch.cuda.is_available():
            batch_rays = batch_rays.cuda()
            batch_pixels = batch_pixels.cuda()
        elif not args.debug:
            raise ValueError("CUDA is not available")

        # renders rays into RGB values
        rgb_c, rgb_f = render(batch_rays, coarse_model, fine_model, bounds, args)

        optimizer.zero_grad()

        print(rgb_c.shape)
        print(batch_pixels.shape)
        print(rgb_c)
        loss = torch.mean((rgb_c - batch_pixels) ** 2).item()
        loss += 0.0 if rgb_f is None else torch.mean((rgb_f - batch_pixels) ** 2).item()

        print(loss)

        optimizer.step()

        step_time = time.time() - step_time
        DECAY_RATE = 0.1
        DECAY_SIZE = 1000

        # update learning rate: https://tinyurl.com/48makybr using exponential decay
        lr = decayed_learning_rate(step, DECAY_SIZE * args.lr_decay, args.lr, decay_rate=DECAY_RATE)
        for g in optimizer.param_groups:
            g['lr'] = lr

        with torch.no_grad():
            if step % args.update_freq == 0:
                print('Step: ' + str(step))
                print('Loss: ' + str(loss))
                print('Step time: ' + str(step_time))

                # source: https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                f = r - a  # free inside reserved

                print('Total GPU Memory: ' + str(t))
                print('Reserved GPU Memory: ' + str(r))
                print('Free GPU Memory: ' + str(f))

            if step % args.video_freq == 0:

                if torch.cuda.is_available():
                    render_poses = render_poses.cuda()
                else:
                    raise ValueError("CUDA is not available")

                pred_frames = render_full(render_poses, [height, width, f], args.save_dir, coarse_model, fine_model,
                                          bounds, args)
                imageio.mimwrite(os.path.join(args.save_dir, 'test_vid_{:d}.mp4'.format(step)),
                                 (255 * np.clip(pred_frames[-1], 0, 1)).astype(np.uint8), fps=30, quality=8)
                print('Writing video at', args.os.path.join(args.save_dir, 'test_vid_{:d}.mp4'.format(step)))

            if step % args.save_freq == 0:
                path = os.path.join(args.base_dir, args.name, '{:d}.pt'.format(i))
                torch.save({
                    'iter': step,
                    'model_coarse_dict': coarse_model.state_dict(),
                    'model_fine_dict': fine_model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            print('========')


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

