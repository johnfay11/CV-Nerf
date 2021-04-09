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


def main():
    args = parse_settings()

    '''
    images: numpy array of shape (total_ims, im_height, im_width, rgba)
    poses: numpy array of shape (total_ims, 4, 4)
    cam_params: list of the form [H, W, f]
    i_split: list of the form [train_indices, val_indices, test_indices]
    '''
    images, poses, render_poses, cam_params, i_split = load_dataset(args)

    train_idx, val_idx, test_idx = i_split

    # unpack camera intrinsics
    height, width, f = cam_params
    height, width = int(height), int(width)

    model = None
    if torch.cuda.is_available():
        model = model.cuda()
        images = torch.Tensor(images).cuda()
        poses = torch.Tensor(poses).cuda()
        render_poses = torch.Tensor(render_poses).cuda()
    elif not args.debug:
        raise ValueError("CUDA is not available")

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    steps = args.iter

    # training loop
    for i in range(steps):

        # select random image
        im_idx = np.random.choice(train_idx)
        im = images[im_idx]

        # extract projection matrix: (https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera)
        pose = poses[im_idx, :3, :4]
        r_origins, r_dirs = compute_rays(height, width, f, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)


if __name__ == '__main__':
    main()
