import configargparse
import torch

from data_helpers import load_blender_data
from model import OutputModel


def parse_settings():
    # use standard config parser
    parser = configargparse.ArgumentParser('args')

    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--name', type=str)
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--data_dir', type=str)

    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='llff or blender')

    parser.add_argument('--n_rays', type=int, default=4096)
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--n_fine_samples', type=int, default=0)

    # position encoding dimensions
    parser.add_argument('--L_pos', type=int, default=10)
    parser.add_argument('--L_ang', type=int, default=4)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--iter', type=int )

    # we might not need this parameter (could set to size of dataset)
    parser.add_argument("--testskip", type=int, default=8)

    # this is useful for reducing the size of blender models
    parser.add_argument('--half_res', action='store_true')
    return parser.parse_args()


def load_dataset(args):
    if args.dataset_type != 'llff' and args.dataset_type != 'blender':
        raise ValueError('Invalid data type. Must be one of llff or blender.')

    # here we can probably plug in the LLFF library and blender extensions.
    # TODO: return images, poses, poses to render, camera intrinsics, maybe other information?

    if args.dtype == 'blender':
        return load_blender_data(args.data_dir, half_res=args.half_res, testskip=args.testskip)
    else:
        raise NotImplementedError("please implement me! (LLFF)")


def main():
    config = parse_settings()
    args = config.parse_args()

    images, poses, cam_params, render_poses, test_im = load_dataset(args)

    # unpack camera intrinsics
    height, width, f = cam_params

    model = None
    if torch.cuda.is_available():
        model = model.cuda()
        images = torch.Tensor(images).cuda()
        poses = torch.Tensor(images).cuda()
    else:
        raise ValueError("CUDA is not available")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    steps = args.iter

    # training loop
    for i in range(steps):
        pass


if __name__ == 'main':
    main()
