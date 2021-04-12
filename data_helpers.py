import os
import imageio
import json

import torch
import numpy as np
import cv2


# translation by t: https://www.cs.cornell.edu/courses/cs4620/2010fa/lectures/03transforms3d.pdf
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# 3D vector rotation by phi: https://mathworld.wolfram.com/RotationMatrix.html
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# 3D vector rotation by theta: https://mathworld.wolfram.com/RotationMatrix.html
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w

    # reflect x and swap y and z
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


# custom Blender loader modified from NERF implementation: https://github.com/bmild/nerf/blob/master/load_blender.py
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        # iterate over frames and load poses (represented as a transformation matrix)
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # image index slits between train, val, and test
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # unpack camera intrinsics
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])

    # focal length calculations: https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # map poses (40 in total) to a sphere, which gives us a full 360 degree view of a synthetic object
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            # resize image; TODO: determine if we can get around using opencv
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    return imgs, poses, render_poses, [H, W, focal], i_split, [2.0, 6.0]

# custom llff loader modified from NERF implementation: https://github.com/bmild/nerf/blob/master/load_llff.py
# allows us to use synthetic data
def load_llff(topdir):
    #location of the file containing poses and bounds - file created using COLMAP and imgs2poses.py from https://github.com/Fyusion/LLFF
    # format of this file:
    # np array (N,17) - each row, contains all pose matrix data + 2 depth values
    # N = number of images
    fpose = os.path.join(topdir,'poses_bounds.npy') 
    poses_bounds = np.load(fpose) #load in file   
    
    # get only poses 
    # format of poses after reshaping + transposing:
    # (3,5,N)
    # 3x4 camera-to-world transform + columns with [height, width, focal length]
    # transpose handles weird storing of rotation in c2w transform matrix
    poses = poses_bounds[:,:-2] 
    poses = poses.reshape([-1,3,5]).transpose([1,2,0]) 
    
    #get only bounds - depth values
    bounds = poses_bounds[:,-2:]
    bounds = bounds.transpose([1,0]) 

    #create list of images to read (contains image paths)
    imgdir = os.path.join(topdir,'images')
    #for each file in imgdir, if the file ends with png,jpg(JPG), add to list
    images = []
    for file in os.listdir(imgdir):
        #TODO: double check this is returning the last 3 files
        if file[-3:] == 'png' or file[-3:] == 'jpg' or file[-3:] == 'JPG':
            images.append(os.path.join(imgdir,file))
    

    # make  sure image size is consistent
    # img_shape = imageio.imread(images[0]).shape
    # poses[:2,4,:] = np.array(img_shape[:2]).reshape([2,1]) #TODO: see how this reshape is happening

    images_read = []
    for file in images:
        if file[-3:] == 'png':
            i = imageio.imread(file,ignoregamma=True) 
        else:
            i = imageio.imread(file)
        images_read.append(i[...,:3]/255.) #normalize images
        
    images = np.stack(images_read,-1) #TODO: figure out what this does
    return poses, bounds, images