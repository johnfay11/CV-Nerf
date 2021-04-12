import numpy as np 
import imageio 
import os 

#what is minify doing???????? looks like it resizes? removes duplicates? is this absolutely necessary?
#should I add factoring back in?********
#TODO: inform thaat the loading of llff data only needs the data directory

#llff loader modified from NERF implementation: https://github.com/bmild/nerf/blob/master/load_llff.py
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
    for file in sorted(os.listdir(imgdir)):
        #TODO: double check this is returning the last 3 files
        if file[-3:] == 'png' or file[-3:] == 'jpg' or file[-3:] == 'JPG':
            images.append(os.path.join(imgdir,file))
    

    # make  sure image size is consistent
    # img_shape = imageio.imread(images[0]).shape
    # poses[:2,4,:] = np.array(img_shape[:2]).reshape([2,1]) #TODO: see how this reshape is happening

    # images_read = []
    # def imread(f):
    #     if f.endswith('png'):
    #         return imageio.imread(f, ignoregamma=True)
    #     else:
    #         return imageio.imread(f)
        
    # imgs = imgs = [imread(f)[...,:3]/255. for f in images]
    # images = np.stack(imgs, -1)  
    images_read = []
    j=0
    for file in images:
        j+=1
        print(j)
        # print(file)
        if j < 5:
            if file[-3:] == 'png':
                i = imageio.imread(file,ignoregamma=True) 
            else:
                i = imageio.imread(file)
        # if j >= 95:
        #     print(file)
            images_read.append(i/255.)

    
    # images_norm = []
    # for i in images_read:
    #     img = i[...,:3]/255.
    #     images_norm.append(img)
    # print('finished')
       # images_read.append(i[...,:3]/255.) #normalize images
        
    images = np.stack(images_read,-1) #TODO: figure out what this does

    return poses, bounds, images

def view_matrix(z,up,pos):
    #TODO: walkthrough
    # z:camera eye - target point
    # up: average pose
    # pos: position of camera
    # math from https://www.3dgep.com/understanding-the-view-matrix/#:~:text=The%20view%20matrix%20on%20the,space%20in%20the%20vertex%20program.
    v2 = z/np.linalg.norm(z) #zaxis

    v0 = np.cross(up,v2) #xaxis
    v0 = v0/np.linalg.norm(v0)

    v1 = np.cross(v2,v0) #yaxis
    v1 = v1/np.linalg.norm(v1)

    m = np.stack([v0,v1,v2,pos],1) #orientation matrix
    return m 

# def point2cam(points,c2w):
#     tt = np.matmul(c2w[:3,:3].T, (points-c2w[:3,:3])[...,np.newaxis])[...,0] #TODO: see what's happening here
#     return tt

def avg_poses(poses):
    #get height, width, focal point (last column in the 3x5 poses matrices)
    hwf = poses[0,:3,-1:] 

    center = poses[:,:3,3].mean(0)

    v2 = poses[:,:3,2].sum(0)
    v2 = v2/np.linalg.norm(v2)

    up = poses[:,:3,1].sum(0)
    
    m = view_matrix(v2,up,center)
    
    c2w = np.concatenate([m,hwf],1)

    #returns camera to world transformation matrix
    return c2w 

def recenter(poses):
    #TODO: see if tou need this
    poses2 = poses+0
    b = np.reshape([0,0,0,1.],[1,4]) #TODO: see what's happening here 
    c2w = avg_poses(poses)
    c2w = np.concatenate([c2w[:3,:4],b],-2)
    b = np.tile(np.reshape(b,[1,1,4]),[poses.shape[0],1,1]) #look at args for reshape, concatenate, np.array, etc.
    poses = np.concatenate([poses[:,:3,:4],b],-2)

    poses = np.linalg.inv(c2w) @ poses
    poses2[:,:3,:4] = poses[:,:3,:4]
    return poses2

def render_path_spiral(c2w,up,rads,focal,zdelta,zrate,rots,N):
    render_poses = []
    r  = np.array(list(rads) + [1.])
    hwf =  c2w[:,4:5] #TODO: check that this is getting the right thing

    for theta in np.linspace(0.,2.*np.pi *rots,N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta),-np.sin(theta),-np.sin(theta),-np.sin(theta*zrate),1.]) * r)
        z = c - np.dot(c2w[:3,:4],np.array([0,0,-focal,1.]))
        z = z / np.linalg.norm(z)
        m = view_matrix(z,up,c)
        render = np.concatenate([m,hwf],1)
        render_poses.append(render)
    return render_poses

def load_llff_data(topdir):
    poses,bounds,images = load_llff(topdir)

    #issues w/ order of rotation matrix in poses + moving variable dim to axis 0
    poses = np.concatenate([poses[:,1:2,:], -poses[:,0:1,:],poses[:,2:,:]],1)
    poses = np.moveaxis(poses,-1,0).astype(np.float32)
    images = np.moveaxis(images,-1,0).astype(np.float32)
    bounds = np.moveaxis(bounds,-1,0).astype(np.float32)

    #rescale 
    sc = 1./(np.min(bounds) * .75)
    poses[:,:3,3] *= sc
    bounds *= sc

    #recenter poses
    poses = recenter(poses) #check size after 

    #find spiral path 
    c2w = avg_poses(poses)

    #avg pose
    up = poses[:,:3,1].sum(0)
    up = up/np.linalg.norm(up)

    #find focus depth: 
    close_d = np.min(bounds)*0.9
    inf_d = np.max(bounds)*5. 
    mean_dz = 1./(((1.-.75)/close_d + .75/inf_d))
    focal = mean_dz

    #radius for path 

    zd = close_d * .2
    tt = poses[:,:3,3] #what does this stand for 
    r = np.percentile(np.abs(tt),90,0)
    c2w_path  = c2w

    #path_zflat?
    #check N_rots and N_views
    render_poses = render_path_spiral(c2w_path,up,r,focal,zd,zrate=.5,rots=2,N=120)

    c2w = avg_poses(poses)
    dist = np.sum(np.square(c2w[:3,3]-poses[:,:3,3]),-1)
    i_test = np.argmin(dist)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)


            #images, poses, render_poses, hwf, i_test, bounds
    return images,poses[:,:3,:4],render_poses,poses[0,:3,-1],i_test, bounds


def main():
    topdir = '/Users/danielawiepert/Downloads/gerrard-hall'
    images,poses,render_poses,hwf,i_test,bounds = load_llff_data(topdir)
    print('outputs')


if __name__ == "__main__":
    main()

