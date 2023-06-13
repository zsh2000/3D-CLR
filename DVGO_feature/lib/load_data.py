import numpy as np

from .load_hm3d_sample import load_hm3d_sample_data

def load_data(args):

    K, depths = None, None
    near_clip = None


    if args.dataset_type == 'hm3d_sample':
        images, features, poses, render_poses, [H, W, focal], i_test = load_hm3d_sample_data(
                args.datadir)
        hwf = [H, W, focal]
        if not isinstance(i_test, list):
            i_test = [i_test]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        near = 0
        near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, features=features, depths=depths,
        irregular_shape=irregular_shape,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

