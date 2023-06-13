import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import pickle
import glob
 
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()
 
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()
 
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()
 
 
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w
 

from tqdm import tqdm

def load_hm3d_sample_data(basedir, half_res=True, testskip=1):
    half_res = True
    all_imgs = []
    all_poses = []
    all_features = []
    img_dir = '/gpfs/u/home/LMCG/LMCGzhnf/scratch-shared/new_bbox_hm3d/' + basedir

    img_list = glob.glob(os.path.join(img_dir, "*.png"))
    for img_idx in range(len(img_list)):
        if 200 < len(img_list) < 500:
            if img_idx % 2 != 0:
                continue

        elif len(img_list) >= 500:
            if img_idx % 5 != 0:
                continue
        img_name = str(img_idx) + ".png"
        img_file = img_dir +  '/' + img_name
        pos_file = img_dir + '/' + img_name.replace(".png", ".json")
        feat_file = img_dir + '_result/features_' + img_name.replace(".png", ".npy.npz")
        if not os.path.exists(feat_file):
            continue

        img = np.array(cv2.imread(img_file)).astype(np.float32)/255.0
        pos = np.array(json.load(open(pos_file))["pose"]).astype(np.float32)
        try:
            feat = np.load(feat_file)['arr_0'].astype(np.float16).transpose(1,2,0)
        except:
            continue
        all_imgs.append(img)
        all_poses.append(pos)
        all_features.append(feat)
    
    imgs = np.stack(all_imgs, 0)[...,:3]

    imgs[...,:3] = np.clip(imgs[...,:3], a_min=None,a_max=0.95)

    poses = np.stack(all_poses, 0)

    features = np.stack(all_features, 0)
    i_test = [10]
 
    H, W = imgs[0].shape[:2]
    camera_angle_x = np.radians(90)

    focal = 256.0 / np.tan(np.deg2rad(90.0) / 2)

    render_poses = all_poses
    render_poses = np.stack(render_poses, 0)
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
 
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, features, poses, render_poses, [H, W, focal], i_test
 
