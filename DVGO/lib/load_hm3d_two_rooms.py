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

def load_hm3d_two_rooms(basedir, half_res=True, testskip=1):
    # with open("./data/habitat_test_apt_2.pkl", "rb") as loader_pkl:
    # with open(os.path.join("/data/vision/torralba/scratch/chuang/L/nerf-pytorch/processed_pkls/", basedir), "rb") as loader_pkl:
    #     data = pickle.load(loader_pkl)
    # with open("/home/aluo/Tools/nerf-pytorch/data/habitat.pkl", "rb") as loader_pkl:
    #     data = pickle.load(loader_pkl)
    # with open("/home/aluo/Tools/nerf-pytorch/data/habitat_test.pkl", "rb") as loader_pkl:
    #     data2 = pickle.load(loader_pkl)
    all_imgs = []
    all_poses = []
    #jsn_file = '/gpfs/u/home/LMCG/LMCGzhnf/scratch/lxs/habitat-lab/sample_25_three_rooms/'+basedir+'.json'
    #img_dir = '/gpfs/u/home/LMCG/LMCGzhnf/scratch/lxs/habitat-lab/single_rooms/'

    root_dir = '/gpfs/u/home/LMCG/LMCGzhnf/scratch-shared/new_bbox_hm3d/'
    #    img_dir = '/gpfs/u/home/LMCG/LMCGzhnf/scratch-shared/new_scenes_temp/' + basedir + '_add_temp'
    x = basedir.split("_")

    img_dir_1 = root_dir + x[0] + '_' + x[1]
    img_dir_2 = root_dir + x[0] + '_' + x[2]
    img_list_1 = glob.glob(os.path.join(img_dir_1, "*.png"))
    img_list_2 = glob.glob(os.path.join(img_dir_2, "*.png"))
#    img_dir = '/gpfs/u/home/LMCG/LMCGzhnf/scratch/lxs/habitat-lab/whole_houses/'
#    ky = basedir
#    ids = np.random.choice(1000, 250)
    # features = np.zeros((500, 256, 256, 512), dtype=np.float16)
    # counts = [0]i
    #jsn_data = json.load(open(jsn_file))#{basedir: [_ for _ in range(1000)]} #json.load(open(jsn_file))
    #for ky in jsn_data.keys():
    cnt = 0
    for img_name in tqdm(img_list_1): #jsn_data[ky]):
        cnt += 1
        if cnt%2==0:
            continue
        img_file = img_name
        pos_file = img_file.replace(".png", ".json")
        img = np.array(cv2.imread(img_file)).astype(np.float32)/255.0
        pos = np.array(json.load(open(pos_file))["pose"]).astype(np.float32)
            # pose = pose @ np.array(([[1.0000000000000002, 0.0, 0.0, 0.0], [0.0, -1.0000000000000002, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
        all_imgs.append(img)
        all_poses.append(pos)


    cnt = 0 
    for img_name in tqdm(img_list_2): #jsn_data[ky]):
        cnt += 1
        if cnt%2==0:
            continue
        img_file = img_name
        pos_file = img_file.replace(".png", ".json")
        img = np.array(cv2.imread(img_file)).astype(np.float32)/255.0
        pos = np.array(json.load(open(pos_file))["pose"]).astype(np.float32)
            # pose = pose @ np.array(([[1.0000000000000002, 0.0, 0.0, 0.0], [0.0, -1.0000000000000002, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
        all_imgs.append(img)
        all_poses.append(pos)

    # all_poses = json.load(open("/home/evelyn/Desktop/CLIP-NERF/LargeScaleNeRFPytorch-feat-zelin-incorporate_zesong_block/replica_all/train/00/cameras.json"))
    # all_poses = [(np.array(pose['Rt'] @ np.array(pose['K']))).astype(np.float32) for pose in all_poses]
        # all_imgs.append(np.array(data[k][0]).astype(np.float32)/255.0)
        # all_poses.append(np.array(data[k][2]).astype(np.float32))

        # base = basedir.replace("_dump_test_real_tester.pkl", "")
        # feature = np.load("/data/vision/torralba/scratch/chuang/L/nerf-pytorch/processed_pkls/img/%s/features/%d_feature.npy"%(base, k)).astype(np.float16).transpose(1,2,0)
        # features[k] = feature
        
    # for k in range(360):
    #     all_imgs.append(np.array(data2[k][0]).astype(np.float32)/255.0)
    #     all_poses.append(np.array(data2[k][2]).astype(np.float32))
 
    # counts = [0, 200, 300, 500]
    # counts = [0, 150, 150, 150]
    imgs = np.stack(all_imgs, 0)[...,:3]

    # print (imgs.shape)
    # dump= np.all(imgs[:,:,:,:3]!=0, axis=3).astype(np.uint8)
    # imgs 
    # print (dump.shape)

    imgs[...,:3] = np.clip(imgs[...,:3], a_min=None,a_max=0.95)

    poses = np.stack(all_poses, 0)
    # features = np.transpose(features, (0, 2, 3, 1))
    i_test = [10]
    # i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
 
    H, W = imgs[0].shape[:2]
    camera_angle_x = np.radians(90)
    # focal = .5 * W / np.tan(.5 * camera_angle_x)

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
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
 
    return imgs, poses, render_poses, [H, W, focal], i_test
 
